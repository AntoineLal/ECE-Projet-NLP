"""Bonus B : chatbot adaptatif. Il modifie le profil et relance le moteur NLP existant.

Architecture (à retenir pour la soutenance)
-------------------------------------------
1. `interpret`        lit le message et produit un CONTRAT : un dictionnaire aux champs fixes.
                      Deux lecteurs : les règles (toujours) et Gemini (en JSON). Le contrat est validé par
                      `validate_contract` : clés inconnues, valeurs hors listes ou hors bornes sont rejetées.
2. `apply_contract`   fusionne le contrat dans le profil. Les exclusions strictes (« je ne veux pas », « aucun »,
                      « uniquement ») et les préférences souples (« je préfère », « j'aime ») sont stockées dans
                      des champs distincts. Une valeur explicitement corrigée remplace l'ancienne.
3. `engine.recommend` moteur existant, NON remplacé : il applique les filtres stricts AVANT le classement.
4. `compare_results`  compare le top 3 avant / après et explique chaque entrée et sortie avec les critères du moteur.
5. `diagnose_empty`   si aucun sport n'est admissible, indique quelle contrainte retirer, sans jamais la retirer soi-même.

Le LLM ne calcule aucun score et ne produit aucun classement : il ne fait que lire le message.
Le texte des messages n'est plus ajouté au profil : les modifications passent uniquement par des champs structurés.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import replace

from bonus_llm import CallBudget, LLMError, Transport, call_llm
from engine import ALL_GOALS, BLOCK_LABELS, BLOCKS, FAMILIES, UserProfile
from extraction import extract_signals
from llm_config import LLMSettings
from preprocess import fold
from telemetry import get_logger

FORMATS = ["individuel", "duo", "équipe", "groupe"]
ENVIRONMENTS = ["intérieur", "extérieur"]
LEVELS = ["débutant", "occasionnel", "régulier", "confirmé"]
IMPACTS = ["faible", "moyenne", "élevée"]
ZONES_ALL = ["dos", "épaules", "bras", "abdominaux", "cuisses", "hanches et fessiers", "mollets", "pieds et chevilles",
             "mains et poignets", "avant-bras"]
JOINTS = ["genoux", "hanches", "chevilles", "pieds", "lombaires", "cervicales", "epaules", "coudes", "poignets"]
INTENTIONS = ["mise_a_jour", "question", "autre"]
QUESTION_TYPES = ["pourquoi", "comparer", "criteres"]

# ------------------------------------------------------------------ contrat : structure fixe
CONTRACT_SHAPE = {
    "intention": "enum:INTENTIONS",
    "objectifs_ajoutes": "list:goals",
    "preferences_souples": {"familles": "list:families", "formats": "list:formats", "environnement": "enum:env|null"},
    "exclusions_strictes": {"familles": "list:families", "formats": "list:formats", "categories": "list:categories",
                            "sports": "list:sports", "environnements": "list:env", "eau": "bool|null",
                            "contact": "bool|null", "seul_uniquement": "bool|null"},
    "acceptations": {"familles": "list:families", "formats": "list:formats", "categories": "list:categories",
                     "sports": "list:sports", "environnements": "list:env", "eau": "bool|null", "contact": "bool|null"},
    "contraintes_physiques": {"impacts": "enum:impacts|null", "zones_a_menager": "list:zones",
                              "articulations_douloureuses": "list:joints"},
    "budget_max_eur": "int0-300|levée|null",
    "seances_par_semaine": "int1-7|null",
    "duree_minutes": "int20-180|null",
    "niveau": "enum:levels|null",
    "intensite": "int-1-1|null",
    "corrections": "list:corrections",
    "precisions": "list:str",
    "question": "dict:question|null",
}


def vocabulary(engine) -> dict:
    """Valeurs admises pour les champs à listes : sports et catégories viennent du dataset lui-même."""
    cats = sorted({t for lst in engine.df["categories_l"] for t in lst})
    return {"sports": list(engine.df["nom_sport"]), "categories": cats}


def _fail(rejected: list, path: str, why: str):
    rejected.append(f"{path} ({why})")
    return None


def _str_list(value, allowed, path, rejected, max_len=None):
    if value is None:
        return []
    if not isinstance(value, list):
        return _fail(rejected, path, "liste attendue")
    out = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            _fail(rejected, f"{path}[{str(item)[:30]}]", "valeur hors liste")
            continue
        if item not in out:
            out.append(item)
    return out


def _bool(value, path, rejected):
    if value is None or isinstance(value, bool):
        return value
    return _fail(rejected, path, "booléen attendu")


def _int(value, lo, hi, path, rejected):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return _fail(rejected, path, "entier attendu")
    if not lo <= value <= hi:
        return _fail(rejected, path, f"hors bornes {lo}-{hi}")
    return value


def validate_contract(raw, vocab: dict) -> tuple[dict, list[str]]:
    """Valide le contrat (JSON du LLM ou règles). Retourne (contrat propre, motifs de rejet).

    Une clé inconnue, une valeur hors liste ou hors bornes est rejetée et signalée : elle n'atteint jamais le profil.
    """
    rejected: list[str] = []
    if not isinstance(raw, dict):
        return _empty_contract(), ["réponse non structurée"]
    for key in raw:
        if key not in CONTRACT_SHAPE:
            rejected.append(f"{key} (champ inconnu)")
    c = _empty_contract()
    intention = raw.get("intention", "autre")
    if intention in INTENTIONS:
        c["intention"] = intention
    else:
        _fail(rejected, "intention", "valeur inconnue")
    c["objectifs_ajoutes"] = _str_list(raw.get("objectifs_ajoutes"), ALL_GOALS, "objectifs_ajoutes", rejected) or []

    ps = raw.get("preferences_souples") or {}
    if isinstance(ps, dict):
        c["preferences_souples"]["familles"] = _str_list(ps.get("familles"), list(FAMILIES), "preferences_souples.familles", rejected) or []
        c["preferences_souples"]["formats"] = _str_list(ps.get("formats"), FORMATS, "preferences_souples.formats", rejected) or []
        env = ps.get("environnement")
        c["preferences_souples"]["environnement"] = env if env in ENVIRONMENTS else (
            None if env is None else _fail(rejected, "preferences_souples.environnement", "intérieur ou extérieur"))
    es = raw.get("exclusions_strictes") or {}
    if isinstance(es, dict):
        ex = c["exclusions_strictes"]
        ex["familles"] = _str_list(es.get("familles"), list(FAMILIES), "exclusions_strictes.familles", rejected) or []
        ex["formats"] = _str_list(es.get("formats"), FORMATS, "exclusions_strictes.formats", rejected) or []
        ex["categories"] = _str_list(es.get("categories"), vocab["categories"], "exclusions_strictes.categories", rejected) or []
        ex["sports"] = _str_list(es.get("sports"), vocab["sports"], "exclusions_strictes.sports", rejected) or []
        ex["environnements"] = _str_list(es.get("environnements"), ENVIRONMENTS, "exclusions_strictes.environnements", rejected) or []
        ex["eau"] = _bool(es.get("eau"), "exclusions_strictes.eau", rejected)
        ex["contact"] = _bool(es.get("contact"), "exclusions_strictes.contact", rejected)
        ex["seul_uniquement"] = _bool(es.get("seul_uniquement"), "exclusions_strictes.seul_uniquement", rejected)
    ac = raw.get("acceptations") or {}
    if isinstance(ac, dict):
        a = c["acceptations"]
        a["familles"] = _str_list(ac.get("familles"), list(FAMILIES), "acceptations.familles", rejected) or []
        a["formats"] = _str_list(ac.get("formats"), FORMATS, "acceptations.formats", rejected) or []
        a["categories"] = _str_list(ac.get("categories"), vocab["categories"], "acceptations.categories", rejected) or []
        a["sports"] = _str_list(ac.get("sports"), vocab["sports"], "acceptations.sports", rejected) or []
        a["environnements"] = _str_list(ac.get("environnements"), ENVIRONMENTS, "acceptations.environnements", rejected) or []
        a["eau"] = _bool(ac.get("eau"), "acceptations.eau", rejected)
        a["contact"] = _bool(ac.get("contact"), "acceptations.contact", rejected)
    cp = raw.get("contraintes_physiques") or {}
    if isinstance(cp, dict):
        imp = cp.get("impacts")
        c["contraintes_physiques"]["impacts"] = imp if imp in IMPACTS else (
            None if imp is None else _fail(rejected, "contraintes_physiques.impacts", "faible, moyenne ou élevée"))
        c["contraintes_physiques"]["zones_a_menager"] = _str_list(cp.get("zones_a_menager"), ZONES_ALL, "contraintes_physiques.zones_a_menager", rejected) or []
        c["contraintes_physiques"]["articulations_douloureuses"] = _str_list(cp.get("articulations_douloureuses"), JOINTS, "contraintes_physiques.articulations_douloureuses", rejected) or []
    budget = raw.get("budget_max_eur")
    if budget == "levée":
        c["budget_max_eur"] = "levée"
    else:
        c["budget_max_eur"] = _int(budget, 0, 300, "budget_max_eur", rejected)
    c["seances_par_semaine"] = _int(raw.get("seances_par_semaine"), 1, 7, "seances_par_semaine", rejected)
    c["duree_minutes"] = _int(raw.get("duree_minutes"), 20, 180, "duree_minutes", rejected)
    niv = raw.get("niveau")
    c["niveau"] = niv if niv in LEVELS else (None if niv is None else _fail(rejected, "niveau", "valeur inconnue"))
    c["intensite"] = _int(raw.get("intensite"), -1, 1, "intensite", rejected)
    corr = raw.get("corrections") or []
    if isinstance(corr, list):
        for item in corr[:5]:
            if isinstance(item, dict) and all(isinstance(item.get(k), str) for k in ("champ", "ancienne", "nouvelle")):
                c["corrections"].append({k: item[k][:80] for k in ("champ", "ancienne", "nouvelle")})
            else:
                _fail(rejected, "corrections", "format attendu : champ, ancienne, nouvelle")
    precis = raw.get("precisions") or []
    if isinstance(precis, list):
        c["precisions"] = [str(p)[:200] for p in precis if isinstance(p, str)][:3]
    q = raw.get("question")
    if isinstance(q, dict) and q.get("type") in QUESTION_TYPES:
        sports = q.get("sports") if isinstance(q.get("sports"), list) else []
        c["question"] = {"type": q["type"], "sports": [s for s in sports if isinstance(s, str) and s in vocab["sports"]][:2]}
    elif q is not None:
        _fail(rejected, "question", "type attendu : pourquoi, comparer ou criteres")
    return c, rejected


def _empty_contract() -> dict:
    return {
        "intention": "autre", "objectifs_ajoutes": [],
        "preferences_souples": {"familles": [], "formats": [], "environnement": None},
        "exclusions_strictes": {"familles": [], "formats": [], "categories": [], "sports": [], "environnements": [],
                                "eau": None, "contact": None, "seul_uniquement": None},
        "acceptations": {"familles": [], "formats": [], "categories": [], "sports": [], "environnements": [],
                         "eau": None, "contact": None},
        "contraintes_physiques": {"impacts": None, "zones_a_menager": [], "articulations_douloureuses": []},
        "budget_max_eur": None, "seances_par_semaine": None, "duree_minutes": None, "niveau": None, "intensite": None,
        "corrections": [], "precisions": [], "question": None,
    }


def merge_contracts(a: dict, b: dict) -> dict:
    """Fusion de deux contrats validés. Listes : union. Valeurs simples : a (LLM) prime, sinon b (règles)."""
    def merge(x, y):
        if isinstance(x, dict):
            return {k: merge(x.get(k), (y or {}).get(k)) for k in x}
        if isinstance(x, list):
            items = (x or []) + (y or [])
            if any(isinstance(i, dict) for i in items):  # corrections : pas de déduplication
                return items
            return list(dict.fromkeys(items))
        return x if x not in (None, "autre") else y
    out = merge(a, b)
    out["corrections"] = (a.get("corrections") or []) + (b.get("corrections") or [])
    out["precisions"] = list(dict.fromkeys((a.get("precisions") or []) + (b.get("precisions") or [])))[:3]
    out["question"] = a.get("question") or b.get("question")
    out["intention"] = a.get("intention") if a.get("intention") not in (None, "autre") else b.get("intention", "autre")
    return out


def _filled(value) -> bool:
    """Une valeur est « renseignée » si elle n'est ni None ni une liste vide. False compte : c'est un refus."""
    return value is not None and value != []


def contract_is_empty(c: dict) -> bool:
    values = [c["objectifs_ajoutes"], c["preferences_souples"]["familles"], c["preferences_souples"]["formats"],
              c["preferences_souples"]["environnement"], *c["exclusions_strictes"].values(),
              *c["acceptations"].values(), *c["contraintes_physiques"].values(), c["budget_max_eur"],
              c["seances_par_semaine"], c["duree_minutes"], c["niveau"], c["intensite"], c["question"]]
    return not any(_filled(v) for v in values)


# ------------------------------------------------------------------ lecteur 1 : règles (toujours disponible)
ACTIVITY_RULES = [
    # (motif, champ strict, valeur)
    (r"aquatique|dans l.eau|natation|nager|piscine|nautique", "eau", True),
    (r"sports? collectifs?|sports? d.equipe|jeux? d.equipe|sports? en equipe", "categories", "sport collectif"),
    (r"course a pied|courir|\bcourse\b|footing|jogging|running", "categories", "course"),
    (r"sauts?\b|sauter", "categories", "saut"),
    (r"sports? de combat|boxe|judo|karate|arts? martiaux", "familles", "Combat"),
    (r"raquette|tennis|padel|badminton|squash", "familles", "Raquette"),
    (r"sports? de contact|contact physique|contact avec", "contact", False),
]
NEGATION = r"ne (?:veux|voudrais|souhaite|souhaiterais|veut|peux) (?:pas|aucun|aucune|plus)|aucun|aucune|jamais|je refuse|eviter|evite|exclu|sans\b|pas de\b|pas d'|uniquement|seulement|seul"
ACCEPT = r"accepte|j'accepte|je veux bien|ca me va|pas de probleme|finalement.*(?:veux|peux|accepte)"
SOFT = r"n'aime pas|n aime pas|pas trop|moins"
PRECISION_TRIGGER = r"n'aime pas|n aime pas|ne veux pas|je prefere|je veux plutot|eviter"


def _segments(message: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.;!?]|\bmais\b|\bet\b(?=\s+je\b)", fold(message)) if s.strip()]


def rules_contract(message: str, vocab: dict) -> dict:
    """Lecture par règles : mêmes champs que le contrat du LLM. Une négation rend l'exclusion stricte."""
    c = _empty_contract()
    sig = extract_signals(message)
    c["objectifs_ajoutes"] = sorted(g for g in sig.goals if g in ALL_GOALS)
    for seg in _segments(message):
        negated = re.search(NEGATION, seg) is not None
        accepted = re.search(ACCEPT, seg) is not None
        for pattern, field, value in ACTIVITY_RULES:
            if not re.search(pattern, seg):
                continue
            if accepted:
                target = c["acceptations"]
            elif negated and not re.search(r"n'?aime pas", seg):
                target = c["exclusions_strictes"]
            elif re.search(SOFT, seg):
                target = None  # « je n'aime pas » : préférence négative souple, traitée comme une précision
            else:
                target = c["preferences_souples"]
            if target is None:
                c["precisions"].append(f"Vous n'aimez pas « {value if isinstance(value, str) else field} » : "
                                       "voulez-vous l'exclure complètement ou simplement le défavoriser ?")
                continue
            if field == "eau":
                target["eau"] = value if target is c["acceptations"] else True
            elif field == "contact":
                target["contact"] = value if target is c["acceptations"] else False
            elif field == "categories":
                target["categories"] = list(dict.fromkeys(target["categories"] + [value]))
            elif field == "familles":
                target["familles"] = list(dict.fromkeys(target["familles"] + [value]))
        if re.search(r"individuel|seul\b|solo|tout seul", seg):
            if re.search(r"uniquement|seulement|que des", seg):
                c["exclusions_strictes"]["seul_uniquement"] = True
            elif not negated:
                c["preferences_souples"]["formats"] = list(dict.fromkeys(c["preferences_souples"]["formats"] + ["individuel"]))
        if re.search(r"sans equipe|pas en equipe|pas d'equipe", seg):
            c["exclusions_strictes"]["formats"] = list(dict.fromkeys(c["exclusions_strictes"]["formats"] + ["équipe"]))
        for word, env in (("interieur|en salle|dedans", "intérieur"), ("exterieur|dehors|plein air", "extérieur")):
            if re.search(word, seg):
                if negated and re.search(r"uniquement|seulement|ne veux pas|pas", seg):
                    other = "extérieur" if env == "intérieur" else "intérieur"
                    if re.search(r"uniquement|seulement", seg):
                        c["exclusions_strictes"]["environnements"] = list(dict.fromkeys(
                            c["exclusions_strictes"]["environnements"] + [other]))
                    else:
                        c["exclusions_strictes"]["environnements"] = list(dict.fromkeys(
                            c["exclusions_strictes"]["environnements"] + [env]))
                elif not negated:
                    if c["preferences_souples"]["environnement"] and c["preferences_souples"]["environnement"] != env:
                        c["precisions"].append("Vous mentionnez les deux environnements : lequel préférez-vous ?")
                    c["preferences_souples"]["environnement"] = env
        if re.search(r"impacts? (?:eleves?|forts?|importants?|marques?)|articulations? fragiles?", seg):
            c["contraintes_physiques"]["impacts"] = "faible"
        if re.search(r"moins intense|plus doux|plus douce|plus calme|moins fatigant|plus leger", seg):
            c["intensite"] = -1
        elif re.search(r"plus intense|plus dur|plus soutenu", seg):
            c["intensite"] = 1
        m = re.search(r"\b(\d)\s*(?:seances?|fois)\b", seg)
        if m and 1 <= int(m.group(1)) <= 7:
            c["seances_par_semaine"] = int(m.group(1))
        m = re.search(r"\b(\d{2,3})\s*(?:min|minutes)\b", seg)
        if m and 20 <= int(m.group(1)) <= 180:
            c["duree_minutes"] = int(m.group(1))
        if re.search(r"budget|par seance|euros?|€", seg):
            m = re.search(r"(\d{1,3})\s*(?:euros?|€)", seg)
            if m and 0 <= int(m.group(1)) <= 300:
                c["budget_max_eur"] = int(m.group(1))
            if re.search(r"plus de limite|sans limite|plus limite", seg):
                c["budget_max_eur"] = "levée"
        if re.search(r"debutant|je reprends", seg):
            c["niveau"] = "débutant"
        elif re.search(r"confirme|haut niveau", seg):
            c["niveau"] = "confirmé"
        if re.search(r"blesse|mal au|mal a |douleur|fragile", seg):
            for joint, rx in (("genoux", "genou"), ("chevilles", "cheville"), ("hanches", "hanche"),
                              ("lombaires", "dos|lombaire"), ("epaules", "epaule"), ("coudes", "coude"),
                              ("poignets", "poignet"), ("cervicales", "nuque|cervical")):
                if re.search(rx, seg):
                    c["contraintes_physiques"]["articulations_douloureuses"].append(joint)
        if re.search(PRECISION_TRIGGER, seg) and not any(
                [c["exclusions_strictes"]["categories"], c["exclusions_strictes"]["familles"], c["exclusions_strictes"]["eau"],
                 c["preferences_souples"]["familles"], c["preferences_souples"]["formats"], c["preferences_souples"]["environnement"],
                 c["exclusions_strictes"]["environnements"], c["exclusions_strictes"]["formats"], c["intensite"],
                 c["contraintes_physiques"]["impacts"], c["budget_max_eur"], c["seances_par_semaine"], c["precisions"]]):
            c["precisions"].append("Pouvez-vous préciser ce que vous souhaitez éviter ou privilégier : un type d'activité, "
                                   "un format (seul, en équipe…) ou un environnement (intérieur, extérieur) ?")
    if not contract_is_empty(c):
        c["intention"] = "mise_a_jour"
    if c["precisions"]:
        c["precisions"] = list(dict.fromkeys(c["precisions"]))[:3]
    return c


# ------------------------------------------------------------------ lecteur 2 : Gemini (JSON strict)
CONTRACT_PROMPT = """Tu lis UN message d'un utilisateur d'une application de recommandation sportive et tu le traduis
en un objet JSON qui respecte EXACTEMENT ce schéma. Tu réponds uniquement par le JSON, sans texte autour.

Définitions importantes :
- exclusions_strictes : refus explicites et définitifs (« je ne veux pas », « aucun », « jamais », « sans »,
  « uniquement », « éviter »). Ce sont des contraintes dures.
- preferences_souples : goûts exprimés (« je préfère », « j'aime », « plutôt »). Ce sont des critères de score, pas des interdits.
- acceptations : levée d'une exclusion antérieure (« finalement j'accepte… », « je veux bien… »).
- corrections : chaque fois qu'une information REMPLACE une précédente (ex. « finalement extérieur » après « intérieur »).
- precisions : une question à poser SEULEMENT si le message est ambigu ; sinon liste vide.
- N'invente jamais une information absente du message. Ne donne aucun score ni classement.

Schéma (valeurs autorisées entre crochets) :
{
 "intention": "mise_a_jour" | "question" | "autre",
 "objectifs_ajoutes": [OBJECTIFS],
 "preferences_souples": {"familles": [FAMILLES], "formats": [FORMATS], "environnement": "intérieur" | "extérieur" | null},
 "exclusions_strictes": {"familles": [FAMILLES], "formats": [FORMATS], "categories": [CATEGORIES], "sports": [SPORTS],
                         "environnements": ["intérieur" | "extérieur"], "eau": true | null, "contact": false | null,
                         "seul_uniquement": true | null},
 "acceptations": {"familles": [FAMILLES], "formats": [FORMATS], "categories": [CATEGORIES], "sports": [SPORTS],
                  "environnements": [...], "eau": false | null, "contact": true | null},
 "contraintes_physiques": {"impacts": "faible" | "moyenne" | "élevée" | null, "zones_a_menager": [ZONES],
                           "articulations_douloureuses": [ARTICULATIONS]},
 "budget_max_eur": entier 0-300 | "levée" | null,
 "seances_par_semaine": entier 1-7 | null,
 "duree_minutes": entier 20-180 | null,
 "niveau": "débutant" | "occasionnel" | "régulier" | "confirmé" | null,
 "intensite": -1 | 0 | 1 | null,            (-1 = moins intense, 1 = plus intense)
 "corrections": [{"champ": "...", "ancienne": "...", "nouvelle": "..."}],
 "precisions": ["question"],
 "question": null | {"type": "pourquoi" | "comparer" | "criteres", "sports": [SPORTS]}
}
Pour « sport collectif » utilise la catégorie « sport collectif » (dans categories) ; pour la course à pied, « course ».
"""


def _prompt(vocab: dict, profile: UserProfile) -> str:
    text = CONTRACT_PROMPT.replace("OBJECTIFS", ", ".join(ALL_GOALS)).replace("FAMILLES", ", ".join(FAMILIES))
    text = text.replace("FORMATS", ", ".join(FORMATS)).replace("ZONES", ", ".join(ZONES_ALL))
    text = text.replace("ARTICULATIONS", ", ".join(JOINTS)).replace("CATEGORIES", "une catégorie du dataset")
    text = text.replace("SPORTS", "un sport du dataset")
    current = {"formats_souhaites": profile.formats, "environnement_souhaite": profile.env_preference,
               "familles_exclues": profile.excluded_families, "categories_exclues": profile.excluded_categories,
               "sans_eau": profile.no_water, "sans_contact": profile.no_contact, "seul_uniquement": profile.only_solo,
               "budget_max_eur": profile.budget_seance_max, "seances_par_semaine": profile.sessions_per_week}
    return text + "\nProfil actuel (contexte seulement, ne pas le recopier) : " + json.dumps(current, ensure_ascii=False)


def llm_contract(message: str, profile: UserProfile, vocab: dict, settings: LLMSettings,
                 transport: Transport | None = None) -> dict:
    """Demande le contrat à Gemini. Lève LLMError ou ValueError si la réponse n'est pas du JSON exploitable."""
    text = call_llm(_prompt(vocab, profile), f"Message : « {message[:600]} »", settings, json_mode=True, transport=transport)
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        raise ValueError("réponse sans JSON")
    return json.loads(match.group(0))


def interpret(message: str, profile: UserProfile, settings: LLMSettings, vocab: dict, *,
              budget: CallBudget | None = None, transport: Transport | None = None) -> dict:
    """Lit le message : règles toujours, Gemini si disponible. Retourne contrat validé, rejets, source et note."""
    rules, rules_rej = validate_contract(rules_contract(message, vocab), vocab)
    use_llm = settings.chat_use_llm and settings.provider != "off" and not (
        settings.provider == "gemini" and not settings.has_gemini_key)
    note, rejected, source = "", list(rules_rej), "règles"
    contract = rules
    if use_llm and (budget is None or budget.available()):
        if budget is not None:
            budget.consume()
        try:
            llm, llm_rej = validate_contract(llm_contract(message, profile, vocab, settings, transport), vocab)
            contract = merge_contracts(llm, rules)
            rejected += llm_rej
            source = "IA + règles"
        except (LLMError, ValueError) as e:
            kind = getattr(e, "kind", "réponse illisible")
            get_logger().warning("chatbot : lecture IA indisponible (%s), règles seules", kind)
            note = "Lecture par règles : l'IA n'était pas disponible."
    elif use_llm:
        note = "Lecture par règles : limite d'appels IA atteinte pour cette session."
    return {"contract": contract, "rejected": rejected, "source": source, "note": note}


# ------------------------------------------------------------------ fusion dans le profil
def _scalar_change(label, old, new, changes):
    if old != new:
        changes.append(f"{label} : {old if old is not None else '—'} → {new if new is not None else '—'}")


def apply_contract(profile: UserProfile, c: dict) -> tuple[UserProfile, list[str], list[str]]:
    """Applique le contrat validé. Retourne (nouveau profil, changements, contradictions résolues).

    Règles :
    - une valeur explicitement corrigée remplace l'ancienne (environnement, formats) ;
    - une exclusion stricte retire la préférence souple opposée (et inversement pour une acceptation) ;
    - ce qui n'est pas mentionné est conservé.
    """
    changes, contradictions = [], []
    new = {}
    fams = list(profile.families)
    exf = list(profile.excluded_families)
    fmts = list(profile.formats)
    exfmt = list(profile.excluded_formats)
    excat = list(profile.excluded_categories)
    exspo = list(profile.excluded_sports)
    exenv = list(profile.excluded_environments)
    avoid = list(profile.avoid_families)
    goals = list(profile.goals)
    zones = list(profile.spare_zones)
    joints = list(profile.injured_joints)

    # objectifs
    for g in c["objectifs_ajoutes"]:
        if g not in goals:
            goals.append(g)
    # préférences souples
    ps = c["preferences_souples"]
    if ps["environnement"] and ps["environnement"] != profile.env_preference:
        old = profile.env_preference or "aucune"
        changes.append(f"Environnement préféré : {old} → {ps['environnement']}")
        if old != "aucune":
            contradictions.append(f"« {old} » remplacé par « {ps['environnement']} »")
        new["env_preference"] = ps["environnement"]
        if ps["environnement"] in exenv:
            exenv.remove(ps["environnement"])
    if ps["formats"]:
        if sorted(ps["formats"]) != sorted(fmts):
            changes.append(f"Formats préférés : {', '.join(fmts) or '—'} → {', '.join(ps['formats'])}")
        fmts = list(ps["formats"])
        for f in fmts:
            if f in exfmt:
                exfmt.remove(f)
                contradictions.append(f"exclusion du format « {f} » levée par votre préférence")
    for f in ps["familles"]:
        if f not in fams:
            fams.append(f)
            changes.append(f"Famille appréciée : + {f}")
        if f in exf:
            exf.remove(f)
            contradictions.append(f"exclusion de « {f} » levée")
        if f in avoid:
            avoid.remove(f)
    # exclusions strictes
    ex = c["exclusions_strictes"]
    for f in ex["familles"]:
        if f in fams:
            fams.remove(f)
            contradictions.append(f"« {f} » retiré des préférences souples")
        if f not in exf:
            exf.append(f)
            changes.append(f"Exclu : famille {f}")
    for f in ex["formats"]:
        if f in fmts:
            fmts.remove(f)
            contradictions.append(f"format « {f} » retiré des préférences")
        if f not in exfmt:
            exfmt.append(f)
            changes.append(f"Exclu : format {f}")
    for cat in ex["categories"]:
        if cat not in excat:
            excat.append(cat)
            changes.append(f"Exclu : catégorie « {cat} »")
    for sp in ex["sports"]:
        if sp not in exspo:
            exspo.append(sp)
            changes.append(f"Exclu : sport {sp}")
    for env in ex["environnements"]:
        if env not in exenv:
            exenv.append(env)
            changes.append(f"Exclu : environnement {env}")
        if profile.env_preference == env:
            new["env_preference"] = None
            contradictions.append(f"préférence « {env} » retirée (exclusion stricte)")
    if ex["eau"] is True and not profile.no_water:
        new["no_water"] = True
        changes.append("Exclu : activités dans l'eau")
    if ex["contact"] is False and not profile.no_contact:
        new["no_contact"] = True
        changes.append("Exclu : sports à contact physique")
    if ex["seul_uniquement"] is True:
        if not profile.only_solo:
            new["only_solo"] = True
            changes.append("Pratique seul uniquement")
        # « uniquement individuels » exclut aussi les sports classés « sport collectif » dans le dataset
        if "sport collectif" not in excat:
            excat.append("sport collectif")
            changes.append("Exclu : catégorie « sport collectif » (pratique seul uniquement)")
    # acceptations : levée d'une exclusion
    ac = c["acceptations"]
    for f in ac["familles"]:
        if f in exf:
            exf.remove(f)
            changes.append(f"Accepté : famille {f}")
    for cat in ac["categories"]:
        if cat in excat:
            excat.remove(cat)
            changes.append(f"Accepté : catégorie « {cat} »")
            contradictions.append(f"exclusion de « {cat} » levée")
    for sp in ac["sports"]:
        if sp in exspo:
            exspo.remove(sp)
            changes.append(f"Accepté : sport {sp}")
    for f in ac["formats"]:
        if f in exfmt:
            exfmt.remove(f)
            changes.append(f"Accepté : format {f}")
    for env in ac["environnements"]:
        if env in exenv:
            exenv.remove(env)
            changes.append(f"Accepté : environnement {env}")
    if profile.only_solo and ("sport collectif" in ac["categories"] or "équipe" in ac["formats"]
                              or "Sports collectifs" in ac["familles"]):
        new["only_solo"] = False  # accepter les collectifs lève « uniquement individuels »
        changes.append("Accepté : pratique en équipe (fin de « pratique seul uniquement »)")
        contradictions.append("« uniquement individuels » levé par votre acceptation des collectifs")
    if ac["eau"] is True and profile.no_water:
        new["no_water"] = False
        changes.append("Accepté : activités dans l'eau")
        contradictions.append("exclusion de l'eau levée")
    if ac["contact"] is True and profile.no_contact:
        new["no_contact"] = False
        changes.append("Accepté : sports à contact physique")
        contradictions.append("exclusion du contact levée")
    # contraintes physiques
    cp = c["contraintes_physiques"]
    if cp["impacts"] and cp["impacts"] != profile.impact_tolerance:
        changes.append(f"Tolérance aux impacts : {profile.impact_tolerance} → {cp['impacts']}")
        new["impact_tolerance"] = cp["impacts"]
    for z in cp["zones_a_menager"]:
        if z not in zones:
            zones.append(z)
            changes.append(f"Zone à ménager : + {z}")
    for j in cp["articulations_douloureuses"]:
        if j not in joints:
            joints.append(j)
            changes.append(f"Articulation douloureuse : + {j}")
    # scalaires
    if c["budget_max_eur"] is not None:
        value = None if c["budget_max_eur"] == "levée" else c["budget_max_eur"]
        if value != profile.budget_seance_max:
            _scalar_change("Budget max par séance (€)", profile.budget_seance_max, value, changes)
            new["budget_seance_max"] = value
    if c["seances_par_semaine"] and c["seances_par_semaine"] != profile.sessions_per_week:
        _scalar_change("Séances par semaine", profile.sessions_per_week, c["seances_par_semaine"], changes)
        new["sessions_per_week"] = c["seances_par_semaine"]
    if c["duree_minutes"] and c["duree_minutes"] != profile.minutes:
        _scalar_change("Durée d'une séance (min)", profile.minutes, c["duree_minutes"], changes)
        new["minutes"] = c["duree_minutes"]
    if c["niveau"] and c["niveau"] != profile.experience:
        _scalar_change("Niveau", profile.experience, c["niveau"], changes)
        new["experience"] = c["niveau"]
    if c["intensite"]:
        shift = max(-2, min(2, profile.intensity_shift + c["intensite"]))
        if shift != profile.intensity_shift:
            _scalar_change("Intensité souhaitée (décalage)", profile.intensity_shift, shift, changes)
            new["intensity_shift"] = shift
    for corr in c["corrections"]:
        contradictions.append(f"{corr['champ']} : « {corr['ancienne']} » remplacé par « {corr['nouvelle']} »")

    new.update({"families": fams, "excluded_families": exf, "formats": fmts, "excluded_formats": exfmt,
                "excluded_categories": excat, "excluded_sports": exspo, "excluded_environments": exenv,
                "avoid_families": avoid, "goals": goals, "spare_zones": zones, "injured_joints": joints})
    return replace(profile, **new), changes, list(dict.fromkeys(contradictions))


def profile_view(profile: UserProfile) -> dict:
    """Vue lisible du profil, pour l'affichage : objectifs, préférences souples, contraintes strictes."""
    strict = []
    strict += [f"exclu : famille {f}" for f in profile.excluded_families]
    strict += [f"exclu : format {f}" for f in profile.excluded_formats]
    strict += [f"exclu : catégorie « {c} »" for c in profile.excluded_categories]
    strict += [f"exclu : sport {s}" for s in profile.excluded_sports]
    strict += [f"exclu : environnement {e}" for e in profile.excluded_environments]
    if profile.no_water:
        strict.append("sans activité dans l'eau")
    if profile.no_contact:
        strict.append("sans contact physique")
    if profile.only_solo:
        strict.append("pratique seul uniquement")
    if profile.impact_tolerance == "faible":
        strict.append("impacts élevés exclus (tolérance faible)")
    if profile.injured_joints:
        strict.append("articulations douloureuses : " + ", ".join(profile.injured_joints))
    if profile.budget_seance_max is not None:
        strict.append(f"budget max {profile.budget_seance_max} € par séance")
    soft = []
    if profile.env_preference:
        soft.append(f"environnement préféré : {profile.env_preference} (favorisé, non exclusif)")
    if profile.formats:
        soft.append("formats préférés : " + ", ".join(profile.formats))
    if profile.families:
        soft.append("familles appréciées : " + ", ".join(profile.families))
    if profile.avoid_families:
        soft.append("familles peu appréciées : " + ", ".join(profile.avoid_families))
    return {
        "objectifs": list(profile.goals) + [f"zone {z}" for z in profile.goal_zones],
        "preferences": soft,
        "contraintes_strictes": strict,
        "niveau": profile.experience, "seances": profile.sessions_per_week, "duree": profile.minutes,
    }


def profile_fingerprint(profile: UserProfile) -> str:
    return repr(profile)


# ------------------------------------------------------------------ comparaison et explications
def _top_criteria(a: dict, high: bool, k: int = 3) -> list[str]:
    items = []
    for b in BLOCKS:
        for lab, sc in a["criteria"][b]:
            if (high and sc >= 0.7) or (not high and sc < 0.45):
                items.append((sc if high else -sc, f"{lab} ({BLOCK_LABELS[b].lower()})"))
    return [t for _, t in sorted(items, reverse=high)[:k]]


def _anatomy(out: dict, name: str) -> dict | None:
    return next((a for a in out["anatomy"] if a["name"] == name), None)


def top_summary(out: dict, k: int = 3) -> list[dict]:
    """Top k avec score global, critères de correspondance et scores par bloc."""
    rows = []
    for res in out["results"][:k]:
        a = _anatomy(out, res.name)
        rows.append({"sport": res.name, "rang": a["rank"], "score": round(res.total * 100),
                     "blocs": {BLOCK_LABELS[b]: (None if res.blocks[b] is None else round(res.blocks[b] * 100)) for b in BLOCKS},
                     "criteres": _top_criteria(a, True)})
    return rows


def compare_results(old_out: dict, new_out: dict, k: int = 3) -> dict:
    """Compare le top k avant / après, et explique entrées, sorties et mouvements avec les données du moteur."""
    old = [r.name for r in old_out["results"][:k]]
    new = [r.name for r in new_out["results"][:k]]
    old_all = {a["name"]: a for a in old_out["anatomy"]}
    new_all = {a["name"]: a for a in new_out["anatomy"]}
    entries, exits = [], []
    for name in new:
        if name not in old:
            a = new_all[name]
            why = _top_criteria(a, True) or ["bon équilibre entre vos critères"]
            prev = old_all.get(name)
            if prev and not prev["retained"]:
                why = [f"devenu admissible : {', '.join(prev['excluded_why'])} n'est plus vrai"] + why
            entries.append({"sport": name, "raisons": why})
    for name in old:
        if name not in new:
            a = new_all.get(name)
            if a and not a["retained"]:
                why = ["exclu par votre nouvelle contrainte : " + "; ".join(a["excluded_why"])]
            elif a:
                why = [f"classement passé de {old_all[name]['rank']} à {a['rank']} (score {round(a['total'] * 100)} %)"]
            else:
                why = ["hors du classement"]
            exits.append({"sport": name, "raisons": why})
    return {"avant": old, "apres": new, "entrees": entries, "sorties": exits, "identique": old == new,
            "top_avant": top_summary(old_out, k), "top_apres": top_summary(new_out, k)}


def diagnose_empty(engine, profile: UserProfile, limit: int = 4) -> list[str]:
    """Si aucun sport n'est admissible : quelle contrainte, si elle était levée, rendrait des sports admissibles.

    Le calcul ne modifie rien : il indique seulement des pistes. La contrainte n'est jamais retirée automatiquement.
    """
    candidates = []
    for f in profile.excluded_families:
        candidates.append((f"exclusion de la famille {f}", replace(profile, excluded_families=[x for x in profile.excluded_families if x != f])))
    for c in profile.excluded_categories:
        candidates.append((f"exclusion de la catégorie « {c} »", replace(profile, excluded_categories=[x for x in profile.excluded_categories if x != c])))
    for fmt in profile.excluded_formats:
        candidates.append((f"exclusion du format {fmt}", replace(profile, excluded_formats=[x for x in profile.excluded_formats if x != fmt])))
    for e in profile.excluded_environments:
        candidates.append((f"exclusion de l'environnement {e}", replace(profile, excluded_environments=[x for x in profile.excluded_environments if x != e])))
    if profile.no_water:
        candidates.append(("le refus des activités dans l'eau", replace(profile, no_water=False)))
    if profile.no_contact:
        candidates.append(("le refus du contact physique", replace(profile, no_contact=False)))
    if profile.only_solo:
        candidates.append(("la pratique seul uniquement", replace(profile, only_solo=False)))
    if profile.impact_tolerance == "faible":
        candidates.append(("la tolérance faible aux impacts", replace(profile, impact_tolerance="moyenne")))
    if profile.budget_seance_max is not None:
        candidates.append(("le budget maximum", replace(profile, budget_seance_max=None)))
    suggestions = []
    for label, p2 in candidates:
        keep, _ = engine.filter_hard(engine.resolve(p2))
        if int(keep.sum()) > 0:
            suggestions.append(f"{label} : {int(keep.sum())} sport(s) redeviendraient admissibles")
    return suggestions[:limit]


# ------------------------------------------------------------------ questions sur les sports
def find_sports(message: str, out: dict | None) -> list[str]:
    if out is None:
        return []
    f = " " + fold(message) + " "
    ranked = [a["name"] for a in sorted(out["anatomy"], key=lambda a: (a["rank"] or 999, a["name"]))]
    hits = []
    for name in ranked:
        full = fold(name)
        first = full.split()[0]
        if f" {full} " in f or (len(first) >= 5 and f" {first} " in f):
            hits.append((0 if f" {full} " in f else 1, ranked.index(name), len(name), name))
    return [h[3] for h in sorted(hits)]


def detect_question(message: str, out: dict) -> dict | None:
    f = fold(message)
    sports = find_sports(message, out)
    if re.search(r"difference|compar|versus|\bvs\b", f) and len(sports) >= 2:
        return {"type": "comparer", "sports": sports[:2]}
    if re.search(r"pourquoi|qu.est.ce qui|explique", f):
        return {"type": "pourquoi", "sports": sports[:1]}
    if re.search(r"critere|influenc|facteurs|mon score", f):
        return {"type": "criteres", "sports": []}
    return None


def answer_question(q: dict, out: dict, profile: UserProfile) -> str:
    kind, sports = q["type"], q.get("sports", [])
    if kind == "criteres":
        weights = " · ".join(f"{BLOCK_LABELS[b]} {out['weights'][b] * 100:.0f} %" for b in BLOCKS)
        txt = f"Votre score repose sur quatre blocs, pondérés ainsi : {weights}.\n\n"
        if out["results"]:
            a = _anatomy(out, out["results"][0].name)
            txt += (f"Pour **{a['name']}** (n°1) : points forts → {', '.join(_top_criteria(a, True)) or '—'} ; "
                    f"points d'attention → {', '.join(_top_criteria(a, False)) or 'aucun'}.")
        return txt
    if not sports:
        return "Je n'ai pas reconnu de sport dans votre question. Citez un sport, par exemple « Pourquoi me recommandes-tu la natation ? »."
    if kind == "pourquoi":
        a = _anatomy(out, sports[0])
        if a is None:
            return "Ce sport n'est pas dans le dataset."
        if not a["retained"]:
            return f"**{a['name']}** est écarté par vos contraintes : " + " ; ".join(a["excluded_why"]) + "."
        head = (f"**{a['name']}** est classé n°{a['rank']} avec un score de {a['total'] * 100:.0f} %"
                + (" (dans votre top 3)." if a["rank"] <= 3 else " (hors top 3)."))
        return (head + f"\n\n- **Ce qui le favorise** : {', '.join(_top_criteria(a, True)) or 'aucun point fort marqué'}."
                + f"\n- **Points d'attention** : {', '.join(_top_criteria(a, False) + a['pen_why']) or 'aucun'}.")
    if kind == "comparer":
        a, b = _anatomy(out, sports[0]), _anatomy(out, sports[1])
        def rk(x):
            return f"n°{x['rank']}" if x["retained"] else "écarté : " + " ; ".join(x["excluded_why"])
        def pct(x, v):
            return f"{v * 100:.0f} %" if (x["retained"] and v is not None) else "—"
        rows = ["| | " + a["name"] + " | " + b["name"] + " |", "|---|---|---|", f"| Rang | {rk(a)} | {rk(b)} |",
                f"| Score global | {pct(a, a['total'])} | {pct(b, b['total'])} |"]
        for blk in BLOCKS:
            rows.append(f"| {BLOCK_LABELS[blk]} | {pct(a, a['blocks'][blk])} | {pct(b, b['blocks'][blk])} |")
        ra, rb = a["row"], b["row"]
        rows += [f"| Intensité (sur 5) | {int(ra.intensite_loisir_1_5)} | {int(rb.intensite_loisir_1_5)} |",
                 f"| Impacts (sur 3) | {int(ra.impacts_1_3)} | {int(rb.impacts_1_3)} |",
                 f"| Contact | {ra.contact} | {rb.contact} |",
                 f"| Environnement | {ra.environnement.replace('|', ', ')} | {rb.environnement.replace('|', ', ')} |",
                 f"| Formats | {ra.format_pratique.replace('|', ', ')} | {rb.format_pratique.replace('|', ', ')} |"]
        return "\n".join(rows)
    return "Je ne sais pas répondre à cette question."


HELP_TEXT = ("Je peux :\n- modifier votre profil : « je ne veux aucun sport aquatique », « je veux uniquement des sports "
             "individuels », « je préfère l'extérieur », « finalement j'accepte les sports collectifs », « mon budget est de "
             "15 € par séance » ;\n- répondre à vos questions : « Pourquoi me recommandes-tu la natation ? », « Quelle différence "
             "entre le tennis et le badminton ? », « Quels critères ont influencé mon score ? ».")


# ------------------------------------------------------------------ point d'entrée
def process_message(message: str, profile: UserProfile, out: dict, engine, weights: dict | None,
                    settings: LLMSettings, *, budget: CallBudget | None = None,
                    transport: Transport | None = None) -> dict:
    """Traite un message de bout en bout.

    kind = "maj"            : profil modifié, moteur relancé (profile, out, changes, contradictions, comparison…)
    kind = "sans_changement": message lu, aucun critère modifié : aucun recalcul
    kind = "reponse"        : réponse à une question sur les sports (aucun recalcul)
    kind = "precision"      : message ambigu, une question est posée
    kind = "aide"           : message non reconnu
    """
    vocab = vocabulary(engine)
    question = detect_question(message, out)
    if question:
        return {"kind": "reponse", "text": answer_question(question, out, profile), "source": "moteur", "note": ""}
    read = interpret(message, profile, settings, vocab, budget=budget, transport=transport)
    c = read["contract"]
    base = {"source": read["source"], "note": read["note"], "rejected": read["rejected"]}
    if c["question"]:
        return {**base, "kind": "reponse", "text": answer_question(c["question"], out, profile)}
    if contract_is_empty(c):
        if c["precisions"]:
            return {**base, "kind": "precision", "text": "\n".join(f"- {q}" for q in c["precisions"])}
        return {**base, "kind": "aide", "text": HELP_TEXT}
    new_profile, changes, contradictions = apply_contract(profile, c)
    if profile_fingerprint(new_profile) == profile_fingerprint(profile):
        return {**base, "kind": "sans_changement", "text": "Ce message ne modifie aucun critère de votre profil : "
                                                           "il est bien pris en compte, aucun recalcul n'a été nécessaire."}
    new_out = engine.recommend(new_profile, weights)  # moteur existant, non dupliqué
    comparison = compare_results(out, new_out)
    n_kept = int(sum(a["retained"] for a in new_out["anatomy"]))
    suggestions = diagnose_empty(engine, new_profile) if n_kept == 0 else []
    get_logger().info("chatbot : %d changement(s), %d sport(s) admissible(s)", len(changes), n_kept)
    return {**base, "kind": "maj", "profile": new_profile, "out": new_out, "changes": changes,
            "contradictions": contradictions, "comparison": comparison, "view": profile_view(new_profile),
            "n_kept": n_kept, "suggestions": suggestions, "precisions": c["precisions"],
            "text": ("Aucun sport du dataset ne respecte toutes vos contraintes." if n_kept == 0 else "")}


def reset_profile(base: UserProfile) -> UserProfile:
    """Réinitialise UNIQUEMENT les modifications de la conversation : retour au profil du questionnaire."""
    return base
