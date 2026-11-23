"""Bonus B : chatbot adaptatif. Il comprend un besoin formulé librement et relance le moteur NLP existant.

Architecture (à retenir pour la soutenance)
-------------------------------------------
1. `interpret`      lit le message en langage naturel et produit un CONTRAT (dictionnaire aux champs fixes).
                    Deux lecteurs : les règles (toujours disponibles) et Gemini (JSON). Le contrat est validé par
                    `validate_contract` : clés inconnues, valeurs hors vocabulaire ou hors bornes sont rejetées.
                    Chaque compréhension est reformulée dans `interpretations` (« ce que j'ai compris »).
2. `apply_contract` fusionne le contrat dans le profil :
                    - exclusions STRICTES : lieux indisponibles, groupes de sports, sports, catégories, formats…
                    - préférences SOUPLES : elles favorisent un sport dans le score, sans jamais l'exclure ;
                    - une valeur corrigée remplace l'ancienne ; une exclusion retire la préférence opposée.
3. `engine.recommend` moteur existant, NON remplacé : filtres stricts AVANT le classement, puis scores SBERT/TF-IDF.
4. `compare_results` compare le top 3 avant / après et explique entrées et sorties avec les données du moteur.
5. `diagnose_empty` si aucun sport n'est admissible : indique quelle contrainte lever, sans jamais la lever.

Le LLM ne calcule aucun score et ne produit aucun classement : il lit le message et propose des champs du contrat.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace

from bonus_llm import CallBudget, LLMError, Transport, call_llm
from engine import ALL_GOALS, BLOCK_LABELS, BLOCKS, FAMILIES, GROUP_LABEL, GROUPS, INFRA_DEFS, UserProfile
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

CONTRACT_SHAPE = {
    "intention": "enum", "objectifs_ajoutes": "goals",
    "preferences_souples": {"familles": "families", "formats": "formats", "environnement": "env|null",
                            "groupes": "groups"},
    "exclusions_strictes": {"familles": "families", "formats": "formats", "categories": "categories",
                            "sports": "sports", "environnements": "env", "groupes": "groups", "eau": "bool|null",
                            "contact": "bool|null", "seul_uniquement": "bool|null"},
    "indisponibilites": {"lieux": "places"},
    "acceptations": {"familles": "families", "formats": "formats", "categories": "categories", "sports": "sports",
                     "environnements": "env", "groupes": "groups", "lieux": "places", "eau": "bool|null",
                     "contact": "bool|null"},
    "contraintes_physiques": {"impacts": "impacts|null", "zones_a_menager": "zones", "articulations_douloureuses": "joints"},
    "budget_max_eur": "int|levée|null", "seances_par_semaine": "int|null", "duree_minutes": "int|null",
    "niveau": "levels|null", "intensite": "int|null", "corrections": "list", "interpretations": "list",
    "precisions": "list", "question": "dict|null",
}


def vocabulary(engine) -> dict:
    """Valeurs admises : sports et catégories viennent du dataset ; lieux et groupes sont des vocabulaires contrôlés."""
    cats = sorted({t for lst in engine.df["categories_l"] for t in lst})
    return {"sports": list(engine.df["nom_sport"]), "categories": cats, "places": list(INFRA_DEFS), "groups": list(GROUPS)}


def _fail(rejected: list, path: str, why: str):
    rejected.append(f"{path} ({why})")
    return None


def _str_list(value, allowed, path, rejected):
    if value is None:
        return []
    if not isinstance(value, list):
        return _fail(rejected, path, "liste attendue") or []
    out = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            _fail(rejected, f"{path}[{str(item)[:30]}]", "valeur hors vocabulaire")
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


def _empty_contract() -> dict:
    return {
        "intention": "autre", "objectifs_ajoutes": [],
        "preferences_souples": {"familles": [], "formats": [], "environnement": None, "groupes": []},
        "exclusions_strictes": {"familles": [], "formats": [], "categories": [], "sports": [], "environnements": [],
                                "groupes": [], "eau": None, "contact": None, "seul_uniquement": None},
        "indisponibilites": {"lieux": []},
        "acceptations": {"familles": [], "formats": [], "categories": [], "sports": [], "environnements": [],
                         "groupes": [], "lieux": [], "eau": None, "contact": None},
        "contraintes_physiques": {"impacts": None, "zones_a_menager": [], "articulations_douloureuses": []},
        "budget_max_eur": None, "seances_par_semaine": None, "duree_minutes": None, "niveau": None, "intensite": None,
        "corrections": [], "interpretations": [], "precisions": [], "question": None,
    }


def validate_contract(raw, vocab: dict) -> tuple[dict, list[str]]:
    """Valide le contrat (JSON de Gemini ou règles). Retourne (contrat propre, motifs de rejet).

    Une clé inconnue, une valeur hors vocabulaire ou hors bornes est rejetée et signalée : elle n'atteint jamais le profil.
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
    c["objectifs_ajoutes"] = _str_list(raw.get("objectifs_ajoutes"), ALL_GOALS, "objectifs_ajoutes", rejected)

    def section(name):
        val = raw.get(name)
        return val if isinstance(val, dict) else {}

    ps = section("preferences_souples")
    p = c["preferences_souples"]
    p["familles"] = _str_list(ps.get("familles"), list(FAMILIES), "preferences_souples.familles", rejected)
    p["formats"] = _str_list(ps.get("formats"), FORMATS, "preferences_souples.formats", rejected)
    p["groupes"] = _str_list(ps.get("groupes"), vocab["groups"], "preferences_souples.groupes", rejected)
    env = ps.get("environnement")
    p["environnement"] = env if env in ENVIRONMENTS else (None if env is None else _fail(rejected, "preferences_souples.environnement", "intérieur ou extérieur"))

    es = section("exclusions_strictes")
    e = c["exclusions_strictes"]
    e["familles"] = _str_list(es.get("familles"), list(FAMILIES), "exclusions_strictes.familles", rejected)
    e["formats"] = _str_list(es.get("formats"), FORMATS, "exclusions_strictes.formats", rejected)
    e["categories"] = _str_list(es.get("categories"), vocab["categories"], "exclusions_strictes.categories", rejected)
    e["sports"] = _str_list(es.get("sports"), vocab["sports"], "exclusions_strictes.sports", rejected)
    e["environnements"] = _str_list(es.get("environnements"), ENVIRONMENTS, "exclusions_strictes.environnements", rejected)
    e["groupes"] = _str_list(es.get("groupes"), vocab["groups"], "exclusions_strictes.groupes", rejected)
    e["eau"] = _bool(es.get("eau"), "exclusions_strictes.eau", rejected)
    e["contact"] = _bool(es.get("contact"), "exclusions_strictes.contact", rejected)
    e["seul_uniquement"] = _bool(es.get("seul_uniquement"), "exclusions_strictes.seul_uniquement", rejected)

    ind = section("indisponibilites")
    c["indisponibilites"]["lieux"] = _str_list(ind.get("lieux"), vocab["places"], "indisponibilites.lieux", rejected)

    ac = section("acceptations")
    a = c["acceptations"]
    a["familles"] = _str_list(ac.get("familles"), list(FAMILIES), "acceptations.familles", rejected)
    a["formats"] = _str_list(ac.get("formats"), FORMATS, "acceptations.formats", rejected)
    a["categories"] = _str_list(ac.get("categories"), vocab["categories"], "acceptations.categories", rejected)
    a["sports"] = _str_list(ac.get("sports"), vocab["sports"], "acceptations.sports", rejected)
    a["environnements"] = _str_list(ac.get("environnements"), ENVIRONMENTS, "acceptations.environnements", rejected)
    a["groupes"] = _str_list(ac.get("groupes"), vocab["groups"], "acceptations.groupes", rejected)
    a["lieux"] = _str_list(ac.get("lieux"), vocab["places"], "acceptations.lieux", rejected)
    a["eau"] = _bool(ac.get("eau"), "acceptations.eau", rejected)
    a["contact"] = _bool(ac.get("contact"), "acceptations.contact", rejected)

    cp = section("contraintes_physiques")
    k = c["contraintes_physiques"]
    imp = cp.get("impacts")
    k["impacts"] = imp if imp in IMPACTS else (None if imp is None else _fail(rejected, "contraintes_physiques.impacts", "faible, moyenne ou élevée"))
    k["zones_a_menager"] = _str_list(cp.get("zones_a_menager"), ZONES_ALL, "contraintes_physiques.zones_a_menager", rejected)
    k["articulations_douloureuses"] = _str_list(cp.get("articulations_douloureuses"), JOINTS, "contraintes_physiques.articulations_douloureuses", rejected)

    budget = raw.get("budget_max_eur")
    c["budget_max_eur"] = "levée" if budget == "levée" else _int(budget, 0, 300, "budget_max_eur", rejected)
    c["seances_par_semaine"] = _int(raw.get("seances_par_semaine"), 1, 7, "seances_par_semaine", rejected)
    c["duree_minutes"] = _int(raw.get("duree_minutes"), 20, 180, "duree_minutes", rejected)
    niv = raw.get("niveau")
    c["niveau"] = niv if niv in LEVELS else (None if niv is None else _fail(rejected, "niveau", "valeur inconnue"))
    c["intensite"] = _int(raw.get("intensite"), -1, 1, "intensite", rejected)

    for item in (raw.get("corrections") or [])[:5] if isinstance(raw.get("corrections"), list) else []:
        if isinstance(item, dict) and all(isinstance(item.get(x), str) for x in ("champ", "ancienne", "nouvelle")):
            c["corrections"].append({x: item[x][:80] for x in ("champ", "ancienne", "nouvelle")})
        else:
            _fail(rejected, "corrections", "format attendu : champ, ancienne, nouvelle")
    for item in (raw.get("interpretations") or [])[:6] if isinstance(raw.get("interpretations"), list) else []:
        if isinstance(item, dict) and isinstance(item.get("extrait"), str) and isinstance(item.get("comprend"), str):
            c["interpretations"].append({"extrait": item["extrait"][:120], "comprend": item["comprend"][:240]})
        else:
            _fail(rejected, "interpretations", "format attendu : extrait, comprend")
    if isinstance(raw.get("precisions"), list):
        c["precisions"] = [str(x)[:200] for x in raw["precisions"] if isinstance(x, str)][:3]
    q = raw.get("question")
    if isinstance(q, dict) and q.get("type") in QUESTION_TYPES:
        sports = q.get("sports") if isinstance(q.get("sports"), list) else []
        c["question"] = {"type": q["type"], "sports": [x for x in sports if isinstance(x, str) and x in vocab["sports"]][:2]}
    elif q is not None:
        _fail(rejected, "question", "type attendu : pourquoi, comparer ou criteres")
    return c, rejected


def merge_contracts(a: dict, b: dict) -> dict:
    """Fusion : listes unies (sans doublon, sans écraser les dictionnaires des corrections et interprétations),
    valeurs simples : le LLM (a) prime, sinon les règles (b)."""
    def merge(x, y):
        if isinstance(x, dict):
            return {k: merge(x.get(k), (y or {}).get(k)) for k in x}
        if isinstance(x, list):
            items = (x or []) + (y or [])
            if any(isinstance(i, dict) for i in items):
                return items
            return list(dict.fromkeys(items))
        return x if x not in (None, "autre") else y
    out = merge(a, b)
    out["corrections"] = (a.get("corrections") or []) + (b.get("corrections") or [])
    out["interpretations"] = (a.get("interpretations") or []) + (b.get("interpretations") or [])
    out["precisions"] = list(dict.fromkeys((a.get("precisions") or []) + (b.get("precisions") or [])))[:3]
    out["question"] = a.get("question") or b.get("question")
    out["intention"] = a.get("intention") if a.get("intention") not in (None, "autre") else b.get("intention", "autre")
    return out


def _filled(value) -> bool:
    """Une valeur est renseignée si elle n'est ni None ni une liste vide. False compte : c'est un refus."""
    return value is not None and value != []


def contract_is_empty(c: dict) -> bool:
    flat = [c["objectifs_ajoutes"], c["preferences_souples"]["familles"], c["preferences_souples"]["formats"],
            c["preferences_souples"]["environnement"], c["preferences_souples"]["groupes"],
            *c["exclusions_strictes"].values(), c["indisponibilites"]["lieux"], *c["acceptations"].values(),
            *c["contraintes_physiques"].values(), c["budget_max_eur"], c["seances_par_semaine"], c["duree_minutes"],
            c["niveau"], c["intensite"], c["question"]]
    return not any(_filled(v) for v in flat)


# ------------------------------------------------------------------ lecteur 1 : règles (sans IA)
LIEU_PATTERNS = {
    "mer": r"\bmer\b|ocean|littoral|plage|bord de (?:la )?mer|vagues|(?<!a )\bcotes?\b(?!\s+(?:de chez|de moi|de ma|d'ici|de notre|de nous))",
    "plan_eau": r"\blac\b|riviere|plan d'eau|\betang|canal|eau vive",
    "piscine": r"piscine|bassin",
    "stade": r"stade|terrain de (?:foot|rugby|sport|football)|\bpiste\b(?!\s+(?:cyclable|de ski))|athletisme|pelouse|terrain de jeu",
    "gymnase": r"gymnase|salle omnisport",
    "court": r"\bcourts?\b|terrain de tennis",
    "fronton": r"fronton",
    "salle_musculation": r"salle de musculation|salle de sport|salle de fitness|musculation|crossfit",
    "cardio_salle": r"cardio|tapis de course|rameur|elliptique|spinning",
    "dojo": r"dojo|arts martiaux|salle de combat|salle de boxe|salle d'armes|salle d'escrime|tatami",
    "studio_danse": r"studio de danse|salle de danse",
    "espace_calme": r"yoga|salle calme|espace calme|studio calme",
    "neige": r"montagne|neige|station de ski|piste de ski|domaine skiable|\bski\b",
    "glace": r"patinoire|\bglace\b",
    "mur_escalade": r"salle d'escalade|mur d'escalade|escalade en salle",
    "paroi_naturelle": r"falaise|rocher|paroi|via ferrata",
    "route": r"\broute\b",
    "piste_cyclable": r"piste cyclable|voie verte",
    "velodrome": r"velodrome|piste de velo",
    "manege": r"manege|centre equestre|\bcheval\b|chevaux",
    "skatepark": r"skatepark|skate ?park",
    "bowling_billard": r"bowling|billard|flechettes|\bbar\b",
    "stand_tir": r"stand de tir|tir a l'arc|carabine",
    "boulodrome": r"boulodrome|petanque",
    "parcours_golf": r"\bgolf\b",
    "nature": r"foret|sentier|nature|campagne|chemin",
    "rue_parc": r"\bparc\b|\brue\b|trottoir|playground|urbain|\bville\b|quartier",
}
GROUP_PATTERNS = {
    "sports_de_balle": r"sports? de balle|sports? de ballon|jeux? de balle|jeux? de ballon|sports? avec (?:un |une )?ballon",
    "sports_collectifs": r"sports? collectifs?|sports? d'equipe|jeux? d'equipe|sports? en equipe|jeux? collectifs?",
    "sports_de_raquette": r"sports? de raquette|raquettes?\b",
    "sports_de_combat": r"sports? de combat|combats?\b|arts? martiaux|boxe|judo|karate|\blutte\b|kick|muay",
    "sports_de_glisse": r"sports? de glisse|\bglisse\b",
    "sports_aquatiques": r"aquatiques?|\bnager\b|natation|\bnage\b|\bwater|dans l'eau|sur l'eau",
    "sports_equestres": r"equitation|\bcheval\b|dressage",
    "sports_de_cible": r"sports? de cible|\btir\b",
    "course_a_pied": r"course a pied|courir|\bcourse\b|footing|jogging|running|\bsprint",
    "danse": r"\bdanse\b|danser|choregraph",
    "cyclisme": r"cyclisme|\bvelo\b|bicyclette|cycliste|\bvtt\b",
    "gymnastique": r"gymnastique|acrobat",
    "escalade": r"escalade|grimper|grimpe",
    "sports_de_contact": r"sports? de contact",
}
ALIASES = {"foot": "Football", "basket": "Basketball", "rugby": "Rugby à XV", "vtt": "VTT cross-country",
           "surf": "Surf", "ski": "Ski alpin", "ping-pong": "Tennis de table"}
NEGATION = (r"\bpas (?:de|d'|du|des|a |en|dans|aucun|une?)|\bpas\b|\bsans\b|\baucun|\baucune|\bjamais\b|\beviter|\bevite\b"
            r"|\bexclu|\bn'ai pas|\bn'aime pas|\bje n'aime|\bne veux|\bne peux|\bne souhaite|\buniquement|\bseulement|\bne fais pas")
SOFT = r"plutot moins|pas trop|un peu moins|moins de|peu de|modere"
ACCEPT = r"\baccepte|je veux bien|ca me va|pas de probleme|\bje suis ok|finalement.*(?:veux|peux|accepte|ok)"
SPLIT = r"[.;!?]|\bmais\b|\bsauf\b|\bpar contre\b|\bcependant\b|\bet (?=pas |sans |je (?:n'|ne) )"
NEG_LOCAL = (r"\bpas\b|\bsans\b|\bjamais\b|\baucun|\bn'ai pas|\bn'aime pas|\bje n'aime|\bne veux|\bne peux|\bne souhaite"
             r"|\bevit|\bni\b|\bexclu|\bne fais pas|\bne sais pas|\bpas accès|\bsais pas")


def _neg_before(f: str, pos: int, window: int = 32) -> bool:
    """Une négation est-elle juste avant la mention (dans les 32 caractères qui précèdent) ?"""
    return re.search(NEG_LOCAL, f[max(0, pos - window):pos]) is not None


def _clauses(message: str) -> list[str]:
    return [c.strip() for c in re.split(SPLIT, message, flags=re.IGNORECASE) if c.strip()]


def _norm(text: str) -> str:
    """Minuscules sans accents, ponctuation remplacée par des espaces : « longe-côte, » devient « longe cote »."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", fold(text)).split())


def _sports_in(text_folded: str, names: list[str]) -> list[str]:
    found = []
    text_folded = _norm(text_folded)
    f = " " + text_folded + " "
    for alias, real in ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text_folded) and real in names:
            found.append(real)
    for name in names:
        full = _norm(name)
        first = full.split()[0]
        if f" {full} " in f or (len(first) >= 5 and f" {first} " in f):
            found.append(name)
    return list(dict.fromkeys(found))


def examples(engine, kind: str, key: str, k: int = 4) -> list[str]:
    """Quelques sports concernés par un lieu ou un groupe : ils servent à expliquer la compréhension."""
    df = engine.df
    if kind == "place":
        mask = df["infrastructures_l"].apply(lambda lst: key in lst)
    else:
        mask = df["categories_l"].apply(lambda lst: bool(set(lst) & GROUPS.get(key, set())))
    return list(df.loc[mask, "nom_sport"])[:k]


def rules_contract(message: str, vocab: dict, engine) -> dict:
    """Lecture par règles. Chaque clause est analysée : négation, acceptation, souplesse, puis entités trouvées."""
    c = _empty_contract()
    names = vocab["sports"]
    sig = extract_signals(message)
    c["objectifs_ajoutes"] = sorted(g for g in sig.goals if g in ALL_GOALS)
    for clause in _clauses(message):
        f = fold(clause)
        negated = re.search(NEGATION, f) is not None
        soft = re.search(SOFT, f) is not None
        accepted = re.search(ACCEPT, f) is not None
        touched = False
        # lieux indisponibles (« je n'ai pas de mer à côté »)
        for place, rx in LIEU_PATTERNS.items():
            m = re.search(rx, f)
            if m:
                touched = True
                neg = _neg_before(f, m.start()) or negated
                if accepted:
                    c["acceptations"]["lieux"].append(place)
                elif neg and not soft:
                    c["indisponibilites"]["lieux"].append(place)
                else:
                    c["precisions"].append(f"Pratiquez-vous ou souhaitez-vous éviter les activités liées à « {place.replace('_', ' ')} » ?")
        # groupes de sports (« je n'aime pas les sports de balle », « j'aime les sports de combat »)
        for group, rx in GROUP_PATTERNS.items():
            m = re.search(rx, f)
            if m:
                touched = True
                neg = _neg_before(f, m.start()) or (negated and not re.search(r"\bj'?\s*aim|\bj'adore|\bj'apprecie", f))
                if accepted:
                    c["acceptations"]["groupes"].append(group)
                elif neg and not soft:
                    c["exclusions_strictes"]["groupes"].append(group)
                elif neg and soft:
                    c["precisions"].append(f"Voulez-vous exclure complètement les {GROUP_LABEL.get(group, group)} ou seulement les défavoriser ?")
                else:
                    c["preferences_souples"]["groupes"].append(group)
        # sports précis (« je n'aime pas le surf », « finalement j'accepte le foot »)
        for sport in _sports_in(f, names):
            touched = True
            pos = f.find(fold(sport).split()[0].lower()) if fold(sport).split() else 0
            if accepted:
                c["acceptations"]["sports"].append(sport)
            elif _neg_before(f, max(pos, 0)) or negated:
                c["exclusions_strictes"]["sports"].append(sport)
        if re.search(r"ne sais pas nager|pas nager|sais pas nager|peur de l'eau", f):
            touched = True
            c["exclusions_strictes"]["eau"] = True
        if re.search(r"contact", f) and (negated or re.search(r"pas de contact|sans contact", f)):
            touched = True
            c["exclusions_strictes"]["contact"] = False
        if re.search(r"\b(individuel|seul|solo|tout seul)s?\b", f):
            touched = True
            if re.search(r"uniquement|seulement|que des", f):
                c["exclusions_strictes"]["seul_uniquement"] = True
            elif not negated:
                c["preferences_souples"]["formats"].append("individuel")
        if re.search(r"sans equipe|pas en equipe|pas d'equipe", f):
            touched = True
            c["exclusions_strictes"]["formats"].append("équipe")
        for word, env in (("interieur|en salle|dedans", "intérieur"), ("exterieur|dehors|plein air", "extérieur")):
            m = re.search(word, f)
            if m:
                touched = True
                other = "extérieur" if env == "intérieur" else "intérieur"
                if _neg_before(f, m.start()) and re.search(r"uniquement|seulement", f):
                    c["exclusions_strictes"]["environnements"].append(other)
                elif _neg_before(f, m.start()):
                    c["exclusions_strictes"]["environnements"].append(env)
                else:
                    c["preferences_souples"]["environnement"] = env
        if re.search(r"\bchocs?\b|impacts?|articulations? fragiles?", f) and (negated or "fragile" in f or "evit" in f):
            touched = True
            c["contraintes_physiques"]["impacts"] = "faible"
        if re.search(r"moins intense|plus doux|plus douce|plus calme|moins fatigant|plus leger", f):
            touched = True
            c["intensite"] = -1
        elif re.search(r"plus intense|plus dur|plus soutenu", f):
            touched = True
            c["intensite"] = 1
        m = re.search(r"\b(\d)\s*(?:seances?|fois)\b", f)
        if m and 1 <= int(m.group(1)) <= 7:
            touched = True
            c["seances_par_semaine"] = int(m.group(1))
        m = re.search(r"\b(\d{2,3})\s*(?:min|minutes)\b", f)
        if m and 20 <= int(m.group(1)) <= 180:
            touched = True
            c["duree_minutes"] = int(m.group(1))
        if re.search(r"budget|par seance|euros?|€", f):
            touched = True
            if re.search(r"plus de limite|sans limite|plus limite", f):
                c["budget_max_eur"] = "levée"
            else:
                m = re.search(r"(\d{1,3})\s*(?:euros?|€)", f)
                if m and 0 <= int(m.group(1)) <= 300:
                    c["budget_max_eur"] = int(m.group(1))
        if re.search(r"debutant|je reprends", f):
            touched = True
            c["niveau"] = "débutant"
        elif re.search(r"confirme|haut niveau", f):
            touched = True
            c["niveau"] = "confirmé"
        if re.search(r"blesse|mal au|mal a |douleur|fragile", f):
            for joint, jrx in (("genoux", "genou"), ("chevilles", "cheville"), ("hanches", "hanche"),
                               ("lombaires", "dos|lombaire"), ("epaules", "epaule"), ("coudes", "coude"),
                               ("poignets", "poignet"), ("cervicales", "nuque|cervical")):
                if re.search(jrx, f):
                    touched = True
                    c["contraintes_physiques"]["articulations_douloureuses"].append(joint)
        if not touched and (negated or soft) and not accepted:
            c["precisions"].append(f"Je n'ai pas reconnu ce que vous souhaitez éviter dans « {clause[:70]} ». "
                                   "Pouvez-vous préciser l'activité, le lieu ou le format concerné ?")
        if not touched and not negated and not soft and not accepted and len(f.split()) > 3:
            pass
    # compréhension explicite de chaque élément (« ce que j'ai compris »)
    for place in dict.fromkeys(c["indisponibilites"]["lieux"]):
        ex = ", ".join(examples(engine, "place", place))
        c["interpretations"].append({"extrait": place, "comprend":
            f"Pas d'accès à {INFRA_DEFS[place]} : j'exclus les sports qui en ont besoin" + (f" (ex. {ex})." if ex else ".")})
    for group in dict.fromkeys(c["exclusions_strictes"]["groupes"]):
        ex = ", ".join(examples(engine, "group", group))
        c["interpretations"].append({"extrait": group, "comprend":
            f"Vous ne voulez pas de {GROUP_LABEL.get(group, group)} : j'exclus ces sports" + (f" (ex. {ex})." if ex else ".")})
    for group in dict.fromkeys(c["preferences_souples"]["groupes"]):
        c["interpretations"].append({"extrait": group, "comprend":
            f"Vous appréciez les {GROUP_LABEL.get(group, group)} : je les favorise sans les imposer."})
    for sport in dict.fromkeys(c["exclusions_strictes"]["sports"]):
        c["interpretations"].append({"extrait": sport, "comprend": f"Vous ne voulez pas pratiquer {sport} : j'exclus ce sport."})
    if c["exclusions_strictes"]["eau"]:
        c["interpretations"].append({"extrait": "eau", "comprend": "Vous ne nagez pas : j'exclus les activités dans l'eau."})
    if c["preferences_souples"]["environnement"]:
        env = c["preferences_souples"]["environnement"]
        c["interpretations"].append({"extrait": env, "comprend": f"Vous préférez pratiquer en {env} : je favorise ces sports sans en exclure."})
    if c["exclusions_strictes"]["seul_uniquement"]:
        c["interpretations"].append({"extrait": "seul", "comprend": "Pratique seul uniquement : j'exclus les sports qui ne se pratiquent pas seul, et les sports collectifs."})
    c["precisions"] = list(dict.fromkeys(c["precisions"]))[:3]
    if not contract_is_empty(c):
        c["intention"] = "mise_a_jour"
    return c


# ------------------------------------------------------------------ lecteur 2 : Gemini (JSON strict)
CONTRACT_PROMPT = """Tu lis UN message d'un utilisateur d'une application de recommandation sportive, écrit en langage libre,
et tu le traduis en un objet JSON qui respecte EXACTEMENT le schéma. Tu réponds uniquement par le JSON.

Ce que tu dois comprendre, même formulé indirectement :
- un lieu INDISPONIBLE (« je n'ai pas de mer à côté », « pas de stade près de chez moi ») : indisponibilites.lieux
  Un lieu est choisi parmi : LIEUX. Exemples : mer (surf, longe-côte…), stade (football, rugby, lancer du poids, piste…),
  piscine, plan_eau (lac, rivière), neige (ski, snowboard…), dojo (arts martiaux, boxe…).
- un groupe de sports refusé (« je n'aime pas les sports de balle ») : exclusions_strictes.groupes
  Un groupe est choisi parmi : GROUPES.
- un goût pour un groupe (« j'aime les sports de combat ») : preferences_souples.groupes (favorise, n'exclut pas).
- une exclusion STRICTE (« je ne veux pas », « aucun », « jamais », « sans », « uniquement », « éviter ») : exclusions_strictes.
- une préférence SOUPLE (« je préfère », « j'aime », « plutôt ») : preferences_souples.
- une acceptation (« finalement j'accepte », « je veux bien ») : acceptations.
- une correction (« finalement intérieur » après « extérieur ») : corrections.
- interpretations : pour chaque élément compris, une phrase {"extrait": "fragment du message", "comprend": "ce que tu as compris, en français simple"}.
- precisions : une question SEULEMENT si le message est vraiment ambigu. Ne demande pas ce qui est clair.
- N'invente jamais une information absente du message. Ne donne aucun score ni classement.

Schéma :
{
 "intention": "mise_a_jour" | "question" | "autre",
 "objectifs_ajoutes": [OBJECTIFS],
 "preferences_souples": {"familles": [FAMILLES], "formats": ["individuel","duo","équipe","groupe"], "environnement": "intérieur"|"extérieur"|null, "groupes": [GROUPES]},
 "exclusions_strictes": {"familles": [FAMILLES], "formats": [...], "categories": [CATEGORIES], "sports": [SPORTS],
                         "environnements": ["intérieur"|"extérieur"], "groupes": [GROUPES], "eau": true|null,
                         "contact": false|null, "seul_uniquement": true|null},
 "indisponibilites": {"lieux": [LIEUX]},
 "acceptations": {"familles": [...], "formats": [...], "categories": [...], "sports": [...], "environnements": [...],
                  "groupes": [...], "lieux": [LIEUX], "eau": false|null, "contact": true|null},
 "contraintes_physiques": {"impacts": "faible"|"moyenne"|"élevée"|null, "zones_a_menager": [ZONES],
                           "articulations_douloureuses": [ARTICULATIONS]},
 "budget_max_eur": entier 0-300 | "levée" | null,
 "seances_par_semaine": entier 1-7 | null,
 "duree_minutes": entier 20-180 | null,
 "niveau": "débutant"|"occasionnel"|"régulier"|"confirmé"|null,
 "intensite": -1 | 0 | 1 | null,
 "corrections": [{"champ": "...", "ancienne": "...", "nouvelle": "..."}],
 "interpretations": [{"extrait": "...", "comprend": "..."}],
 "precisions": ["question"],
 "question": null | {"type": "pourquoi"|"comparer"|"criteres", "sports": [SPORTS]}
}
"""


def _prompt(vocab: dict, profile: UserProfile) -> str:
    text = (CONTRACT_PROMPT.replace("OBJECTIFS", ", ".join(ALL_GOALS)).replace("FAMILLES", ", ".join(FAMILIES))
            .replace("GROUPES", ", ".join(vocab["groups"])).replace("LIEUX", ", ".join(vocab["places"]))
            .replace("ZONES", ", ".join(ZONES_ALL)).replace("ARTICULATIONS", ", ".join(JOINTS))
            .replace("CATEGORIES", "une catégorie du dataset").replace("SPORTS", "un sport du dataset"))
    current = {"formats_souhaites": profile.formats, "environnement_souhaite": profile.env_preference,
               "groupes_exclus": profile.excluded_groups, "groupes_apprecies": profile.liked_groups,
               "lieux_indisponibles": profile.excluded_infrastructures, "familles_exclues": profile.excluded_families,
               "categories_exclues": profile.excluded_categories, "sans_eau": profile.no_water,
               "sans_contact": profile.no_contact, "seul_uniquement": profile.only_solo,
               "budget_max_eur": profile.budget_seance_max, "seances_par_semaine": profile.sessions_per_week}
    return text + "\nProfil actuel (contexte seulement, ne pas le recopier) : " + json.dumps(current, ensure_ascii=False)


def llm_contract(message: str, profile: UserProfile, vocab: dict, settings: LLMSettings,
                 transport: Transport | None = None) -> dict:
    text = call_llm(_prompt(vocab, profile), f"Message : « {message[:800]} »", settings, json_mode=True, transport=transport)
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        raise ValueError("réponse sans JSON")
    return json.loads(match.group(0))


def interpret(message: str, profile: UserProfile, settings: LLMSettings, vocab: dict, engine, *,
              budget: CallBudget | None = None, transport: Transport | None = None) -> dict:
    """Lit le message : règles toujours, Gemini si disponible. Retourne contrat validé, rejets, source et note."""
    rules, rules_rej = validate_contract(rules_contract(message, vocab, engine), vocab)
    use_llm = settings.chat_use_llm and settings.provider != "off" and not (
        settings.provider == "gemini" and not settings.has_gemini_key)
    note, rejected, source, contract = "", list(rules_rej), "règles", rules
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
def apply_contract(profile: UserProfile, c: dict) -> tuple[UserProfile, list[str], list[str]]:
    """Applique le contrat validé. Retourne (nouveau profil, changements, contradictions résolues)."""
    changes, contradictions, new = [], [], {}
    fams, exf = list(profile.families), list(profile.excluded_families)
    fmts, exfmt = list(profile.formats), list(profile.excluded_formats)
    excat, exspo = list(profile.excluded_categories), list(profile.excluded_sports)
    exenv, avoid = list(profile.excluded_environments), list(profile.avoid_families)
    exlieux, exgroups = list(profile.excluded_infrastructures), list(profile.excluded_groups)
    liked = list(profile.liked_groups)
    goals, zones, joints = list(profile.goals), list(profile.spare_zones), list(profile.injured_joints)

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
    for g in ps["groupes"]:
        if g not in liked:
            liked.append(g)
            changes.append(f"Groupe apprécié : + {GROUP_LABEL.get(g, g)} (favorisé, non exclusif)")
        if g in exgroups:
            exgroups.remove(g)
            contradictions.append(f"exclusion des {GROUP_LABEL.get(g, g)} levée par votre préférence")
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
    for g in ex["groupes"]:
        if g not in exgroups:
            exgroups.append(g)
            changes.append(f"Exclu : {GROUP_LABEL.get(g, g)}")
        if g in liked:
            liked.remove(g)
            contradictions.append(f"préférence pour les {GROUP_LABEL.get(g, g)} retirée (exclusion stricte)")
    for lieu in c["indisponibilites"]["lieux"]:
        if lieu not in exlieux:
            exlieux.append(lieu)
            changes.append(f"Indisponible : {INFRA_DEFS[lieu]}")
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
        if "sport collectif" not in excat:
            excat.append("sport collectif")
            changes.append("Exclu : catégorie « sport collectif » (pratique seul uniquement)")
    # acceptations : levée d'exclusions
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
    for g in ac["groupes"]:
        if g in exgroups:
            exgroups.remove(g)
            changes.append(f"Accepté : {GROUP_LABEL.get(g, g)}")
            contradictions.append(f"exclusion des {GROUP_LABEL.get(g, g)} levée")
    for lieu in ac["lieux"]:
        if lieu in exlieux:
            exlieux.remove(lieu)
            changes.append(f"Disponible à nouveau : {INFRA_DEFS[lieu]}")
    for f in ac["formats"]:
        if f in exfmt:
            exfmt.remove(f)
            changes.append(f"Accepté : format {f}")
    for env in ac["environnements"]:
        if env in exenv:
            exenv.remove(env)
            changes.append(f"Accepté : environnement {env}")
    if profile.only_solo and ("sport collectif" in ac["categories"] or "équipe" in ac["formats"]
                              or "Sports collectifs" in ac["familles"] or "sports_collectifs" in ac["groupes"]):
        new["only_solo"] = False
        changes.append("Accepté : pratique en équipe (fin de « pratique seul uniquement »)")
        contradictions.append("« uniquement individuels » levé par votre acceptation des collectifs")
    if ac["eau"] is True and profile.no_water:
        new["no_water"] = False
        changes.append("Accepté : activités dans l'eau")
        contradictions.append("exclusion de l'eau levée")
    if ac["contact"] is True and profile.no_contact:
        new["no_contact"] = False
        changes.append("Accepté : sports à contact physique")
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
    # valeurs simples
    if c["budget_max_eur"] is not None:
        value = None if c["budget_max_eur"] == "levée" else c["budget_max_eur"]
        if value != profile.budget_seance_max:
            changes.append(f"Budget max par séance : {profile.budget_seance_max or '—'} → {value or 'levé'} €")
            new["budget_seance_max"] = value
    if c["seances_par_semaine"] and c["seances_par_semaine"] != profile.sessions_per_week:
        changes.append(f"Séances par semaine : {profile.sessions_per_week} → {c['seances_par_semaine']}")
        new["sessions_per_week"] = c["seances_par_semaine"]
    if c["duree_minutes"] and c["duree_minutes"] != profile.minutes:
        changes.append(f"Durée d'une séance : {profile.minutes} → {c['duree_minutes']} min")
        new["minutes"] = c["duree_minutes"]
    if c["niveau"] and c["niveau"] != profile.experience:
        changes.append(f"Niveau : {profile.experience} → {c['niveau']}")
        new["experience"] = c["niveau"]
    if c["intensite"]:
        shift = max(-2, min(2, profile.intensity_shift + c["intensite"]))
        if shift != profile.intensity_shift:
            changes.append("Intensité souhaitée : " + ("plus douce" if shift < 0 else "plus soutenue" if shift > 0 else "niveau habituel"))
            new["intensity_shift"] = shift
    for corr in c["corrections"]:
        contradictions.append(f"{corr['champ']} : « {corr['ancienne']} » remplacé par « {corr['nouvelle']} »")

    new.update({"families": fams, "excluded_families": exf, "formats": fmts, "excluded_formats": exfmt,
                "excluded_categories": excat, "excluded_sports": exspo, "excluded_environments": exenv,
                "excluded_infrastructures": exlieux, "excluded_groups": exgroups, "liked_groups": liked,
                "avoid_families": avoid, "goals": goals, "spare_zones": zones, "injured_joints": joints})
    if "Eau" in exf and not profile.no_water and "no_water" not in new:
        new["no_water"] = True
    return replace(profile, **new), changes, list(dict.fromkeys(contradictions))


def profile_view(profile: UserProfile) -> dict:
    """Vue lisible du profil : objectifs, préférences souples, contraintes strictes."""
    strict = []
    strict += [f"lieu indisponible : {INFRA_DEFS.get(p, p)}" for p in profile.excluded_infrastructures]
    strict += [f"exclu : {GROUP_LABEL.get(g, g)}" for g in profile.excluded_groups]
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
    if profile.liked_groups:
        soft.append("groupes appréciés : " + ", ".join(GROUP_LABEL.get(g, g) for g in profile.liked_groups))
    if profile.families:
        soft.append("familles appréciées : " + ", ".join(profile.families))
    return {"objectifs": list(profile.goals) + [f"zone {z}" for z in profile.goal_zones], "preferences": soft,
            "contraintes_strictes": strict, "niveau": profile.experience, "seances": profile.sessions_per_week,
            "duree": profile.minutes}


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
    rows = []
    for res in out["results"][:k]:
        a = _anatomy(out, res.name)
        rows.append({"sport": res.name, "rang": a["rank"], "score": round(res.total * 100),
                     "blocs": {BLOCK_LABELS[b]: (None if res.blocks[b] is None else round(res.blocks[b] * 100)) for b in BLOCKS},
                     "criteres": _top_criteria(a, True)})
    return rows


def compare_results(old_out: dict, new_out: dict, k: int = 3) -> dict:
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
                why = [f"redevenu admissible : {'; '.join(prev['excluded_why'])} n'est plus vrai"] + why
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
    """Si aucun sport n'est admissible : quelle contrainte, levée, rendrait des sports admissibles. Rien n'est modifié."""
    candidates = []
    for lieu in profile.excluded_infrastructures:
        candidates.append((f"l'indisponibilité de {INFRA_DEFS[lieu]}",
                           replace(profile, excluded_infrastructures=[x for x in profile.excluded_infrastructures if x != lieu])))
    for g in profile.excluded_groups:
        candidates.append((f"l'exclusion des {GROUP_LABEL.get(g, g)}",
                           replace(profile, excluded_groups=[x for x in profile.excluded_groups if x != g])))
    for f in profile.excluded_families:
        candidates.append((f"l'exclusion de la famille {f}",
                           replace(profile, excluded_families=[x for x in profile.excluded_families if x != f])))
    for c in profile.excluded_categories:
        candidates.append((f"l'exclusion de la catégorie « {c} »",
                           replace(profile, excluded_categories=[x for x in profile.excluded_categories if x != c])))
    for sp in profile.excluded_sports:
        candidates.append((f"l'exclusion de {sp}", replace(profile, excluded_sports=[x for x in profile.excluded_sports if x != sp])))
    for fmt in profile.excluded_formats:
        candidates.append((f"l'exclusion du format {fmt}", replace(profile, excluded_formats=[x for x in profile.excluded_formats if x != fmt])))
    for e in profile.excluded_environments:
        candidates.append((f"l'exclusion de l'environnement {e}", replace(profile, excluded_environments=[x for x in profile.excluded_environments if x != e])))
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
    names = [a["name"] for a in out["anatomy"]]
    ranked = sorted(names, key=lambda n: ((_anatomy(out, n)["rank"] or 999), n))
    return _sports_in(fold(message), ranked)


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
                 f"| Lieux nécessaires | {ra.infrastructures.replace('|', ' ou ')} | {rb.infrastructures.replace('|', ' ou ')} |",
                 f"| Formats | {ra.format_pratique.replace('|', ', ')} | {rb.format_pratique.replace('|', ', ')} |"]
        return "\n".join(rows)
    return "Je ne sais pas répondre à cette question."


HELP_TEXT = ("Décrivez simplement ce que vous voulez, comme vous le diriez à quelqu'un. Par exemple :\n"
             "- « Je n'ai pas de mer à côté de chez moi. » · « Pas de stade près de chez moi. »\n"
             "- « Je n'aime pas les sports de balle. » · « J'aime les sports de combat. »\n"
             "- « Je préfère pratiquer seul, en extérieur, pas plus de 15 € par séance. »\n"
             "- « Finalement j'accepte les sports collectifs. » · « Pourquoi me recommandes-tu la natation ? »")


# ------------------------------------------------------------------ point d'entrée
def process_message(message: str, profile: UserProfile, out: dict, engine, weights: dict | None,
                    settings: LLMSettings, *, budget: CallBudget | None = None,
                    transport: Transport | None = None) -> dict:
    """Traite un message de bout en bout.

    kind = "maj"            : profil modifié, moteur relancé (profile, out, changes, interpretations, comparison…)
    kind = "sans_changement": message compris, aucun critère modifié : aucun recalcul
    kind = "reponse"        : réponse à une question sur les sports (aucun recalcul)
    kind = "precision"      : message ambigu, une question est posée
    kind = "aide"           : message non reconnu
    """
    vocab = vocabulary(engine)
    question = detect_question(message, out)
    if question:
        return {"kind": "reponse", "text": answer_question(question, out, profile), "source": "moteur", "note": "",
                "interpretations": []}
    read = interpret(message, profile, settings, vocab, engine, budget=budget, transport=transport)
    c = read["contract"]
    base = {"source": read["source"], "note": read["note"], "rejected": read["rejected"],
            "interpretations": c["interpretations"]}
    if c["question"]:
        return {**base, "kind": "reponse", "text": answer_question(c["question"], out, profile)}
    if contract_is_empty(c):
        if c["precisions"]:
            return {**base, "kind": "precision", "text": "", "precisions": c["precisions"]}
        if read["rejected"] and read["source"] != "règles":
            unknown = "; ".join(read["rejected"][:3])
            return {**base, "kind": "precision", "text": "", "precisions": [
                f"Je n'ai pas pu rattacher votre demande à une option connue ({unknown}). Pouvez-vous la reformuler ?"]}
        return {**base, "kind": "aide", "text": HELP_TEXT, "precisions": []}
    new_profile, changes, contradictions = apply_contract(profile, c)
    if profile_fingerprint(new_profile) == profile_fingerprint(profile):
        return {**base, "kind": "sans_changement", "precisions": c["precisions"],
                "text": "Votre message est bien compris, mais il ne modifie aucun critère de votre profil : aucun recalcul n'a été nécessaire."}
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
