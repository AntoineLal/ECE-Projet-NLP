"""Bonus B : assistant conversationnel qui affine le profil et relance le moteur NLP existant.

Architecture (à retenir pour la soutenance)
-------------------------------------------
1. `interpret`      lit le message : règles (extraction.py) + LLM optionnel en JSON, puis VALIDATION stricte.
2. `apply_updates`  fusionne les modifications validées dans le profil, sans effacer ce qui est déjà connu.
3. `engine.recommend` (moteur existant, NON remplacé) recalcule filtres, scores et classement.
4. `compare_rankings` compare l'ancien et le nouveau top 3 ; `describe_change` l'explique avec les données.
5. `answer_question` répond aux questions sur les sports à partir des résultats calculés (pas du LLM).

Le LLM ne fait que LIRE le message et proposer des champs de profil. Il ne produit jamais de score ni de classement.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import replace

from bonus_llm import CallBudget, LLMError, Transport, call_llm
from engine import ALL_GOALS, BLOCK_LABELS, BLOCKS, FAMILIES, IMPACT_TOL, TEXT_FAM, UserProfile
from extraction import extract_signals
from llm_config import LLMSettings
from preprocess import fold
from telemetry import get_logger

FORMATS = ["individuel", "duo", "équipe", "groupe"]
ENVIRONMENTS = ["intérieur", "extérieur", "indifférent"]
ZONES_ALL = ["dos", "épaules", "bras", "abdominaux", "cuisses", "hanches et fessiers", "mollets", "pieds et chevilles",
             "mains et poignets", "avant-bras"]
INTENTIONS = {"mise_a_jour", "question", "autre"}
QUESTION_TYPES = {"pourquoi", "comparer", "criteres"}
LABELS = {
    "formats": "Formats", "environnement": "Environnement", "families": "Familles appréciées",
    "avoid_families": "Familles à éviter", "goals": "Objectifs", "spare_zones": "Zones à ménager",
    "no_contact": "Sans contact", "no_water": "Sans eau", "impact_tolerance": "Tolérance aux impacts",
    "budget_seance_max": "Budget max par séance", "sessions_per_week": "Séances par semaine",
    "minutes": "Durée d'une séance", "intensity_shift": "Intensité souhaitée",
}


# ------------------------------------------------------------------ validation des champs autorisés
def _as_list(v, allowed: list[str]) -> list[str] | None:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        return None
    out = []
    for x in v:
        x = x.strip()
        if x not in allowed:
            return None  # une valeur hors liste rend la liste entière suspecte : on la rejette
        if x not in out:
            out.append(x)
    return out


def _env(v):
    m = {"interieur": "intérieur", "exterieur": "extérieur", "indifferent": "indifférent"}
    v = m.get(fold(v), v) if isinstance(v, str) else v
    return v if v in ENVIRONMENTS else None


def _fmt_list(v):
    m = {"equipe": "équipe"}
    if isinstance(v, list):
        v = [m.get(fold(x), x) if isinstance(x, str) else x for x in v]
    return _as_list(v, FORMATS)


def _bool(v):
    return v if isinstance(v, bool) else None


def _int_in(v, lo, hi, nullable=False):
    if v is None and nullable:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        return "rejet"
    return v if lo <= v <= hi else "rejet"


FIELD_SPECS = {
    "formats": lambda v: _fmt_list(v),
    "environnement": _env,
    "families": lambda v: _as_list(v, list(FAMILIES)),
    "avoid_families": lambda v: _as_list(v, list(FAMILIES)),
    "goals": lambda v: _as_list(v, ALL_GOALS),
    "spare_zones": lambda v: _as_list(v, ZONES_ALL),
    "no_contact": _bool,
    "no_water": _bool,
    "impact_tolerance": lambda v: v if v in IMPACT_TOL else None,
    # null = « budget levé » : valeur valide, convertie en None lors de l'application
    "budget_seance_max": lambda v: "levée" if v is None else _int_in(v, 0, 300),
    "sessions_per_week": lambda v: _int_in(v, 1, 7),
    "minutes": lambda v: _int_in(v, 20, 180),
    "intensity_delta": lambda v: _int_in(v, -1, 1),
}


def validate_updates(raw) -> dict:
    """Valide la sortie (JSON) du LLM ou des règles. Ne garde que les champs autorisés et les valeurs valides.

    Retourne {"intention", "updates", "rejected", "questions", "question"}.
    """
    res = {"intention": "autre", "updates": {}, "rejected": [], "questions": [], "question": None}
    if not isinstance(raw, dict):
        res["rejected"].append("réponse non structurée")
        return res
    intention = raw.get("intention", "autre")
    res["intention"] = intention if intention in INTENTIONS else "autre"
    mods = raw.get("modifications", {}) or {}
    if isinstance(mods, dict):
        for key, val in mods.items():
            spec = FIELD_SPECS.get(key)
            if spec is None:
                res["rejected"].append(key)
                continue
            value = spec(val)
            if value is None or value == "rejet":
                res["rejected"].append(key)
            else:
                res["updates"][key] = value
    q = raw.get("question")
    if isinstance(q, dict) and q.get("type") in QUESTION_TYPES:
        sports = q.get("sports", []) if isinstance(q.get("sports", []), list) else []
        res["question"] = {"type": q["type"], "sports": [str(s)[:80] for s in sports[:2]]}
    precis = raw.get("precisions", []) or []
    if isinstance(precis, list):
        res["questions"] = [str(p)[:200] for p in precis if isinstance(p, str)][:3]
    return res


# ------------------------------------------------------------------ lecture par règles (sans IA)
NUMBER_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7}


def detect_question(message: str, out: dict) -> dict | None:
    f = fold(message)
    sports = find_sports(message, out)
    if re.search(r"difference|compar|versus|\bvs\b", f) and len(sports) >= 2:
        return {"type": "comparer", "sports": sports[:2]}
    if re.search(r"pourquoi|qu.est.ce qui|explique", f):
        return {"type": "pourquoi", "sports": sports[:1]}
    if re.search(r"critere|influenc|facteurs|mon score|quoi.*score", f):
        return {"type": "criteres", "sports": []}
    return None


def interpret_rules(message: str, profile: UserProfile) -> dict:
    """Lecture du message par règles : réutilise l'extraction du moteur (négations comprises)."""
    f = fold(message)
    sig = extract_signals(message)
    upd, precis = {}, []
    avoid = sorted({TEXT_FAM[k] for k in sig.disliked if k in TEXT_FAM})
    likes = sorted({TEXT_FAM[k] for k in sig.prefs if k in TEXT_FAM})
    if avoid:
        upd["avoid_families"] = avoid
    if likes:
        upd["families"] = likes
    fmt_like = sorted({k.split(":")[1] for k in sig.prefs if k.startswith("format:")})
    if fmt_like:
        upd["formats"] = fmt_like
    env_like = {k.split(":")[1] for k in sig.prefs if k.startswith("env:")}
    env_dis = {k.split(":")[1] for k in sig.disliked if k.startswith("env:")}
    if len(env_like) > 1:
        precis.append("Préférez-vous pratiquer en intérieur ou en extérieur ?")
    elif env_like:
        upd["environnement"] = env_like.pop()
    elif env_dis and len(env_dis) == 1:
        upd["environnement"] = "extérieur" if env_dis.pop() == "intérieur" else "intérieur"
    if "no_contact" in sig.constraints:
        upd["no_contact"] = True
    if "no_water" in sig.constraints or "Eau" in avoid:
        upd["no_water"] = True
    if "joint_sensitive" in sig.constraints:
        upd["impact_tolerance"] = "faible"
    if sig.budget is not None:
        upd["budget_seance_max"] = int(sig.budget)
    if sig.goals:
        upd["goals"] = sorted(g for g in sig.goals if g in ALL_GOALS)
    if sig.spare_zones and re.search(r"mal|douleur|bless|fragil", f):
        upd["spare_zones"] = sorted(z for z in sig.spare_zones if z in ZONES_ALL)
    m = re.search(r"\b(\d)\s*(?:seances?|fois)\b", f)
    if m is None:
        m2 = re.search(r"\b(un|une|deux|trois|quatre|cinq|six|sept)\s*(?:seances?|fois)\s*(?:par|/)?\s*semaine", f)
        n = NUMBER_WORDS.get(m2.group(1)) if m2 else None
    else:
        n = int(m.group(1))
    if n and 1 <= n <= 7:
        upd["sessions_per_week"] = n
    m = re.search(r"\b(\d{2,3})\s*(?:min|minutes)\b", f)
    if m and 20 <= int(m.group(1)) <= 180:
        upd["minutes"] = int(m.group(1))
    if re.search(r"moins intense|plus doux|plus douce|plus calme|plus leger|plus legere|plus tranquille|moins fatigant", f):
        upd["intensity_delta"] = -1
    elif re.search(r"plus intense|plus dur|plus soutenu|plus sportif", f):
        upd["intensity_delta"] = 1
    negative = re.search(r"n'aime pas|je n'aime|ne veux pas|pas de |sans |je prefere|je veux plutot", f)
    if not upd and not precis and negative:
        precis.append("Pouvez-vous préciser ce que vous souhaitez éviter ou privilégier : un type d'activité, "
                      "un format (seul, en équipe…) ou un environnement (intérieur, extérieur) ?")
    raw = {"intention": "mise_a_jour" if upd else "autre", "modifications": upd, "precisions": precis}
    return validate_updates(raw)


# ------------------------------------------------------------------ lecture par LLM (JSON validé)
INTENT_SYSTEM = """Tu analyses un message d'un utilisateur d'une application de recommandation sportive.
Tu réponds UNIQUEMENT par un objet JSON valide, sans texte autour.
Schéma : {"intention": "mise_a_jour" | "question" | "autre",
          "modifications": {...},
          "question": {"type": "pourquoi" | "comparer" | "criteres", "sports": ["nom du sport"]},
          "precisions": ["question à poser si le message est ambigu"]}
Champs autorisés dans "modifications" (aucun autre) :
- formats : liste parmi individuel, duo, équipe, groupe
- environnement : intérieur | extérieur | indifférent
- families : liste de familles appréciées, parmi : FAMILLES
- avoid_families : liste de familles à éviter, parmi : FAMILLES
- goals : liste parmi : OBJECTIFS
- spare_zones : liste parmi : ZONES
- no_contact : true | false ; no_water : true | false
- impact_tolerance : faible | moyenne | élevée
- budget_seance_max : entier en euros, ou null si la limite de budget est levée
- sessions_per_week : entier de 1 à 7 ; minutes : entier de 20 à 180
- intensity_delta : -1 (moins intense), 0, ou 1 (plus intense)
Règles : n'inclus que ce qui est explicitement dit ; n'invente jamais une caractéristique physique ; si une demande
est ambiguë, n'inclus pas le champ et pose une question dans "precisions" ; n'indique aucun score ni classement."""


def _intent_prompt(message: str, profile: UserProfile) -> str:
    current = {"formats": profile.formats, "environnement": profile.environnement,
               "familles_appreciees": profile.families, "familles_a_eviter": profile.avoid_families,
               "sans_contact": profile.no_contact, "sans_eau": profile.no_water,
               "budget_max_eur": profile.budget_seance_max, "seances_par_semaine": profile.sessions_per_week,
               "duree_min": profile.minutes}
    system = (INTENT_SYSTEM.replace("FAMILLES", ", ".join(FAMILIES)).replace("OBJECTIFS", ", ".join(ALL_GOALS))
              .replace("ZONES", ", ".join(ZONES_ALL)))
    return system + "\n\nProfil actuel (pour contexte, ne pas le recopier) : " + json.dumps(current, ensure_ascii=False)


def interpret_llm(message: str, profile: UserProfile, settings: LLMSettings, transport: Transport | None = None) -> dict:
    """Demande au LLM un JSON d'intention. Lève LLMError ou ValueError si la réponse n'est pas exploitable."""
    text = call_llm(_intent_prompt(message, profile), f"Message de l'utilisateur : « {message[:600]} »",
                    settings, json_mode=True, transport=transport)
    match = re.search(r"\{.*\}", text or "", flags=re.S)  # on isole l'objet JSON même entouré de texte parasite
    if not match:
        raise ValueError("réponse sans JSON")
    return validate_updates(json.loads(match.group(0)))


def interpret(message: str, profile: UserProfile, settings: LLMSettings, *, out: dict | None = None,
              budget: CallBudget | None = None, transport: Transport | None = None) -> dict:
    """Interprète un message : règles d'abord (toujours), puis LLM si disponible. Les deux résultats sont fusionnés,
    le LLM ayant la priorité sur les valeurs qu'il fournit. Retourne aussi `source` et `note` pour l'affichage."""
    question = detect_question(message, out) if out is not None else None
    rules = interpret_rules(message, profile)
    res = {**rules, "source": "règles", "note": ""}
    if question is not None:
        return {**res, "intention": "question", "question": question}
    use_llm = settings.chat_use_llm and settings.provider != "off" and not (
        settings.provider == "gemini" and not settings.has_gemini_key)
    if use_llm and (budget is None or budget.available()):
        if budget is not None:
            budget.consume()
        try:
            llm = interpret_llm(message, profile, settings, transport)
            merged = {**rules["updates"], **llm["updates"]}
            intention = llm["intention"] if llm["intention"] != "autre" else rules["intention"]
            if intention == "question" and llm["question"] is None:
                intention = "autre"
            res = {"intention": intention, "updates": merged, "rejected": llm["rejected"],
                   "questions": llm["questions"] or rules["questions"], "question": llm["question"],
                   "source": "IA + règles", "note": ""}
        except (LLMError, ValueError) as e:
            kind = getattr(e, "kind", "réponse illisible")
            get_logger().warning("chatbot : analyse LLM indisponible (%s), règles seules", kind)
            res["note"] = "Analyse par règles (l'IA n'était pas disponible)."
    elif use_llm:
        res["note"] = "Analyse par règles (limite d'appels IA atteinte pour cette session)."
    return res


def find_sports(message: str, out: dict | None) -> list[str]:
    """Sports cités dans le message, du plus précis au plus général. Préférence aux sports déjà classés."""
    if out is None:
        return []
    f = " " + fold(message) + " "
    ranked = [a["name"] for a in sorted(out["anatomy"], key=lambda a: (a["rank"] or 999, a["name"]))]
    hits = []
    for name in ranked:
        full = fold(name)
        first = full.split()[0]
        if f" {full} " in f or (len(first) >= 5 and f" {first} " in f):
            hits.append((0 if f" {full} " in f else 1, ranked.index(name), name))
    return [h[2] for h in sorted(hits, key=lambda h: (h[0], h[1], len(h[2])))]


# ------------------------------------------------------------------ fusion dans le profil
def apply_updates(profile: UserProfile, updates: dict, message: str = "") -> tuple[UserProfile, list[str]]:
    """Applique les modifications validées. Règle : ce qui n'est pas mentionné est conservé.

    - formats et environnement : une nouvelle valeur remplace l'ancienne ;
    - familles : une nouvelle famille s'ajoute, et s'enlève des familles à éviter (et inversement) ;
    - intensité : le décalage s'accumule, borné à [-2, 2] ;
    - le message est ajouté au texte du profil pour que l'analyse NLP le prenne en compte.
    """
    new = {}
    changes = []
    fams, avoid = list(profile.families), list(profile.avoid_families)
    if "families" in updates:
        for f in updates["families"]:
            if f not in fams:
                fams.append(f)
            if f in avoid:
                avoid.remove(f)
    if "avoid_families" in updates:
        for f in updates["avoid_families"]:
            if f not in avoid:
                avoid.append(f)
            if f in fams:
                fams.remove(f)
    if fams != profile.families:
        changes.append(f"{LABELS['families']} : {', '.join(fams) or '—'}")
    if avoid != profile.avoid_families:
        changes.append(f"{LABELS['avoid_families']} : {', '.join(avoid) or '—'}")
    new["families"], new["avoid_families"] = fams, avoid
    if "Eau" in avoid and not profile.no_water:
        updates = {**updates, "no_water": True}
    if "formats" in updates and updates["formats"] != profile.formats:
        new["formats"] = updates["formats"]
        changes.append(f"{LABELS['formats']} : {', '.join(updates['formats'])}")
    if "environnement" in updates and updates["environnement"] != profile.environnement:
        new["environnement"] = updates["environnement"]
        changes.append(f"{LABELS['environnement']} : {updates['environnement']}")
    if "goals" in updates:
        goals = list(dict.fromkeys(profile.goals + updates["goals"]))
        if goals != profile.goals:
            new["goals"] = goals
            changes.append(f"{LABELS['goals']} : {', '.join(goals)}")
    if "spare_zones" in updates:
        zones = list(dict.fromkeys(profile.spare_zones + updates["spare_zones"]))
        if zones != profile.spare_zones:
            new["spare_zones"] = zones
            changes.append(f"{LABELS['spare_zones']} : {', '.join(zones)}")
    for key, attr in (("no_contact", "no_contact"), ("no_water", "no_water"), ("impact_tolerance", "impact_tolerance"),
                      ("budget_seance_max", "budget_seance_max"), ("sessions_per_week", "sessions_per_week"),
                      ("minutes", "minutes")):
        if key not in updates:
            continue
        value = None if updates[key] == "levée" else updates[key]
        if value != getattr(profile, attr):
            new[attr] = value
            shown = 'levée' if value is None and key == 'budget_seance_max' else ('oui' if value is True else 'non' if value is False else value)
            changes.append(f"{LABELS[key]} : {shown}")
    if "intensity_delta" in updates and updates["intensity_delta"]:
        shift = max(-2, min(2, profile.intensity_shift + updates["intensity_delta"]))
        if shift != profile.intensity_shift:
            new["intensity_shift"] = shift
            changes.append(f"{LABELS['intensity_shift']} : " + ("plus douce" if shift < 0 else "plus soutenue" if shift > 0 else "niveau habituel"))
    if message and new:
        new["text"] = (profile.text + " " + message).strip()
    return replace(profile, **new), changes


# ------------------------------------------------------------------ comparaison avant / après
def compare_rankings(old_out: dict, new_out: dict, k: int = 3) -> dict:
    old = [r.name for r in old_out["results"][:k]]
    new = [r.name for r in new_out["results"][:k]]
    old_scores = {r.name: r.total for r in old_out["results"]}
    new_scores = {r.name: r.total for r in new_out["results"]}
    old_all = {a["name"]: a["rank"] for a in old_out["anatomy"]}
    new_all = {a["name"]: a["rank"] for a in new_out["anatomy"]}
    moves = []
    for name in new:
        before = old_all.get(name)
        after = new_all.get(name)
        if name in old_scores and name in new_scores:
            delta = (new_scores[name] - old_scores[name]) * 100
        else:
            delta = None
        moves.append({"sport": name, "avant": before, "apres": after,
                      "delta_score": None if delta is None else round(delta, 1)})
    return {"avant": old, "apres": new, "entrees": [n for n in new if n not in old],
            "sorties": [n for n in old if n not in new], "mouvements": moves,
            "identique": old == new}


def describe_change(cmp: dict, changes: list[str]) -> str:
    lines = []
    if changes:
        lines.append("**Modifications du profil :** " + " ; ".join(changes) + ".")
    if cmp["identique"]:
        lines.append("Le top 3 reste identique après cette modification.")
        return "\n\n".join(lines)
    lines.append("**Nouveau top 3 :** " + " · ".join(f"{i}. {n}" for i, n in enumerate(cmp["apres"], 1)))
    if cmp["entrees"]:
        lines.append("**Entrent dans le top 3 :** " + ", ".join(cmp["entrees"]) + ".")
    if cmp["sorties"]:
        lines.append("**Sortent du top 3 :** " + ", ".join(cmp["sorties"]) + ".")
    for m in cmp["mouvements"]:
        if m["avant"] is not None and m["apres"] is not None and m["avant"] != m["apres"]:
            sens = "monte" if m["apres"] < m["avant"] else "descend"
            lines.append(f"- {m['sport']} {sens} : du rang {m['avant']} au rang {m['apres']}.")
    return "\n\n".join(lines)


# ------------------------------------------------------------------ réponses sur les sports
def _anatomy(out: dict, name: str) -> dict:
    return next(a for a in out["anatomy"] if a["name"] == name)


def _top_criteria(a: dict, high: bool, k: int = 3) -> list[str]:
    items = []
    for b in BLOCKS:
        for lab, sc in a["criteria"][b]:
            if (high and sc >= 0.7) or (not high and sc < 0.45):
                items.append((sc if high else -sc, f"{lab} ({BLOCK_LABELS[b].lower()})"))
    return [t for _, t in sorted(items, reverse=high)[:k]]


def answer_question(q: dict, out: dict, profile: UserProfile) -> str:
    kind, sports = q["type"], q.get("sports", [])
    if kind == "criteres":
        top = out["results"][0].name if out["results"] else None
        weights = " · ".join(f"{BLOCK_LABELS[b]} {out['weights'][b] * 100:.0f} %" for b in BLOCKS)
        txt = f"Votre score repose sur quatre blocs, pondérés ainsi : {weights}.\n\n"
        if top:
            a = _anatomy(out, top)
            txt += f"Pour **{top}** (n°1) : points forts → {', '.join(_top_criteria(a, True)) or '—'} ; "
            txt += f"points d'attention → {', '.join(_top_criteria(a, False)) or 'aucun'}."
        return txt
    if not sports:
        return ("Je n'ai pas reconnu de sport dans votre question. Citez un sport de la liste, par exemple "
                "« Pourquoi me recommandes-tu la natation ? ».")
    if kind == "pourquoi":
        a = _anatomy(out, sports[0])
        if not a["retained"]:
            return f"**{a['name']}** a été écarté : " + " ; ".join(a["excluded_why"]) + "."
        head = (f"**{a['name']}** est classé n°{a['rank']} avec un score de {a['total'] * 100:.0f} %"
                + (" (dans votre top 5)." if a["in_top"] else " (hors top 5)."))
        pos = _top_criteria(a, True)
        neg = _top_criteria(a, False)
        txt = head + f"\n\n- **Ce qui le favorise** : {', '.join(pos) or 'aucun point fort marqué'}."
        txt += f"\n- **Points d'attention** : {', '.join(neg + a['pen_why']) or 'aucun'}."
        return txt
    if kind == "comparer":
        a, b = _anatomy(out, sports[0]), _anatomy(out, sports[1])
        rows = ["| | " + a["name"] + " | " + b["name"] + " |", "|---|---|---|"]
        def pct(x):
            return "—" if x is None else f"{x * 100:.0f} %"
        def rk(x):
            return f"n°{x['rank']}" if x['retained'] else "écarté : " + " ; ".join(x['excluded_why'])
        rows.append(f"| Rang | {rk(a)} | {rk(b)} |")
        rows.append(f"| Score global | {pct(a['total']) if a['retained'] else '—'} | {pct(b['total']) if b['retained'] else '—'} |")
        gaps = []
        for blk in BLOCKS:
            va = a['blocks'][blk] if a['retained'] else None
            vb = b['blocks'][blk] if b['retained'] else None
            rows.append(f"| {BLOCK_LABELS[blk]} | {pct(va)} | {pct(vb)} |")
            if va is not None and vb is not None:
                gaps.append((abs(a["blocks"][blk] - b["blocks"][blk]), blk))
        ra, rb = a["row"], b["row"]
        rows.append(f"| Intensité (sur 5) | {int(ra.intensite_loisir_1_5)} | {int(rb.intensite_loisir_1_5)} |")
        rows.append(f"| Impacts (sur 3) | {int(ra.impacts_1_3)} | {int(rb.impacts_1_3)} |")
        rows.append(f"| Contact | {ra.contact} | {rb.contact} |")
        rows.append(f"| Environnement | {ra.environnement.replace('|', ', ')} | {rb.environnement.replace('|', ', ')} |")
        rows.append(f"| Formats | {ra.format_pratique.replace('|', ', ')} | {rb.format_pratique.replace('|', ', ')} |")
        txt = "\n".join(rows)
        if gaps:
            _, blk = max(gaps)
            winner = a["name"] if (a["blocks"][blk] or 0) >= (b["blocks"][blk] or 0) else b["name"]
            txt += f"\n\nL'écart le plus marqué concerne **{BLOCK_LABELS[blk].lower()}**, où **{winner}** est mieux placé."
        return txt
    return "Je ne sais pas répondre à cette question."


def unknown_sport_hint(message: str, names: list[str]) -> str:
    close = difflib.get_close_matches(fold(message), [fold(n) for n in names], n=1, cutoff=0.6)
    return f"Je n'ai pas trouvé ce sport dans les {len(names)} disciplines." + (f" Vouliez-vous dire {close[0]} ?" if close else "")


def process_message(message: str, profile: UserProfile, out: dict, engine, weights: dict, settings: LLMSettings, *,
                    budget: CallBudget | None = None, transport: Transport | None = None) -> dict:
    """Traite un message de bout en bout. Retourne un dictionnaire décrivant la réponse à afficher.

    kind = "maj"      : profil modifié, moteur relancé (contient profile, out, changes, comparison)
    kind = "reponse"  : réponse à une question sur les sports (aucun recalcul)
    kind = "precision": message ambigu, l'assistant demande une précision
    kind = "aide"     : message non reconnu
    """
    res = interpret(message, profile, settings, out=out, budget=budget, transport=transport)
    base = {"source": res["source"], "note": res["note"]}
    if res["intention"] == "question" and res["question"]:
        return {**base, "kind": "reponse", "text": answer_question(res["question"], out, profile)}
    if res["updates"]:
        new_profile, changes = apply_updates(profile, res["updates"], message)
        if not changes:
            return {**base, "kind": "aide", "text": "Ce message ne modifie pas votre profil : je l'ai bien lu, mais "
                                                    "l'information était déjà prise en compte."}
        new_out = engine.recommend(new_profile, weights)  # moteur existant, non dupliqué
        cmp = compare_rankings(out, new_out)
        get_logger().info("chatbot : profil mis à jour (%d champ(s)), top 3 %s", len(changes),
                          "modifié" if not cmp["identique"] else "inchangé")
        return {**base, "kind": "maj", "text": describe_change(cmp, changes), "profile": new_profile,
                "out": new_out, "changes": changes, "comparison": cmp}
    if res["questions"]:
        return {**base, "kind": "precision", "text": "\n".join(f"- {q}" for q in res["questions"])}
    return {**base, "kind": "aide", "text": HELP_TEXT}


HELP_TEXT = ("Je peux :\n- modifier votre profil : « je n'aime pas les sports aquatiques », « je préfère des activités "
             "individuelles en extérieur », « je m'entraîne 4 fois par semaine », « un budget de 15 € par séance » ;\n"
             "- répondre à vos questions : « Pourquoi me recommandes-tu la natation ? », « Quelle différence entre le tennis "
             "et le badminton ? », « Quels critères ont influencé mon score ? ».")
