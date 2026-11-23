"""Bonus A : explications personnalisées d'une recommandation, générées par un LLM sous contrôle strict.

Principe (à retenir pour la soutenance)
---------------------------------------
Le LLM ne calcule RIEN. Le moteur NLP calcule les scores ; ce module :
    1. construit un « dossier » structuré à partir des résultats réels du moteur (`build_payload`) ;
    2. demande au LLM de rédiger une explication à partir de ce dossier uniquement ;
    3. VÉRIFIE la réponse (`validate_text`) : chiffres autorisés, aucun autre sport, aucune formulation médicale ;
    4. si l'IA est indisponible ou si la réponse est rejetée, affiche l'explication classique déjà produite par
       l'application (les « bullets » de `engine.py`). L'application fonctionne donc sans IA.

Fournisseur supporté (configuré dans `llm_config.py`) : Google Gemini (niveau gratuit).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Callable

import requests

from engine import BLOCK_LABELS, BLOCKS
from llm_config import LLMSettings
from preprocess import fold
from telemetry import get_logger

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Un transport est une fonction (url, en-têtes, corps, délai) -> (code HTTP, JSON). Injectée dans les tests
# pour simuler le LLM sans consommer de quota.
Transport = Callable[[str, dict, dict, float], tuple[int, dict]]

USER_MESSAGES = {
    "no_key": "aucune clé API Gemini n'est configurée",
    "quota": "le quota gratuit est épuisé ou la limite d'appels est atteinte",
    "provider": "le fournisseur a renvoyé une erreur",
    "model_missing": "le modèle configuré n'est pas disponible chez le fournisseur",
    "timeout": "le délai de réponse a été dépassé",
    "disabled": "l'IA est désactivée",
    "budget": "le nombre d'appels autorisés pour cette session est atteint",
    "truncated": "la réponse de l'IA a été coupée avant la fin",
}


class LLMError(Exception):
    """Erreur d'appel au LLM. `kind` est une clé de USER_MESSAGES, jamais un détail technique affiché tel quel."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(kind)
        self.kind = kind
        self.detail = detail  # journalisé, non affiché à l'utilisateur


def _http_post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict]:
    """Appel HTTP réel (requests). Les en-têtes contiennent la clé : ne jamais les journaliser."""
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        data = r.json()
    except ValueError:
        data = {}
    return r.status_code, data


def _send_with_retry(send, url, headers, body, timeout):
    """Un seul nouvel essai si le service est surchargé (503), avec une courte pause. Les autres erreurs ne sont pas relancées."""
    for attempt in (1, 2):
        try:
            status, data = send(url, headers, body, timeout)
        except requests.Timeout as e:
            raise LLMError("timeout", str(type(e).__name__)) from e
        except requests.RequestException as e:
            raise LLMError("provider", type(e).__name__) from e
        if status != 503 or attempt == 2:
            return status, data
        time.sleep(2)


def call_llm(system: str, user: str, settings: LLMSettings, *, json_mode: bool = False,
             transport: Transport | None = None) -> str:
    """Envoie une requête au fournisseur configuré et renvoie le texte de la réponse.

    Lève LLMError(kind) en cas d'échec ; l'appelant décide alors du repli.
    """
    send = transport or _http_post
    t0 = time.perf_counter()
    if settings.provider == "off":
        raise LLMError("disabled")

    if settings.provider == "gemini":
        if not settings.has_gemini_key:
            raise LLMError("no_key")
        url = GEMINI_URL.format(model=settings.gemini_model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": settings.gemini_key}
        # Les modèles Gemini 3 consomment des tokens de « réflexion » sur le même budget que la réponse :
        # 600 tokens coupaient le texte (finishReason MAX_TOKENS). On donne donc plus de place et une réflexion légère.
        # marge confortable : la réflexion interne (≈ 1 000 tokens) et le texte partagent ce budget
        gen = {"temperature": 0.2, "maxOutputTokens": 4096}
        if settings.gemini_model.startswith("gemini-3"):
            gen["thinkingConfig"] = {"thinkingLevel": "low"}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen}
        status, data = _send_with_retry(send, url, headers, body, settings.timeout_s)
        if status == 429:
            raise LLMError("quota", f"HTTP {status}")
        if status == 404:
            raise LLMError("model_missing", f"HTTP {status}")
        if status >= 400:
            raise LLMError("provider", f"HTTP {status}")  # 400/401/403 : clé ou requête refusée
        try:
            candidate = data["candidates"][0]
            text = candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("provider", "réponse sans texte (filtrée ou vide)") from e
        # finishReason indique POURQUOI la génération s'est arrêtée. STOP = fin normale.
        # MAX_TOKENS = limite atteinte : le texte est coupé, il ne doit jamais être affiché.
        reason = candidate.get("finishReason", "STOP")
        if reason == "MAX_TOKENS":
            raise LLMError("truncated", f"finishReason={reason}, {len(text)} caractères")
        if reason != "STOP":
            raise LLMError("provider", f"finishReason={reason}")

    else:
        raise LLMError("disabled")

    get_logger().info("LLM %s · %d ms · %d caractères reçus", settings.describe().split(" (")[0],
                      int((time.perf_counter() - t0) * 1000), len(text or ""))
    return text or ""


# ------------------------------------------------------------------ dossier transmis au LLM
def _pct(x) -> int | None:
    return None if x is None else int(round(float(x) * 100))


def _numbers(obj, acc: set) -> set:
    """Collecte tous les nombres du dossier : seuls ceux-là peuvent apparaître dans l'explication."""
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, acc)
    return acc


def constraint_lines(R, row) -> list[str]:
    """Contraintes déjà vérifiées par le moteur, formulées en phrases simples."""
    out = []
    if R.no_contact:
        out.append(f"Sans contact physique : ce sport a un niveau de contact {int(row.niveau_contact)} sur 3.")
    if R.no_water:
        out.append("Sans activité dans l'eau : respecté." if not (set(row.categories.split("|")) & {"sport aquatique", "nage", "sport nautique", "pagaie", "rame", "navigation à voile", "glisse sur l'eau"}) else "Sans activité dans l'eau : ce sport se pratique dans l'eau.")
    for j in R.injured_joints:
        out.append(f"Articulation douloureuse ({j}) : ce sport la charge à {int(row[f'art_{j}'])} sur 3.")
    if R.p.budget_seance_max is not None:
        out.append(f"Budget maximum {R.p.budget_seance_max} € par séance ; ce sport coûte de {int(row.budget_seance_min_eur)} à {int(row.budget_seance_max_eur)} €.")
    return out


def build_payload(out: dict, sport_name: str) -> dict:
    """Dossier structuré pour le LLM, construit UNIQUEMENT à partir des résultats calculés par le moteur.

    Le texte libre de l'utilisateur n'est pas transmis : seuls les champs structurés le sont (minimisation des données).
    """
    a = next(x for x in out["anatomy"] if x["name"] == sport_name)
    row, R = a["row"], out["resolved"]
    forts, faibles = [], []
    for b in BLOCKS:
        for lab, sc in a["criteria"][b]:
            if sc >= 0.7:
                forts.append({"critere": f"{BLOCK_LABELS[b]} : {lab}", "score_pct": _pct(sc)})
            elif sc < 0.45:
                faibles.append({"critere": f"{BLOCK_LABELS[b]} : {lab}", "score_pct": _pct(sc)})
    payload = {
        "sport": {
            "nom": row.nom_sport,
            "categories": row.categories.replace("|", ", "),
            "description": row.description_fr,
            "lieu": row.lieu_pratique,
            "formats": row.format_pratique.replace("|", ", "),
            "environnement": row.environnement.replace("|", ", "),
            "contact": row.contact,
            "budget_seance_eur": [int(row.budget_seance_min_eur), int(row.budget_seance_max_eur)],
        },
        "rang_recommandation": a["rank"],
        "score_global_pct": _pct(a["total"]),
        "scores_par_bloc_pct": {BLOCK_LABELS[b]: _pct(a["blocks"][b]) for b in BLOCKS},
        "poids_des_blocs_pct": {BLOCK_LABELS[b]: _pct(a["weights"][b]) for b in BLOCKS},
        "criteres_forts": sorted(forts, key=lambda c: -c["score_pct"])[:6],
        "criteres_faibles": sorted(faibles, key=lambda c: c["score_pct"])[:4],
        "garde_fous_appliques": a["pen_why"],
        "contraintes_verifiees": constraint_lines(R, row),
        "profil": {
            "niveau": R.exp_label,
            "endurance_sur_5": R.abilities["endurance"],
            "force_sur_5": R.abilities["force"],
            "mobilite_sur_5": R.abilities["souplesse"],
            "intensite_cible_sur_5": round(float(R.target_intensity), 1),
            "seances_par_semaine": R.p.sessions_per_week,
            "duree_seance_min": R.p.minutes,
            "objectifs": R.goals,
            "zones_a_renforcer": R.goal_zones,
            "formats_souhaites": R.formats,
            "environnement_souhaite": R.envs or [R.p.environnement],
            "familles_appreciees": R.families,
            "familles_a_eviter": R.disliked_families,
            "articulations_douloureuses": R.injured_joints,
            "zones_a_menager": R.spare_hard + R.spare_soft,
            "sans_contact": R.no_contact,
            "sans_eau": R.no_water,
            "budget_max_seance_eur": R.p.budget_seance_max,
        },
    }
    payload["nombres_autorises"] = sorted(_numbers(payload, set()))
    return payload


# ------------------------------------------------------------------ consignes et vérification
PROMPT_VERSION = "v2"  # changer cette valeur invalide le cache des anciennes explications

SYSTEM_PROMPT = """Tu rédiges, en français, une explication personnalisée d'une recommandation sportive, pour un utilisateur non technique.

Règles strictes :
1. Tu n'inventes RIEN : chiffres, scores, caractéristiques ou sports absents des données JSON fournies.
2. Tu reprends EXACTEMENT le score global et les scores par bloc donnés, sans les arrondir ni les modifier.
3. Tu n'es pas médecin : aucun diagnostic, aucune prescription d'exercices. En cas de douleur, conseille de consulter un professionnel de santé.
4. Tu ne cites aucun autre sport que celui fourni, et tu n'emploies pas de jargon (ni « bloc », « pondération », « similarité », « score structuré »).

Structure attendue, en 5 à 8 phrases complètes :
a) Pourquoi ce sport correspond à votre profil : adéquation avec les capacités physiques déclarées (endurance, force, mobilité, niveau).
b) Comment il répond à vos objectifs déclarés (objectifs et zones à renforcer).
c) Comment il respecte vos préférences (formats, environnement, types d'activités appréciés).
d) Les limites ou points d'attention (critères faibles, garde-fous, contraintes), sans rien inventer.
e) Une conclusion pertinente, qui résume la recommandation en une phrase.

Termine toujours par une phrase complète qui se termine par un point.""" + ""


def build_user_message(payload: dict) -> str:
    return "Données (JSON, seules sources autorisées) :\n" + json.dumps(payload, ensure_ascii=False, indent=1)


FORBIDDEN_PATTERNS = [r"diagnostic", r"diagnostiqu", r"vous souffrez", r"vous etes atteint", r"prescri", r"traitement"]


def validate_text(text: str, payload: dict, all_sport_names: list[str]) -> tuple[bool, str]:
    """Vérifie la réponse du LLM. Retourne (ok, motif de rejet).

    Ordre des contrôles : d'abord la sécurité et l'exactitude (médical, autres sports, chiffres, score),
    puis la forme (longueur, phrase complète).
    """
    if not text or not text.strip():
        return False, "réponse vide"
    folded = fold(text)
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, folded):
            return False, "formulation médicale détectée"
    own = payload["sport"]["nom"]
    for name in all_sport_names:
        if name != own and len(name) >= 6 and fold(name) in folded:
            return False, f"mentionne un autre sport ({name})"
    allowed = set(payload["nombres_autorises"]) | {1.0, 2.0, 3.0, 4.0, 5.0}
    found = set()
    for tok in re.findall(r"\d+(?:[.,]\d+)?", text):
        value = float(tok.replace(",", "."))
        if value not in allowed:
            return False, f"chiffre non fourni par le moteur ({tok})"
        found.add(value)
    # cohérence : le score global du moteur doit apparaître tel quel dans l'explication
    if float(payload["score_global_pct"]) not in found:
        return False, "score global absent ou différent de celui du moteur"
    if len(text) > 1500:
        return False, "réponse trop longue"
    if len(text.strip()) < 120:
        return False, "réponse trop courte"
    if text.strip()[-1] not in ".!?»)":
        return False, "phrase incomplète (texte coupé)"
    return True, ""


def template_explanation(payload: dict) -> str:
    """Explication complète construite UNIQUEMENT à partir des résultats du moteur (sans IA).

    Utilisée quand l'IA est indisponible ou quand sa réponse est tronquée ou rejetée. Chaque phrase est complète,
    et tous les chiffres viennent du dossier calculé par le moteur.
    """
    s, p = payload["sport"], payload["profil"]
    b = payload["scores_par_bloc_pct"]
    score = payload["score_global_pct"]
    rank = payload["rang_recommandation"]
    adjective = ("très pertinente" if score >= 70 else "pertinente" if score >= 50 else "à envisager avec prudence")
    out = [f"{s['nom']} est recommandé au rang {rank} avec un score global de {score} %, ce qui le rend {adjective} pour votre profil."]
    forts = [c["critere"].split(" : ", 1)[-1] for c in payload["criteres_forts"][:2]]
    if forts:
        out.append(f"Il s'appuie sur des points forts pour vous : {' et '.join(forts)}.")
    if b.get("Condition physique") is not None:
        out.append(f"Sur le plan physique, son adéquation avec votre niveau « {p['niveau']} » est évaluée à {b['Condition physique']} %.")
    objectifs = p.get("objectifs", []) + [f"zone {z}" for z in p.get("zones_a_renforcer", [])]
    if objectifs and b.get("Objectifs") is not None:
        out.append(f"Il répond à vos objectifs ({', '.join(objectifs)}) à hauteur de {b['Objectifs']} %.")
    prefs = []
    if p.get("formats_souhaites"):
        prefs.append("format " + " ou ".join(p["formats_souhaites"]))
    if p.get("environnement_souhaite"):
        prefs.append("environnement " + " ou ".join(p["environnement_souhaite"]))
    if prefs and b.get("Préférences") is not None:
        out.append(f"Il correspond à vos préférences ({', '.join(prefs)}) : {b['Préférences']} %.")
    if b.get("Contraintes") is not None:
        out.append(f"Vos contraintes déclarées sont prises en compte à {b['Contraintes']} %.")
    limites = [c["critere"].split(" : ", 1)[-1] for c in payload["criteres_faibles"][:2]] + payload["garde_fous_appliques"]
    if limites:
        out.append(f"Points d'attention : {' ; '.join(limites)}.")
    contraintes = payload.get("contraintes_verifiees", [])
    if contraintes:
        out.append(f"Les contraintes déclarées sont vérifiées : {' '.join(c.rstrip('.') + '.' for c in contraintes[:2])}")
    out.append("En résumé, il s'agit d'une piste à essayer en progressant, et en cas de douleur, consultez un professionnel de santé.")
    return " ".join(sentence.rstrip() for sentence in out)


# ------------------------------------------------------------------ budget d'appels et repli
class CallBudget:
    """Compte les appels LLM d'une session : limite le quota consommé (stocké dans st.session_state par l'appelant)."""

    def __init__(self, limit: int, used: int = 0):
        self.limit, self.used = limit, used

    def available(self) -> bool:
        return self.used < self.limit

    def consume(self) -> None:
        self.used += 1


def classic_lines(out: dict, sport_name: str) -> list[dict]:
    """Explication classique déjà produite par l'application (puces du moteur)."""
    for res in out["results"]:
        if res.name == sport_name:
            return res.bullets
    return []


def explain_sport(out: dict, sport_name: str, settings: LLMSettings, *, budget: CallBudget | None = None,
                  transport: Transport | None = None) -> dict:
    """Produit l'explication complète d'un sport recommandé.

    Retourne toujours un texte complet :
      - source "llm"    : texte rédigé par Gemini, vérifié (fin de phrase, score exact, chiffres autorisés) ;
      - source "moteur" : explication construite à partir des résultats NLP (IA absente, quota, erreur, texte
                          tronqué ou rejeté). `reason` indique pourquoi l'IA n'a pas été utilisée.
    `lines` contient en plus les puces classiques du moteur.
    """
    payload = build_payload(out, sport_name)
    lines = classic_lines(out, sport_name)
    template = template_explanation(payload)

    def fallback(kind_or_msg: str) -> dict:
        msg = USER_MESSAGES.get(kind_or_msg, kind_or_msg)
        return {"source": "moteur", "text": template, "reason": msg, "payload": payload, "lines": lines}

    if settings.provider == "off":
        return fallback("disabled")
    if settings.provider == "gemini" and not settings.has_gemini_key:
        return fallback("no_key")
    if budget is not None and not budget.available():
        return fallback("budget")
    if budget is not None:
        budget.consume()
    try:
        text = call_llm(SYSTEM_PROMPT, build_user_message(payload), settings, transport=transport)
    except LLMError as e:
        get_logger().warning("explication IA indisponible : %s (%s)", e.kind, e.detail)
        return fallback(e.kind)
    ok, why = validate_text(text, payload, [a["name"] for a in out["anatomy"]])
    if not ok:
        get_logger().warning("explication LLM rejetée : %s", why)
        return fallback(f"réponse rejetée par la vérification ({why})")
    return {"source": "llm", "text": text.strip(), "reason": "", "payload": payload, "lines": lines}


def cache_key(out: dict, sport_name: str, settings: LLMSettings) -> str:
    """Clé de cache : une même recommandation n'est expliquée qu'une fois (économie de quota)."""
    p = build_payload(out, sport_name)
    raw = PROMPT_VERSION + "|" + settings.describe() + "|" + json.dumps(p, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
