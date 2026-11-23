"""Bonus A : explications personnalisées d'une recommandation, générées par un LLM sous contrôle strict.

Principe (à retenir pour la soutenance)
---------------------------------------
Le LLM ne calcule RIEN. Le moteur NLP calcule les scores ; ce module :
    1. construit un « dossier » structuré à partir des résultats réels du moteur (`build_payload`) ;
    2. demande au LLM de rédiger une explication à partir de ce dossier uniquement ;
    3. VÉRIFIE la réponse (`validate_text`) : chiffres autorisés, aucun autre sport, aucune formulation médicale ;
    4. si l'IA est indisponible ou si la réponse est rejetée, affiche l'explication classique déjà produite par
       l'application (les « bullets » de `engine.py`). L'application fonctionne donc sans IA.

Fournisseurs supportés (configurés dans `llm_config.py`) : Google Gemini (niveau gratuit) et Ollama (local).
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
    "unreachable": "le serveur local (Ollama) est injoignable",
    "disabled": "l'IA est désactivée",
    "budget": "le nombre d'appels autorisés pour cette session est atteint",
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
        gen = {"temperature": 0.2, "maxOutputTokens": 600}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen}
        try:
            status, data = send(url, headers, body, settings.timeout_s)
        except requests.Timeout as e:
            raise LLMError("timeout", str(type(e).__name__)) from e
        except requests.RequestException as e:
            raise LLMError("provider", type(e).__name__) from e
        if status == 429:
            raise LLMError("quota", f"HTTP {status}")
        if status == 404:
            raise LLMError("model_missing", f"HTTP {status}")
        if status >= 400:
            raise LLMError("provider", f"HTTP {status}")  # 400/401/403 : clé ou requête refusée
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("provider", "réponse sans texte (filtrée ou vide)") from e

    elif settings.provider == "ollama":
        url = f"{settings.ollama_url}/api/chat"
        body = {"model": settings.ollama_model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": 0.2}}
        if json_mode:
            body["format"] = "json"
        try:
            status, data = send(url, {"Content-Type": "application/json"}, body, settings.timeout_s)
        except requests.Timeout as e:
            raise LLMError("timeout", str(type(e).__name__)) from e
        except requests.ConnectionError as e:
            raise LLMError("unreachable", str(type(e).__name__)) from e
        except requests.RequestException as e:
            raise LLMError("provider", type(e).__name__) from e
        if status == 404:
            raise LLMError("model_missing", f"HTTP {status}")
        if status >= 400:
            raise LLMError("provider", f"HTTP {status}")
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise LLMError("provider", "réponse Ollama inattendue") from e
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
SYSTEM_PROMPT = """Tu expliques à un utilisateur non technique pourquoi un sport lui est recommandé.
Règles strictes :
1. Tu n'inventes RIEN : chiffres, scores, caractéristiques ou sports absents des données fournies.
2. Tu n'es pas médecin : aucun diagnostic, aucune prescription d'exercices. En cas de douleur, invite à consulter un professionnel de santé.
3. Tu ne modifies pas la recommandation : tu ne cites aucun autre sport que celui fourni.
4. Tu écris en français, en 4 à 6 phrases au maximum, sans jargon (pas de « bloc », « pondération », « similarité »).
5. Structure : pourquoi ce sport correspond au profil ; puis les points d'attention ou limites ; éventuellement un conseil prudent.
6. N'emploie que les nombres présents dans les données."""


def build_user_message(payload: dict) -> str:
    return "Données (JSON, seules sources autorisées) :\n" + json.dumps(payload, ensure_ascii=False, indent=1)


FORBIDDEN_PATTERNS = [r"diagnostic", r"diagnostiqu", r"vous souffrez", r"vous etes atteint", r"prescri", r"traitement"]


def validate_text(text: str, payload: dict, all_sport_names: list[str]) -> tuple[bool, str]:
    """Vérifie la réponse du LLM. Retourne (ok, motif de rejet)."""
    if not text or not text.strip():
        return False, "réponse vide"
    if len(text) > 1200:
        return False, "réponse trop longue"
    folded = fold(text)
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, folded):
            return False, "formulation médicale détectée"
    own = payload["sport"]["nom"]
    for name in all_sport_names:
        if name != own and len(name) >= 6 and fold(name) in folded:
            return False, f"mentionne un autre sport ({name})"
    allowed = set(payload["nombres_autorises"]) | {1.0, 2.0, 3.0, 4.0, 5.0}
    for tok in re.findall(r"\d+(?:[.,]\d+)?", text):
        if float(tok.replace(",", ".")) not in allowed:
            return False, f"chiffre non fourni par le moteur ({tok})"
    return True, ""


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
    """Produit l'explication d'un sport recommandé.

    Retourne {"source": "llm" | "classique", "text": str | None, "reason": str, "payload": dict, "lines": list}.
    `source == "classique"` signifie que l'IA n'a pas été utilisée : l'appelant affiche alors `lines`.
    """
    payload = build_payload(out, sport_name)
    lines = classic_lines(out, sport_name)

    def fallback(kind_or_msg: str) -> dict:
        msg = USER_MESSAGES.get(kind_or_msg, kind_or_msg)
        return {"source": "classique", "text": None, "reason": msg, "payload": payload, "lines": lines}

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
    return hashlib.sha1((settings.describe() + json.dumps(p, sort_keys=True)).encode("utf-8")).hexdigest()
