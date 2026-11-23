"""Configuration du fournisseur LLM (bonus IA).

Pourquoi un module à part ?
    Toute la configuration vient d'un seul endroit : fournisseur (Gemini ou désactivé), modèle, limites
    d'appels. La clé API n'est JAMAIS écrite dans le code : elle est lue dans le fichier `.env` (exclu de Git)
    ou dans les secrets Streamlit (`.streamlit/secrets.toml`, lui aussi exclu de Git).

Variables reconnues (voir `.env.example`) :
    LLM_PROVIDER      gemini | off                   (défaut : gemini)
    GEMINI_API_KEY    clé gratuite Google AI Studio  (ne jamais la committer)
    GEMINI_MODEL      identifiant du modèle Gemini   (défaut : gemini-3.5-flash, configurable)
    LLM_TIMEOUT_S     délai maximal d'un appel       (défaut : 60)
    LLM_MAX_CALLS     appels LLM max par session     (défaut : 15) : limite le quota consommé
    LLM_CHAT_USE_LLM  1 = le chatbot utilise le LLM pour lire les messages, 0 = règles seules (défaut : 1)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"


@dataclass
class LLMSettings:
    provider: str = "gemini"
    gemini_key: str = field(default="", repr=False)   # repr=False : la clé n'apparaît jamais dans les logs
    gemini_model: str = "gemini-3.5-flash"
    timeout_s: float = 60.0
    max_calls_per_session: int = 15
    chat_use_llm: bool = True

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_key.strip())

    def describe(self) -> str:
        """Résumé lisible, sans secret."""
        if self.provider == "gemini":
            return f"Gemini · {self.gemini_model}" + ("" if self.has_gemini_key else " (clé API absente)")
        return "désactivé"


def _load_dotenv_file() -> None:
    """Charge `.env` sans écraser les variables déjà définies (python-dotenv si présent, sinon lecture simple)."""
    if not ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
    except ImportError:  # python-dotenv absent : lecture minimaliste de type CLE=valeur
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _streamlit_secret(name: str) -> str:
    """Lit un secret Streamlit s'il existe (fonctionne seulement dans une application Streamlit)."""
    try:
        import streamlit as st
        return str(st.secrets.get(name, "") or "")
    except Exception:  # pas de fichier secrets, ou exécution hors Streamlit
        return ""


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def load_settings() -> LLMSettings:
    """Construit la configuration à partir de l'environnement (puis de .env et des secrets Streamlit)."""
    _load_dotenv_file()
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"gemini", "off"}:
        provider = "off"
    key = os.environ.get("GEMINI_API_KEY", "") or _streamlit_secret("GEMINI_API_KEY")
    return LLMSettings(
        provider=provider,
        gemini_key=key,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash",
        timeout_s=_float("LLM_TIMEOUT_S", 60.0),
        max_calls_per_session=_int("LLM_MAX_CALLS", 15),
        chat_use_llm=os.environ.get("LLM_CHAT_USE_LLM", "1").strip() not in {"0", "false", "non"},
    )
