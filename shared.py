"""Ressources partagées entre la page principale, l'admin et le benchmark.

Chaque modèle NLP est chargé une seule fois et mis en cache séparément (st.cache_resource).
"""
import time

import streamlit as st

from engine import Recommender
from telemetry import get_logger

# Modèles comparés. hf=None : TF-IDF seul, sans réseau de neurones.
MODELS = {
    "tfidf": {"label": "TF-IDF seul (lexical)", "hf": None},
    "minilm": {"label": "SBERT MiniLM multilingue", "hf": "paraphrase-multilingual-MiniLM-L12-v2"},
    "camembert": {"label": "CamemBERT (sentence-camembert, français)", "hf": "dangvantuan/sentence-camembert-base"},
    "mpnet": {"label": "MPNet multilingue", "hf": "paraphrase-multilingual-mpnet-base-v2"},
}
DEFAULT_MODEL = "minilm"


def model_label(key: str) -> str:
    return MODELS.get(key, {}).get("label", key)


def normalize_model_key(raw: str | None) -> str:
    """Accepte les anciennes étiquettes de logs (« SBERT+TFIDF », « TFIDF ») et renvoie une clé de MODELS."""
    if raw in MODELS:
        return raw
    return {"SBERT+TFIDF": "minilm", "TFIDF": "tfidf"}.get(raw, "inconnu")


class SentenceEmbedder:
    def __init__(self, key: str):
        from sentence_transformers import SentenceTransformer
        self.key = key
        self.name = MODELS[key]["hf"]
        self.model = SentenceTransformer(self.name)

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


def build_engine(key: str):
    """Construit un moteur pour le modèle donné. Retourne (moteur, erreur, secondes)."""
    lg, t = get_logger(), time.perf_counter()
    if MODELS[key]["hf"] is None:
        return Recommender(None), None, time.perf_counter() - t
    try:
        eng = Recommender(SentenceEmbedder(key))
        lg.info("Moteur démarré avec %s", MODELS[key]["hf"])
        return eng, None, time.perf_counter() - t
    except Exception as e:  # modèle indisponible : repli TF-IDF
        lg.error("Modèle %s indisponible, repli TF-IDF : %s", key, e)
        return Recommender(None), str(e), time.perf_counter() - t


@st.cache_resource(show_spinner="Chargement du modèle et indexation des 106 sports…")
def get_engine(key: str = DEFAULT_MODEL):
    """Retourne (moteur, erreur, secondes de démarrage) pour le modèle choisi, en cache."""
    return build_engine(key)
