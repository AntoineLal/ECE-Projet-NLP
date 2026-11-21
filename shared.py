"""Ressources partagées entre la page principale et l'admin (modèle SBERT, moteur)."""
import time

import streamlit as st

from engine import Recommender
from telemetry import get_logger


class SbertEmbedder:
    name = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name)

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


@st.cache_resource(show_spinner="Chargement du modèle SBERT et indexation des 70 sports…")
def get_engine():
    """Retourne (moteur, erreur, secondes de démarrage)."""
    lg, t = get_logger(), time.perf_counter()
    try:
        eng, err = Recommender(SbertEmbedder()), None
        lg.info("Moteur démarré avec SBERT (%s)", SbertEmbedder.name)
    except Exception as e:
        eng, err = Recommender(None), str(e)
        lg.error("SBERT indisponible, repli TF-IDF : %s", e)
    return eng, err, time.perf_counter() - t
