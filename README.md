# ECE-Projet-NLP — Recommandation intelligente d'activités sportives

Thème 7 : recommandation d'activités sportives fondée sur l'analyse sémantique du profil physique et des objectifs
personnels. Outil d'aide à la **découverte** d'activités, pas un dispositif médical.

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Le premier lancement télécharge le modèle SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) et met en cache les
embeddings des 70 sports dans `data/.cache/`. Sans modèle (hors-ligne), l'application se replie sur TF-IDF seul.

## Pipeline NLP

| Étape | Technique | Fichier |
|---|---|---|
| Prétraitement | normalisation Unicode, minuscules, tokenisation, stop-words FR, racinisation Snowball | `preprocess.py` |
| Extraction de signaux | lexiques + négation (« pas de contact »), zones à renforcer vs. zones douloureuses, ajustement des auto-évaluations (« je me fatigue vite ») | `extraction.py` |
| Routage | chaque phrase est affectée aux blocs condition / objectifs / préférences / contraintes | `extraction.py` |
| Embeddings | SBERT multilingue, 1 texte par bloc et par sport | `engine.py` |
| Similarité lexicale | TF-IDF (uni+bigrammes) sur textes racinisés | `engine.py` |
| Similarité | cosinus, normalisée min-max sur les 70 sports | `engine.py` |
| Filtrage | exclusion des sports incompatibles avec les contraintes explicites (contact, eau, impacts, zones, budget, environnement) | `engine.py` |
| Agrégation | somme pondérée des 4 blocs | `engine.py` |
| Explicabilité | critères par bloc + phrase de l'utilisateur la plus proche de la description | `engine.py` |

## Scoring

Score d'un bloc = **40 % structuré** (échelles 1–5, étiquettes `objectifs`, `format_pratique`, `impacts`… du jeu de
données) + **60 % sémantique** (70 % SBERT + 30 % TF-IDF).

Pondération par défaut (modifiable dans l'interface) :

| Bloc | Poids | Justification |
|---|---|---|
| Objectifs | 35 % | motif principal de la recherche |
| Condition physique | 30 % | faisabilité et sécurité (niveau, intensité cible) |
| Préférences | 20 % | adhésion à long terme |
| Contraintes | 15 % | déjà traitées en amont par le filtrage dur |

## Données

`data/disciplines.xlsx` : 70 disciplines (voir la feuille « Guide des champs »). Annotations de prototype pédagogique,
non validées médicalement.

## Hors périmètre (volontairement)

Chatbot, LLM, RAG et agent IA (extensions facultatives) ne sont pas implémentés ; le moteur fonctionne seul.
