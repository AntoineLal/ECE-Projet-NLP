"""📖 Présentation du projet : sujet, données, entraînement des modèles, calcul du score, comparaison TF-IDF / SBERT."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark import gold_metrics, jaccard, run_model  # noqa: E402
from engine import (BLOCK_LABELS, BLOCKS, DEFAULT_WEIGHTS, SBERT_SHARE, STRUCT_SHARE, Recommender,  # noqa: E402
                    UserProfile)
from preprocess import normalize, preprocess, tokenize  # noqa: E402
from shared import DEFAULT_MODEL, MODELS, get_engine, model_label  # noqa: E402

st.set_page_config(page_title="Présentation du projet", page_icon="📖", layout="wide")

EXAMPLE = ("Je suis débutant en sport, je me fatigue rapidement lorsque je cours, mais je peux marcher assez longtemps. "
           "Je souhaite améliorer mon endurance, renforcer mon dos et pratiquer une activité qui ne sollicite pas trop "
           "mes articulations. Je préfère les activités individuelles et je suis disponible trois fois par semaine.")

eng_sb, err_sb, load_sb = get_engine(DEFAULT_MODEL)
eng_tf, _, load_tf = get_engine("tfidf")
df = eng_sb.df
N = len(df)

st.title("📖 Présentation du projet")
st.caption("Recommandation d'activités sportives par analyse sémantique d'un profil écrit en langage naturel.")
if err_sb:
    st.warning("SBERT n'a pas pu être chargé, les parties SBERT sont indisponibles : " + err_sb[:150])
tabs = st.tabs(["1 · Le sujet", "2 · Les données", "3 · Entraînement", "4 · Score de similarité", "5 · TF-IDF vs SBERT"])

# ====================================================================== 1. SUJET
with tabs[0]:
    st.header("🏃 Le sujet : le sport")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"""
**Problème** : choisir un sport est difficile : il en existe des centaines, et le bon choix dépend de la condition
physique, des objectifs, des préférences et des contraintes (blessures, budget, météo…).

**Idée** : la personne **décrit son profil avec ses propres mots**. Le système le comprend (NLP), le compare à
**{N} disciplines** et renvoie le **top 5** avec une explication de chaque choix.

**Outil d'aide à la découverte** : ce n'est ni un diagnostic médical ni une prescription.
""")
    with c2:
        st.markdown("""
**Chaîne de traitement**
1. Texte libre + échelles (endurance, force…)
2. Extraction des signaux (objectifs, rejets, blessures)
3. Filtrage des sports incompatibles
4. Score par bloc (structuré + sémantique)
5. Somme pondérée → top 5
""")

# ====================================================================== 2. DONNÉES
with tabs[1]:
    st.header("🗂️ Les données")
    m = st.columns(4)
    m[0].metric("Disciplines", N)
    m[1].metric("Champs par discipline", 40)
    cats = pd.Series([c for l in df["categories_l"] for c in l])
    m[2].metric("Catégories distinctes", cats.nunique())
    m[3].metric("Annotation", "manuelle")
    st.write("Un sport = une ligne du fichier `data/sports.xlsx`, avec une **description en français** et des champs structurés :")
    st.markdown("""
| Famille de champs | Exemples |
|---|---|
| Texte | description, catégories, mots-clés, lieu de pratique |
| Capacités (échelle 1-5) | endurance, force, souplesse, coordination, équilibre, intensité |
| Contraintes | impacts (1-3), contact, atteinte des articulations, météo |
| Pratique | format (solo, équipe), environnement, équipement, budget |
""")
    g1, g2 = st.columns(2)
    with g1:
        top = cats.value_counts().head(12).iloc[::-1]
        f = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "Nombre de sports", "y": ""},
                   title="Catégories les plus fréquentes")
        f.update_layout(height=340, margin=dict(t=40))
        st.plotly_chart(f, use_container_width=True)
    with g2:
        f = px.histogram(df, x="intensite_loisir_1_5", nbins=5, labels={"intensite_loisir_1_5": "Intensité (1-5)"},
                         title="Intensité de loisir")
        f.update_layout(height=340, margin=dict(t=40), yaxis_title="Nombre de sports", bargap=0.1)
        st.plotly_chart(f, use_container_width=True)
    st.subheader("Un sport devient 4 textes (un par bloc)")
    pick = st.selectbox("Sport", df["nom_sport"].tolist(), key="pres_sport")
    row = df[df["nom_sport"] == pick].iloc[0]
    for b in BLOCKS:
        st.markdown(f"**{BLOCK_LABELS[b]}** : {Recommender._sport_text(b, row)}")
    st.caption("Ces 4 textes sont ce que TF-IDF et SBERT comparent au profil de l'utilisateur. "
               "Les annotations sont pédagogiques, non validées médicalement.")

# ====================================================================== 3. ENTRAÎNEMENT
with tabs[2]:
    st.header("⚙️ Entraînement des modèles")
    st.write("Les deux modèles ne sont pas entraînés de la même façon : TF-IDF est **ajusté sur nos données**, "
             "SBERT est **pré-entraîné** puis utilisé tel quel (pas de fine-tuning).")
    a, b_ = st.columns(2)

    with a:
        st.subheader("TF-IDF : ajusté sur nos sports")
        st.markdown("""
**Prétraitement** : minuscules → tokenisation → suppression des stop-words → racinisation (Snowball français).

**Ajustement** : un `TfidfVectorizer` par bloc, `fit` sur les textes des sports, `sublinear_tf=True`, n-grammes (1, 2).
""")
        st.latex(r"\mathrm{tf}(t,d)=1+\ln f_{t,d}\qquad \mathrm{idf}(t)=\ln\frac{1+N}{1+n_t}+1")
        st.latex(r"w_{t,d}=\mathrm{tf}(t,d)\times\mathrm{idf}(t)\quad\text{puis } \vec d \leftarrow \vec d/\lVert\vec d\rVert_2")
        st.caption("N = nombre de textes, nₜ = nombre de textes contenant le terme t, f = fréquence du terme dans le texte.")
        txt = st.text_input("Essayer le prétraitement", "Je me fatigue rapidement lorsque je cours", key="pres_pre")
        st.write("Normalisé :", normalize(txt))
        st.write("Tokens :", tokenize(txt))
        st.write("Racines sans stop-words :", preprocess(txt))
        vocab = {BLOCK_LABELS[k]: len(eng_tf.tfidf[k].vocabulary_) for k in BLOCKS}
        st.dataframe(pd.DataFrame({"Bloc": vocab.keys(), "Taille du vocabulaire": vocab.values(),
                                   "Matrice": [str(eng_tf.tfidf_mat[k].shape) for k in BLOCKS]}),
                     hide_index=True, use_container_width=True)

    with b_:
        st.subheader("SBERT : pré-entraîné, puis encodage")
        hf = MODELS[DEFAULT_MODEL]["hf"]
        st.markdown(f"""
**Modèle** : `{hf}` (Sentence-BERT, 12 couches Transformer, 50+ langues).

**Pré-entraînement (déjà fait, hors projet)** : réseau siamois ; deux phrases de même sens passent dans le même
encodeur, et on rapproche leurs vecteurs (distillation sur des paires de paraphrases).

**Ce que nous faisons** : pour chaque texte, *tokens → Transformer → moyenne des vecteurs de tokens (mean pooling)
→ normalisation L2*. Les {N * 4} textes de sports sont encodés une fois puis mis en cache (`data/.cache/*.npy`).
""")
        st.latex(r"\vec e=\dfrac{\frac{1}{L}\sum_{i=1}^{L}\vec h_i}{\lVert\frac{1}{L}\sum_{i}\vec h_i\rVert_2}\in\mathbb{R}^{384}")
        if eng_sb.embedder is not None:
            shp = {BLOCK_LABELS[k]: str(eng_sb.emb[k].shape) for k in BLOCKS}
            st.dataframe(pd.DataFrame({"Bloc": shp.keys(), "Matrice d'embeddings (sports × dimensions)": shp.values()}),
                         hide_index=True, use_container_width=True)
            st.caption(f"Chargement + indexation : {load_sb:.1f} s (TF-IDF seul : {load_tf:.2f} s).")
    st.info("Comme les vecteurs sont normalisés (norme 1), le produit scalaire est directement le cosinus.")

# ====================================================================== 4. SCORE
with tabs[3]:
    st.header("🧮 Calcul du score de similarité")
    st.subheader("Formules")
    st.latex(r"\cos(\vec u,\vec s)=\frac{\vec u\cdot\vec s}{\lVert\vec u\rVert\,\lVert\vec s\rVert}")
    st.markdown("**1. Similarité brute** d'un bloc du profil avec le bloc du sport (cosinus TF-IDF et cosinus SBERT), "
                "puis **2. normalisation min-max** sur tous les sports (les cosinus SBERT sont dans une plage étroite) :")
    st.latex(r"\tilde c = \dfrac{c-c_{\min}}{c_{\max}-c_{\min}}")
    st.markdown(f"**3. Score sémantique** = {SBERT_SHARE:.0%} SBERT + {1 - SBERT_SHARE:.0%} TF-IDF. "
                f"**4. Score du bloc** = {STRUCT_SHARE:.0%} structuré (échelles, étiquettes) + {1 - STRUCT_SHARE:.0%} sémantique. "
                "**5. Total** = somme pondérée des blocs.")
    st.latex(r"S_{bloc}=0{,}4\,S_{struct}+0{,}6\,\big(0{,}7\,\tilde c_{SBERT}+0{,}3\,\tilde c_{TFIDF}\big)\qquad "
             r"S_{total}=\dfrac{\sum_b w_b\,S_b}{\sum_b w_b}\times f_{pénalité}")
    st.caption("Poids par défaut : " + ", ".join(f"{BLOCK_LABELS[b].lower()} {DEFAULT_WEIGHTS[b]:.0%}" for b in BLOCKS))

    st.subheader("Exemple concret")
    text = st.text_area("Profil", EXAMPLE, height=130, key="pres_text")
    if len(text.split()) < 8:
        st.warning("Écrivez au moins une phrase.")
    elif eng_sb.embedder is None:
        st.warning("SBERT indisponible : exemple impossible.")
    else:
        prof = UserProfile(text=text)
        out = eng_sb.recommend(prof)
        R, texts = out["resolved"], out["texts"]
        sem, sb_raw, tf_raw = eng_sb.sem_scores(texts)
        blk = st.selectbox("Bloc détaillé", BLOCKS, format_func=BLOCK_LABELS.get, index=1, key="pres_blk")
        st.markdown(f"**Texte du bloc « {BLOCK_LABELS[blk]} » construit à partir du profil :**")
        st.code(texts[blk] or "(vide)", language=None)
        if blk in sem:
            order = np.argsort(-sem[blk])[:6]
            rows = []
            for i in order:
                rows.append({"Sport": df.loc[i, "nom_sport"], "cos TF-IDF": tf_raw[blk][i], "cos SBERT": sb_raw[blk][i],
                             "TF-IDF norm.": (tf_raw[blk][i] - tf_raw[blk].min()) / max(np.ptp(tf_raw[blk]), 1e-9),
                             "SBERT norm.": (sb_raw[blk][i] - sb_raw[blk].min()) / max(np.ptp(sb_raw[blk]), 1e-9),
                             "Sémantique (0,7·SBERT + 0,3·TF-IDF)": sem[blk][i]})
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
            st.caption(f"Min / max des cosinus sur les {N} sports : SBERT {sb_raw[blk].min():.3f} / {sb_raw[blk].max():.3f} ; "
                       f"TF-IDF {tf_raw[blk].min():.3f} / {tf_raw[blk].max():.3f}.")
        else:
            st.info("Ce bloc est vide pour ce profil : seul le score structuré est utilisé.")

        best = out["anatomy"][0]
        st.markdown(f"**Calcul complet pour le n°1 : {best['name']}**")
        tbl, tot_w = [], 0.0
        for b in BLOCKS:
            s, m_, v = best["struct"][b], best["sem"][b], best["blocks"][b]
            tbl.append({"Bloc": BLOCK_LABELS[b], "Structuré": s, "Sémantique": m_,
                        "Score bloc = 0,4·struct + 0,6·sém": v, "Poids normalisé": best["weights"][b],
                        "Contribution": best["contrib"][b]})
        st.dataframe(pd.DataFrame(tbl).round(3), hide_index=True, use_container_width=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Somme des contributions", f"{best['base']:.3f}")
        s2.metric("Facteur de pénalité", f"×{best['pen_factor']:.2f}")
        s3.metric("Score final", f"{best['total'] * 100:.0f} %")
        if best["pen_why"]:
            st.caption("Pénalités : " + " ; ".join(best["pen_why"]))
        st.markdown("**Top 5 obtenu :** " + " · ".join(f"{i}. {r.name} ({r.total * 100:.0f} %)" for i, r in enumerate(out["results"], 1)))

# ====================================================================== 5. COMPARAISON
with tabs[4]:
    st.header("⚖️ TF-IDF vs SBERT")
    st.markdown("""
| | **TF-IDF** | **SBERT** |
|---|---|---|
| Principe | compte les mots communs (pondérés par leur rareté) | vecteur de sens appris sur de gros corpus |
| Entraînement | ajusté sur nos textes de sports | pré-entraîné, non modifié |
| Synonymes | ❌ « courir » ≠ « footing » | ✅ rapproche les sens |
| Rapidité | très rapide | plus lent (réseau de neurones) |
| Explicable | oui (mots) | moins |
""")
    st.subheader("Même profil, deux moteurs complets")
    ctext = st.text_area("Profil", EXAMPLE, height=110, key="cmp_text")
    if len(ctext.split()) >= 8 and eng_sb.embedder is not None:
        o_tf, o_sb = eng_tf.recommend(UserProfile(text=ctext)), eng_sb.recommend(UserProfile(text=ctext))
        c1, c2 = st.columns(2)
        for col, lab, o in ((c1, model_label("tfidf"), o_tf), (c2, model_label(DEFAULT_MODEL), o_sb)):
            col.markdown(f"**{lab}** · {o['timings']['total']:.0f} ms")
            for i, r in enumerate(o["results"], 1):
                col.write(f"{i}. {r.name} — `{r.total * 100:.0f} %`")
        t_tf, t_sb = [r.name for r in o_tf["results"]], [r.name for r in o_sb["results"]]
        st.metric("Accord des deux top 5 (Jaccard)", f"{jaccard(t_tf, t_sb):.2f}",
                  help="Nombre de sports communs / nombre de sports distincts. 1 = identiques, 0 = aucun point commun.")
        st.caption("Rappel : les deux moteurs appliquent les mêmes filtres et le même score structuré ; "
                   "seule la partie sémantique change.")

        st.markdown("**Similarité brute seule** (sans règles, sur la description du sport) :")
        meth = o_sb["methods"]
        cc = st.columns(2)
        for col, name in zip(cc, ["TF-IDF (lexical)", "SBERT (sémantique)"]):
            col.markdown(f"*{name}*")
            for i, (sp, sc) in enumerate(meth[name], 1):
                col.write(f"{i}. {sp} — `{sc:.2f}`")

    st.subheader("Mesures sur les jeux de test")

    @st.cache_data(show_spinner="Évaluation des deux modèles…")
    def evaluate(key: str) -> dict:
        eng, _, _ = get_engine(key)
        rows = run_model(eng)
        gm = gold_metrics(eng)
        ms = np.array([r["ms"] for r in rows])
        return {"pass": sum(r["ok"] for r in rows), "n": len(rows), "p5": gm["p5_mean"],
                "para": gm["para_stability"], "ms": float(ms.mean())}

    if eng_sb.embedder is None:
        st.info("SBERT indisponible : comparaison chiffrée impossible.")
    else:
        res = {k: evaluate(k) for k in ("tfidf", DEFAULT_MODEL)}
        tab = pd.DataFrame([{"Modèle": model_label(k), "Scénarios de contraintes": f"{v['pass']}/{v['n']}",
                             "Précision@5": round(v["p5"], 2), "Robustesse aux paraphrases": round(v["para"], 2),
                             "Latence moyenne (ms)": round(v["ms"])} for k, v in res.items()])
        st.dataframe(tab, hide_index=True, use_container_width=True)
        fig = go.Figure()
        for k, v in res.items():
            fig.add_bar(name=model_label(k), x=["Précision@5", "Robustesse"], y=[v["p5"], v["para"]])
        fig.update_layout(barmode="group", yaxis_range=[0, 1], height=340, legend_orientation="h", legend_y=1.2,
                          margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Scénarios** : filtres stricts (genou fragile, pas de contact, budget…), traités par des règles, donc "
                   "indépendants du modèle. **Précision@5** : part du top 5 dans la famille attendue (14 intentions). "
                   "**Robustesse** : accord des top 5 entre trois formulations d'une même intention. "
                   "Jeu de test modeste : un écart inférieur à 0,05 n'est pas significatif.")
        d_p, d_r = res[DEFAULT_MODEL]["p5"] - res["tfidf"]["p5"], res[DEFAULT_MODEL]["para"] - res["tfidf"]["para"]
        st.success(f"SBERT apporte {d_p:+.2f} de précision@5 et {d_r:+.2f} de robustesse face à TF-IDF, pour "
                   f"{res[DEFAULT_MODEL]['ms'] - res['tfidf']['ms']:+.0f} ms de latence. C'est pourquoi le score final "
                   f"combine les deux ({SBERT_SHARE:.0%} SBERT, {1 - SBERT_SHARE:.0%} TF-IDF).")
