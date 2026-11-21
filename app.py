"""Application Streamlit : recommandation d'activités sportives par analyse sémantique du profil.

Lancement :  streamlit run app.py
"""
import plotly.graph_objects as go
import streamlit as st

from engine import (BLOCK_LABELS, BLOCKS, CAP_LABEL, DEFAULT_WEIGHTS, EXPERIENCE, 
                    Recommender, UserProfile)

st.set_page_config(page_title="Recommandation sportive", page_icon="🏃", layout="wide")

EXAMPLE = ("Je suis débutant en sport, je me fatigue rapidement lorsque je cours, mais je peux marcher assez longtemps. "
           "Je souhaite améliorer mon endurance, renforcer mon dos et pratiquer une activité qui ne sollicite pas trop "
           "mes articulations. Je préfère les activités individuelles et je suis disponible trois fois par semaine.")
ZONES = ["dos", "épaules", "bras", "abdominaux", "cuisses", "hanches et fessiers", "mollets", "pieds et chevilles",
         "mains et poignets", "avant-bras"]
BLOCK_COLORS = {"condition": "#2E7D6B", "objectifs": "#3B6FB6", "preferences": "#E0913A", "contraintes": "#8E5BB5"}


class SbertEmbedder:
    name = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name)

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


@st.cache_resource(show_spinner="Chargement du modèle SBERT et indexation des 70 sports…")
def get_engine():
    try:
        return Recommender(SbertEmbedder()), None
    except Exception as e:  # modèle indisponible (hors-ligne…) : repli TF-IDF seul
        return Recommender(None), str(e)


engine, sbert_error = get_engine()

# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("⚙️ Paramètres")
    st.caption("Modèle sémantique : " + ("SBERT multilingue + TF-IDF" if engine.embedder else "TF-IDF seul (SBERT indisponible)"))
    if sbert_error:
        st.warning("SBERT n'a pas pu être chargé : " + sbert_error[:150])
    with st.expander("Pondération des blocs"):
        st.caption("Par défaut : objectifs 35 %, condition 30 %, préférences 20 %, contraintes 15 %. "
                   "Les valeurs sont normalisées pour sommer à 100 %.")
        w = {b: st.slider(BLOCK_LABELS[b], 0, 100, int(DEFAULT_WEIGHTS[b] * 100), 5) for b in BLOCKS}
        tot = sum(w.values()) or 1
        weights = {b: v / tot for b, v in w.items()}
    st.info("Outil d'aide à la **découverte** d'activités. Il ne s'agit ni d'un diagnostic médical ni d'une prescription "
            "d'exercices. Consultez un professionnel de santé en cas de doute.")

st.title("🏃 Quel sport est fait pour vous ?")
st.write("Décrivez votre profil, nous le comparons sémantiquement à **70 disciplines** et vous proposons le **top 5**.")

if "txt" not in st.session_state:
    st.session_state.txt = ""

# ---------------------------------------------------------------- formulaire
st.button("📝 Remplir le texte avec l'exemple", on_click=lambda: st.session_state.update(txt=EXAMPLE))

with st.form("profil"):
    st.subheader("1 · Décrivez-vous avec vos mots")
    st.caption("C'est le cœur de l'analyse : condition physique, objectifs, préférences (individuel/équipe, intérieur/"
               "extérieur, types de sport…), limitations (articulations, dos, contact, eau…), budget, disponibilité.")
    text = st.text_area("Votre profil sportif", key="txt", height=200,
                        placeholder="Ex. : Je suis débutant, je me fatigue vite en courant mais je marche longtemps. "
                                    "Je veux améliorer mon endurance et renforcer mon dos, sans trop solliciter mes articulations…")

    st.subheader("2 · Auto-évaluation rapide")
    c1, c2 = st.columns(2)
    with c1:
        endurance = st.slider("Endurance (1 = très faible, 5 = excellente)", 1, 5, 3)
        force = st.slider("Force musculaire", 1, 5, 3)
        mobilite = st.slider("Mobilité / souplesse", 1, 5, 3)
    with c2:
        motivation = st.slider("Motivation", 1, 5, 3)
        experience = st.select_slider("Niveau d'activité physique actuel", list(EXPERIENCE), value="occasionnel")
        sessions = st.slider("Séances par semaine", 1, 7, 3)
        minutes = st.slider("Durée d'une séance (min)", 20, 180, 60, 10)

    submitted = st.form_submit_button("🔍 Trouver mes sports", type="primary", use_container_width=True)

if not submitted:
    st.stop()
if len(text.split()) < 8:
    st.warning("Écrivez quelques phrases (objectifs, préférences, limitations) : l'analyse repose surtout sur votre texte.")
    st.stop()

profile = UserProfile(text=text, endurance=endurance, force=force, mobilite=mobilite, motivation=motivation,
                      experience=experience, sessions_per_week=sessions, minutes=minutes)
out = engine.recommend(profile, weights)
R, results = out["resolved"], out["results"]

# ---------------------------------------------------------------- synthèse du profil
st.divider()
st.header("Synthèse de votre profil")
m = st.columns(5)
m[0].metric("Niveau", R.exp_label)
m[1].metric("Endurance", f"{R.abilities['endurance']}/5")
m[2].metric("Force", f"{R.abilities['force']}/5")
m[3].metric("Mobilité", f"{R.abilities['souplesse']}/5")
m[4].metric("Intensité cible", f"{R.target_intensity:.1f}/5")

lines = []
if R.goals or R.goal_zones:
    lines.append("**Objectifs** : " + ", ".join(R.goals + [f"zone {z}" for z in R.goal_zones]))
pref = R.formats + R.envs + [f.lower() for f in R.families]
if pref:
    lines.append("**Préférences** : " + ", ".join(pref))
if R.disliked_families or R.disliked_formats:
    lines.append("**À éviter (texte)** : " + ", ".join(R.disliked_families + R.disliked_formats))
cons = [f"impacts : tolérance {R.impact_tol}"]
if R.no_contact: cons.append("sans contact")
if R.no_water: cons.append("sans eau")
if R.spare_hard or R.spare_soft: cons.append("zones à ménager : " + ", ".join(R.spare_hard + R.spare_soft))
if profile.budget_seance_max is not None: cons.append(f"budget ≤ {profile.budget_seance_max} €")
lines.append("**Contraintes** : " + " ; ".join(cons))
lines.append(f"**Disponibilité** : {sessions} séance(s) de {minutes} min par semaine")
st.markdown("  \n".join(lines))
if R.notes:
    with st.expander("🔎 Ce que l'analyse de votre texte a modifié"):
        for n in R.notes:
            st.write("• " + n)
with st.expander("🧪 Prétraitement et signaux détectés dans le texte"):
    from preprocess import normalize, preprocess
    st.write("**Texte normalisé** :", normalize(text) or "—")
    st.write("**Racines retenues (sans stop-words)** :", " ".join(preprocess(text)) or "—")
    sg = R.sig
    st.write("**Objectifs détectés** :", sorted(sg.goals) or "—", "| **Zones visées** :", sorted(sg.goal_zones) or "—",
             "| **Zones à ménager** :", sorted(sg.spare_zones) or "—")
    st.write("**Préférences** :", sorted(sg.prefs) or "—", "| **Rejets** :", sorted(sg.disliked) or "—",
             "| **Contraintes** :", sorted(sg.constraints) or "—")

# ---------------------------------------------------------------- top 5
st.divider()
st.header("🏆 Vos 5 sports recommandés")
if not results:
    st.error("Aucune activité ne satisfait toutes vos contraintes. Assouplissez-en certaines.")
    st.stop()
if len(results) < 5:
    st.warning(f"Seulement {len(results)} activité(s) compatibles avec vos contraintes.")

for rank, res in enumerate(results, 1):
    r = res.row
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.subheader(f"{rank}. {res.name}")
        h1.caption(f"{r.categories.replace('|', ' · ')}  —  {r.lieu_pratique}")
        h2.metric("Affinité", f"{res.total * 100:.0f} %")
        st.write(r.description_fr)
        bc = st.columns(4)
        for col, b in zip(bc, BLOCKS):
            v = res.blocks[b]
            col.progress(float(v) if v is not None else 0.0, text=f"{BLOCK_LABELS[b]} : {v * 100:.0f} %" if v is not None else f"{BLOCK_LABELS[b]} : n/a")
        with st.expander("Pourquoi ce sport ?"):
            for b in res.bullets:
                st.markdown(f"{'✅' if b['ok'] else '⚠️'} **{b['block']}** — {b['text']}")
            st.caption(f"Budget séance {int(r.budget_seance_min_eur)}–{int(r.budget_seance_max_eur)} € · "
                       f"matériel {int(r.budget_equipement_min_eur)}–{int(r.budget_equipement_max_eur)} € · "
                       f"équipement : {r.equipement.replace('|', ', ')}")

# ---------------------------------------------------------------- visualisations
st.divider()
st.header("📊 Visualisations")
t1, t2, t3, t4 = st.tabs(["Scores par bloc", "Profil vs sport", "Carte des affinités", "Comparaison des méthodes NLP"])

with t1:
    fig = go.Figure()
    for b in BLOCKS:
        fig.add_bar(name=BLOCK_LABELS[b], x=[x.name for x in results],
                    y=[(x.blocks[b] or 0) * 100 for x in results], marker_color=BLOCK_COLORS[b])
    fig.update_layout(barmode="group", yaxis_title="Score (%)", yaxis_range=[0, 100], legend_orientation="h",
                      legend_y=1.15, height=420, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Score d'un bloc = 40 % score structuré (déduit du texte et des échelles) + 60 % similarité "
               "sémantique (70 % SBERT + 30 % TF-IDF, normalisée sur les 70 sports). Total = somme pondérée des blocs.")

with t2:
    pick = st.selectbox("Sport à comparer", [x.name for x in results])
    sel = next(x for x in results if x.name == pick).row
    axes = ["endurance", "force", "souplesse", "coordination", "equilibre", "intensite_loisir"]
    labels = [CAP_LABEL.get(a, "intensité") for a in axes]
    goal_cap = {"endurance": "endurance", "force": "force", "souplesse": "souplesse", "équilibre": "equilibre",
                "coordination": "coordination"}
    declared = [R.abilities.get(a) for a in ("endurance", "force", "souplesse")] + [None, None, R.target_intensity]
    wanted = [5 if any(goal_cap.get(g) == a for g in R.goals) else 0 for a in axes[:5]] + [None]
    sport_v = [sel[f"{a}_1_5"] for a in axes]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=sport_v + sport_v[:1], theta=labels + labels[:1], fill="toself", name=pick,
                                  line_color="#3B6FB6"))
    fig.add_trace(go.Scatterpolar(r=[v if v is not None else 0 for v in declared] + [declared[0]], theta=labels + labels[:1],
                                  name="Vos capacités (intensité = cible)", line=dict(color="#2E7D6B", dash="dot")))
    fig.add_trace(go.Scatterpolar(r=[v or 0 for v in wanted] + [wanted[0]], theta=labels + labels[:1], name="Objectifs visés",
                                  line=dict(color="#E0913A", dash="dash")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5])), height=460,
                      legend_orientation="h", legend_y=-0.1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Coordination et équilibre ne sont pas demandés dans les échelles : seuls les objectifs les concernent.")

with t3:
    top = out["full"][out["full"]["retenu"]].head(12)
    z = [[(v or 0) * 100 for v in top[b]] for b in BLOCKS]
    fig = go.Figure(go.Heatmap(z=z, x=top["sport"], y=[BLOCK_LABELS[b] for b in BLOCKS], colorscale="Viridis",
                               zmin=0, zmax=100, text=[[f"{v:.0f}" for v in row] for row in z], texttemplate="%{text}",
                               colorbar_title="%"))
    fig.update_layout(height=360, xaxis_tickangle=-35, margin=dict(b=120))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Les 12 meilleures activités compatibles avec vos contraintes, score par bloc.")

with t4:
    st.write("Même profil, trois méthodes de classement (activités filtrées par vos contraintes dans les trois cas) :")
    cols = st.columns(len(out["methods"]))
    for col, (name, items) in zip(cols, out["methods"].items()):
        col.markdown(f"**{name}**")
        for i, (sp, sc) in enumerate(items, 1):
            col.write(f"{i}. {sp} — `{sc:.2f}`")
    st.caption("TF-IDF ne voit que les mots communs ; SBERT capte le sens mais ignore échelles et contraintes ; "
               "le modèle hybride combine les deux avec les règles du jeu de données.")

with st.expander(f"🚫 Activités écartées par vos contraintes ({len(out['excluded'])})"):
    for sp, why in out["excluded"].items():
        st.write(f"**{sp}** — " + ", ".join(why))
