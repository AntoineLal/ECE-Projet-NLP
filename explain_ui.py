"""Interface « Pourquoi ? » : explique, pour chacun des 106 sports, pourquoi il est recommandé, classé ou écarté."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import BLOCK_LABELS, BLOCKS

ICON_BLOCK = {"condition": "💪", "objectifs": "🎯", "preferences": "❤️", "contraintes": "🛡️"}
BLOCK_HELP = {
    "condition": "Vos capacités et votre niveau sont-ils adaptés à l'effort demandé ?",
    "objectifs": "Le sport fait-il progresser ce que vous voulez développer ?",
    "preferences": "Correspond-il à vos goûts (format, lieu, type d'activité) ?",
    "contraintes": "Respecte-t-il vos limites (impacts, zones à ménager, budget…) ?",
}

# Raison technique du filtre -> phrase claire pour l'utilisateur
REASON_TEXT = [
    ("contact réglementé", "Vous ne souhaitez pas de sport de contact."),
    ("pratique dans l'eau", "Vous avez écarté les activités dans l'eau."),
    ("impacts fréquents", "Vos articulations demandent d'éviter les chocs répétés."),
    ("repose sur des sauts", "Vous voulez éviter les sauts."),
    ("uniquement en extérieur", "Vous voulez pratiquer en intérieur, et ce sport se pratique uniquement dehors."),
    ("uniquement en intérieur", "Vous voulez pratiquer en extérieur, et ce sport se pratique uniquement en salle."),
    ("sollicite la zone blessée", "Vous êtes blessé(e) : ce sport sollicite la zone concernée."),
    ("sollicite fortement une zone à ménager", "Ce sport sollicite fortement une zone que vous voulez ménager."),
    ("budget séance", "Son coût par séance dépasse votre budget."),
    ("impacts marqués", "Vous cherchez une pratique douce, or ce sport comporte des chocs marqués."),
    ("intensité soutenue", "Vous cherchez une pratique douce, or ce sport est trop intense."),
]


def plain_reason(r: str) -> str:
    for key, txt in REASON_TEXT:
        if r.startswith(key) or key in r:
            return f"{txt}  \n<small>Règle appliquée : *{r}*</small>"
    return r


def level(v):
    return "🟢" if v >= 0.7 else ("🟠" if v >= 0.4 else "🔴")


def pct(v):
    return "—" if v is None else f"{v * 100:.0f} %"


def verdict(a, fifth):
    """Retourne (niveau, titre, phrase) pour le bandeau principal."""
    n_ok = None
    if not a["retained"]:
        return "error", f"❌ Écarté : {a['name']}", "Un filtre de sécurité l'a retiré avant tout classement."
    if a["in_top"]:
        return "success", f"✅ Recommandé n°{a['rank']} : {a['name']}", f"Score d'affinité {a['total'] * 100:.0f} %."
    gap = (fifth["total"] - a["total"]) * 100 if fifth else 0
    return "warning", f"🟡 Compatible mais non retenu (n°{a['rank']}) : {a['name']}", \
        f"Score {a['total'] * 100:.0f} %, soit {gap:.1f} points sous le 5ᵉ du classement."


def waterfall(a, fifth_total):
    xs, ys, meas = [], [], []
    for b in BLOCKS:
        xs.append(f"{ICON_BLOCK[b]} {BLOCK_LABELS[b]}")
        ys.append(a["contrib"][b] * 100)
        meas.append("relative")
    pen = (a["total"] - a["base"]) * 100
    xs += ["🚧 Garde-fous", "Score final"]
    ys += [pen, 0]
    meas += ["relative", "total"]
    fig = go.Figure(go.Waterfall(x=xs, y=ys, measure=meas, text=[f"{v:+.1f}" for v in ys[:-1]] + [f"{a['total'] * 100:.1f}"],
                                 textposition="outside", connector={"line": {"color": "#999"}},
                                 increasing={"marker": {"color": "#2E7D6B"}}, decreasing={"marker": {"color": "#C8553D"}},
                                 totals={"marker": {"color": "#3B6FB6"}}))
    if fifth_total is not None:
        fig.add_hline(y=fifth_total * 100, line_dash="dot", line_color="#E0913A",
                      annotation_text="5ᵉ du classement", annotation_position="top left")
    fig.update_layout(height=340, margin=dict(t=30, b=10), yaxis_title="points sur 100", showlegend=False)
    return fig


def render(out, profile_text: str):
    A = out["anatomy"]
    kept = [a for a in A if a["retained"]]
    fifth = next((a for a in kept if a["rank"] == 5), None)
    st.header("🔍 Pourquoi ce résultat ? Comprendre chaque décision")
    st.caption("Choisissez n'importe quel sport (recommandé, écarté ou non retenu) pour voir exactement ce qui a joué.")

    # ---- entonnoir
    c = st.columns([1, 0.15, 1, 0.15, 1])
    c[0].metric("Activités étudiées", out["n_total"])
    c[1].markdown("### →")
    c[2].metric("Compatibles avec vos contraintes", out["n_kept"], delta=f"-{out['n_total'] - out['n_kept']} écartées",
                delta_color="off")
    c[3].markdown("### →")
    c[4].metric("Recommandées", len(out["results"]))
    if out.get("relaxed"):
        st.warning("Peu d'activités respectaient votre blessure : des sports très doux qui sollicitent la zone concernée "
                   "ont été acceptés en dernier recours (marqués ⚠). Validez-les avec un professionnel de santé.")

    # ---- sélection
    f = st.radio("Afficher", ["Tous", "✅ Recommandés", "🟡 Non retenus", "❌ Écartés"], horizontal=True, key="expl_filter")
    if f == "✅ Recommandés":
        pool = [a for a in A if a["in_top"]]
    elif f == "🟡 Non retenus":
        pool = [a for a in A if a["retained"] and not a["in_top"]]
    elif f == "❌ Écartés":
        pool = [a for a in A if not a["retained"]]
    else:
        pool = A

    def label(a):
        tag = f"✅ n°{a['rank']}" if a["in_top"] else (f"🟡 n°{a['rank']}" if a["retained"] else "❌ écarté")
        return f"{tag} · {a['name']}"

    if not pool:
        st.info("Aucun sport dans cette catégorie.")
        return
    by_id = {x["sport_id"]: x for x in A}
    sid = st.selectbox("Sport à expliquer", [x["sport_id"] for x in pool], format_func=lambda i: label(by_id[i]),
                       key=f"expl_sport_{f}")
    a = by_id[sid]

    kind, title, sub = verdict(a, fifth)
    box = {"success": st.success, "warning": st.warning, "error": st.error}[kind]
    box(f"**{title}**  \n{sub}")

    # ---- cas écarté
    if not a["retained"]:
        st.markdown("#### Pourquoi a-t-il été écarté ?")
        for r in a["excluded_why"]:
            st.markdown("- " + plain_reason(r), unsafe_allow_html=True)
        st.caption("Les filtres sont des règles strictes appliquées *avant* le calcul des scores : "
                   "un sport écarté n'est jamais classé, quel que soit son score.")
        row = a["row"]
        with st.expander("Voir la fiche du sport"):
            st.write(row.description_fr)
            st.caption(f"Catégories : {row.categories.replace('|', ' · ')} · intensité {int(row.intensite_loisir_1_5)}/5 · "
                       f"impacts {int(row.impacts_1_3)}/3 · contact : {row.contact} · environnement : {row.environnement.replace('|', ', ')}")
        return

    # ---- cas retenu : décomposition du score
    st.markdown("#### Comment son score est construit")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(waterfall(a, fifth["total"] if fifth else None), use_container_width=True)
    with right:
        st.markdown("**Lecture**")
        st.markdown("- Chaque barre verte = ce que le bloc apporte au score (note du bloc × son poids).\n"
                    "- La barre rouge = les **garde-fous** (pénalités pour vos limites).\n"
                    "- Le trait orange = le score du 5ᵉ sport : le seuil pour entrer dans le top 5.")
        if a["pen_why"]:
            st.markdown("**🚧 Garde-fous appliqués**")
            for w in a["pen_why"]:
                st.markdown(f"- {w}")
        else:
            st.markdown("**🚧 Garde-fous** : aucun, ce sport respecte toutes vos limites.")

    # ---- détail par bloc
    st.markdown("#### Détail par bloc")
    cols = st.columns(4)
    for col, b in zip(cols, BLOCKS):
        v = a["blocks"][b]
        with col.container(border=True):
            st.markdown(f"**{ICON_BLOCK[b]} {BLOCK_LABELS[b]}**")
            st.caption(BLOCK_HELP[b])
            st.metric("Note du bloc", pct(v), help=f"Poids dans le score : {a['weights'][b] * 100:.0f} %")
            st.caption(f"📏 Règles : **{pct(a['struct'][b])}** · 🧠 Sens du texte : **{pct(a['sem'][b])}**")
            for lab, sc in a["criteria"][b]:
                st.markdown(f"{level(sc)} {lab}")
            if not a["criteria"][b]:
                st.caption("Rien de précis à comparer dans votre texte pour ce bloc.")

    # ---- comparaison avec le 5e
    if not a["in_top"] and fifth:
        st.markdown(f"#### Pourquoi pas dans le top 5 ? Comparaison avec {fifth['name']} (5ᵉ)")
        rows = [{"Bloc": f"{ICON_BLOCK[b]} {BLOCK_LABELS[b]}", a["name"]: (a["blocks"][b] or 0) * 100,
                 fifth["name"]: (fifth["blocks"][b] or 0) * 100,
                 "Écart (pts)": ((a["blocks"][b] or 0) - (fifth["blocks"][b] or 0)) * 100} for b in BLOCKS]
        rows.append({"Bloc": "🚧 Garde-fous (facteur)", a["name"]: a["pen_factor"] * 100, fifth["name"]: fifth["pen_factor"] * 100,
                     "Écart (pts)": (a["pen_factor"] - fifth["pen_factor"]) * 100})
        st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, use_container_width=True)

    # ---- classement complet
    with st.expander("📋 Tableau complet des 106 sports"):
        tbl = pd.DataFrame([{
            "Statut": "✅ top 5" if x["in_top"] else ("🟡 non retenu" if x["retained"] else "❌ écarté"),
            "Rang": x["rank"], "Sport": x["name"],
            "Score": round(x["total"] * 100, 1) if x["retained"] else None,
            **{BLOCK_LABELS[b]: None if x["blocks"][b] is None or not x["retained"] else round(x["blocks"][b] * 100) for b in BLOCKS},
            "Raison d'écart": " ; ".join(x["excluded_why"]),
        } for x in A])
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=420,
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")})
