"""🛰️ Centre de contrôle : runtimes, logs, historique rejouable, carte des sports, labo de sensibilité, tests."""
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import BLOCK_LABELS, BLOCKS, DEFAULT_WEIGHTS, WATER_TAGS, UserProfile  # noqa: E402
from shared import get_engine  # noqa: E402
from telemetry import APP_LOG, RUNS, get_logger, read_runs  # noqa: E402

st.set_page_config(page_title="Admin · Centre de contrôle", page_icon="🛰️", layout="wide")

# ---------------------------------------------------------------- accès
pwd = os.environ.get("ADMIN_PASSWORD")
if pwd and st.session_state.get("_admin_ok") is not True:
    st.title("🔒 Accès administrateur")
    typed = st.text_input("Mot de passe", type="password")
    if typed and typed == pwd:
        st.session_state["_admin_ok"] = True
        st.rerun()
    st.stop()

engine, sbert_error, boot_s = get_engine()
runs = read_runs()
STAGES = ["extraction", "filtrage", "embeddings+cosinus", "scoring", "explications+méthodes"]
STAGE_COL = dict(zip(STAGES, ["#3B6FB6", "#8E5BB5", "#E0913A", "#2E7D6B", "#C8553D"]))

st.title("🛰️ Centre de contrôle")
st.caption("Supervision du moteur de recommandation : performances, journaux, historique rejouable et tests.")

tot = [r["timings_ms"]["total"] for r in runs]
k = st.columns(6)
k[0].metric("Runs enregistrés", len(runs))
k[1].metric("Latence médiane", f"{np.median(tot):.0f} ms" if tot else "—")
k[2].metric("Latence p95", f"{np.percentile(tot, 95):.0f} ms" if tot else "—")
k[3].metric("Démarrage moteur", f"{boot_s:.1f} s")
k[4].metric("Moteur", "SBERT + TF-IDF" if engine.embedder else "TF-IDF seul")
k[5].metric("Dernier run", datetime.fromtimestamp(runs[-1]["ts"]).strftime("%d/%m %H:%M") if runs else "—")
if sbert_error:
    st.error("SBERT indisponible : " + sbert_error[:200])

tabs = st.tabs(["📡 Vue d'ensemble", "⏱️ Runtime", "🗂️ Historique & replay", "📜 Logs", "🗺️ Carte des sports",
                "🎛️ Labo de sensibilité", "✅ Tests", "🧬 Dataset"])


def need_runs():
    if not runs:
        st.info("Aucun run enregistré. Lancez une recommandation depuis la page principale.")
        return True
    return False


def run_label(r):
    return f"{datetime.fromtimestamp(r['ts']).strftime('%d/%m %H:%M:%S')} · {r['id']} · {r['text'][:50]}…"


# ================================================================ 1. vue d'ensemble
with tabs[0]:
    if not need_runs():
        c1, c2 = st.columns(2)
        top1 = Counter(r["top"][0]["name"] for r in runs if r["top"])
        anyc = Counter(t["name"] for r in runs for t in r["top"])
        d = pd.DataFrame({"sport": list(anyc), "Dans le top 5": list(anyc.values())})
        d["Numéro 1"] = d["sport"].map(top1).fillna(0)
        d = d.sort_values("Dans le top 5").tail(12)
        fig = go.Figure()
        fig.add_bar(y=d["sport"], x=d["Dans le top 5"], orientation="h", name="Dans le top 5", marker_color="#3B6FB6")
        fig.add_bar(y=d["sport"], x=d["Numéro 1"], orientation="h", name="Numéro 1", marker_color="#E0913A")
        fig.update_layout(barmode="overlay", title="Sports les plus recommandés", height=380, legend_orientation="h")
        c1.plotly_chart(fig, use_container_width=True)

        sig = Counter()
        for r in runs:
            s = r["signals"]
            sig.update(f"objectif · {g}" for g in s["goals"])
            sig.update(f"préf · {x}" for x in s["formats"] + s["envs"] + [f.lower() for f in s["families"]])
            for key, lab in (("no_contact", "sans contact"), ("no_water", "sans eau"), ("no_jump", "sans sauts")):
                if s[key]:
                    sig[f"contrainte · {lab}"] += 1
            if s["impact_tol"] == "faible":
                sig["contrainte · articulations"] += 1
        sd = pd.DataFrame(sig.most_common(12), columns=["signal", "occurrences"]).iloc[::-1]
        c2.plotly_chart(px.bar(sd, x="occurrences", y="signal", orientation="h", title="Signaux détectés dans les textes",
                               height=380, color_discrete_sequence=["#2E7D6B"]), use_container_width=True)

        c3, c4 = st.columns(2)
        ex = Counter(m for r in runs for ws in r["excluded"].values() for m in ws)
        if ex:
            ed = pd.DataFrame(ex.most_common(8), columns=["motif d'exclusion", "fois"]).iloc[::-1]
            c3.plotly_chart(px.bar(ed, x="fois", y="motif d'exclusion", orientation="h", title="Motifs de filtrage",
                                   height=330, color_discrete_sequence=["#8E5BB5"]), use_container_width=True)
        agree = []
        for r in runs:
            h = set(r["methods"].get("Hybride complet (blocs + règles)", []))
            for name, lst in r["methods"].items():
                if not name.startswith("Hybride"):
                    agree.append((name, len(h & set(lst))))
        if agree:
            ag = pd.DataFrame(agree, columns=["méthode", "recoupement"]).groupby("méthode").mean().reset_index()
            c4.plotly_chart(px.bar(ag, x="méthode", y="recoupement", title="Recoupement moyen avec l'hybride (sur 5)",
                                   range_y=[0, 5], height=330, color_discrete_sequence=["#E0913A"]), use_container_width=True)

# ================================================================ 2. runtime
with tabs[1]:
    if not need_runs():
        df = pd.DataFrame([{"run": r["id"], **r["timings_ms"]} for r in runs]).tail(40)
        fig = go.Figure()
        for s in STAGES:
            if s in df:
                fig.add_bar(x=df["run"], y=df[s], name=s, marker_color=STAGE_COL[s])
        fig.update_layout(barmode="stack", title="Décomposition du temps par run (ms)", height=380, legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        mean = df[[s for s in STAGES if s in df]].mean().sort_values()
        c1.plotly_chart(px.bar(x=mean.values, y=mean.index, orientation="h", title="Temps moyen par étape (ms)",
                               labels={"x": "ms", "y": ""}, height=320, color_discrete_sequence=["#3B6FB6"]),
                        use_container_width=True)
        c2.plotly_chart(px.histogram(df, x="total", nbins=15, title="Distribution des latences (ms)", height=320,
                                     color_discrete_sequence=["#2E7D6B"]), use_container_width=True)
        slow = df.sort_values("total").iloc[-1]
        st.caption(f"Étape la plus coûteuse en moyenne : **{mean.index[-1]}** ({mean.iloc[-1]:.0f} ms). "
                   f"Run le plus lent : `{slow['run']}` ({slow['total']:.0f} ms).")

# ================================================================ 3. historique & replay
with tabs[2]:
    if not need_runs():
        tbl = pd.DataFrame([{"id": r["id"], "date": datetime.fromtimestamp(r["ts"]).strftime("%d/%m %H:%M:%S"),
                             "texte": r["text"][:70], "n°1": r["top"][0]["name"] if r["top"] else "—",
                             "exclus": r["n_excluded"], "ms": r["timings_ms"]["total"]} for r in reversed(runs)])
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=220)
        sel = st.selectbox("Inspecter un run", list(reversed(runs)), format_func=run_label)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Texte analysé**")
            st.info(sel["text"] or "—")
            st.markdown("**Top 5 enregistré**")
            st.dataframe(pd.DataFrame([{"sport": t["name"], "score": round(t["score"] * 100, 1),
                                        **{BLOCK_LABELS[b]: None if v is None else round(v * 100) for b, v in t["blocks"].items()}}
                                       for t in sel["top"]]), hide_index=True, use_container_width=True)
            with st.expander(f"Exclus ({sel['n_excluded']})"):
                st.json(sel["excluded"])
        with c2:
            st.markdown("**Signaux extraits**")
            st.json(sel["signals"], expanded=False)
            st.markdown("**Ajustements** : " + (" ; ".join(sel["notes"]) or "aucun"))
            st.markdown("**Temps (ms)** : " + ", ".join(f"{k} {v}" for k, v in sel["timings_ms"].items()))
        if st.button("🔁 Rejouer avec le code actuel (détection de régression)"):
            p = UserProfile(text=sel["text"], **sel["profile"])
            o = engine.recommend(p, sel["weights"])
            old, new = [t["name"] for t in sel["top"]], [r.name for r in o["results"]]
            if old == new:
                st.success("Résultat identique : " + ", ".join(new))
            else:
                st.warning("Le classement a changé depuis ce run.")
                st.dataframe(pd.DataFrame({"rang": range(1, 6), "enregistré": old + [""] * (5 - len(old)),
                                           "actuel": new + [""] * (5 - len(new))}), hide_index=True)

# ================================================================ 4. logs
with tabs[3]:
    @st.fragment(run_every="5s")
    def logs_view():
        auto = st.session_state.get("_auto_logs", True)
        c1, c2, c3 = st.columns([2, 2, 1])
        lv = c1.multiselect("Niveaux", ["DEBUG", "INFO", "WARNING", "ERROR"], default=["INFO", "WARNING", "ERROR"], key="lv")
        q = c2.text_input("Recherche", key="logq")
        n = c3.number_input("Lignes", 20, 1000, 200, 20, key="logn")
        lines = APP_LOG.read_text(encoding="utf-8").splitlines() if APP_LOG.exists() else []
        sel_lines = [ln for ln in lines if any(f"| {l} |" in ln for l in lv) and q.lower() in ln.lower()][-int(n):]
        counts = Counter(ln.split(" | ")[1] for ln in lines if " | " in ln)
        st.caption("  ".join(f"**{k}** {v}" for k, v in counts.items()) + "  · rafraîchi toutes les 5 s")
        st.code("\n".join(sel_lines) or "(aucun log)", language="log", height=380)

    logs_view()
    if APP_LOG.exists():
        st.download_button("⬇️ Télécharger app.log", APP_LOG.read_bytes(), "app.log")
    if RUNS.exists():
        st.download_button("⬇️ Télécharger runs.jsonl", RUNS.read_bytes(), "runs.jsonl")

# ================================================================ 5. carte
with tabs[4]:
    if engine.embedder is None:
        st.info("Nécessite SBERT.")
    else:
        from sklearn.decomposition import PCA
        X = engine.emb["desc"]
        P = PCA(2, random_state=0).fit(X)
        xy = P.transform(X)
        df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "sport": engine.df.nom_sport,
                           "famille": engine.df.categories_l.apply(lambda c: c[0])})
        fig = px.scatter(df, x="x", y="y", color="famille", hover_name="sport", height=560)
        fig.update_traces(marker=dict(size=9, opacity=0.75))
        if runs:
            sel = st.selectbox("Superposer un run", list(reversed(runs)), format_func=run_label, key="mapsel")
            names = [t["name"] for t in sel["top"]]
            d2 = df[df.sport.isin(names)]
            fig.add_trace(go.Scatter(x=d2.x, y=d2.y, mode="markers+text", text=d2.sport, textposition="top center",
                                     marker=dict(size=18, symbol="star", color="gold", line=dict(width=1, color="black")),
                                     name="Top 5"))
            u = P.transform(np.asarray(engine.embedder.encode([sel["text"]])))
            fig.add_trace(go.Scatter(x=u[:, 0], y=u[:, 1], mode="markers+text", text=["Vous"], textposition="bottom center",
                                     marker=dict(size=20, symbol="x", color="red"), name="Texte utilisateur"))
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Projection PCA des embeddings SBERT des descriptions (variance expliquée : "
                   f"{P.explained_variance_ratio_.sum() * 100:.0f} %). Plus un sport est proche de la croix, plus sa description ressemble au texte.")

# ================================================================ 6. labo de sensibilité
with tabs[5]:
    if not need_runs():
        sel = st.selectbox("Run de référence", list(reversed(runs)), format_func=run_label, key="labsel")
        cols = st.columns(4)
        w = {b: cols[i].slider(BLOCK_LABELS[b], 0, 100, int(DEFAULT_WEIGHTS[b] * 100), 5, key=f"lab_{b}") for i, b in enumerate(BLOCKS)}
        s = sum(w.values()) or 1
        o = engine.recommend(UserProfile(text=sel["text"], **sel["profile"]), {b: v / s for b, v in w.items()})
        base = [t["name"] for t in sel["top"]]
        rows = []
        for i, r in enumerate(o["results"], 1):
            was = base.index(r.name) + 1 if r.name in base else None
            arrow = "🆕" if was is None else ("▲" if was > i else "▼" if was < i else "＝")
            rows.append({"rang": i, "sport": r.name, "score": round(r.total * 100, 1), "mouvement": arrow,
                         "rang initial": was or "hors top 5"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Si le top 5 bouge beaucoup avec de petits changements de poids, la recommandation est fragile.")

# ================================================================ 7. tests
def _top(o, df):
    """Lignes du jeu de données correspondant au top 5 retourné."""
    return df[df.nom_sport.isin([r.name for r in o["results"]])]


PERSONAS = [
    ("Genou fragile : pas de sauts ni d'impacts fréquents",
     "J'ai mal au genou, je dois éviter les sauts et les chocs. Je veux me détendre et travailler mon équilibre.",
     lambda o, df: all(df.loc[df.nom_sport == r.name, "impacts_1_3"].iloc[0] < 3 and "saut" not in df.loc[df.nom_sport == r.name, "categories_l"].iloc[0] for r in o["results"])),
    ("Ne sait pas nager : aucun sport aquatique",
     "Je ne sais pas nager. Je veux améliorer mon endurance, plutôt en extérieur.",
     lambda o, df: not any(set(df.loc[df.nom_sport == r.name, "categories_l"].iloc[0]) & WATER_TAGS for r in o["results"])),
    ("Pas de contact : aucun contact réglementé",
     "Je ne veux surtout pas de sport de contact, je cherche à développer ma force et mon cardio.",
     lambda o, df: all(df.loc[df.nom_sport == r.name, "contact"].iloc[0] != "réglementé" for r in o["results"])),
    ("Budget 10 € : respecte le plafond",
     "Mon budget est de 10 euros par séance maximum. Je veux me remettre à l'endurance.",
     lambda o, df: all(df.loc[df.nom_sport == r.name, "budget_seance_min_eur"].iloc[0] <= 10 for r in o["results"])),
    ("Exemple de l'énoncé : un sport doux d'endurance dans le top 5",
     "Je suis débutant en sport, je me fatigue rapidement lorsque je cours, mais je peux marcher assez longtemps. "
     "Je souhaite améliorer mon endurance, renforcer mon dos et pratiquer une activité qui ne sollicite pas trop mes articulations. "
     "Je préfère les activités individuelles et je suis disponible trois fois par semaine.",
     lambda o, df: bool(((_top(o, df).endurance_1_5 >= 4) & (_top(o, df).impacts_1_3 == 1)).any())),
    ("Passionné de combat : au moins 3 sports de combat",
     "Je suis confirmé, j'adore les sports de combat et la boxe, je veux de l'intensité et progresser techniquement.",
     lambda o, df: sum(bool({"sport de combat", "combat de percussion", "combat de préhension"} & set(df.loc[df.nom_sport == r.name, "categories_l"].iloc[0])) for r in o["results"]) >= 3),
    ("Amateur d'équipe : au moins 3 sports collectifs",
     "J'aime les sports collectifs, jouer en équipe avec des amis, courir et me défouler.",
     lambda o, df: sum("sport collectif" in df.loc[df.nom_sport == r.name, "categories_l"].iloc[0] for r in o["results"]) >= 3),
    ("Épaule blessée, tennis pratiqué : doux, intérieur, sans épaule ni raquette",
     "Je pratique la salle de sport et le tennis mais je me suis blessé à l'épaule donc j'aimerais trouver un sport plus calme et plus doux qui pourrait me soulager l'épaule pendant ma réhabilitation. Je préfère un sport d'intérieur qui ne pose pas de problème avec la météo.",
     lambda o, df: all("sport de raquette" not in df.loc[df.nom_sport == r.name, "categories_l"].iloc[0]
                       and df.loc[df.nom_sport == r.name, "intensite_loisir_1_5"].iloc[0] <= 3
                       and "intérieur" in df.loc[df.nom_sport == r.name, "environnement_l"].iloc[0] for r in o["results"])
                  and sum("épaules" in df.loc[df.nom_sport == r.name, "zones_corps_sollicitees_l"].iloc[0] for r in o["results"]) <= 1),
    ("Tendinite au coude : pas de raquette, pas d'avant-bras sollicité",
     "J'ai une tendinite au coude, j'ai mal à l'avant-bras. Je veux garder la forme et travailler mon endurance en douceur.",
     lambda o, df: all("avant-bras" not in df.loc[df.nom_sport == r.name, "zones_corps_sollicitees_l"].iloc[0] for r in o["results"])),
    ("Vertiges : au plus un sport très exigeant en équilibre dans le top 5",
     "J'ai des vertiges et je manque d'équilibre quand je marche. Je veux bouger en douceur et me détendre.",
     lambda o, df: int((_top(o, df).equilibre_1_5 >= 4).sum()) <= 1),
    ("Détente et souplesse : au moins 2 pratiques de mobilité ou de bien-être",
     "Je suis débutant, je veux me détendre et gagner en souplesse. Je préfère pratiquer en intérieur, plutôt seul.",
     lambda o, df: int(_top(o, df).categories_l.apply(lambda c: bool({"mobilité", "relaxation"} & set(c))).sum()) >= 2),
    ("Salle et force : au moins 2 sports de salle orientés force",
     "Je m'entraîne en salle de sport depuis plusieurs années. Je veux prendre de la force et de la puissance, je m'entraîne seul.",
     lambda o, df: int(_top(o, df).apply(lambda r: "sport en salle" in r.categories_l and r.force_1_5 >= 4, axis=1).sum()) >= 2),
    ("Dos en douceur : au moins 2 sports qui travaillent le dos",
     "Je veux renforcer mon dos en douceur, sans trop d'impacts.",
     lambda o, df: int(_top(o, df).zones_corps_sollicitees_l.apply(lambda z: "dos" in z).sum()) >= 2),
    ("Équipe sans contact : aucun sport à contact réglementé, et au moins un sport collectif",
     "Je veux jouer en équipe avec des amis, mais sans aucun contact physique.",
     lambda o, df: bool((_top(o, df).contact != "réglementé").all()) and bool(_top(o, df).categories_l.apply(lambda c: "sport collectif" in c).any())),
    ("Pluie et intérieur : tous les sports proposés se pratiquent en intérieur",
     "Je veux faire du cardio en intérieur, car il pleut souvent où j'habite. Je suis régulier.",
     lambda o, df: bool(_top(o, df).environnement_l.apply(lambda e: "intérieur" in e).all())),
    ("Budget nul : seulement des activités sans coût de séance",
     "Mon budget est de 0 euro, je veux marcher et me promener dehors.",
     lambda o, df: bool((_top(o, df).budget_seance_min_eur <= 0).all())),
]
with tabs[6]:
    st.write("Scénarios types qui vérifient que les contraintes et la cohérence de la recommandation tiennent.")
    if st.button("▶️ Lancer la suite de tests", type="primary"):
        res = []
        for name, text, check in PERSONAS:
            lvl = "confirmé" if "confirmé" in text else ("régulier" if "salle de sport" in text else "occasionnel")
            t0 = time.perf_counter()
            o = engine.recommend(UserProfile(text=text, experience=lvl))
            ok = bool(check(o, engine.df))
            res.append({"": "✅" if ok else "❌", "scénario": name, "top 5": ", ".join(r.name for r in o["results"]),
                        "ms": round((time.perf_counter() - t0) * 1000)})
            get_logger().log(20 if ok else 40, "test '%s' : %s", name, "OK" if ok else "ÉCHEC")
        st.dataframe(pd.DataFrame(res), hide_index=True, use_container_width=True)
        n_ok = sum(r[""] == "✅" for r in res)
        (st.success if n_ok == len(res) else st.error)(f"{n_ok}/{len(res)} scénarios validés")

# ================================================================ 8. dataset
with tabs[7]:
    df = engine.df
    c = st.columns(4)
    c[0].metric("Sports", len(df))
    c[1].metric("Valeurs manquantes", int(df[["description_fr", "categories", "objectifs"]].isna().sum().sum()))
    c[2].metric("Longueur moy. description", f"{df.description_fr.str.split().str.len().mean():.0f} mots")
    c[3].metric("Tags distincts (catégories)", len({t for l in df.categories_l for t in l}))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="intensite_loisir_1_5", color="impacts_1_3", barmode="group",
                                 title="Intensité × impacts", height=320), use_container_width=True)
    cap = df[["endurance_1_5", "force_1_5", "souplesse_1_5", "coordination_1_5", "equilibre_1_5"]].mean().round(2)
    c2.plotly_chart(px.bar(x=[i.replace("_1_5", "") for i in cap.index], y=cap.values, title="Sollicitation moyenne des capacités",
                           height=320, labels={"x": "", "y": "moyenne /5"}, color_discrete_sequence=["#3B6FB6"]),
                    use_container_width=True)
    st.dataframe(df[["sport_id", "nom_sport", "categories", "intensite_loisir_1_5", "difficulte_entree_1_3", "impacts_1_3",
                     "contact", "environnement", "budget_seance_min_eur", "budget_seance_max_eur"]],
                 use_container_width=True, hide_index=True, height=300)
