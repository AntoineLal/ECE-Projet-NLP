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
from shared import MODELS, get_engine, model_label, normalize_model_key  # noqa: E402
from benchmark import read_benchmarks, run_benchmark  # noqa: E402
from gold import GOLD, PARAPHRASES  # noqa: E402
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
                "🎛️ Labo de sensibilité", "✅ Tests", "🧬 Dataset", "🧠 Modèles NLP"])


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
        mods_present = sorted({normalize_model_key(r.get("engine")) for r in runs})
        mod = st.selectbox("Modèle", ["Tous"] + mods_present,
                           format_func=lambda k: "Tous" if k == "Tous" else model_label(k), key="hist_model")
        runs_v = [r for r in runs if mod == "Tous" or normalize_model_key(r.get("engine")) == mod]
        tbl = pd.DataFrame([{"id": r["id"], "modèle": model_label(normalize_model_key(r.get("engine"))),
                             "date": datetime.fromtimestamp(r["ts"]).strftime("%d/%m %H:%M:%S"),
                             "texte": r["text"][:70], "n°1": r["top"][0]["name"] if r["top"] else "—",
                             "exclus": r["n_excluded"], "ms": r["timings_ms"]["total"]} for r in reversed(runs_v)])
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=220)
        sel = st.selectbox("Inspecter un run", list(reversed(runs_v)), format_func=run_label)
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
from personas import PERSONAS  # noqa: E402  (scénarios partagés)

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

# ================================================================ 9. modèles NLP
with tabs[8]:
    st.write("Même jeu d'évaluation, plusieurs modèles de langage : justesse, robustesse et coût.")
    st.caption("**Scénarios** : contraintes (filtres stricts, indépendants du modèle). **Précision@5** : part du top 5 "
               "dans la famille attendue, sur 14 intentions. **Robustesse** : accord des top 5 entre formulations d'une "
               "même intention, sur 5 groupes. Jeu de taille modeste : un écart inférieur à 0,05 n'est pas significatif.")
    if st.button("▶️ Lancer le benchmark sur tous les modèles", type="primary"):
        engines = {k: get_engine(k) for k in MODELS}
        bar = st.progress(0.0, text="Préparation…")

        def _prog(i, n, k):
            bar.progress(i / n, text=f"Évaluation : {model_label(k)}")

        with st.spinner("Évaluation en cours…"):
            run_benchmark(engines=engines, progress=_prog)
        bar.empty()
        st.success("Benchmark terminé et enregistré.")

    history = read_benchmarks()
    if not history:
        st.info("Aucun benchmark enregistré. Lancez-le avec le bouton ci-dessus (le premier passage charge les modèles).")
    else:
        last = history[-1]
        mods = last["models"]
        keys = list(mods)
        names = {k: v["label"] for k, v in mods.items()}
        lab = [names[k] for k in keys]
        st.caption(f"Dernier benchmark : {datetime.fromtimestamp(last['ts']):%d/%m/%Y %H:%M} · "
                   f"{len(history)} benchmark(s) enregistré(s)")

        def avg_agree(k):
            vals = [last["agreement"][f"{k}|{o}"] for o in keys
                    if o != k and last["agreement"].get(f"{k}|{o}") is not None]
            return round(float(np.mean(vals)), 2) if vals else None

        summ = pd.DataFrame([{
            "Modèle": v["label"], "Scénarios": f"{v['pass']}/{v['n']}",
            "Précision@5": round(v["p5_mean"], 2), "Robustesse": round(v["para_stability"], 2),
            "Accord moyen": avg_agree(k), "Latence moy. (ms)": v["ms_mean"], "Latence p95 (ms)": v["ms_p95"],
            "Chargement (s)": v["load_s"], "Erreur": v["error"] or "—",
        } for k, v in mods.items()])
        st.dataframe(summ, hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        long = summ.melt(id_vars="Modèle", value_vars=["Précision@5", "Robustesse"], var_name="Mesure", value_name="Score")
        c1.plotly_chart(px.bar(long, x="Modèle", y="Score", color="Mesure", barmode="group", range_y=[0, 1],
                               title="Justesse et robustesse (0 à 1)", height=380), use_container_width=True)
        c2.plotly_chart(px.scatter(summ, x="Latence moy. (ms)", y="Précision@5", text="Modèle", title="Compromis : précision vs latence",
                                   height=380).update_traces(textposition="top center"), use_container_width=True)

        am = np.array([[last["agreement"].get(f"{a}|{b}", np.nan) for b in keys] for a in keys], dtype=float)
        pm = np.array([[next(g["p5"] for g in mods[k]["gold"] if g["intent"] == i) for k in keys]
                       for i in [g["intent"] for g in mods[keys[0]]["gold"]]])
        intents = [g["intent"] for g in mods[keys[0]]["gold"]]
        c3, c4 = st.columns(2)
        c3.plotly_chart(go.Figure(go.Heatmap(z=am, x=lab, y=lab, zmin=0, zmax=1, colorscale="Blues",
                                             text=np.round(am, 2), texttemplate="%{text}"))
                        .update_layout(title="Accord des top 5 entre modèles (Jaccard)", height=380),
                        use_container_width=True)
        c4.plotly_chart(go.Figure(go.Heatmap(z=pm, x=lab, y=intents, zmin=0, zmax=1, colorscale="RdYlGn",
                                             text=np.round(pm, 2), texttemplate="%{text:.1f}"))
                        .update_layout(title="Précision@5 par intention", height=380),
                        use_container_width=True)

        st.markdown("#### Scénarios de contraintes")
        pn = [p["persona"] for p in mods[keys[0]]["personas"]]
        grid = pd.DataFrame({names[k]: ["✅" if p["ok"] else "❌" for p in mods[k]["personas"]] for k in keys}, index=pn)
        st.dataframe(grid, use_container_width=True)

        st.markdown("#### Détail : ce que chaque modèle propose")
        mode = st.radio("Voir", ["Une intention", "Une formulation (paraphrases)", "Un scénario de contraintes"],
                        horizontal=True, key="bench_mode")
        if mode == "Une intention":
            it = st.selectbox("Intention", intents, key="bench_int")
            rows = []
            for k in keys:
                g = next(g for g in mods[k]["gold"] if g["intent"] == it)
                rows.append({"Modèle": names[k], "Précision@5": g["p5"], **{f"#{i + 1}": t for i, t in enumerate(g["top5"])}})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            expected = next(e for n, _, e in GOLD if n == it)
            st.caption("Familles attendues : " + ", ".join(sorted(expected)))
        elif mode == "Une formulation (paraphrases)":
            gp = st.selectbox("Groupe de formulations", [p["group"] for p in mods[keys[0]]["para"]], key="bench_para")
            texts = next(t for g, t in PARAPHRASES if g == gp)
            for i, t in enumerate(texts, 1):
                st.write(f"**Formulation {i}** : « {t} »")
            rows = []
            for k in keys:
                pr = next(p for p in mods[k]["para"] if p["group"] == gp)
                rows.append({"Modèle": names[k], "Stabilité": pr["stability"],
                             **{f"Formulation {i + 1}": " · ".join(top) for i, top in enumerate(pr["tops"])}})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            sc = st.selectbox("Scénario", pn, key="bench_sc")
            rows = [{"Modèle": names[k], "Résultat": "✅" if p["ok"] else "❌", "Top 5": " · ".join(p["top5"])}
                    for k in keys for p in mods[k]["personas"] if p["persona"] == sc]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if len(history) > 1:
            tr = pd.DataFrame([{"date": datetime.fromtimestamp(h["ts"]), "Modèle": v["label"], "Précision@5": v["p5_mean"]}
                               for h in history for v in h["models"].values() if "p5_mean" in v])
            st.plotly_chart(px.line(tr, x="date", y="Précision@5", color="Modèle", markers=True,
                                    title="Évolution entre benchmarks"), use_container_width=True)

    st.markdown("#### En production : runs par modèle")
    if not runs:
        st.info("Aucun run enregistré.")
    else:
        prod = pd.DataFrame([{"Modèle": model_label(normalize_model_key(r.get("engine"))), "ms": r["timings_ms"]["total"]}
                             for r in runs])
        g = prod.groupby("Modèle")["ms"].agg(["count", "median"]).reset_index()
        g.columns = ["Modèle", "Runs", "Latence médiane (ms)"]
        st.dataframe(g, hide_index=True, use_container_width=True)
        by_text = {}
        for r in runs:
            by_text.setdefault(r["text"], {})[normalize_model_key(r.get("engine"))] = [t["name"] for t in r["top"]]
        ov = []
        for d in by_text.values():
            ks = list(d)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    a, b = set(d[ks[i]]), set(d[ks[j]])
                    ov.append({"Paire": f"{model_label(ks[i])} / {model_label(ks[j])}",
                               "Jaccard": len(a & b) / len(a | b) if a | b else 1.0})
        if ov:
            o2 = pd.DataFrame(ov).groupby("Paire")["Jaccard"].agg(["count", "mean"]).reset_index()
            o2.columns = ["Paire de modèles", "Textes communs", "Accord moyen (Jaccard)"]
            st.dataframe(o2, hide_index=True, use_container_width=True)
        else:
            st.caption("Aucun texte n'a encore été analysé avec deux modèles. Changez de modèle dans la barre latérale "
                       "de l'application pour comparer en production.")
