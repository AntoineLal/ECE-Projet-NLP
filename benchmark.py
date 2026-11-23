"""Benchmark des modèles NLP : mêmes scénarios exécutés sur chaque moteur, résultats persistés.

Mesures par modèle : scénarios validés, latence moyenne et p95, temps de chargement, top 5 obtenu.
Mesure entre modèles : accord des top 5 (indice de Jaccard moyen sur les scénarios).
"""
import json
import time
from pathlib import Path

import numpy as np

from engine import UserProfile
from gold import GOLD, PARAPHRASES
from personas import PERSONAS
from shared import MODELS, build_engine, model_label
from telemetry import get_logger

LOG_DIR = Path(__file__).parent / "data" / "logs"
BENCH = LOG_DIR / "benchmark.jsonl"


def level_for(text: str) -> str:
    """Même règle que le panneau Admin : niveau déduit du texte des scénarios."""
    return "confirmé" if "confirmé" in text else ("régulier" if "salle de sport" in text else "occasionnel")


def run_model(engine, personas=PERSONAS) -> list[dict]:
    rows = []
    for name, text, check in personas:
        t0 = time.perf_counter()
        o = engine.recommend(UserProfile(text=text, experience=level_for(text)))
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"persona": name, "ok": bool(check(o, engine.df)), "ms": round(ms, 1),
                     "top5": [r.name for r in o["results"]], "n_kept": int(o["n_kept"])})
    return rows


def gold_metrics(engine) -> dict:
    """Pertinence (précision@5 sur les intentions) et robustesse aux paraphrases (Jaccard des top 5)."""
    cats = dict(zip(engine.df.nom_sport, engine.df.categories_l))
    rows, p5 = [], []
    for name, text, expected in GOLD:
        o = engine.recommend(UserProfile(text=text))
        top = [r.name for r in o["results"]]
        hits = sum(1 for n in top if set(cats.get(n, [])) & expected)
        prec = hits / len(top) if top else 0.0
        p5.append(prec)
        rows.append({"intent": name, "p5": round(prec, 2), "top5": top})
    para_rows, para_j = [], []
    for group, texts in PARAPHRASES:
        tops = [[r.name for r in engine.recommend(UserProfile(text=t))["results"]] for t in texts]
        pairs = [jaccard(tops[i], tops[j]) for i in range(len(tops)) for j in range(i + 1, len(tops))]
        para_j.append(float(np.mean(pairs)))
        para_rows.append({"group": group, "stability": round(float(np.mean(pairs)), 2), "tops": tops})
    return {"gold": rows, "p5_mean": round(float(np.mean(p5)), 3),
            "para": para_rows, "para_stability": round(float(np.mean(para_j)), 3)}


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def agreement(models: dict) -> dict:
    """Matrice d'accord : Jaccard moyen des top 5 entre chaque paire de modèles."""
    keys = list(models)
    out = {}
    for a in keys:
        for b in keys:
            pairs = [jaccard(ra["top5"], rb["top5"]) for ra, rb in zip(models[a]["personas"], models[b]["personas"])]
            out[f"{a}|{b}"] = round(float(np.mean(pairs)), 3) if pairs else None
    return out


def run_benchmark(keys=None, engines=None, progress=None) -> dict:
    """Exécute le benchmark. `engines` : dict clé -> (moteur, erreur, secondes) déjà chargés (cache Streamlit)."""
    keys = list(keys or MODELS)
    res = {"ts": time.time(), "models": {}}
    for i, key in enumerate(keys):
        if progress:
            progress(i, len(keys), key)
        eng, err, load_s = engines[key] if engines and key in engines else build_engine(key)
        rows = run_model(eng)
        gm = gold_metrics(eng) if eng.df is not None else {}
        ms = np.array([r["ms"] for r in rows])
        res["models"][key] = {
            "label": model_label(key), "error": err, "load_s": round(float(load_s), 2),
            "personas": rows, "pass": int(sum(r["ok"] for r in rows)), "n": len(rows),
            "ms_mean": round(float(ms.mean()), 1), "ms_p95": round(float(np.percentile(ms, 95)), 1),
            "fallback": eng.embedder is None and MODELS[key]["hf"] is not None,
            **gm,
        }
    res["agreement"] = agreement(res["models"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCH, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    get_logger().info("benchmark : %s", ", ".join(f"{k} {v['pass']}/{v['n']}" for k, v in res["models"].items()))
    return res


def read_benchmarks() -> list[dict]:
    if not BENCH.exists():
        return []
    out = []
    for line in BENCH.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


if __name__ == "__main__":  # exécution en ligne de commande
    r = run_benchmark(progress=lambda i, n, k: print(f"[{i + 1}/{n}] {k}…", flush=True))
    for k, v in r["models"].items():
        print(f"{v['label']:45s} scénarios {v['pass']}/{v['n']}  précision@5 {v['p5_mean']:.2f}  "
              f"paraphrases {v['para_stability']:.2f}  latence {v['ms_mean']} ms  chargement {v['load_s']} s")
