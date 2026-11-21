"""Journalisation : runs de recommandation (JSONL) + logs applicatifs (fichier tournant)."""
import json
import logging
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "data" / "logs"
RUNS = LOG_DIR / "runs.jsonl"
APP_LOG = LOG_DIR / "app.log"


def get_logger() -> logging.Logger:
    lg = logging.getLogger("sport-reco")
    if not lg.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        lg.setLevel(logging.DEBUG)
        h = RotatingFileHandler(APP_LOG, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        lg.addHandler(h)
    return lg


def log_run(profile, out, engine_mode: str) -> str:
    R = out["resolved"]
    rec = {
        "id": uuid.uuid4().hex[:8], "ts": time.time(), "engine": engine_mode,
        "text": profile.text,
        "profile": {k: getattr(profile, k) for k in ("endurance", "force", "mobilite", "motivation", "experience",
                                                       "sessions_per_week", "minutes")},
        "signals": {"goals": R.goals, "goal_zones": R.goal_zones, "formats": R.formats, "envs": R.envs,
                    "families": R.families, "disliked": R.disliked_families + R.disliked_formats,
                    "impact_tol": R.impact_tol, "no_contact": R.no_contact, "no_water": R.no_water,
                    "no_jump": R.no_jump, "spare": R.spare_hard + R.spare_soft,
                    "budget": R.p.budget_seance_max, "exp": R.exp_label, "target_intensity": round(R.target_intensity, 2)},
        "notes": R.notes, "weights": out["weights"],
        "top": [{"name": r.name, "id": r.sport_id, "score": round(float(r.total), 4),
                 "blocks": {b: (None if v is None else round(float(v), 3)) for b, v in r.blocks.items()}} for r in out["results"]],
        "excluded": out["excluded"], "n_excluded": len(out["excluded"]),
        "methods": {k: [s for s, _ in v] for k, v in out["methods"].items()},
        "timings_ms": {k: round(v, 1) for k, v in out["timings"].items()},
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    get_logger().info("run %s | %.0f ms | top=%s | %d exclus", rec["id"], rec["timings_ms"]["total"],
                      [t["name"] for t in rec["top"]], rec["n_excluded"])
    return rec["id"]


def read_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    out = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out
