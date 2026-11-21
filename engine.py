"""Moteur de recommandation : filtrage des contraintes, scores par blocs, agrégation pondérée.

Pipeline
    1. Profil utilisateur (formulaire + texte libre) -> signaux extraits du texte (extraction.py)
    2. Filtrage dur des activités incompatibles avec les contraintes explicites
    3. Score de chaque bloc = moyenne d'un score STRUCTURÉ (échelles, étiquettes du jeu de données)
       et d'un score SÉMANTIQUE (SBERT + TF-IDF, similarité cosinus)
    4. Agrégation pondérée des 4 blocs -> top 5 + explications
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from extraction import TextSignals, extract_signals, route_sentences
from preprocess import fold, preprocess_str

DATA_PATH = Path(__file__).parent / "data" / "disciplines.xlsx"
CACHE_DIR = Path(__file__).parent / "data" / ".cache"

BLOCKS = ["condition", "objectifs", "preferences", "contraintes"]
BLOCK_LABELS = {"condition": "Condition physique", "objectifs": "Objectifs",
                "preferences": "Préférences", "contraintes": "Contraintes"}
# Pondération par défaut (justifiée dans le README / l'interface)
DEFAULT_WEIGHTS = {"objectifs": 0.35, "condition": 0.30, "preferences": 0.20, "contraintes": 0.15}
STRUCT_SHARE = 0.5        # part du score structuré dans un bloc (le reste = sémantique)
SBERT_SHARE = 0.7         # dans la partie sémantique : SBERT vs TF-IDF lexical

CAPS = ["endurance", "force", "souplesse", "coordination", "equilibre"]
CAP_LABEL = {"endurance": "endurance", "force": "force", "souplesse": "souplesse",
             "coordination": "coordination", "equilibre": "équilibre"}
GOAL_TO_CAP = {"endurance": "endurance", "force": "force", "souplesse": "souplesse",
               "coordination": "coordination", "équilibre": "equilibre"}
ALL_GOALS = ["endurance", "force", "souplesse", "équilibre", "coordination", "détente", "vitesse", "puissance",
             "précision", "agilité", "jeu collectif", "plein air", "progression technique", "expression corporelle"]

WATER_TAGS = {"sport aquatique", "sport nautique", "glisse sur l'eau", "nage", "pagaie", "rame", "navigation à voile"}
FAMILIES = {
    "Sports de ballon": {"jeu de ballon"},
    "Raquette": {"sport de raquette"},
    "Combat": {"sport de combat", "combat de percussion", "combat de préhension"},
    "Eau": WATER_TAGS,
    "Glisse (neige, glace, roulettes)": {"glisse sur neige", "glisse sur glace", "glisse sur roulettes"},
    "Course et athlétisme": {"course", "athlétisme", "saut", "lancer"},
    "Cyclisme": {"cyclisme"},
    "Gymnastique et danse": {"gymnastique", "danse", "acrobatie", "chorégraphie"},
    "Précision et cible": {"sport de cible", "précision", "jeu de boules", "jeu de quilles"},
    "Marche et randonnée": {"marche", "terrain naturel", "orientation"},
    "Escalade": {"escalade", "franchissement"},
    "Équitation": {"équitation"},
}
TEXT_FAM = {"fam:ballon": "Sports de ballon", "fam:raquette": "Raquette", "fam:combat": "Combat", "fam:eau": "Eau",
            "fam:glisse": "Glisse (neige, glace, roulettes)", "fam:course": "Course et athlétisme",
            "fam:cyclisme": "Cyclisme", "fam:gym-danse": "Gymnastique et danse", "fam:cible": "Précision et cible",
            "fam:marche": "Marche et randonnée", "fam:escalade": "Escalade"}
EXPERIENCE = {"débutant": 1, "occasionnel": 2, "régulier": 3, "confirmé": 4}
IMPACT_TOL = ["faible", "moyenne", "élevée"]
INTENSITY_LABEL = {1: "faible", 2: "légère", 3: "modérée", 4: "soutenue", 5: "très soutenue"}
IMPACT_LABEL = {1: "faibles ou rares", 2: "présents", 3: "fréquents ou marqués"}
LEVEL_LABEL = {1: "prise en main simple", 2: "bases techniques à apprendre", 3: "initiation technique structurée"}


def _n(s: str) -> str:
    return str(s).replace("’", "'").strip().lower()


def _split(v) -> list[str]:
    return [_n(x) for x in str(v).split("|") if str(x).strip()] if pd.notna(v) else []


@dataclass
class UserProfile:
    text: str = ""
    endurance: int = 3
    force: int = 3
    mobilite: int = 3
    motivation: int = 3
    experience: str = "occasionnel"
    goals: list = field(default_factory=list)
    goal_zones: list = field(default_factory=list)
    formats: list = field(default_factory=list)         # individuel / duo / équipe / groupe
    environnement: str = "indifférent"
    families: list = field(default_factory=list)
    impact_tolerance: str = "moyenne"
    spare_zones: list = field(default_factory=list)
    no_contact: bool = False
    no_water: bool = False
    budget_seance_max: int | None = None
    sessions_per_week: int = 3
    minutes: int = 60


@dataclass
class Resolved:
    """Profil final après fusion formulaire + signaux du texte, avec traçabilité."""
    p: UserProfile
    sig: TextSignals
    abilities: dict
    exp_level: int
    exp_label: str
    goals: list
    goal_zones: list
    formats: list
    envs: list
    families: list
    disliked_families: list
    disliked_formats: list
    impact_tol: str
    spare_hard: list
    spare_soft: list
    no_contact: bool
    no_water: bool
    no_outdoor: bool
    no_indoor: bool
    target_intensity: float
    notes: list
    sources: dict


@dataclass
class Result:
    sport_id: str
    name: str
    row: pd.Series
    total: float
    blocks: dict
    structured: dict
    semantic: dict
    bullets: list


def load_sports() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="Disciplines")
    for c in ["categories", "zones_corps_sollicitees", "capacites_principales", "objectifs", "format_pratique",
              "environnement", "equipement", "mots_cles"]:
        df[c + "_l"] = df[c].apply(_split)
    df["contact"] = df["contact"].astype(str).str.strip().str.lower()
    return df


def _clip(x, lo=0.0, hi=1.0):
    return float(min(hi, max(lo, x)))


def _minmax(a: np.ndarray) -> np.ndarray:
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 1e-9 else np.full_like(a, 0.5)


class Recommender:
    def __init__(self, embedder=None):
        """embedder : objet avec .encode(list[str]) -> ndarray L2-normalisé (SBERT). None = TF-IDF seul."""
        self.df = load_sports()
        self.embedder = embedder
        self.block_texts = {b: [self._sport_text(b, r) for _, r in self.df.iterrows()] for b in BLOCKS}
        self.desc_texts = self.df["description_fr"].tolist()
        self.tfidf = {}
        self.tfidf_mat = {}
        for b in BLOCKS:
            v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1)
            self.tfidf_mat[b] = v.fit_transform([preprocess_str(t) for t in self.block_texts[b]])
            self.tfidf[b] = v
        self.tfidf["desc"] = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        self.tfidf_mat["desc"] = self.tfidf["desc"].fit_transform([preprocess_str(t) for t in self.desc_texts])
        self.emb = {}
        if embedder is not None:
            for b in BLOCKS:
                self.emb[b] = self._cached_embed(b, self.block_texts[b])
            self.emb["desc"] = self._cached_embed("desc", self.desc_texts)

    # ---------------- textes décrivant chaque sport, par bloc ----------------
    @staticmethod
    def _sport_text(block: str, r: pd.Series) -> str:
        if block == "condition":
            return (f"{r.nom_sport}. {r.description_fr} Intensité {INTENSITY_LABEL[int(r.intensite_loisir_1_5)]}. "
                    f"Difficulté d'entrée : {LEVEL_LABEL[int(r.difficulte_entree_1_3)]}. "
                    f"Capacités principales : {r.capacites_principales.replace('|', ', ')}.")
        if block == "objectifs":
            return (f"{r.nom_sport}. Objectifs : {r.objectifs.replace('|', ', ')}. "
                    f"Capacités développées : {r.capacites_principales.replace('|', ', ')}. "
                    f"Zones du corps travaillées : {r.zones_corps_sollicitees.replace('|', ', ')}.")
        if block == "preferences":
            return (f"{r.nom_sport}. Catégories : {r.categories.replace('|', ', ')}. "
                    f"Format : {r.format_pratique.replace('|', ', ')}. Environnement : {r.environnement.replace('|', ', ')}. "
                    f"Lieu : {r.lieu_pratique}. Mots-clés : {r.mots_cles.replace('|', ', ')}.")
        contact = {"aucun": "sans contact avec les autres pratiquants", "occasionnel": "contact occasionnel possible",
                   "réglementé": "contact réglementé avec les adversaires"}.get(r.contact, r.contact)
        return (f"{r.nom_sport}. Chocs et impacts {IMPACT_LABEL[int(r.impacts_1_3)]}, {contact}. "
                f"Intensité {INTENSITY_LABEL[int(r.intensite_loisir_1_5)]}. "
                f"Matériel : {r.equipement.replace('|', ', ')}. "
                f"Budget par séance : {int(r.budget_seance_min_eur)} à {int(r.budget_seance_max_eur)} euros.")

    def _cached_embed(self, key: str, texts: list[str]) -> np.ndarray:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(("|".join(texts) + getattr(self.embedder, "name", "m")).encode()).hexdigest()[:12]
        f = CACHE_DIR / f"{key}_{h}.npy"
        if f.exists():
            return np.load(f)
        arr = np.asarray(self.embedder.encode(texts))
        np.save(f, arr)
        return arr

    # ---------------- 1. résolution du profil ----------------
    def resolve(self, p: UserProfile) -> Resolved:
        sig = extract_signals(p.text)
        notes, src = [], {}
        ab = {"endurance": p.endurance, "force": p.force, "souplesse": p.mobilite}
        for cap, delta in sig.condition_adj.items():
            new = int(min(5, max(1, ab[cap] + delta)))
            if new != ab[cap]:
                notes.append(f"{CAP_LABEL[cap].capitalize()} ajustée de {ab[cap]} à {new} d'après votre texte")
                ab[cap] = new
        exp = p.experience
        if sig.level_hint == "débutant" and EXPERIENCE[exp] > 1:
            notes.append(f"Niveau « {exp} » ramené à « débutant » d'après votre texte")
            exp = "débutant"
        elif sig.level_hint == "confirmé" and EXPERIENCE[exp] < 4:
            notes.append(f"Niveau « {exp} » relevé à « confirmé » d'après votre texte")
            exp = "confirmé"
        L = EXPERIENCE[exp]
        goals = list(dict.fromkeys(list(p.goals) + [g for g in ALL_GOALS if g in sig.goals]))
        for g in goals:
            src[g] = "formulaire" if g in p.goals else "texte"
        goal_zones = list(dict.fromkeys(list(p.goal_zones) + sorted(sig.goal_zones)))

        formats = set(p.formats) | {k.split(":")[1] for k in sig.prefs if k.startswith("format:")}
        disliked_formats = {k.split(":")[1] for k in sig.disliked if k.startswith("format:")}
        formats -= disliked_formats
        envs = []
        if p.environnement != "indifférent":
            envs = [p.environnement]
        env_txt = {k.split(":")[1] for k in sig.prefs if k.startswith("env:")}
        if not envs and len(env_txt) == 1:
            envs = list(env_txt)
        families = set(p.families) | {TEXT_FAM[k] for k in sig.prefs if k in TEXT_FAM}
        dislike_fam = {TEXT_FAM[k] for k in sig.disliked if k in TEXT_FAM}
        families -= dislike_fam

        tol = p.impact_tolerance
        if "joint_sensitive" in sig.constraints and tol != "faible":
            notes.append("Articulations / chocs mentionnés dans votre texte : tolérance aux impacts ramenée à « faible »")
            tol = "faible"
        no_contact = p.no_contact or "no_contact" in sig.constraints
        no_water = p.no_water or "no_water" in sig.constraints
        if no_contact and not p.no_contact:
            notes.append("Refus du contact détecté dans votre texte")
        if no_water and not p.no_water:
            notes.append("Refus de l'eau détecté dans votre texte")
        no_out = "no_outdoor" in sig.constraints or p.environnement == "intérieur"
        no_in = "no_indoor" in sig.constraints or p.environnement == "extérieur"

        target = float(np.clip(1 + 0.5 * ab["endurance"] + 0.5 * L + 0.25 * (p.motivation - 3), 1, 5))
        return Resolved(p=p, sig=sig, abilities=ab, exp_level=L, exp_label=exp, goals=goals, goal_zones=goal_zones,
                        formats=sorted(formats), envs=envs, families=sorted(families),
                        disliked_families=sorted(dislike_fam), disliked_formats=sorted(disliked_formats),
                        impact_tol=tol, spare_hard=list(p.spare_zones),
                        spare_soft=sorted(set(sig.spare_zones) - set(p.spare_zones)),
                        no_contact=no_contact, no_water=no_water, no_outdoor=no_out and p.environnement != "extérieur",
                        no_indoor=no_in and p.environnement != "intérieur",
                        target_intensity=target, notes=notes, sources=src)

    # ---------------- 2. filtrage dur ----------------
    def filter_hard(self, R: Resolved) -> tuple[np.ndarray, dict]:
        keep = np.ones(len(self.df), bool)
        reasons: dict[str, list[str]] = {}
        for i, r in self.df.iterrows():
            why = []
            if R.no_contact and r.contact == "réglementé":
                why.append("contact réglementé")
            if R.no_water and (set(r.categories_l) & WATER_TAGS):
                why.append("pratique dans l'eau")
            if R.impact_tol == "faible" and r.impacts_1_3 >= 3:
                why.append("impacts fréquents")
            if R.p.environnement == "intérieur" and "intérieur" not in r.environnement_l:
                why.append("uniquement en extérieur")
            if R.p.environnement == "extérieur" and "extérieur" not in r.environnement_l:
                why.append("uniquement en intérieur")
            if R.no_outdoor and R.p.environnement == "indifférent" and "intérieur" not in r.environnement_l:
                why.append("uniquement en extérieur")
            if R.no_indoor and R.p.environnement == "indifférent" and "extérieur" not in r.environnement_l:
                why.append("uniquement en intérieur")
            ov = set(_n(z) for z in R.spare_hard) & set(r.zones_corps_sollicitees_l)
            if ov and (r.impacts_1_3 >= 3 or r.force_1_5 >= 4):
                why.append("sollicite fortement une zone à ménager (" + ", ".join(sorted(ov)) + ")")
            if R.p.budget_seance_max is not None and r.budget_seance_min_eur > R.p.budget_seance_max:
                why.append(f"budget séance dès {int(r.budget_seance_min_eur)} €")
            if why:
                keep[i] = False
                reasons[r.nom_sport] = why
        return keep, reasons

    # ---------------- 3a. scores structurés ----------------
    def _goal_match(self, goal: str, r) -> tuple[float, str]:
        tags = r.objectifs_l
        s, why = 0.0, ""
        if _n(goal) in tags:
            s, why = 1.0, f"objectif « {goal} » explicitement visé"
        cap = GOAL_TO_CAP.get(goal)
        if cap and r[f"{cap}_1_5"] / 5 > s:
            s, why = r[f"{cap}_1_5"] / 5, f"{goal} sollicitée {int(r[f'{cap}_1_5'])}/5"
        if goal == "détente" and r.intensite_loisir_1_5 <= 2 and s < 0.7:
            s, why = 0.7, "intensité légère, propice à la détente"
        if goal == "plein air" and "extérieur" in r.environnement_l and s < 0.8:
            s, why = 0.8, "pratiquée en extérieur"
        if goal == "jeu collectif" and ({"équipe", "groupe"} & set(r.format_pratique_l)) and s < 0.8:
            s, why = 0.8, "pratique en équipe ou en groupe"
        if goal in ("vitesse", "puissance") and r.intensite_loisir_1_5 >= 4 and s < 0.5:
            s, why = 0.5, "effort soutenu"
        return s, why

    def struct_scores(self, R: Resolved, r) -> tuple[dict, dict]:
        det: dict[str, list] = {b: [] for b in BLOCKS}     # (libellé, score, positif?)
        # --- condition
        gaps = [max(0, r[f"{c}_1_5"] - (R.abilities[a] + 1)) / 4
                for c, a in (("endurance", "endurance"), ("force", "force"), ("souplesse", "souplesse"))]
        cap_fit = 1 - float(np.mean(gaps))
        allowed = {1: 1, 2: 2}.get(R.exp_level, 3)
        lvl_fit = 1 - 0.35 * max(0, r.difficulte_entree_1_3 - allowed)
        diff = r.intensite_loisir_1_5 - R.target_intensity
        int_fit = 1 - (max(0, diff - 0.5) / 3 + max(0, -diff - 1.0) / 4)
        cond = _clip(0.5 * cap_fit + 0.25 * lvl_fit + 0.25 * int_fit)
        det["condition"] = [("exigences physiques vs capacités", cap_fit), ("difficulté d'entrée vs niveau", lvl_fit),
                            ("intensité vs intensité cible", int_fit)]
        # --- objectifs
        gs = []
        for g in R.goals:
            s, why = self._goal_match(g, r)
            gs.append(s)
            det["objectifs"].append((f"{g} — {why}" if why else f"{g} — peu associé", s))
        for z in R.goal_zones:
            hit = _n(z) in r.zones_corps_sollicitees_l
            s = (0.5 + 0.5 * r.force_1_5 / 5) if hit else 0.0
            gs.append(s)
            det["objectifs"].append((f"zone « {z} » " + ("sollicitée" if hit else "peu sollicitée"), s))
        obj = float(np.mean(gs)) if gs else None
        # --- préférences
        ps = []
        if R.formats:
            ok = bool(set(R.formats) & set(r.format_pratique_l))
            ps.append(1.0 if ok else 0.0)
            det["preferences"].append(("format " + "/".join(r.format_pratique_l) + (" ✓" if ok else " ✗"), float(ok)))
        if R.envs:
            ok = R.envs[0] in r.environnement_l
            ps.append(1.0 if ok else 0.0)
            det["preferences"].append((f"environnement {'/'.join(r.environnement_l)}", float(ok)))
        if R.families:
            tags = set(r.categories_l)
            hit = [f for f in R.families if {_n(t) for t in FAMILIES[f]} & tags]
            ps.append(1.0 if hit else 0.0)
            det["preferences"].append(("famille " + (", ".join(hit) if hit else "non appréciée"), float(bool(hit))))
        bad = [f for f in R.disliked_families if {_n(t) for t in FAMILIES[f]} & set(r.categories_l)]
        bad += [f for f in R.disliked_formats if f in r.format_pratique_l]
        if bad:
            ps.append(0.0)
            det["preferences"].append(("à éviter selon votre texte : " + ", ".join(bad), 0.0))
        pref = float(np.mean(ps)) if ps else None
        # --- contraintes (partie souple)
        cs = []
        imp = int(r.impacts_1_3)
        s_imp = {"faible": {1: 1, 2: 0.5, 3: 0}, "moyenne": {1: 1, 2: 1, 3: 0.6}, "élevée": {1: 1, 2: 1, 3: 1}}[R.impact_tol][imp]
        cs.append(s_imp)
        det["contraintes"].append((f"impacts {IMPACT_LABEL[imp]}", s_imp))
        soft = {_n(z) for z in R.spare_soft} | {_n(z) for z in R.spare_hard}
        if soft:
            ov = soft & set(r.zones_corps_sollicitees_l)
            s = 1 - len(ov) / max(1, len(soft)) * (0.5 + 0.5 * r.force_1_5 / 5)
            cs.append(s)
            det["contraintes"].append(("zones à ménager " + ("sollicitées : " + ", ".join(sorted(ov)) if ov else "épargnées"), s))
        if R.no_contact or r.contact != "aucun":
            s = {"aucun": 1.0, "occasionnel": 0.7 if R.no_contact else 0.9, "réglementé": 0.5}[r.contact]
            if R.no_contact:
                cs.append(s)
                det["contraintes"].append((f"contact : {r.contact}", s))
        if R.p.budget_seance_max is not None:
            b = R.p.budget_seance_max
            s = 1.0 if r.budget_seance_max_eur <= b else _clip((b - r.budget_seance_min_eur) / max(1, r.budget_seance_max_eur - r.budget_seance_min_eur)) * 0.8 + 0.2
            cs.append(s)
            det["contraintes"].append((f"budget séance {int(r.budget_seance_min_eur)}–{int(r.budget_seance_max_eur)} € (plafond {b} €)", s))
        if R.p.sessions_per_week >= 4 and (imp == 3 or r.intensite_loisir_1_5 == 5):
            cs.append(0.6)
            det["contraintes"].append(("séances fréquentes + effort/impacts élevés : récupération", 0.6))
        cons = float(np.mean(cs))
        return {"condition": cond, "objectifs": obj, "preferences": pref, "contraintes": cons}, det

    # ---------------- 3b. scores sémantiques ----------------
    def user_block_texts(self, R: Resolved) -> dict[str, str]:
        p = R.p
        lev = {1: "faible", 2: "plutôt faible", 3: "moyen", 4: "bon", 5: "très bon"}
        routed = route_sentences(p.text)
        form = {
            "condition": f"Je suis {R.exp_label} en sport. Mon endurance est {lev[R.abilities['endurance']]}, ma force est "
                         f"{lev[R.abilities['force']]} et ma mobilité est {lev[R.abilities['souplesse']]}.",
            "objectifs": ("Je souhaite travailler : " + ", ".join(R.goals + [f"{z}" for z in R.goal_zones]) + ".") if (R.goals or R.goal_zones) else "",
            "preferences": " ".join(filter(None, [
                ("Je préfère les activités " + ", ".join(R.formats) + ".") if R.formats else "",
                (f"Je préfère pratiquer en {R.envs[0]}.") if R.envs else "",
                ("J'aime : " + ", ".join(R.families).lower() + ".") if R.families else ""])),
            "contraintes": " ".join(filter(None, [
                {"faible": "Je dois éviter les chocs et impacts sur mes articulations.", "moyenne": "", "élevée": ""}[R.impact_tol],
                "Je ne veux pas de sport de contact." if R.no_contact else "",
                ("Je dois ménager : " + ", ".join(R.spare_hard + R.spare_soft) + ".") if (R.spare_hard or R.spare_soft) else "",
                f"Je dispose de {p.sessions_per_week} séances de {p.minutes} minutes par semaine."])),
        }
        return {b: " ".join(filter(None, [form[b]] + routed[b])).strip() for b in BLOCKS}

    def sem_scores(self, texts: dict[str, str]) -> tuple[dict, dict, dict]:
        """Retourne (score_sem_normalisé[b], sbert_brut[b], tfidf_brut[b]) sur tous les sports."""
        out, sb, tf = {}, {}, {}
        for b in BLOCKS:
            t = texts[b]
            if not t:
                continue
            cos_tf = (self.tfidf_mat[b] @ self.tfidf[b].transform([preprocess_str(t)]).T).toarray().ravel()
            tf[b] = cos_tf
            if self.embedder is not None:
                cos_sb = self.emb[b] @ np.asarray(self.embedder.encode([t]))[0]
                sb[b] = cos_sb
                tf_part = _minmax(cos_tf) if cos_tf.max() > 0 else _minmax(cos_sb)
                out[b] = SBERT_SHARE * _minmax(cos_sb) + (1 - SBERT_SHARE) * tf_part
            else:
                out[b] = _minmax(cos_tf)
        return out, sb, tf

    # ---------------- 4. recommandation ----------------
    def recommend(self, p: UserProfile, weights: dict | None = None, top_k: int = 5) -> dict:
        w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        R = self.resolve(p)
        keep, excluded = self.filter_hard(R)
        texts = self.user_block_texts(R)
        sem, sb_raw, tf_raw = self.sem_scores(texts)

        rows = []
        for i, r in self.df.iterrows():
            st, det = self.struct_scores(R, r)
            blocks, wsum, acc = {}, 0.0, 0.0
            for b in BLOCKS:
                s, m = st[b], sem[b][i] if b in sem else None
                if s is not None and m is not None:
                    v = STRUCT_SHARE * s + (1 - STRUCT_SHARE) * m
                else:
                    v = s if s is not None else m
                blocks[b] = v
                if v is not None:
                    acc += w[b] * v
                    wsum += w[b]
            rows.append((i, acc / wsum if wsum else 0.0, blocks, st, det))

        # phrase utilisateur la plus proche (explicabilité)
        sent_best = self._best_sentences(p.text)
        order = sorted([x for x in rows if keep[x[0]]], key=lambda x: -x[1])
        results = []
        for i, tot, blocks, st, det in order[:top_k]:
            r = self.df.loc[i]
            sem_i = {b: float(sem[b][i]) for b in sem}
            results.append(Result(r.sport_id, r.nom_sport, r, tot, blocks, st, sem_i,
                                  self._bullets(R, r, det, sb_raw, tf_raw, i, sent_best)))
        # classement complet (visualisation) + comparaison des méthodes
        full = pd.DataFrame([{"sport": self.df.loc[i, "nom_sport"], "sport_id": self.df.loc[i, "sport_id"],
                              "total": tot, **{b: blocks[b] for b in BLOCKS}, "retenu": bool(keep[i])}
                             for i, tot, blocks, _, _ in rows]).sort_values("total", ascending=False)
        return {"resolved": R, "results": results, "excluded": excluded, "texts": texts,
                "full": full, "methods": self._methods(R, texts, keep, rows), "weights": w}

    def _methods(self, R, texts, keep, rows, k=5) -> dict:
        g = " ".join(t for t in texts.values() if t)
        out = {}
        tf = (self.tfidf_mat["desc"] @ self.tfidf["desc"].transform([preprocess_str(g)]).T).toarray().ravel()
        out["TF-IDF (lexical)"] = tf
        if self.embedder is not None:
            out["SBERT (sémantique)"] = self.emb["desc"] @ np.asarray(self.embedder.encode([g]))[0]
        out["Hybride complet (blocs + règles)"] = np.array([x[1] for x in rows])
        res = {}
        for name, sc in out.items():
            idx = [i for i in np.argsort(-sc) if keep[i]][:k]
            res[name] = [(self.df.loc[i, "nom_sport"], float(sc[i])) for i in idx]
        return res

    def _best_sentences(self, text: str):
        from preprocess import split_sentences
        sents = split_sentences(text)
        if not sents or self.embedder is None:
            return None
        e = np.asarray(self.embedder.encode(sents))
        return sents, e @ self.emb["desc"].T      # (n_phrases, n_sports)

    def _bullets(self, R, r, det, sb_raw, tf_raw, i, sent_best) -> list[dict]:
        out = []
        good = [d for d in det["objectifs"] if d[1] >= 0.6]
        if R.goals or R.goal_zones:
            out.append({"block": "Objectifs", "ok": bool(good), "text":
                        ("Répond à : " + " ; ".join(d[0] for d in good)) if good else "Peu d'objectifs directement couverts"})
        out.append({"block": "Condition physique", "ok": det["condition"][0][1] >= 0.7, "text":
                    f"Intensité {INTENSITY_LABEL[int(r.intensite_loisir_1_5)]} ({int(r.intensite_loisir_1_5)}/5, cible ≈ {R.target_intensity:.1f}), "
                    f"entrée : {LEVEL_LABEL[int(r.difficulte_entree_1_3)]}. Endurance {int(r.endurance_1_5)}/5, force {int(r.force_1_5)}/5, "
                    f"souplesse {int(r.souplesse_1_5)}/5."})
        pr = [d[0] for d in det["preferences"] if d[1] >= 0.6]
        if det["preferences"]:
            out.append({"block": "Préférences", "ok": bool(pr), "text": ("Correspond à : " + " ; ".join(pr)) if pr else "Ne correspond pas à vos préférences déclarées"})
        out.append({"block": "Contraintes", "ok": det["contraintes"][0][1] >= 0.6, "text":
                    (f"Impacts {IMPACT_LABEL[int(r.impacts_1_3)]}, contact {r.contact}. " + " ; ".join(d[0] for d in det["contraintes"][1:])).strip()})
        if sent_best is not None:
            sents, sim = sent_best
            j = int(np.argmax(sim[:, i]))
            out.append({"block": "Sémantique", "ok": True, "text":
                        f"Votre phrase « {sents[j]} » est la plus proche de la description (cos SBERT = {sim[j, i]:.2f})."})
        return out
