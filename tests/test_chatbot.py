"""Tests du bonus B : chatbot adaptatif.

Lancement :  python -m unittest discover -s tests -v
Gemini est SIMULÉ : chaque message reçoit le JSON que le modèle produirait. Aucun quota n'est consommé.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bonus_chatbot as bc  # noqa: E402
from engine import UserProfile  # noqa: E402
from llm_config import LLMSettings  # noqa: E402
from test_bonus import fixture  # noqa: E402


# =================================================================== bonus B : scénarios du cahier des charges
# Les messages sont lus par un Gemini SIMULÉ : le transport renvoie le JSON que Gemini produirait.
# Aucun appel réel, donc aucune consommation de quota.
WATER_TAGS_T = {"sport aquatique", "nage", "sport nautique", "pagaie", "rame", "navigation à voile", "glisse sur l'eau"}


def simulated_gemini(contract: dict):
    """Transport simulé : renvoie `contract` comme réponse JSON de Gemini."""
    def t(url, headers, body, timeout):
        return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps(contract, ensure_ascii=False)}]},
                                     "finishReason": "STOP"}]}
    return t


def contract(**fields) -> dict:
    base = {"intention": "mise_a_jour"}
    base.update(fields)
    return base


class CountingEngine:
    """Enveloppe le moteur pour compter les recalculs (vérifie l'absence de recalcul inutile)."""

    def __init__(self, engine):
        self.inner, self.calls = engine, 0

    def recommend(self, p, w=None, top_k=5):
        self.calls += 1
        return self.inner.recommend(p, w, top_k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ChatScenarios(unittest.TestCase):
    """Les 8 tests obligatoires, plus les cas de contradiction, de sans-changement et de réinitialisation."""

    @classmethod
    def setUpClass(cls):
        cls.engine = fixture()["rec"]
        cls.df = cls.engine.df

    def start(self, text="Je m'entraîne en salle depuis un an et je veux progresser.", experience="régulier", **kw):
        p = UserProfile(text=text, experience=experience, **kw)
        return p, self.engine.recommend(p, None)

    def send(self, msg, profile, out, llm=None):
        transport = simulated_gemini(llm) if llm is not None else None
        settings = LLMSettings(provider="gemini", gemini_key="FAKE-TEST-KEY-000000") if llm is not None else LLMSettings(provider="off")
        res = bc.process_message(msg, profile, out, self.engine, None, settings, transport=transport)
        if res["kind"] == "maj":
            return res, res["profile"], res["out"]
        return res, profile, out

    def admissible(self, out):
        return [a["name"] for a in out["anatomy"] if a["retained"]]

    def cats(self, name):
        return set(self.df.loc[self.df.nom_sport == name, "categories_l"].iloc[0])

    # --- Test 1
    def test_1_no_aquatic_sport(self):
        p, o = self.start()
        res, p, o = self.send("Je ne veux aucun sport aquatique.", p, o,
                              contract(exclusions_strictes={"eau": True}))
        self.assertEqual(res["kind"], "maj")
        self.assertFalse([n for n in self.admissible(o) if self.cats(n) & WATER_TAGS_T])
        self.assertFalse([r.name for r in o["results"] if self.cats(r.name) & WATER_TAGS_T])

    def test_1_no_aquatic_sport_by_rules_without_llm(self):
        p, o = self.start()
        res, p, o = self.send("Je ne veux aucun sport aquatique.", p, o)
        self.assertFalse([n for n in self.admissible(o) if self.cats(n) & WATER_TAGS_T])

    # --- Test 2
    def test_2_only_individual_sports_excludes_team_sports(self):
        p, o = self.start()
        res, p, o = self.send("Je veux uniquement des sports individuels.", p, o,
                              contract(exclusions_strictes={"seul_uniquement": True}))
        self.assertEqual(res["kind"], "maj")
        collectifs = [n for n in self.admissible(o) if "sport collectif" in self.cats(n)]
        self.assertEqual(collectifs, [], "aucun sport collectif ne doit rester admissible")
        self.assertTrue(all("individuel" in r.row.format_pratique.split("|") for r in o["results"]))

    # --- Test 3 : préférence souple, pas d'exclusion
    def test_3_outdoor_preference_is_soft(self):
        p, o = self.start()
        before_kept = set(self.admissible(o))
        res, p2, o2 = self.send("Je préfère les sports en extérieur.", p, o,
                                contract(preferences_souples={"environnement": "extérieur"}))
        self.assertEqual(res["kind"], "maj")
        self.assertEqual(p2.env_preference, "extérieur")
        self.assertEqual(set(self.admissible(o2)), before_kept, "une préférence ne doit exclure aucun sport")
        # un sport d'extérieur doit voir son score augmenter (critère de préférence)
        ext = [a for a in o2["anatomy"] if a["retained"] and "extérieur" in a["row"].environnement_l
               and a["name"] in {x["name"] for x in o["anatomy"] if x["retained"]}]
        before = {a["name"]: a["total"] for a in o["anatomy"]}
        gains = [a["name"] for a in ext if a["total"] > before[a["name"]]]
        self.assertTrue(gains, "les sports compatibles avec la préférence doivent être favorisés")

    # --- Test 4 : impacts élevés
    def test_4_high_impact_sports_are_filtered(self):
        p, o = self.start()
        res, p2, o2 = self.send("Je veux éviter les sports à impacts élevés.", p, o,
                                contract(contraintes_physiques={"impacts": "faible"}))
        self.assertEqual(res["kind"], "maj")
        self.assertEqual(p2.impact_tolerance, "faible")
        high = [n for n in self.admissible(o2) if int(self.df.loc[self.df.nom_sport == n, "impacts_1_3"].iloc[0]) >= 3]
        self.assertEqual(high, [])

    # --- Test 5 : levée d'une exclusion
    def test_5_accepting_team_sports_lifts_the_exclusion(self):
        p, o = self.start()
        res, p, o = self.send("Je veux uniquement des sports individuels.", p, o,
                              contract(exclusions_strictes={"seul_uniquement": True}))
        n_before = len(self.admissible(o))
        res, p, o = self.send("Finalement, j'accepte aussi les sports collectifs.", p, o,
                              contract(acceptations={"categories": ["sport collectif"]}))
        self.assertEqual(res["kind"], "maj")
        self.assertNotIn("sport collectif", p.excluded_categories)
        self.assertFalse(p.only_solo)
        self.assertGreater(len(self.admissible(o)), n_before)
        self.assertTrue(any("sport collectif" in self.cats(n) for n in self.admissible(o)))

    # --- Test 6 : objectif + exclusion d'une activité
    def test_6_endurance_goal_without_running(self):
        p, o = self.start()
        res, p, o = self.send("Je veux améliorer mon endurance sans faire de course à pied.", p, o,
                              contract(objectifs_ajoutes=["endurance"],
                                       exclusions_strictes={"categories": ["course"]}))
        self.assertIn("endurance", p.goals)
        self.assertIn("course", p.excluded_categories)
        self.assertFalse([n for n in self.admissible(o) if "course" in self.cats(n)])
        self.assertFalse([r.name for r in o["results"] if "course" in self.cats(r.name)])

    # --- Test 7 : messages successifs
    def test_7_successive_messages_keep_earlier_constraints(self):
        p, o = self.start()
        steps = [
            ("Je ne veux aucun sport aquatique.", contract(exclusions_strictes={"eau": True})),
            ("Je préfère les activités individuelles.", contract(preferences_souples={"formats": ["individuel"]})),
            ("Mon budget est de 15 euros par séance.", contract(budget_max_eur=15)),
            ("Je m'entraîne 4 fois par semaine.", contract(seances_par_semaine=4)),
        ]
        for msg, llm in steps:
            res, p, o = self.send(msg, p, o, llm)
            self.assertEqual(res["kind"], "maj", msg)
        self.assertTrue(p.no_water)
        self.assertEqual(p.budget_seance_max, 15)
        self.assertEqual(p.formats, ["individuel"])
        self.assertEqual(p.sessions_per_week, 4)
        self.assertFalse([n for n in self.admissible(o) if self.cats(n) & WATER_TAGS_T])

    # --- Test 8 : aucun sport admissible
    def test_8_no_admissible_sport_is_explained_not_relaxed(self):
        p, o = self.start()
        msgs = [
            ("Je ne veux aucun sport aquatique.", contract(exclusions_strictes={"eau": True})),
            ("Je ne veux aucun sport de contact.", contract(exclusions_strictes={"contact": False})),
            ("Je veux éviter les sports à impacts élevés.", contract(contraintes_physiques={"impacts": "faible"})),
            ("Je veux rester en intérieur.", contract(exclusions_strictes={"environnements": ["extérieur"]})),
            ("Mon budget est de 0 euro.", contract(budget_max_eur=0)),
            ("Je veux uniquement des sports individuels.", contract(exclusions_strictes={"seul_uniquement": True})),
            ("Je ne veux pas de course à pied.", contract(exclusions_strictes={"categories": ["course"]})),
            ("Je ne veux pas de sports de combat.", contract(exclusions_strictes={"familles": ["Combat"]})),
            ("Je ne veux pas de raquette.", contract(exclusions_strictes={"familles": ["Raquette"]})),
        ]
        for msg, llm in msgs:
            res, p, o = self.send(msg, p, o, llm)
            self.assertEqual(res["kind"], "maj", msg)
        remaining = self.admissible(o)
        self.assertGreater(len(remaining), 0, "la combinaison doit encore laisser un sport, pour le test suivant")
        # l'utilisateur refuse le dernier sport admissible : aucun sport ne doit rester
        res, p, o = self.send("Je ne veux pas de " + remaining[0] + ".", p, o,
                              contract(exclusions_strictes={"sports": [remaining[0]]}))
        self.assertEqual(res["kind"], "maj")
        self.assertEqual(res["n_kept"], 0)
        self.assertEqual(self.admissible(o), [])
        self.assertEqual(o["results"], [], "aucun sport ne doit être affiché")
        self.assertTrue(res["suggestions"], "des pistes doivent être proposées")
        self.assertTrue(all("redeviendraient" in s for s in res["suggestions"]))
        # les contraintes ne sont jamais relâchées automatiquement
        self.assertTrue(p.no_water)
        self.assertIn("extérieur", p.excluded_environments)
        self.assertEqual(p.budget_seance_max, 0)
        self.assertIn(remaining[0], p.excluded_sports)
        self.assertFalse(any("assouplis" in n for n in o["resolved"].notes))

    # --- contradictions et corrections
    def test_correction_replaces_previous_preference(self):
        p, o = self.start()
        res, p, o = self.send("Je préfère pratiquer en intérieur.", p, o,
                              contract(preferences_souples={"environnement": "intérieur"}))
        res, p, o = self.send("Finalement, je veux pratiquer en extérieur.", p, o,
                              contract(preferences_souples={"environnement": "extérieur"},
                                       corrections=[{"champ": "environnement", "ancienne": "intérieur", "nouvelle": "extérieur"}]))
        self.assertEqual(p.env_preference, "extérieur")
        self.assertTrue(any("intérieur" in c for c in res["contradictions"]))
        self.assertTrue(any("extérieur" in c for c in res["changes"]))

    def test_exclusion_removes_opposite_preference(self):
        p, o = self.start(formats=["équipe"])
        res, p, o = self.send("Je ne veux pas d'équipe.", p, o,
                              contract(exclusions_strictes={"formats": ["équipe"]}))
        self.assertNotIn("équipe", p.formats)
        self.assertIn("équipe", p.excluded_formats)

    # --- sans changement : aucun recalcul
    def test_message_without_criteria_does_not_recalculate(self):
        p, o = self.start()
        counter = CountingEngine(self.engine)
        settings = LLMSettings(provider="off")
        res = bc.process_message("Bonjour, merci pour cette application.", p, o, counter, None, settings)
        self.assertNotEqual(res["kind"], "maj")
        self.assertEqual(counter.calls, 0)

    def test_repeated_identical_message_is_no_change(self):
        p, o = self.start()
        res, p, o = self.send("Je ne veux aucun sport aquatique.", p, o, contract(exclusions_strictes={"eau": True}))
        counter = CountingEngine(self.engine)
        again = bc.process_message("Je ne veux aucun sport aquatique.", p, o, counter, None, LLMSettings(provider="off"))
        self.assertEqual(again["kind"], "sans_changement")
        self.assertEqual(counter.calls, 0)

    # --- réinitialisation : seulement les modifications de conversation
    def test_reset_returns_to_questionnaire_profile(self):
        base, o = self.start(text="Je m'entraîne en salle.", budget_seance_max=20)
        res, p, o = self.send("Je ne veux aucun sport aquatique.", base, o, contract(exclusions_strictes={"eau": True}))
        self.assertTrue(p.no_water)
        restored = bc.reset_profile(base)
        self.assertFalse(restored.no_water)
        self.assertEqual(restored.budget_seance_max, 20)
        self.assertEqual(restored.text, base.text)

    # --- le LLM ne peut ni inventer un score, ni injecter un champ inconnu
    def test_llm_cannot_inject_scores_or_unknown_fields(self):
        p, o = self.start()
        res, p2, o2 = self.send("Je veux du foot.", p, o,
                                contract(score_global=99, excluded_sports_admin=["x"],
                                         preferences_souples={"familles": ["Sports de ballon"]}))
        self.assertEqual(res["kind"], "maj")
        self.assertTrue(any("score_global" in r for r in res["rejected"]))
        self.assertTrue(any("excluded_sports_admin" in r for r in res["rejected"]))
        self.assertIn("Sports de ballon", p2.families)

    def test_llm_invalid_values_are_rejected(self):
        vocab = bc.vocabulary(self.engine)
        clean, rejected = bc.validate_contract(contract(budget_max_eur=9999, seances_par_semaine=12,
                                                        exclusions_strictes={"categories": ["n'importe quoi"]}), vocab)
        self.assertIsNone(clean["budget_max_eur"])
        self.assertIsNone(clean["seances_par_semaine"])
        self.assertEqual(clean["exclusions_strictes"]["categories"], [])
        self.assertGreaterEqual(len(rejected), 3)

    def test_invalid_llm_json_falls_back_to_rules(self):
        def broken(url, headers, body, timeout):
            return 200, {"candidates": [{"content": {"parts": [{"text": "Désolé, pas de JSON ici."}]}, "finishReason": "STOP"}]}
        p, o = self.start()
        settings = LLMSettings(provider="gemini", gemini_key="FAKE-TEST-KEY-000000")
        res = bc.process_message("Je ne veux aucun sport aquatique.", p, o, self.engine, None, settings, transport=broken)
        self.assertEqual(res["kind"], "maj")
        self.assertTrue(res["profile"].no_water or "sports_aquatiques" in res["profile"].excluded_groups)
        self.assertIn("règles", res["note"])

    def test_ambiguous_message_asks_clarification(self):
        p, o = self.start()
        res = bc.process_message("Je n'aime pas le sport.", p, o, self.engine, None, LLMSettings(provider="off"))
        self.assertEqual(res["kind"], "precision")

    # --- démonstration avant / après (affichée pendant les tests)
    def test_before_after_demonstration(self):
        p, o = self.start()
        before = [r.name for r in o["results"]]
        res, p2, o2 = self.send("Je ne veux aucun sport aquatique, je veux uniquement des sports individuels et je préfère l'extérieur.",
                                p, o, contract(exclusions_strictes={"eau": True, "seul_uniquement": True},
                                               preferences_souples={"environnement": "extérieur"}))
        after = [r.name for r in o2["results"]]
        print("\n--- Démonstration avant / après la conversation ---")
        print("Avant :", " | ".join(before))
        print("Après :", " | ".join(after))
        print("Sorties :", [s["sport"] for s in res["comparison"]["sorties"]])
        print("Entrées :", [e["sport"] for e in res["comparison"]["entrees"]])
        self.assertNotEqual(before, after)


class EngineWithoutBonuses(unittest.TestCase):
    def test_engine_runs_without_any_bonus_module(self):
        code = ("import sys; sys.path.insert(0, r'%s'); import engine; "
                "print('bonus' in ' '.join(sys.modules))") % str(ROOT)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(out.stdout.strip(), "False", out.stderr[-300:])
        self.assertTrue(fixture()["out"]["results"])


if __name__ == "__main__":
    unittest.main()
