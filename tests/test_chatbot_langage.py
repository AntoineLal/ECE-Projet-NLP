"""Compréhension de phrases libres : le chatbot doit comprendre le besoin, pas reconnaître des phrases préfaites.

- Partie 1 : règles seules (aucune IA). Chaque phrase doit produire la bonne exclusion, ou une question claire.
- Partie 2 : formulations détournées. Gemini est SIMULÉ : le transport renvoie le contrat qu'il produirait.

Lancement :  python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bonus_chatbot as bc  # noqa: E402
from engine import UserProfile  # noqa: E402
from llm_config import LLMSettings  # noqa: E402
from test_bonus import fixture  # noqa: E402

OFF = LLMSettings(provider="off")


def fake_gemini(contract: dict):
    def t(url, headers, body, timeout):
        return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps(contract, ensure_ascii=False)}]},
                                     "finishReason": "STOP"}]}
    return t


class FreeLanguageRules(unittest.TestCase):
    """Phrases libres comprises par les règles, sans aucune IA."""

    @classmethod
    def setUpClass(cls):
        cls.engine = fixture()["rec"]
        cls.df = cls.engine.df
        cls.base = UserProfile(text="Je m'entraîne en salle.", experience="régulier")
        cls.out = cls.engine.recommend(cls.base, None)

    def run_msg(self, msg):
        return bc.process_message(msg, self.base, self.out, self.engine, None, OFF)

    def admissible(self, res):
        out = res.get("out", self.out)
        return {a["name"] for a in out["anatomy"] if a["retained"]}

    def test_no_sea_nearby_removes_sea_sports(self):
        res = self.run_msg("Je n'ai pas de mer à côté de chez moi, donc éviter le surf, le longe-côte, etc.")
        self.assertEqual(res["kind"], "maj")
        self.assertIn("mer", res["profile"].excluded_infrastructures)
        kept = self.admissible(res)
        self.assertNotIn("Surf", kept)
        self.assertNotIn("Longe-côte", kept)
        self.assertNotIn("Beach-volley", kept)
        self.assertIn("Football", kept)  # un sport sans besoin de mer reste proposé

    def test_no_stadium_nearby_removes_field_sports(self):
        res = self.run_msg("Je n'ai pas de stade à côté de chez moi.")
        kept = self.admissible(res)
        for name in ["Football", "Rugby à XV", "Sprint", "Lancer du poids"]:
            self.assertNotIn(name, kept, name)
        self.assertIn("Tennis", kept)  # le tennis se pratique sur un court, pas dans un stade

    def test_no_stadium_with_informal_wording(self):
        res = self.run_msg("pas de terrain de foot ni de piste d'athlé dans le coin")
        self.assertIn("stade", res["profile"].excluded_infrastructures)

    def test_dislike_ball_sports_excludes_the_group(self):
        res = self.run_msg("Je n'aime pas les sports de balle.")
        self.assertIn("sports_de_balle", res["profile"].excluded_groups)
        kept = self.admissible(res)
        self.assertNotIn("Football", kept)
        self.assertNotIn("Tennis", kept)

    def test_liking_combat_sports_favours_without_excluding(self):
        res = self.run_msg("J'adore les sports de combat.")
        self.assertIn("sports_de_combat", res["profile"].liked_groups)
        self.assertNotIn("sports_de_combat", res["profile"].excluded_groups)
        before = {a["name"]: a["total"] for a in self.out["anatomy"]}
        after = {a["name"]: a["total"] for a in res["out"]["anatomy"]}
        self.assertGreater(after["Judo"], before["Judo"])

    def test_no_pool_removes_swimming(self):
        res = self.run_msg("Je n'ai pas de piscine près de chez moi.")
        self.assertIn("piscine", res["profile"].excluded_infrastructures)
        self.assertNotIn("Natation en bassin", self.admissible(res))

    def test_cannot_swim_removes_water_sports(self):
        res = self.run_msg("Je ne sais pas nager.")
        self.assertTrue(res["profile"].no_water)
        self.assertFalse([n for n in self.admissible(res) if {"sport aquatique", "nage"} & set(self.df.loc[self.df.nom_sport == n, "categories_l"].iloc[0])])

    def test_no_lake_or_river(self):
        res = self.run_msg("Pas de lac ni de rivière dans ma région.")
        self.assertIn("plan_eau", res["profile"].excluded_infrastructures)
        self.assertNotIn("Kayak en eau calme", self.admissible(res))

    def test_no_mountain_removes_snow_sports(self):
        res = self.run_msg("Je n'ai pas de montagne à proximité.")
        self.assertIn("neige", res["profile"].excluded_infrastructures)
        self.assertNotIn("Ski alpin", self.admissible(res))

    def test_only_individual_indoor(self):
        res = self.run_msg("Je veux seulement des sports individuels en intérieur.")
        self.assertTrue(res["profile"].only_solo)
        self.assertEqual(res["profile"].env_preference, "intérieur")

    def test_no_contact(self):
        res = self.run_msg("Pas de sport de contact physique pour moi.")
        self.assertTrue(res["profile"].no_contact)

    def test_avoid_shocks(self):
        res = self.run_msg("Je veux éviter les sports avec beaucoup de chocs.")
        self.assertEqual(res["profile"].impact_tolerance, "faible")

    def test_each_understood_element_is_explained(self):
        res = self.run_msg("Pas de stade près de chez moi et je n'aime pas les sports de balle.")
        texts = " ".join(i["comprend"] for i in res["interpretations"])
        self.assertIn("stade", texts.lower())
        self.assertIn("sports de balle", texts.lower())


class FreeLanguageWithSimulatedGemini(unittest.TestCase):
    """Formulations détournées : seule une lecture de type LLM peut les rattacher au bon lieu ou groupe."""

    @classmethod
    def setUpClass(cls):
        cls.engine = fixture()["rec"]
        cls.df = cls.engine.df
        cls.base = UserProfile(text="Je m'entraîne en salle.", experience="régulier")
        cls.out = cls.engine.recommend(cls.base, None)
        cls.settings = LLMSettings(provider="gemini", gemini_key="FAKE-TEST-KEY-000000")

    def send_with(self, msg, contract):
        res = bc.process_message(msg, self.base, self.out, self.engine, None, self.settings,
                                 transport=fake_gemini(contract))
        return res

    def test_indirect_sea_reference(self):
        contract = {"intention": "mise_a_jour",
                    "indisponibilites": {"lieux": ["mer"]},
                    "interpretations": [{"extrait": "je vis loin de tout point d'eau salé",
                                         "comprend": "Pas d'accès à la mer : j'exclus les sports qui en ont besoin."}]}
        res = self.send_with("Je vis loin de tout point d'eau salée, impossible de faire du surf.", contract)
        self.assertEqual(res["kind"], "maj")
        self.assertIn("mer", res["profile"].excluded_infrastructures)
        self.assertEqual(res["source"], "IA + règles")
        self.assertNotIn("Surf", {a["name"] for a in res["out"]["anatomy"] if a["retained"]})

    def test_indirect_group_reference(self):
        contract = {"intention": "mise_a_jour",
                    "exclusions_strictes": {"groupes": ["sports_de_raquette"]},
                    "interpretations": [{"extrait": "les trucs avec une raquette me gonflent",
                                         "comprend": "Vous ne voulez pas de sports de raquette : j'exclus ces sports."}]}
        res = self.send_with("Les trucs avec une raquette, très peu pour moi.", contract)
        self.assertIn("sports_de_raquette", res["profile"].excluded_groups)
        self.assertNotIn("Tennis", {a["name"] for a in res["out"]["anatomy"] if a["retained"]})

    def test_llm_cannot_invent_a_place(self):
        contract = {"intention": "mise_a_jour", "indisponibilites": {"lieux": ["heliport"]}}
        res = self.send_with("Pas d'héliport près de chez moi.", contract)
        self.assertTrue(any("heliport" in r for r in res["rejected"]))
        self.assertEqual(res["kind"], "precision", "le chatbot doit demander une précision, pas deviner")
        self.assertIn("héliport", " ".join(res["precisions"]))

    def test_ambiguous_message_asks_a_question_instead_of_guessing(self):
        res = bc.process_message("Il y a trop de bruit chez moi.", self.base, self.out, self.engine, None, OFF)
        self.assertIn(res["kind"], ("precision", "aide"))
        self.assertNotEqual(res["kind"], "maj")


class ConversationalReply(unittest.TestCase):
    """Le modèle répond en langage naturel, avec l'historique de la conversation."""

    @classmethod
    def setUpClass(cls):
        cls.engine = fixture()["rec"]
        cls.base = UserProfile(text="Je m'entraîne en salle.", experience="régulier")
        cls.out = cls.engine.recommend(cls.base, None)
        cls.settings = LLMSettings(provider="gemini", gemini_key="FAKE-TEST-KEY-000000")

    def test_model_reply_is_returned_to_the_user(self):
        contract = {"reponse": "Bien noté, je mets de côté les activités nautiques.",
                    "intention": "mise_a_jour",
                    "exclusions_strictes": {"groupes": ["sports_aquatiques"]},
                    "interpretations": [{"extrait": "je n'aime pas l'eau", "comprend": "Pas d'activités dans l'eau."}]}
        res = bc.process_message("je n'aime pas l'eau", self.base, self.out, self.engine, None, self.settings,
                                 transport=fake_gemini(contract))
        self.assertEqual(res["text"], "Bien noté, je mets de côté les activités nautiques.")
        self.assertEqual(res["kind"], "maj")

    def test_history_is_sent_to_the_model(self):
        seen = {}

        def spy(url, headers, body, timeout):
            seen["prompt"] = body["systemInstruction"]["parts"][0]["text"] + body["contents"][0]["parts"][0]["text"]
            return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps({"reponse": "D'accord."})}]},
                                         "finishReason": "STOP"}]}
        history = [{"role": "user", "text": "Je préfère les sports en salle."},
                   {"role": "assistant", "text": "Noté, je favorise les sports en salle."}]
        bc.process_message("Et en extérieur, alors ?", self.base, self.out, self.engine, None, self.settings,
                           transport=spy, history=history)
        self.assertIn("Je préfère les sports en salle.", seen["prompt"])
        self.assertIn("Et en extérieur, alors ?", seen["prompt"])

    def test_longe_cote_is_not_a_sea_requirement(self):
        res = bc.process_message("je n'aime pas le longe cote", self.base, self.out, self.engine, None,
                                 LLMSettings(provider="off"))
        self.assertEqual(res["profile"].excluded_infrastructures, [])
        self.assertIn("Longe-côte", [a["name"] for a in res["out"]["anatomy"] if not a["retained"]])

    def test_dislike_water_excludes_aquatic_group(self):
        res = bc.process_message("je n'aime pas l'eau", self.base, self.out, self.engine, None,
                                 LLMSettings(provider="off"))
        self.assertIn("sports_aquatiques", res["profile"].excluded_groups)
        self.assertNotIn("Natation en bassin", [a["name"] for a in res["out"]["anatomy"] if a["retained"]])


if __name__ == "__main__":
    unittest.main()
