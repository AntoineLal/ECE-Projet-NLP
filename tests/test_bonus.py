"""Tests des bonus IA (A : explications, B : assistant conversationnel).

Lancement :   python -m unittest discover -s tests -v

Aucun appel réel à Gemini ou Ollama : le LLM est SIMULÉ par un faux transport (fonction qui renvoie une réponse
préparée). Les tests ne consomment donc aucun quota. Le moteur utilisé ici est le TF-IDF (rapide, sans téléchargement) ;
la logique de filtres et de scores est identique avec SBERT.
"""
import copy
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

import bonus_chatbot as bc  # noqa: E402
import bonus_llm as bl  # noqa: E402
from engine import Recommender, UserProfile  # noqa: E402
from llm_config import LLMSettings  # noqa: E402

FAKE_KEY = "FAKE-TEST-KEY-000000"  # clé factice : vérifie qu'elle ne fuit jamais


def gemini_ok(text: str):
    """Transport simulé : réponse Gemini réussie contenant `text`."""
    def t(url, headers, body, timeout):
        return 200, {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return t


def gemini_status(code: int):
    def t(url, headers, body, timeout):
        return code, {"error": {"message": "erreur simulée"}}
    return t


def raising(exc):
    def t(url, headers, body, timeout):
        raise exc
    return t


class CountingTransport:
    """Enveloppe un transport pour compter les appels (vérifie la limite de quota)."""

    def __init__(self, inner):
        self.inner, self.calls, self.urls, self.headers = inner, 0, [], []

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        self.urls.append(url)
        self.headers.append(headers)
        return self.inner(url, headers, body, timeout)


def settings_gemini(key: str = FAKE_KEY, **kw) -> LLMSettings:
    return LLMSettings(provider="gemini", gemini_key=key, **kw)


def settings_off() -> LLMSettings:
    return LLMSettings(provider="off")


_CACHE = {}


def fixture() -> dict:
    """Moteur TF-IDF, profil de référence et classement calculés une seule fois pour toute la suite."""
    if not _CACHE:
        rec = Recommender(None)
        profile = UserProfile(
            text="Je veux nager dans l'eau pour travailler tout le corps. Je m'entraîne 4 fois par semaine.",
            sessions_per_week=4, experience="occasionnel")
        _CACHE.update(rec=rec, profile=profile, out=rec.recommend(profile, None))
    return _CACHE


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = Recommender(None)  # moteur TF-IDF : rapide, même règles de filtrage et de score
        cls.profile = UserProfile(
            text="Je veux nager dans l'eau pour travailler tout le corps. Je m'entraîne 4 fois par semaine.",
            sessions_per_week=4, experience="occasionnel")
        cls.out = cls.rec.recommend(cls.profile, None)


# =================================================================== configuration
class TestConfiguration(unittest.TestCase):
    def test_key_is_hidden_from_repr(self):
        s = settings_gemini()
        self.assertNotIn(FAKE_KEY, repr(s))
        self.assertNotIn(FAKE_KEY, s.describe())

    def test_missing_key_is_detected(self):
        self.assertFalse(LLMSettings(provider="gemini", gemini_key="").has_gemini_key)
        self.assertIn("clé API absente", LLMSettings(provider="gemini", gemini_key="").describe())

    def test_payload_never_contains_the_key(self):
        out = fixture()["out"]
        payload = bl.build_payload(out, out["results"][0].name)
        self.assertNotIn(FAKE_KEY, str(payload))


# =================================================================== bonus A : explications
class TestExplanations(Base):
    def setUp(self):
        self.name = self.out["results"][0].name
        self.payload = bl.build_payload(self.out, self.name)

    def test_1_payload_uses_real_engine_scores(self):
        """Les chiffres transmis sont ceux calculés par le moteur, pas des valeurs inventées."""
        a = next(x for x in self.out["anatomy"] if x["name"] == self.name)
        self.assertEqual(self.payload["score_global_pct"], round(a["total"] * 100))
        for b in bl.BLOCKS:
            self.assertEqual(self.payload["scores_par_bloc_pct"][bl.BLOCK_LABELS[b]],
                             None if a["blocks"][b] is None else round(a["blocks"][b] * 100))
        self.assertEqual(self.payload["sport"]["nom"], self.name)

    def test_1_llm_text_with_real_numbers_is_accepted(self):
        text = (f"{self.name} obtient un score global de {self.payload['score_global_pct']} %, "
                "ce qui correspond à votre envie de nager et à votre fréquence de séances.")
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "llm")
        self.assertIn("score global", res["text"])

    def test_8_invented_score_is_rejected_and_classic_shown(self):
        """Un chiffre absent du moteur fait rejeter la réponse : l'explication classique est affichée."""
        text = f"{self.name} atteint un score de 99 %, un excellent choix."
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "classique")
        self.assertIn("chiffre non fourni", res["reason"])
        self.assertEqual(res["lines"], bl.classic_lines(self.out, self.name))

    def test_8_llm_cannot_switch_to_another_sport(self):
        other = self.out["results"][1].name
        text = f"Pour vous, {other} serait encore mieux que {self.name}."
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "classique")
        self.assertIn("autre sport", res["reason"])

    def test_no_medical_diagnosis(self):
        text = f"Votre douleur révèle un diagnostic de tendinite ; {self.name} est adapté."
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "classique")
        self.assertIn("formulation médicale", res["reason"])

    def test_2_missing_key_falls_back_without_calling_api(self):
        counter = CountingTransport(gemini_ok("texte"))
        res = bl.explain_sport(self.out, self.name, settings_gemini(key=""), transport=counter)
        self.assertEqual(res["source"], "classique")
        self.assertIn("aucune clé", res["reason"])
        self.assertEqual(counter.calls, 0)

    def test_2_quota_exhausted_falls_back(self):
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_status(429))
        self.assertEqual((res["source"], "quota" in res["reason"]), ("classique", True))

    def test_2_provider_error_falls_back(self):
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_status(500))
        self.assertEqual(res["source"], "classique")
        self.assertIn("erreur", res["reason"])

    def test_2_timeout_falls_back(self):
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=raising(requests.Timeout()))
        self.assertIn("délai", res["reason"])

    def test_2_ollama_unreachable_falls_back(self):
        s = LLMSettings(provider="ollama")
        res = bl.explain_sport(self.out, self.name, s, transport=raising(requests.ConnectionError()))
        self.assertIn("injoignable", res["reason"])

    def test_2_disabled_provider_uses_classic(self):
        res = bl.explain_sport(self.out, self.name, settings_off())
        self.assertEqual((res["source"], res["reason"]), ("classique", "l'IA est désactivée"))

    def test_session_budget_limits_calls(self):
        counter = CountingTransport(gemini_ok("Un texte sans chiffre."))
        budget = bl.CallBudget(limit=2)
        for _ in range(4):
            bl.explain_sport(self.out, self.name, settings_gemini(), budget=budget, transport=counter)
        self.assertEqual(counter.calls, 2)

    def test_api_key_sent_in_header_not_in_url(self):
        counter = CountingTransport(gemini_ok("Un texte."))
        bl.explain_sport(self.out, self.name, settings_gemini(), transport=counter)
        self.assertNotIn(FAKE_KEY, counter.urls[0])
        self.assertEqual(counter.headers[0]["x-goog-api-key"], FAKE_KEY)

    def test_8_llm_does_not_change_the_ranking(self):
        """Appeler le LLM (accepté ou rejeté) ne modifie ni le classement ni les scores."""
        before = [(r.name, round(r.total, 6)) for r in self.out["results"]]
        bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok("Texte accepté, rien de plus."))
        bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok("Score 99 %"))
        after = [(r.name, round(r.total, 6)) for r in self.out["results"]]
        self.assertEqual(before, after)


# =================================================================== bonus B : compréhension
class TestUnderstanding(Base):
    SPEC_EXAMPLE = ("Finalement, je n'aime pas les sports aquatiques et je préfère les activités individuelles "
                    "en extérieur.")

    def test_3_positive_and_negative_preferences_by_rules(self):
        """Exemple du cahier des charges : négation (avoid), préférence (format, environnement)."""
        r = bc.interpret_rules(self.SPEC_EXAMPLE, self.profile_base())
        self.assertIn("Eau", r["updates"]["avoid_families"])
        self.assertEqual(r["updates"]["formats"], ["individuel"])
        self.assertEqual(r["updates"]["environnement"], "extérieur")
        self.assertTrue(r["updates"]["no_water"])

    def test_3_llm_json_is_validated(self):
        """Champ hors liste blanche, valeur hors bornes : rejetés. Les champs valides sont gardés."""
        raw = ('{"intention":"mise_a_jour","modifications":{"formats":["individuel"],"taille_cm":180,'
               '"sessions_per_week":12,"budget_seance_max":15},"precisions":[]}')
        res = bc.interpret(self.SPEC_EXAMPLE, self.profile_base(), settings_gemini(),
                           transport=gemini_ok(raw))
        self.assertEqual(res["updates"]["formats"], ["individuel"])
        self.assertEqual(res["updates"]["budget_seance_max"], 15)
        self.assertNotIn("taille_cm", res["updates"])
        self.assertNotIn("sessions_per_week", res["updates"])
        self.assertIn("taille_cm", res["rejected"])

    def test_8_llm_cannot_inject_scores(self):
        raw = '{"intention":"mise_a_jour","modifications":{"score_global":99},"precisions":[]}'
        res = bc.interpret("Je veux du foot", self.profile_base(), settings_gemini(), transport=gemini_ok(raw))
        self.assertNotIn("score_global", res["updates"])
        self.assertIn("score_global", res["rejected"])

    def test_3_invalid_json_falls_back_to_rules(self):
        res = bc.interpret(self.SPEC_EXAMPLE, self.profile_base(), settings_gemini(),
                           transport=gemini_ok("Désolé, je ne peux pas répondre en JSON."))
        self.assertIn("Eau", res["updates"]["avoid_families"])
        self.assertIn("règles", res["note"])

    def test_4_ambiguous_message_asks_a_question(self):
        res = bc.interpret("Je n'aime pas le sport", self.profile_base(), settings_off())
        self.assertEqual(res["updates"], {})
        self.assertTrue(res["questions"])

    def test_4_contradictory_environment_asks_a_question(self):
        res = bc.interpret_rules("Je veux pratiquer en intérieur et en extérieur, pas de préférence.",
                                 self.profile_base())
        self.assertTrue(res["questions"], "une demande contradictoire doit déclencher une question")

    def test_budget_null_lifts_the_limit(self):
        raw = '{"intention":"mise_a_jour","modifications":{"budget_seance_max":null},"precisions":[]}'
        res = bc.interpret("Mon budget n'est plus limité.", self.profile_base(), settings_gemini(),
                           transport=gemini_ok(raw))
        self.assertIn("budget_seance_max", res["updates"])
        new, _ = bc.apply_updates(self.profile_base().__class__(text="x", budget_seance_max=15), res["updates"])
        self.assertIsNone(new.budget_seance_max)

    def profile_base(self):
        return UserProfile(text="Je m'entraîne en salle.", sessions_per_week=4)


# =================================================================== bonus B : mise à jour du profil
class TestProfileUpdate(Base):
    def test_5_previous_information_is_kept(self):
        p = UserProfile(text="Je m'entraîne 4 fois par semaine en intérieur.", sessions_per_week=4,
                        environnement="intérieur", formats=["groupe"])
        new, changes = bc.apply_updates(p, {"budget_seance_max": 12}, "Mon budget est de 12 euros.")
        self.assertEqual(new.sessions_per_week, 4)
        self.assertEqual(new.environnement, "intérieur")
        self.assertEqual(new.formats, ["groupe"])
        self.assertEqual(new.budget_seance_max, 12)
        self.assertIn("Mon budget est de 12 euros.", new.text)
        self.assertIn("Mon budget est de 12 euros.", new.text)
        self.assertTrue(new.text.startswith("Je m'entraîne 4 fois"))
        self.assertTrue(changes)

    def test_5_new_negation_removes_previous_like(self):
        p = UserProfile(text="x", families=["Eau", "Course et athlétisme"])
        new, _ = bc.apply_updates(p, {"avoid_families": ["Eau"]})
        self.assertNotIn("Eau", new.families)
        self.assertIn("Eau", new.avoid_families)
        self.assertTrue(new.no_water)  # une famille aquatique refusée active aussi le refus de l'eau
        self.assertIn("Course et athlétisme", new.families)

    def test_5_intensity_accumulates_and_is_bounded(self):
        p = UserProfile(text="x")
        p, _ = bc.apply_updates(p, {"intensity_delta": -1})
        p, _ = bc.apply_updates(p, {"intensity_delta": -1})
        p, _ = bc.apply_updates(p, {"intensity_delta": -1})
        self.assertEqual(p.intensity_shift, -2)

    def test_5_unchanged_value_is_not_reported(self):
        p = UserProfile(text="x", sessions_per_week=4)
        _, changes = bc.apply_updates(p, {"sessions_per_week": 4})
        self.assertEqual(changes, [])


# =================================================================== recalcul réel
class TestRecalculation(Base):
    def test_6_new_constraint_really_changes_the_ranking(self):
        """Avant : sports aquatiques dans le top. Après « je n'aime pas l'eau » : le moteur réel recalcule."""
        before = self.out
        res = bc.process_message("Finalement, je n'aime pas les sports aquatiques et je préfère les activités "
                                 "individuelles en extérieur.", self.profile, before, self.rec, None, settings_off())
        self.assertEqual(res["kind"], "maj")
        cmp = res["comparison"]
        print("\n--- Comparaison avant / après la conversation ---")
        print("Avant :", " | ".join(cmp["avant"]))
        print("Après :", " | ".join(cmp["apres"]))
        print("Entrées :", cmp["entrees"], "| Sorties :", cmp["sorties"])
        self.assertFalse(cmp["identique"])
        self.assertTrue(cmp["sorties"])
        aquatic = {"sport aquatique", "nage", "sport nautique", "pagaie", "rame", "navigation à voile", "glisse sur l'eau"}
        for r in res["out"]["results"]:
            cats = set(r.row.categories.split("|"))
            self.assertFalse(cats & aquatic, f"{r.name} est aquatique")
        top3 = [r.row.format_pratique.split("|") for r in res["out"]["results"][:3]]
        self.assertTrue(any("individuel" in f for f in top3), "aucun sport individuel dans le top 3")

    def test_6_intensity_change_lowers_the_target(self):
        res = bc.process_message("Je préfère un sport moins intense.", self.profile, self.out, self.rec, None,
                                 settings_off())
        self.assertEqual(res["kind"], "maj")
        self.assertLess(res["out"]["resolved"].target_intensity, self.out["resolved"].target_intensity)

    def test_7_engine_works_without_bonus_modules(self):
        """Le moteur n'importe aucun module bonus : il fonctionne seul."""
        code = ("import sys; sys.path.insert(0, r'%s'); import engine; "
                "print('bonus' in ' '.join(sys.modules))") % str(ROOT)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(out.stdout.strip(), "False", out.stderr[-300:])
        self.assertTrue(fixture()["out"]["results"])


# =================================================================== questions sur les sports
class TestQuestions(Base):
    def test_6_why_question_is_answered_from_computed_data(self):
        res = bc.process_message("Pourquoi me recommandes-tu la natation ?", self.profile, self.out, self.rec,
                                 None, settings_off())
        self.assertEqual(res["kind"], "reponse")
        self.assertIn("Natation", res["text"])
        self.assertIn("score", res["text"])

    def test_6_comparison_question_returns_table(self):
        res = bc.process_message("Quelle est la différence entre le tennis et le badminton pour mon profil ?",
                                 self.profile, self.out, self.rec, None, settings_off())
        self.assertEqual(res["kind"], "reponse")
        self.assertIn("| Score global |", res["text"])
        self.assertIn("Tennis", res["text"])

    def test_6_criteria_question(self):
        res = bc.process_message("Quels critères ont influencé mon score ?", self.profile, self.out, self.rec,
                                 None, settings_off())
        self.assertIn("Condition physique", res["text"])


# =================================================================== mémoire de conversation (structure)
class TestConversationTurns(Base):
    def test_full_turn_returns_changes_and_source(self):
        res = bc.process_message("Je veux un budget de 15 euros par séance.", self.profile, self.out, self.rec,
                                 None, settings_gemini(), transport=gemini_ok(
                                     '{"intention":"mise_a_jour","modifications":{"budget_seance_max":15}}'))
        self.assertEqual(res["kind"], "maj")
        self.assertEqual(res["source"], "IA + règles")
        self.assertTrue(any("Budget" in c for c in res["changes"]))



# =================================================================== transport HTTP réel (serveur local simulé)
class TestRealHTTPTransport(Base):
    """Passe par requests (vrai transport) vers un petit serveur local qui imite l'API Gemini."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import http.server
        import json as _json
        import threading

        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = _json.loads(self.rfile.read(length) or b"{}")
                seen["headers"], seen["body"], seen["path"] = dict(self.headers), body, self.path
                if "quota" in self.path:
                    code, data = 429, {"error": {"message": "quota"}}
                else:
                    code, data = 200, {"candidates": [{"content": {"parts": [{"text": "Réponse simulée."}]}}]}
                payload = _json.dumps(data).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        cls.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        cls.seen = seen
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}/v1beta/models/{{model}}:generateContent"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_real_request_format_and_success(self):
        old = bl.GEMINI_URL
        bl.GEMINI_URL = self.base
        try:
            text = bl.call_llm("sys", "user", settings_gemini(gemini_model="gemini-test"))
        finally:
            bl.GEMINI_URL = old
        self.assertEqual(text, "Réponse simulée.")
        self.assertIn("gemini-test:generateContent", self.seen["path"])
        self.assertEqual(self.seen["headers"].get("x-goog-api-key"), FAKE_KEY)
        self.assertNotIn(FAKE_KEY, self.seen["path"])  # la clé n'est jamais dans l'URL
        self.assertIn("systemInstruction", self.seen["body"])

    def test_real_quota_error_is_mapped(self):
        old = bl.GEMINI_URL
        bl.GEMINI_URL = self.base.replace("generateContent", "quota:generateContent")
        try:
            with self.assertRaises(bl.LLMError) as cm:
                bl.call_llm("sys", "user", settings_gemini(gemini_model="quota-model"))
        finally:
            bl.GEMINI_URL = old
        self.assertEqual(cm.exception.kind, "quota")


if __name__ == "__main__":
    unittest.main()
