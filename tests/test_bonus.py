"""Tests des bonus IA (A : explications, B : assistant conversationnel).

Lancement :   python -m unittest discover -s tests -v

Aucun appel réel à Gemini : le LLM est SIMULÉ par un faux transport (fonction qui renvoie une réponse
préparée). Les tests ne consomment donc aucun quota. Le moteur utilisé ici est le TF-IDF (rapide, sans téléchargement) ;
la logique de filtres et de scores est identique avec SBERT.
"""
import copy
import re
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
        self.assertEqual(res["source"], "moteur")
        self.assertIn("chiffre non fourni", res["reason"])
        self.assertEqual(res["lines"], bl.classic_lines(self.out, self.name))
        self.assertTrue(res["text"].endswith("."))

    def test_8_llm_cannot_switch_to_another_sport(self):
        other = self.out["results"][1].name
        text = f"Pour vous, {other} serait encore mieux que {self.name}."
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "moteur")
        self.assertIn("autre sport", res["reason"])

    def test_no_medical_diagnosis(self):
        text = f"Votre douleur révèle un diagnostic de tendinite ; {self.name} est adapté."
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "moteur")
        self.assertIn("formulation médicale", res["reason"])

    def test_2_missing_key_falls_back_without_calling_api(self):
        counter = CountingTransport(gemini_ok("texte"))
        res = bl.explain_sport(self.out, self.name, settings_gemini(key=""), transport=counter)
        self.assertEqual(res["source"], "moteur")
        self.assertIn("aucune clé", res["reason"])
        self.assertEqual(counter.calls, 0)

    def test_2_quota_exhausted_falls_back(self):
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_status(429))
        self.assertEqual((res["source"], "quota" in res["reason"]), ("moteur", True))

    def test_2_provider_error_falls_back(self):
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_status(500))
        self.assertEqual(res["source"], "moteur")
        self.assertIn("erreur", res["reason"])

    def test_2_timeout_falls_back(self):
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=raising(requests.Timeout()))
        self.assertIn("délai", res["reason"])

    def test_2_disabled_provider_uses_classic(self):
        res = bl.explain_sport(self.out, self.name, settings_off())
        self.assertEqual((res["source"], res["reason"]), ("moteur", "l'IA est désactivée"))

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




# =================================================================== explications complètes et cohérentes
class TestCompleteExplanations(Base):
    """Une explication affichée doit toujours être complète, cohérente avec les scores, et ne jamais être tronquée."""

    def setUp(self):
        self.name = self.out["results"][0].name
        self.payload = bl.build_payload(self.out, self.name)

    def test_truncated_response_is_never_shown(self):
        """finishReason = MAX_TOKENS : le texte coupé n'est jamais affiché, le repli complet est utilisé."""
        def truncated(url, headers, body, timeout):
            return 200, {"candidates": [{"content": {"parts": [{"text": f"{self.name} obtient un score global de"}]},
                                         "finishReason": "MAX_TOKENS"}]}
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=truncated)
        self.assertEqual(res["source"], "moteur")
        self.assertIn("coupée", res["reason"])
        self.assertTrue(res["text"].endswith("."))
        self.assertNotIn("obtient un score global de", res["text"])  # le texte coupé n'est pas affiché

    def test_sentence_without_final_punctuation_is_rejected(self):
        text = (f"{self.name} obtient un score global de {self.payload['score_global_pct']} %, ce qui correspond "
                "à votre envie de nager et à votre fréquence de séances. Le sport répond aussi à vos objectifs et "
                "aux contraintes déclarées, sans toucher à vos articulations sensibles et en restant")
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "moteur")
        self.assertIn("incomplète", res["reason"])

    def test_global_score_must_match_the_engine(self):
        """Un texte complet mais sans le score exact du moteur est rejeté."""
        wrong = (self.payload["score_global_pct"] + 7) % 100
        text = (f"{self.name} est un sport adapté à votre profil. Il obtient un score global de {wrong} %, "
                "ce qui reflète votre envie de nager. Les points d'attention concernent surtout la fréquence "
                "de vos séances, et il reste une bonne piste à essayer en progressant.")
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "moteur")
        self.assertTrue("score global" in res["reason"] or "chiffre non fourni" in res["reason"], res["reason"])

    def test_template_is_complete_and_uses_only_engine_numbers(self):
        text = bl.template_explanation(self.payload)
        self.assertTrue(text.endswith("."))
        self.assertIn(self.name, text)
        self.assertIn(f"{self.payload['score_global_pct']} %", text)
        allowed = set(self.payload["nombres_autorises"]) | {1.0, 2.0, 3.0, 4.0, 5.0}
        for tok in re.findall(r"\d+(?:[.,]\d+)?", text):
            self.assertIn(float(tok.replace(",", ".")), allowed, f"nombre inventé : {tok}")
        for label, value in self.payload["scores_par_bloc_pct"].items():
            if value is not None:
                self.assertIn(f"{value} %", text, f"score du bloc {label} absent de l'explication")

    def test_template_sentences_all_end_properly(self):
        text = bl.template_explanation(self.payload)
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
        self.assertGreaterEqual(len(sentences), 3)
        for sentence in sentences:
            self.assertIn(sentence.strip()[-1], ".!?", sentence)

    def test_prompt_requires_the_expected_structure(self):
        for word in ["adéquation", "objectifs", "préférences", "limites", "conclusion"]:
            self.assertIn(word, bl.SYSTEM_PROMPT.lower())
        self.assertIn("EXACTEMENT", bl.SYSTEM_PROMPT)

    def test_cache_depends_on_prompt_version(self):
        """Une nouvelle version des consignes invalide les anciennes explications en cache."""
        before = bl.cache_key(self.out, self.name, settings_gemini())
        old = bl.PROMPT_VERSION
        try:
            bl.PROMPT_VERSION = "autre"
            after = bl.cache_key(self.out, self.name, settings_gemini())
        finally:
            bl.PROMPT_VERSION = old
        self.assertNotEqual(before, after)

    def test_full_valid_llm_text_is_kept_intact(self):
        text = (f"{self.name} est recommandé au rang 1 avec un score global de {self.payload['score_global_pct']} %. "
                "Il correspond à votre envie de nager et à votre fréquence de séances.")
        res = bl.explain_sport(self.out, self.name, settings_gemini(), transport=gemini_ok(text))
        self.assertEqual(res["source"], "llm")
        self.assertEqual(res["text"], text)


if __name__ == "__main__":
    unittest.main()
