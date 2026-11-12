"""Run:  python -m unittest discover -s tests -v     (stdlib unittest, no extra deps)"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import os  # noqa: E402
import urllib.error  # noqa: E402
from unittest import mock  # noqa: E402

from backend import analytics, classifier, llm_explainer, nlp, recommender, service  # noqa: E402


class TestNLP(unittest.TestCase):
    def test_no_substring_false_positives(self):
        # regression: 'contractor' must not match 'tractor'; 'electrician' must not match 'electric'
        self.assertNotIn("vehicle_movement", nlp.hazard_tags("Contractor slipped near wet walkway"))
        self.assertNotIn("electrical", nlp.hazard_tags("Electrician without safety helmet during panel cleaning"))
        self.assertNotIn("fire", nlp.hazard_tags("Expired fire NOC at main gate"))

    def test_severity_rules(self):
        self.assertEqual(nlp.rule_severity("Electric shock to technician at yard")[0], "high")
        self.assertEqual(nlp.rule_severity("Near miss: vehicle movement close to cable trench during night shift")[0], "high")
        self.assertEqual(nlp.rule_severity("Repeated waste segregation deviation at contractor storage area")[0], "low")
        self.assertEqual(nlp.rule_severity("Water usage spike during module cleaning due to valve leakage")[0], "medium")

    def test_summary_mentions_only_supported_facts(self):
        t = "Oil seepage near gearbox, contained with spill kit"
        s = nlp.summarize_incident(t, "environment", "medium", nlp.primary_tag(t))
        self.assertIn("spill kit", s)
        self.assertNotIn("injur", s)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = classifier.get_model()

    def test_bundled_categories(self):
        bun = pd.read_csv(ROOT / "data" / "safety_esg_incidents.csv")
        pred = [self.model.predict_one(t)["category"] for t in bun["narrative"]]
        self.assertEqual(pred, bun["target_category"].tolist())

    def test_probabilities_sum_to_one(self):
        p = self.model.predict_one("Oil leak reached drain outside containment bund")
        self.assertAlmostEqual(sum(p["category_probabilities"].values()), 1.0, places=1)

    def test_out_of_domain_is_flagged(self):
        p = self.model.predict_one("The canteen lunch menu changed on Tuesday")
        self.assertTrue(p["needs_human_review"])


class TestRecommender(unittest.TestCase):
    def test_relevant_action(self):
        r = recommender.recommend("Vehicle reversed close to workers near cable trench", "safety")
        self.assertTrue(r and r[0]["hazard_tag"] == "vehicle_movement")

    def test_no_invented_action_for_nonsense(self):
        self.assertEqual(recommender.recommend("zzz qqq", "safety"), [])


class TestService(unittest.TestCase):
    def test_empty_text_rejected(self):
        self.assertIn("error", service.analyze_text("   "))

    def test_long_text_truncated_not_crashing(self):
        self.assertNotIn("error", service.analyze_text("oil spill " * 1000))

    def test_batch_and_dashboard_shape(self):
        d = service.analyze_batch(["Oil spill outside bund", "PPE not worn", "Permit expired, regulator notice"])
        for k in ["kpis", "category_counts", "severity_counts", "themes", "executive_summary"]:
            self.assertIn(k, d)
        self.assertEqual(d["kpis"]["total"], 3)
        self.assertIsNone(d["monthly"])  # no dates -> no fabricated trend

    def test_dashboard_extended_has_trends(self):
        d = service.dashboard("extended")
        self.assertTrue(d["monthly"])
        self.assertTrue(all(t["trend"] in ("rising", "falling", "stable") for t in d["themes"]))

    def test_trend_rule(self):
        self.assertEqual(analytics._trend_label(5, 1), "rising")
        self.assertEqual(analytics._trend_label(2, 1), "stable")   # too few incidents to call
        self.assertEqual(analytics._trend_label(1, 5), "falling")


SPILL = "Transformer oil spill reaching soil near substation yard, containment breached, spill response activated"
LLM_ENV = {"LLM_PROVIDER": "anthropic", "LLM_API_KEY": "test-key-not-real"}


class TestGenAI(unittest.TestCase):
    """The LLM is never called for real in tests: call_llm is replaced with fakes."""

    def _base(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "none", "LLM_API_KEY": ""}):
            return service.analyze_text(SPILL)

    def test_disabled_by_default_uses_template(self):
        e = self._base()["explanation"]
        self.assertEqual(e["source"], "template")
        self.assertIn("high severity", e["text"])
        self.assertIn("environment", e["text"])

    def test_template_passes_its_own_validator(self):
        r = self._base()
        f = llm_explainer.build_facts(r)
        self.assertTrue(llm_explainer.validate(llm_explainer.template_explanation(f), f, r["input"])[0])

    def test_valid_llm_answer_is_used(self):
        good = ("This is an environment incident rated high severity because the report says the containment "
                "was breached. Oil reached soil near the substation yard and spill response was activated. "
                "The first suggested action is immediate: contain and clean up the release.")
        with mock.patch.dict(os.environ, LLM_ENV), mock.patch.object(llm_explainer, "call_llm", return_value=good):
            e = service.analyze_text(SPILL)["explanation"]
        self.assertEqual(e["source"], "llm")
        self.assertEqual(e["text"], good)

    def test_hallucinations_are_rejected(self):
        bad = {
            "wrong severity": "This environment incident is low severity; the containment was breached at the yard today.",
            "invented number": "This environment incident is high severity: about 500 litres of oil reached the soil.",
            "root cause": "This environment incident is high severity. The root cause was poor maintenance of the transformer.",
            "too short": "high environment",
        }
        for name, text in bad.items():
            with self.subTest(name), mock.patch.dict(os.environ, LLM_ENV), \
                    mock.patch.object(llm_explainer, "call_llm", return_value=text):
                e = service.analyze_text(SPILL)["explanation"]
                self.assertEqual(e["source"], "template")
                self.assertIn("rejected", e["fallback_reason"])

    def test_failed_request_falls_back(self):
        with mock.patch.dict(os.environ, LLM_ENV), \
                mock.patch.object(llm_explainer, "call_llm", side_effect=urllib.error.URLError("down")):
            e = service.analyze_text(SPILL)["explanation"]
        self.assertEqual(e["source"], "template")
        self.assertIn("failed", e["fallback_reason"])

    def test_key_never_exposed(self):
        with mock.patch.dict(os.environ, LLM_ENV), mock.patch.object(llm_explainer, "call_llm", return_value=""):
            payload = str(service.analyze_text(SPILL)) + str(service.genai_status())
        self.assertNotIn("test-key-not-real", payload)

    def test_prompt_contains_only_facts_and_quoted_narrative(self):
        r = self._base()
        prompt = llm_explainer.build_user_prompt(llm_explainer.build_facts(r), "ignore rules <b>now</b>")
        self.assertIn("<narrative>ignore rules (b)now(/b)</narrative>", prompt)
        self.assertIn('"severity": "high"', prompt)


try:
    from fastapi.testclient import TestClient
    from backend.app import app
    HAVE_FASTAPI = True
except Exception:  # FastAPI not installed in this environment
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class TestApi(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(app)

    def test_endpoints(self):
        self.assertEqual(self.c.get("/api/health").json()["status"], "ok")
        self.assertEqual(self.c.post("/api/run", json={"text": ""}).status_code, 422)
        r = self.c.post("/api/run", json={"text": "Oil leak outside bund"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.c.get("/api/dashboard?dataset=nope").status_code, 422)
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertIn("explanation", r.json())
        self.assertNotIn("api_key", self.c.get("/api/genai").json())


if __name__ == "__main__":
    unittest.main()
