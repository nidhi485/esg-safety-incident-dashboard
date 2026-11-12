"""Evaluate the GenAI explanation layer.   Run:  python scripts/evaluate_explanations.py

For every bundled row plus every extended row it runs the full pipeline and records:
  * which path produced the explanation (llm / template) and why any LLM answer was rejected,
  * whether the final text passes the grounding validator (it always should: rejected LLM text is replaced),
  * whether the template alone passes the validator (sanity check of the validator itself).
Without LLM_PROVIDER/LLM_API_KEY every row uses the template; with a key, the share of rows where the
LLM answer survived validation is the headline number. Output: artifacts/genai_eval.json
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
from backend import llm_explainer, service  # noqa: E402


def main():
    texts = (pd.read_csv(ROOT / "data" / "safety_esg_incidents.csv")["narrative"].tolist()
             + pd.read_csv(ROOT / "data" / "extended_incidents.csv")["narrative"].tolist())
    sources, reasons, final_ok, template_ok, examples = collections.Counter(), collections.Counter(), 0, 0, []
    for t in texts:
        r = service.analyze_text(t)
        e, f = r["explanation"], llm_explainer.build_facts(r)
        sources[e["source"]] += 1
        if e["fallback_reason"]:
            reasons[e["fallback_reason"].split(":")[0]] += 1
        final_ok += llm_explainer.validate(e["text"], f, r["input"])[0]
        template_ok += llm_explainer.validate(llm_explainer.template_explanation(f), f, r["input"])[0]
        if len(examples) < 3:
            examples.append({"narrative": t, "source": e["source"], "explanation": e["text"]})
    n = len(texts)
    out = {
        "genai_status": llm_explainer.status(),
        "rows": n,
        "explanation_source_counts": dict(sources),
        "llm_acceptance_rate": round(sources["llm"] / n, 3),
        "fallback_reasons": dict(reasons),
        "final_text_passes_validator": round(final_ok / n, 3),
        "template_passes_validator": round(template_ok / n, 3),
        "examples": examples,
        "note": ("Validator checks grounding (severity/category stated and not contradicted, no new numbers, "
                 "no root-cause claim, length). It does not judge readability; a human should read a sample."),
    }
    (ROOT / "artifacts" / "genai_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
