import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from ..answer import answer_question
from ..config import settings

HERE = Path(__file__).parent
GOLDEN = HERE / "golden.json"
RESULTS_DIR = HERE / "results"


def contains_all(text: str, expected: list[str]) -> bool:
    text = text.lower()
    return all(
        any(alt.strip().lower() in text for alt in item.split("|")) for item in expected
    )


def diagnose(row: dict) -> str:
    """Empty string means the case passed. Otherwise, name the failure."""
    if not row["answerable"]:
        return "" if row["correct_refusal"] else "HALLUCINATION"
    if not row["retrieval_hit"]:
        return "RETRIEVAL"
    if not row["answered"]:
        return "FALSE_REFUSAL"
    if not row["correct"]:
        return "WRONG_ANSWER"
    if not row["cited_ok"]:
        return "BAD_CITATION"
    return ""


def evaluate_case(case: dict) -> dict:
    start = time.time()
    result = answer_question(case["question"])
    latency = round(time.time() - start, 1)

    retrieved_docs = {c["document"] for c in result["retrieved"]}
    cited_docs = {c["document"] for c in result["citations"]}
    drafted = result["status"] == "drafted"

    row = {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "status": result["status"],
        "answer": result["answer"],
        "latency_s": latency,
    }

    if case["answerable"]:
        expected_docs = set(case["expected_docs"])
        row["retrieval_hit"] = expected_docs <= retrieved_docs
        row["answered"] = drafted
        row["correct"] = drafted and contains_all(result["answer"], case["expect"])
        row["cited_ok"] = drafted and expected_docs <= cited_docs
    else:
        row["correct_refusal"] = not drafted

    row["diagnosis"] = diagnose(row)
    return row


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if r["answered"]]
    return {
        "cases": len(rows),
        "answer_accuracy": rate(sum(r["correct"] for r in answerable), len(answerable)),
        "retrieval_hit_rate": rate(sum(r["retrieval_hit"] for r in answerable), len(answerable)),
        "false_refusal_rate": rate(len(answerable) - len(answered), len(answerable)),
        "citation_accuracy": rate(sum(r["cited_ok"] for r in answered), len(answered)),
        "refusal_rate": rate(sum(r["correct_refusal"] for r in unanswerable), len(unanswerable)),
        "hallucination_rate": rate(
            sum(not r["correct_refusal"] for r in unanswerable), len(unanswerable)
        ),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 1),
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Cases:                 {summary['cases']}")
    print(f"Answer accuracy:       {pct(summary['answer_accuracy'])}   (answerable questions answered correctly)")
    print(f"Retrieval hit rate:    {pct(summary['retrieval_hit_rate'])}   (right document was retrieved)")
    print(f"False refusal rate:    {pct(summary['false_refusal_rate'])}   (refused but the answer existed)")
    print(f"Citation accuracy:     {pct(summary['citation_accuracy'])}   (cited the right document)")
    print(f"Refusal rate:          {pct(summary['refusal_rate'])}   (unanswerable questions refused)")
    print(f"Hallucination rate:    {pct(summary['hallucination_rate'])}   (unanswerable but answered anyway)")
    print(f"Avg latency:           {summary['avg_latency_s']}s per question")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation")
    parser.add_argument("--only", help="comma-separated case ids, e.g. q01,q08")
    parser.add_argument("--fail-under", type=float, default=None,
                        help="exit with error if accuracy or refusal rate is below this (0-1)")
    args = parser.parse_args()

    cases = json.loads(GOLDEN.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        sys.exit("No matching cases")

    rows = []
    for case in cases:
        row = evaluate_case(case)
        rows.append(row)
        mark = "PASS" if not row["diagnosis"] else "FAIL"
        print(f"{mark}  {row['id']}  {row['question'][:56]:<56}  {row['status']:<12} {row['diagnosis']}")

    summary = summarize(rows)
    print_summary(summary)

    failures = [r for r in rows if r["diagnosis"]]
    if failures:
        print("\nFAILURES")
        for r in failures:
            print(f"\n[{r['id']}] {r['diagnosis']}: {r['question']}")
            print(f"    answer: {r['answer']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = {
        "timestamp": stamp,
        "config": {
            "llm_model": settings.llm_model,
            "embed_model": settings.embed_model,
            "top_k": settings.top_k,
            "candidates": settings.candidates,
        },
        "summary": summary,
        "rows": rows,
    }
    path = RESULTS_DIR / f"{stamp}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")

    if args.fail_under is not None:
        scores = [s for s in (summary["answer_accuracy"], summary["refusal_rate"]) if s is not None]
        if scores and min(scores) < args.fail_under:
            sys.exit(1)


if __name__ == "__main__":
    main()