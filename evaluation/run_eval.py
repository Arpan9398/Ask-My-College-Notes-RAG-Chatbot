"""Command-line retrieval evaluation.

Examples (run from the project root):
    python -m evaluation.run_eval                      # Sentence Transformers + FAISS
    python -m evaluation.run_eval --backend tfidf      # offline baseline, no model download
    python -m evaluation.run_eval --notes-dir my_pdfs --questions my_questions.json --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, EVAL_QUESTIONS_PATH, SAMPLE_NOTES_DIR, get_settings  # noqa: E402
from src.retrieval_eval import run_evaluation  # noqa: E402


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy on sample questions.")
    parser.add_argument("--backend", choices=["minilm", "tfidf"], default="minilm")
    parser.add_argument("--notes-dir", default=str(SAMPLE_NOTES_DIR))
    parser.add_argument("--questions", default=str(EVAL_QUESTIONS_PATH))
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--min-relevance", type=float, default=settings.min_relevance)
    parser.add_argument("--out", default=str(DATA_DIR / "eval_results.json"))
    args = parser.parse_args()

    try:
        result = run_evaluation(args.notes_dir, args.questions, args.backend, args.k, settings,
                                min_relevance=args.min_relevance)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    a, u = result["answerable"], result["unanswerable"]
    print(f"\nBackend: {result['backend_label']}")
    print(f"Documents: {result['num_documents']}  Chunks: {result['num_chunks']}  k={result['k']}")
    for w in result["warnings"]:
        print(f"WARNING: {w}")
    print("\nPer-question results")
    print(f"{'ID':<5}{'Rank':<6}{'Top score':<11}Question  ->  expected")
    for p in result["per_question"]:
        rank = p["rank"] if p["rank"] else ("-" if p["answerable"] else "n/a")
        score = f"{p['top_score']:.3f}" if p["top_score"] is not None else "-"
        print(f"{p['id']:<5}{str(rank):<6}{score:<11}{p['question'][:60]}  ->  {p['expected']}")
    print(f"\nAnswerable questions: {a['n']}")
    print(f"  Hit@1: {_pct(a['hit_at_1'])}   Hit@3: {_pct(a['hit_at_3'])}   Hit@{result['k']}: {_pct(a['hit_at_k'])}   "
          f"MRR: {a['mrr']:.3f}" if a["mrr"] is not None else "  (none)")
    if a["lowest_top_score"] is not None:
        print(f"  Lowest top-1 similarity among answerable questions: {a['lowest_top_score']:.3f}")
    print(f"Unanswerable questions: {u['n']}")
    if u["n"]:
        print(f"  Rejected by relevance threshold ({result['min_relevance']}): "
              f"{u['rejected_by_threshold']}/{u['n']} ({_pct(u['rejection_rate'])})")
        if u["highest_top_score"] is not None:
            print(f"  Highest top-1 similarity among unanswerable questions: {u['highest_top_score']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
