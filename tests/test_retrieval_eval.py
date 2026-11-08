from src.models import RetrievedChunk
from src.retrieval_eval import evaluate, first_correct_rank


def c(source, page, score):
    return RetrievedChunk(text="t", source=source, page=page, score=score)


def test_first_correct_rank():
    results = [c("a.pdf", 1, .9), c("b.pdf", 2, .8), c("b.pdf", 3, .7)]
    assert first_correct_rank(results, "b.pdf", [3]) == 3
    assert first_correct_rank(results, "b.pdf", [2, 3]) == 2
    assert first_correct_rank(results, "c.pdf", [1]) is None
    assert first_correct_rank(results, "a.pdf", [2]) is None  # right file, wrong page = miss


def test_metrics_and_unanswerable_handling():
    table = {
        "q1": [c("a.pdf", 1, .9), c("a.pdf", 2, .5)],  # rank 1
        "q2": [c("a.pdf", 9, .8), c("a.pdf", 2, .5)],  # rank 2
        "q3": [c("a.pdf", 9, .8), c("a.pdf", 8, .5)],  # miss
        "q4": [c("a.pdf", 1, .10)],                      # out of scope, low score -> rejected
        "q5": [c("a.pdf", 1, .60)],                      # out of scope, high score -> not rejected
    }
    questions = [
        {"id": "1", "question": "q1", "expected_source": "a.pdf", "expected_pages": [1]},
        {"id": "2", "question": "q2", "expected_source": "a.pdf", "expected_pages": [2]},
        {"id": "3", "question": "q3", "expected_source": "a.pdf", "expected_pages": [2]},
        {"id": "4", "question": "q4", "expected_source": None},
        {"id": "5", "question": "q5", "expected_source": None},
    ]
    r = evaluate(questions, lambda q, k: table[q][:k], k=2, min_relevance=0.2)
    a, u = r["answerable"], r["unanswerable"]
    assert a["n"] == 3
    assert a["hit_at_1"] == 1 / 3 and a["hit_at_k"] == 2 / 3
    assert abs(a["mrr"] - (1 + 0.5 + 0) / 3) < 1e-9
    assert u["n"] == 2 and u["rejected_by_threshold"] == 1 and u["rejection_rate"] == 0.5
