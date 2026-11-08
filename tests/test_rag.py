from types import SimpleNamespace

from src.llm import explain_llm_error
from src.models import RetrievedChunk
from src.rag import NOT_FOUND_MESSAGE, answer_question, cited_numbers


def chunk(text, source="a.pdf", page=1, score=0.6):
    return RetrievedChunk(text=text, source=source, page=page, score=score)


class FakeIndex:
    def __init__(self, chunks, empty=False):
        self.chunks, self.is_empty = chunks, empty

    def search(self, query, k):
        return self.chunks[:k]


class FakeLLM:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.calls = content, exc, 0

    def invoke(self, messages):
        self.calls += 1
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=self.content)


def run(index, llm, min_rel=0.2):
    return answer_question("What is X?", index, lambda: llm, top_k=4, min_relevance=min_rel)


def test_cited_numbers_parses_groups_and_ignores_out_of_range():
    assert cited_numbers("A [1] and B [2, 3] and C [9] and [1]", 3) == [1, 2, 3]


def test_below_threshold_is_not_found_and_llm_not_called():
    llm = FakeLLM("should not be used")
    ans = run(FakeIndex([chunk("x", score=0.05)]), llm)
    assert not ans.found and ans.text == NOT_FOUND_MESSAGE and llm.calls == 0 and ans.sources == []


def test_model_not_found_token_gives_not_found_without_sources():
    ans = run(FakeIndex([chunk("x")]), FakeLLM("NOT_FOUND"))
    assert not ans.found and ans.text == NOT_FOUND_MESSAGE and ans.sources == []


def test_cited_answer_returns_only_cited_sources():
    chunks = [chunk("one", "a.pdf", 3), chunk("two", "b.pdf", 7)]
    ans = run(FakeIndex(chunks), FakeLLM("X is two [2]."))
    assert ans.found and ans.source_kind == "cited"
    assert [(r, c.source, c.page) for r, c in ans.sources] == [(2, "b.pdf", 7)]


def test_uncited_answer_is_labelled_as_retrieved_not_cited():
    ans = run(FakeIndex([chunk("one")]), FakeLLM("X is one."))
    assert ans.found and ans.source_kind == "retrieved" and len(ans.sources) == 1


def test_list_content_blocks_are_joined():
    ans = run(FakeIndex([chunk("one")]), FakeLLM([{"type": "text", "text": "X is one [1]."}]))
    assert ans.found and ans.text == "X is one [1]."


def test_llm_failure_becomes_readable_error():
    err = type("AuthenticationError", (Exception,), {"status_code": 401})("bad key")
    ans = run(FakeIndex([chunk("one")]), FakeLLM(exc=err))
    assert not ans.found and "API key" in ans.error


def test_no_documents_and_empty_question():
    assert run(FakeIndex([], empty=True), FakeLLM("x")).error == "no documents indexed"
    assert answer_question("  ", FakeIndex([chunk("x")]), lambda: FakeLLM("x"), 4, 0.2).error == "empty question"


def test_explain_llm_error_variants():
    mk = lambda name, code=None: type(name, (Exception,), {"status_code": code})("boom")
    assert "rate limit" in explain_llm_error(mk("RateLimitError", 429))
    assert "not found" in explain_llm_error(mk("NotFoundError", 404))
    assert "server error" in explain_llm_error(mk("InternalServerError", 503))
    assert "timed out" in explain_llm_error(mk("APITimeoutError"))
