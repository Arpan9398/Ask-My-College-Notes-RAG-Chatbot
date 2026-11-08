"""Ask My College Notes - Streamlit chat UI.   Run with:  streamlit run app.py"""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from src.chunking import pages_to_chunks
from src.config import EVAL_QUESTIONS_PATH, SAMPLE_NOTES_DIR, get_settings
from src.llm import build_llm
from src.pdf_loader import PDFLoadError, load_pdf
from src.rag import answer_question
from src.retrieval_eval import run_evaluation
from src.vector_store import KnowledgeBaseError, NotesIndex

settings = get_settings()
st.set_page_config(page_title="Ask My College Notes", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (the first run downloads ~90 MB)...")
def get_index() -> NotesIndex:
    return NotesIndex(get_settings())


def ingest_files(index: NotesIndex, files) -> list[tuple[str, str]]:
    """Process uploaded PDFs one by one; a bad file never blocks the others."""
    results: list[tuple[str, str]] = []
    progress = st.progress(0.0, text="Processing PDFs...")
    for n, uploaded in enumerate(files, start=1):
        name = uploaded.name
        try:
            data = uploaded.getvalue()
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise PDFLoadError(f"The file is larger than the {settings.max_upload_mb} MB limit.")
            digest = hashlib.sha256(data).hexdigest()
            known = index.files.get(name)
            if known and known["sha256"] == digest:
                results.append(("info", f"**{name}** is already indexed."))
                continue
            loaded = load_pdf(data, name)
            chunks = pages_to_chunks(loaded.pages, settings.chunk_size, settings.chunk_overlap)
            status = index.add_document(name, digest, chunks, num_pages=len(loaded.pages))
            note = f" ({loaded.skipped_pages} page(s) had no extractable text and were skipped)" if loaded.skipped_pages else ""
            verb = "Updated" if status == "replaced" else "Indexed"
            results.append(("success", f"{verb} **{name}**: {len(loaded.pages)} pages, {len(chunks)} chunks{note}."))
        except PDFLoadError as exc:
            results.append(("error", f"**{name}**: {exc}"))
        except KnowledgeBaseError as exc:
            results.append(("error", f"**{name}**: {exc}"))
        except Exception as exc:  # last-resort guard so the UI never crashes on one file
            results.append(("error", f"**{name}**: unexpected error ({type(exc).__name__}: {exc})"))
        finally:
            progress.progress(n / len(files), text=f"Processed {n}/{len(files)}")
    progress.empty()
    return results


def render_answer(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    st.markdown(msg["text"])
    sources = msg.get("sources") or []
    if not sources:
        return
    unique = list(dict.fromkeys((s["source"], s["page"]) for s in sources))
    st.caption("📄 " + ("Sources: " if msg["source_kind"] == "cited" else "Closest retrieved passages (not explicitly cited): ")
               + "; ".join(f"{name}, page {page}" for name, page in unique))
    with st.expander("Show source passages"):
        for s in sources:
            st.markdown(f"**[{s['ref']}] {s['source']} - page {s['page']}**  \n_similarity {s['score']:.2f}_")
            st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


# ------------------------------------------------------------------ startup
st.title("📚 Ask My College Notes")
st.caption("Upload your PDF notes and ask questions. Answers use only your notes and show the source file and page.")

try:
    index = get_index()
except KnowledgeBaseError as exc:
    st.error(str(exc))
    st.stop()

if index.load_warning:
    st.warning(index.load_warning)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Your notes")
    uploads = st.file_uploader("Upload PDF notes", type=["pdf"], accept_multiple_files=True)
    if st.button("Add to knowledge base", type="primary", disabled=not uploads, use_container_width=True):
        st.session_state["ingest_results"] = ingest_files(index, uploads)
    for level, message in st.session_state.get("ingest_results", []):
        getattr(st, level)(message)

    st.subheader("Indexed documents")
    if index.is_empty:
        st.caption("Nothing indexed yet.")
    for name, info in list(index.files.items()):
        left, right = st.columns([5, 1])
        left.markdown(f"**{name}**  \n{info['pages']} pages · {info['chunks']} chunks")
        if right.button("🗑", key=f"remove_{name}", help=f"Remove {name}"):
            index.remove_document(name)
            st.session_state.pop("ingest_results", None)
            st.rerun()
    if not index.is_empty and st.button("Clear everything", use_container_width=True):
        index.clear()
        st.session_state.pop("ingest_results", None)
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, min(settings.top_k, 10))
    min_rel = st.slider("Minimum similarity", 0.0, 0.8, min(settings.min_relevance, 0.8), 0.05,
                        help="Passages below this cosine similarity are ignored. If none pass, the app says "
                             "'not found' without calling the LLM.")
    st.caption(f"LLM: `{settings.llm_label}`  \nEmbeddings: `{settings.embedding_model}`")
    view = st.radio("View", ["💬 Chat", "📊 Evaluation"], horizontal=True)

# --------------------------------------------------------------------- chat
if view == "💬 Chat":
    st.session_state.setdefault("messages", [])
    if st.session_state["messages"] and st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["text"])
            else:
                render_answer(message)

    if index.is_empty:
        st.info("👈 Upload one or more PDF notes in the sidebar to get started.")

    question = st.chat_input("Ask a question about your notes...", disabled=index.is_empty)
    if question:
        st.session_state["messages"].append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                result = answer_question(question, index, lambda: build_llm(settings), top_k, min_rel).to_dict()
            render_answer(result)
        st.session_state["messages"].append({"role": "assistant", **result})

# --------------------------------------------------------------- evaluation
else:
    st.subheader("Retrieval evaluation")
    st.write(
        "Checks whether the correct **PDF and page** appears in the top-k retrieved passages for a set of sample "
        f"questions (`{EVAL_QUESTIONS_PATH.name}`) over the bundled sample notes (`{SAMPLE_NOTES_DIR.name}/`). "
        "It builds its own temporary index, so your uploaded notes are not affected, and it evaluates retrieval only, "
        "not the LLM's answers."
    )
    backend_label = st.selectbox(
        "Retriever", ["Sentence Transformers + FAISS (the app's pipeline)", "TF-IDF baseline (offline, for comparison)"]
    )
    backend = "minilm" if backend_label.startswith("Sentence") else "tfidf"
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Indexing sample notes and running questions..."):
            try:
                st.session_state["eval_result"] = run_evaluation(
                    SAMPLE_NOTES_DIR, EVAL_QUESTIONS_PATH, backend=backend, k=top_k,
                    settings=settings, embeddings=index.embeddings, min_relevance=min_rel,
                )
            except Exception as exc:
                st.session_state.pop("eval_result", None)
                st.error(f"Evaluation failed: {exc}")

    res = st.session_state.get("eval_result")
    if res:
        a, u = res["answerable"], res["unanswerable"]
        st.caption(f"{res['backend_label']} · {res['num_documents']} documents · {res['num_chunks']} chunks · "
                   f"k={res['k']} · threshold={res['min_relevance']:.2f} · run at {res['timestamp']}")
        for warning in res["warnings"]:
            st.warning(warning)
        cols = st.columns(4)
        cols[0].metric("Hit@1", pct(a["hit_at_1"]))
        cols[1].metric("Hit@3", pct(a["hit_at_3"]))
        cols[2].metric(f"Hit@{res['k']}", pct(a["hit_at_k"]))
        cols[3].metric("MRR", "n/a" if a["mrr"] is None else f"{a['mrr']:.3f}")
        st.caption(f"Computed over {a['n']} answerable questions. Hit@n = the correct file+page is among the top n results.")
        if u["n"]:
            st.write(f"**Out-of-scope questions:** {u['rejected_by_threshold']}/{u['n']} fall below the similarity "
                     f"threshold ({pct(u['rejection_rate'])}) and would be answered 'not found' without calling the LLM.")
            if u["highest_top_score"] is not None and a["lowest_top_score"] is not None:
                st.caption(f"Highest top-1 similarity among out-of-scope questions: {u['highest_top_score']:.3f}; "
                           f"lowest among in-scope questions: {a['lowest_top_score']:.3f}. Use these to tune the threshold.")
        table = pd.DataFrame(
            [
                {
                    "ID": p["id"],
                    "Question": p["question"],
                    "Expected": p["expected"],
                    "Rank of correct page": p["rank"] if p["rank"] else ("miss" if p["answerable"] else "-"),
                    "Top similarity": None if p["top_score"] is None else round(p["top_score"], 3),
                    "Top result": f"{p['retrieved'][0]['source']} p.{p['retrieved'][0]['page']}" if p["retrieved"] else "-",
                }
                for p in res["per_question"]
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
