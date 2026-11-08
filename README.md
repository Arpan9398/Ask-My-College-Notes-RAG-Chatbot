# 📚 Ask My College Notes

A small Retrieval-Augmented Generation (RAG) chatbot. Upload your PDF lecture notes, ask questions in a chat
window, and get answers that come **only from your notes**, together with the **source PDF name and page number**.
If the notes don't contain the answer, the app says so instead of guessing.

**Stack:** Python · LangChain · Sentence Transformers (`all-MiniLM-L6-v2`) · FAISS · LLM API (Anthropic Claude by
default, OpenAI optional) · Streamlit

## How it works

```
PDF ──pypdf──▶ text per page ──LangChain splitter──▶ chunks (never cross pages)
                                                         │ Sentence Transformers
                                                         ▼
question ──embed──▶ FAISS top-k search ──▶ drop chunks below similarity threshold
                                                         │ (none left? -> "not found", LLM not called)
                                                         ▼
                              LLM answers using ONLY those chunks and cites them as [1], [2]
                                                         ▼
                         answer + source file / page (only the passages the answer cited)
```

* **"Not found" is enforced twice:** a similarity threshold (no LLM call when nothing is relevant) and a prompt
  rule that makes the model reply `NOT_FOUND` when the retrieved notes don't contain the answer.
* **Honest sources:** the app shows the passages the model actually cited. If the model cites nothing, the
  retrieved passages are shown but labelled "not explicitly cited".

## Setup

Requires Python 3.10+ (3.11 or 3.12 recommended).

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1

# 2. install dependencies (first install is large: PyTorch comes with sentence-transformers)
pip install -r requirements.txt

# 3. add your API key
cp .env.example .env                 # Windows: copy .env.example .env
#    then edit .env and set ANTHROPIC_API_KEY (or set LLM_PROVIDER=openai and OPENAI_API_KEY)

# 4. run
streamlit run app.py
```

The first start downloads the embedding model (~90 MB) from Hugging Face, so it needs internet once. After that
it is cached by Hugging Face in your user cache folder (not in this project).

## Using the app

1. Upload one or more PDFs in the sidebar and click **Add to knowledge base**.
2. Ask questions in the chat box. Each answer lists the source file(s) and page(s); expand **Show source passages**
   to read the exact text.
3. Remove a single PDF with 🗑 or wipe everything with **Clear everything**. The index is saved in `data/index/`
   and reloaded the next time you start the app.
4. Tune **top-k** and **minimum similarity** in the sidebar if answers are missing context or too many questions
   are answered "not found".

Try it with the bundled `sample_notes/*.pdf` (e.g. *"What is Belady's anomaly?"*, *"How does a circular queue
avoid wasting space?"*, and an out-of-scope one such as *"How does TCP congestion control work?"*).

## Retrieval evaluation

`evaluation/eval_questions.json` holds sample questions with the expected source PDF and page(s); 14 are
answerable from `sample_notes/`, 3 are deliberately out of scope. The evaluation checks **retrieval only**
(not the wording of LLM answers):

* **Hit@1 / Hit@3 / Hit@k** – share of questions where a chunk from the expected PDF *and* page is among the top 1 / 3 / k results
* **MRR** – mean reciprocal rank of the first correct chunk
* **Out-of-scope rejection** – how many unanswerable questions fall below the similarity threshold

Run it from the command line (no API key needed, results are saved to `data/eval_results.json`):

```bash
python -m evaluation.run_eval                    # Sentence Transformers + FAISS (the app's pipeline)
python -m evaluation.run_eval --backend tfidf    # offline TF-IDF baseline, no model download
python -m evaluation.run_eval --k 5 --min-relevance 0.25
```

…or open the **📊 Evaluation** view in the app. To evaluate your own notes, put PDFs in a folder, write a
questions file in the same format as `eval_questions.json`, and run:

```bash
python -m evaluation.run_eval --notes-dir path/to/pdfs --questions path/to/questions.json
```

> The sample set is small and the sample notes are short, so treat the scores as a smoke test, not a benchmark.
> The default `MIN_RELEVANCE` of 0.20 has not been calibrated on a large corpus; the evaluation prints the highest
> out-of-scope and lowest in-scope similarity so you can pick a better value for your notes.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | – | your key (never commit `.env`) |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | `claude-sonnet-5-5` / `gpt-4o-mini` | model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | any Sentence Transformers model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | characters per chunk / overlap |
| `TOP_K` | `4` | passages retrieved per question |
| `MIN_RELEVANCE` | `0.20` | minimum cosine similarity |
| `MAX_UPLOAD_MB` | `50` | per-file upload limit |

If you change `EMBEDDING_MODEL`, the saved index is detected as incompatible and reset – just re-upload your PDFs.

## Error handling

* Empty, non-PDF, corrupted, password-protected or scanned (no text layer) PDFs produce a clear message; other
  files in the same upload still get indexed.
* Missing/invalid API key, unknown model, rate limit, timeout and connection errors are shown as readable messages
  in the chat instead of a stack trace.
* A damaged or incompatible saved index is reset with a warning rather than crashing the app.

## Project structure

```
app.py                  Streamlit UI (chat, upload, evaluation view)
src/
  config.py             settings from env / .env
  pdf_loader.py         PDF -> text per page, friendly errors
  chunking.py           LangChain splitter, chunks stay within one page
  vector_store.py       Sentence Transformers + FAISS index, persistence, add/remove per file
  llm.py                provider factory + API error messages
  rag.py                retrieval, threshold, prompt, citation handling
  baseline.py           offline TF-IDF retriever (evaluation only)
  retrieval_eval.py     metrics and evaluation runner
evaluation/             eval_questions.json + run_eval.py (CLI)
sample_notes/           two small sample PDFs (regenerate: python scripts/generate_sample_notes.py)
tests/                  unit tests (pip install -r requirements-dev.txt && pytest)
data/                   created at runtime (FAISS index, eval results) - git-ignored
```

## Limitations

* No OCR: scanned PDFs without a text layer can't be read. Equations, tables and diagrams are extracted as plain
  text at best.
* Each question is answered independently (no follow-up memory such as "and what about its complexity?").
* Chunks stay within a page, so an explanation that spans two pages is retrieved as two separate chunks.
* `data/index/index.pkl` is a pickle written by this app; only load index folders you created yourself.
* Your question and the retrieved note passages are sent to the LLM provider you configure.
