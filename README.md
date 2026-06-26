# Adaptive Course Tutor — Stage 2

RAG-powered tutoring assistant. **Part B (Agent & Experience)** — 
Part A (retrieval/ingestion/evaluation) — partner.

## Project Structure

```
adaptive-course-tutor/
├── app.py                   # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── schemas.py           # ALL Pydantic contracts (single source of truth)
│   ├── config.py            # Shared constants + get_llm(), get_client()
│   ├── ingest.py            # PDF → chunks → Qdrant (Part A)
│   ├── retrieve.py          # Qdrant vector retrieval (Part A)
│   ├── profiles.py          # Student profiles + build_system_prompt()
│   ├── graph_state.py       # LangGraph TutorState TypedDict
│   ├── graph.py             # LangGraph: retrieve → generate → END
│   └── memory.py            # Session-log memory system
├── scripts/
│   ├── run_ingest.py        # Ingest a PDF into Qdrant
│   └── eval_retrieval.py    # Hit@k, Recall@k, MRR evaluation (Part A)
├── data/
│   ├── eval/qa.jsonl        # Eval questions (Part A fills this)
│   └── memory/              # Per-student JSON session logs
├── requirements.txt
├── .env.example
└── README.md
```

## Stage Progress

| Stage | Status | Features |
|-------|--------|----------|
| 1 — Foundations | ✅ Done | Schemas, profiles, graph skeleton, Streamlit shell |
| 2 — Memory | ✅ Done | Session logs, memory injection, Qdrant integration |
| 3 — Quiz & Routing | 🔜 | Router node, quiz generation, quiz UI |
| 4 — Evaluation | 🔜 | RAGAS eval, polish |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the LLM (Ollama must be installed)
ollama pull llama3.1

# 3. Ingest your course PDF (edit path in scripts/run_ingest.py first)
python scripts/run_ingest.py

# 4. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501**.

> **Before ingestion:** The app still runs using a mock retriever fallback.
> After running `run_ingest.py`, real Qdrant retrieval activates automatically.

## LLM Configuration

Model is set in `src/config.py`:
```python
LLM_MODEL = "llama3.1"   # change here to swap models everywhere
```

All LLM calls go through `get_llm()` — one place to change.

## Part A ↔ Part B Contract

`src/schemas.py` is the single source of truth. `Chunk` / `Citation` / `RetrievedChunk` must not be changed unilaterally. `RetrievedChunk` extends `Chunk` with a `.similarity` score from Qdrant.
