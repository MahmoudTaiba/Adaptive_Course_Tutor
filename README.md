# Adaptive Course Tutor — Stage 1

RAG-powered tutoring assistant. **Part B (Agent & Experience)** — built by Oracle Sense.

## Project Structure

```
adaptive-course-tutor/
├── schemas.py        # Shared Pydantic contracts (Part A ↔ Part B interface)
├── profiles.py       # Student profiles + prompt-level adaptation
├── graph_state.py    # LangGraph TutorState TypedDict
├── graph.py          # LangGraph: retrieve → generate → END
├── app.py            # Streamlit UI (profile picker + chat)
├── requirements.txt
├── .env.example
└── README.md
```

## Stage Progress

| Stage | Status | Features |
|-------|--------|----------|
| 1 — Foundations | ✅ Done | Schemas, profiles, graph skeleton, Streamlit shell |
| 2 — Quiz & Routing | 🔜 | Router node, quiz generation, quiz UI |
| 3 — Memory | 🔜 | SessionLog persistence, memory injection |
| 4 — Evaluation | 🔜 | Part A integration, RAGAS eval, polish |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

## Stage 1 Demo Checkpoint

1. Open the app
2. Pick a profile from the sidebar (Beginner / Intermediate / Exam Prep)
3. Ask any question (e.g. *"What is gradient descent?"*)
4. You'll get a Claude-generated answer grounded in **mocked** course chunks
5. Expand "📚 Sources used" to see the mock citations

> ⚠️ **Note:** The retriever is mocked. It always returns 3 canned chunks about ML topics regardless of your query. Part A will wire in the real retriever in Stage 2.

## Architecture Notes

### Prompt-Level Adaptation (NOT fine-tuning)
`profiles.py::build_system_prompt()` injects a persona + constraints into the system prompt. Claude reads this and calibrates answer depth, vocabulary, and quiz difficulty accordingly. No model weights are changed.

### Part A ↔ Part B Contract
`schemas.py` is the agreed interface. `Chunk` (produced by Part A) and `Citation` (consumed by Part B) must not be changed unilaterally.

### Mock Retriever Location
In `graph.py`, the function `_mock_retrieve()` contains the mock. Its **signature** (`query: str → List[Chunk]`) is the contract. Part A replaces the body, not the signature.
