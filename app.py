"""
app.py — Streamlit entry point. Imports everything from src/.
Stage 2: Memory Architecture.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph import tutor_graph
from src.graph_state import TutorState
from src.memory import load_memory_block, persist_log, summarize_session
from src.profiles import PROFILES
from src.schemas import StudentProfile

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive Course Tutor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Adaptive Course Tutor")
st.caption("Stage 2 — Memory Architecture")

# ─── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages: list = []
if "profile" not in st.session_state:
    st.session_state.profile: StudentProfile | None = None
if "citations" not in st.session_state:
    st.session_state.citations: list = []
if "memory_block" not in st.session_state:
    st.session_state.memory_block: str = ""
if "memory_loaded_for" not in st.session_state:
    st.session_state.memory_loaded_for: str = ""
if "session_saved" not in st.session_state:
    st.session_state.session_saved: bool = False

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Student Profile")

    profile_choice = st.selectbox(
        "Select your level:",
        options=list(PROFILES.keys()),
        format_func=lambda k: {
            "beginner": "🟢 Beginner",
            "intermediate": "🟡 Intermediate",
            "exam-prep": "🔴 Exam Prep",
        }[k],
    )

    selected_profile = PROFILES[profile_choice]
    st.session_state.profile = selected_profile

    # Load memory when profile changes
    if st.session_state.memory_loaded_for != selected_profile.name:
        st.session_state.memory_block = load_memory_block(selected_profile.name, max_sessions=3)
        st.session_state.memory_loaded_for = selected_profile.name
        st.session_state.messages = []
        st.session_state.session_saved = False

    st.markdown("---")
    st.markdown(f"**Name:** {selected_profile.name}")
    st.markdown(f"**Time budget:** {selected_profile.time_budget} min")
    st.markdown("**Goals:**")
    for g in selected_profile.goals:
        st.markdown(f"- {g}")

    st.markdown("---")
    st.markdown("**🧠 Prior Session Memory**")
    if st.session_state.memory_block:
        st.info(st.session_state.memory_block)
    else:
        st.caption("No prior sessions yet.")

    st.markdown("---")

    if st.button("💾 End session & save memory", use_container_width=True):
        if len(st.session_state.messages) < 2:
            st.warning("Have a conversation first before saving.")
        elif st.session_state.session_saved:
            st.info("Session already saved.")
        else:
            with st.spinner("Summarising session…"):
                log = summarize_session(
                    student_name=selected_profile.name,
                    messages=st.session_state.messages,
                )
                persist_log(log)
                st.session_state.session_saved = True
                st.session_state.memory_block = load_memory_block(
                    selected_profile.name, max_sessions=3
                )
            st.success(f"Saved! Topics: {', '.join(log.topics_covered) or 'none'}")
            if log.weak_topics:
                st.warning(f"Weak topics: {', '.join(log.weak_topics)}")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.citations = []
        st.session_state.session_saved = False
        st.rerun()

    st.markdown("---")
    st.markdown("**Build stage:** `Stage 2 / 4`")
    st.markdown("**Retriever:** Qdrant (mock fallback if empty)")
    st.markdown("**LLM:** Ollama llama3.1")
    st.markdown("**Memory:** ✅ JSON session logs")

# ─── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# ─── Chat input ───────────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask anything about the course material…"):

    if st.session_state.session_saved:
        st.warning("Session was saved. Clear chat to start a new session.")
        st.stop()

    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)

    with st.chat_message("user"):
        st.write(user_input)

    initial_state: TutorState = {
        "messages": st.session_state.messages,
        "profile": st.session_state.profile,
        "memory_block": st.session_state.memory_block,
        "retrieved_chunks": [],
        "citations": [],
        "route": None,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = tutor_graph.invoke(initial_state)

    ai_message = result["messages"][-1]
    st.session_state.messages = result["messages"]
    st.session_state.citations = result.get("citations", [])

    with st.chat_message("assistant"):
        st.write(ai_message.content)

    if st.session_state.citations:
        with st.expander("📚 Sources used", expanded=False):
            for cite in st.session_state.citations:
                st.markdown(f"- **{cite.source}** — page {cite.page} _(chunk: {cite.chunk_id})_")

    st.rerun()
