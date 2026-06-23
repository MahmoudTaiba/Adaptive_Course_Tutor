"""
app.py — Stage 1 Streamlit shell (deliberately minimal).

Stage 1 scope:
  ✅ Profile picker
  ✅ Chat box
  ✅ Citations display
  ❌ No session persistence yet (Stage 3)
  ❌ No quiz UI yet (Stage 2)
  ❌ No memory panel yet (Stage 3)
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import tutor_graph
from graph_state import TutorState
from profiles import PROFILES
from schemas import StudentProfile

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive Course Tutor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Adaptive Course Tutor")
st.caption("Stage 1 — Foundations (mock retriever active)")

# ─── Session state initialisation ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages: list = []

if "profile" not in st.session_state:
    st.session_state.profile: StudentProfile | None = None

if "citations" not in st.session_state:
    st.session_state.citations: list = []

# ─── Sidebar: Profile Picker ───────────────────────────────────────────────────

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

    st.markdown("---")
    st.markdown(f"**Name:** {selected_profile.name}")
    st.markdown(f"**Time budget:** {selected_profile.time_budget} min")
    st.markdown("**Goals:**")
    for g in selected_profile.goals:
        st.markdown(f"- {g}")

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.citations = []
        st.rerun()

    # Stage indicator
    st.markdown("---")
    st.markdown("**Build stage:** `Stage 1 / 4`")
    st.markdown("**Retriever:** ⚠️ Mock")
    st.markdown("**LLM:** Claude Sonnet 4.6")

# ─── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# ─── Chat input ───────────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask anything about the course material…"):

    # Show user message immediately
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)

    with st.chat_message("user"):
        st.write(user_input)

    # Build initial state for this turn
    initial_state: TutorState = {
        "messages": st.session_state.messages,
        "profile": st.session_state.profile,
        "memory_block": "",          # Stage 3: will be populated from SessionLog
        "retrieved_chunks": [],
        "citations": [],
        "route": None,
    }

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = tutor_graph.invoke(initial_state)

    # Extract the latest AI message (last in result["messages"])
    ai_message = result["messages"][-1]
    st.session_state.messages = result["messages"]
    st.session_state.citations = result.get("citations", [])

    # Display AI response
    with st.chat_message("assistant"):
        st.write(ai_message.content)

    # Display citations
    if st.session_state.citations:
        with st.expander("📚 Sources used", expanded=False):
            for cite in st.session_state.citations:
                st.markdown(f"- **{cite.source}** — page {cite.page} _(chunk: {cite.chunk_id})_")

    st.rerun()
