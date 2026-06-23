"""
profiles.py — Predefined student profiles + prompt-level adaptation.

IMPORTANT: Adaptation here is PROMPT-LEVEL only.
We are NOT fine-tuning the model. Instead, we inject a persona/instruction
block into the system prompt that tells Claude *how* to answer:
  - answer depth (how detailed / technical)
  - vocabulary level
  - quiz difficulty
  - tone (encouraging vs. concise vs. exam-focused)

This is a classic prompt-engineering technique called "role + constraint injection".
"""

from schemas import StudentLevel, StudentProfile

# ─── Predefined profiles ───────────────────────────────────────────────────────

PROFILES: dict[str, StudentProfile] = {
    "beginner": StudentProfile(
        name="Alex",
        level=StudentLevel.BEGINNER,
        time_budget=30,
        goals=["Understand core concepts", "Build confidence with basics"],
    ),
    "intermediate": StudentProfile(
        name="Jordan",
        level=StudentLevel.INTERMEDIATE,
        time_budget=45,
        goals=["Deepen understanding", "Connect concepts", "Apply to examples"],
    ),
    "exam-prep": StudentProfile(
        name="Sam",
        level=StudentLevel.EXAM_PREP,
        time_budget=60,
        goals=["Master edge cases", "Practice under exam conditions", "Identify weak areas"],
    ),
}


# ─── Persona templates per level ──────────────────────────────────────────────

_PERSONA_TEMPLATES: dict[StudentLevel, str] = {
    StudentLevel.BEGINNER: """\
You are a patient, friendly tutor working with a beginner student.
- Use simple, everyday language. Avoid jargon; if you must use a term, define it immediately.
- Break explanations into small numbered steps.
- Use analogies and relatable examples.
- Answer depth: SHORT and focused (2–4 sentences per point max).
- Quiz difficulty: EASY — straightforward recall questions, clear distractors.
- Tone: warm, encouraging, never condescending.""",

    StudentLevel.INTERMEDIATE: """\
You are a knowledgeable tutor working with an intermediate student.
- Assume familiarity with basics; skip trivial definitions.
- Connect new ideas to concepts the student already knows.
- Answer depth: MEDIUM — explain the "why" behind concepts, include one concrete example.
- Quiz difficulty: MODERATE — application-level questions, plausible distractors.
- Tone: collegial, direct, intellectually engaging.""",

    StudentLevel.EXAM_PREP: """\
You are a rigorous exam-prep coach working with a student preparing for high-stakes assessment.
- Be precise, concise, and exam-focused. No filler.
- Highlight common misconceptions and edge cases explicitly.
- Answer depth: DETAILED — cover nuances, exceptions, and exam traps.
- Quiz difficulty: HARD — tricky distractors, multi-step reasoning required.
- Tone: focused, efficient, exam-oriented. Push the student to think critically.""",
}


# ─── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(profile: StudentProfile, memory_block: str = "") -> str:
    """
    Compose the full system prompt for a tutoring session.

    Args:
        profile:      The active StudentProfile (drives persona + depth + difficulty).
        memory_block: Optional string of prior session context injected from memory
                      (e.g. "Previously struggled with: gradient descent, overfitting").

    Returns:
        A fully composed system prompt string ready to pass to ChatAnthropic.

    How prompt-level adaptation works:
        1. We pick a persona template based on profile.level.
        2. We append student-specific context (name, goals, time budget).
        3. We optionally inject a memory block summarising past sessions.
        4. Claude reads all of this as its "identity" for the session and
           calibrates every response accordingly — no model weights change.
    """
    persona = _PERSONA_TEMPLATES[profile.level]

    goals_str = "\n".join(f"  - {g}" for g in profile.goals) if profile.goals else "  - General learning"

    student_context = f"""\
--- Student Context ---
Name: {profile.name}
Level: {profile.level.value}
Session time budget: {profile.time_budget} minutes
Goals for this session:
{goals_str}"""

    memory_section = ""
    if memory_block.strip():
        memory_section = f"""\

--- Prior Session Memory ---
{memory_block.strip()}
(Use this to personalise your responses and avoid repeating things the student already knows.)"""

    rag_instruction = """\

--- Retrieval-Augmented Answers ---
You will be provided with retrieved course material chunks inside <context> tags.
Always ground your answers in the provided context.
At the end of each answer, list your sources as: [Source: <doc>, p.<page>].
If the context does not contain enough information, say so honestly — do not hallucinate."""

    return "\n\n".join(filter(None, [persona, student_context, memory_section, rag_instruction]))
