"""
schemas.py — Shared Pydantic contracts (Part A ↔ Part B interface boundary).

These models are the locked "contract" between the retrieval pipeline (Part A)
and the agent/experience layer (Part B). Neither side changes these without
agreement from both partners.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class StudentLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXAM_PREP = "exam-prep"


# ─── Retrieval contracts (owned jointly, produced by Part A) ──────────────────

class Chunk(BaseModel):
    """A single retrieved text chunk from the document store (Part A output)."""
    text: str = Field(..., description="Raw text content of the chunk")
    source_doc: str = Field(..., description="Source document filename or title")
    page: int = Field(..., description="Page number within the source document")
    chunk_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk identifier")


class Citation(BaseModel):
    """A citation linking an answer back to a specific chunk."""
    chunk_id: str = Field(..., description="ID of the Chunk this citation refers to")
    source: str = Field(..., description="Human-readable source name")
    page: int = Field(..., description="Page number in the source document")


# ─── Student & session contracts (owned by Part B) ────────────────────────────

class StudentProfile(BaseModel):
    """Describes a student's learning context; drives prompt adaptation."""
    name: str = Field(..., description="Student's display name")
    level: StudentLevel = Field(..., description="Learning level")
    time_budget: int = Field(..., description="Available study time in minutes per session")
    goals: List[str] = Field(default_factory=list, description="Learning goals for the session")


class SessionLog(BaseModel):
    """Persisted record of one tutoring session."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    student_name: str
    topics_covered: List[str] = Field(default_factory=list)
    weak_topics: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Quiz contracts (owned by Part B) ─────────────────────────────────────────

class QuizItem(BaseModel):
    """A single multiple-choice question."""
    question: str
    options: List[str] = Field(..., min_length=2, description="Answer choices (min 2)")
    correct_index: int = Field(..., description="0-based index of the correct option")
    topic: str = Field(..., description="Topic this question tests")
    explanation: str = Field(default="", description="Why the correct answer is right")


class Quiz(BaseModel):
    """A collection of QuizItems on a single topic."""
    topic: str
    items: List[QuizItem] = Field(default_factory=list)
