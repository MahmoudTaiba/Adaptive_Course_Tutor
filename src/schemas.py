"""
src/schemas.py — SINGLE source of truth for all Pydantic contracts.

Part A schemas (Chunk, Citation, RetrievedChunk) — owned jointly.
Part B schemas (StudentProfile, SessionLog, QuizItem, Quiz) — owned by Part B.

DO NOT duplicate these anywhere else.
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


# ─── Part A contracts ─────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """A single text chunk from the document store."""
    chunk_id: str           # deterministic: f"{source_doc}_p{page}_{idx}"
    text: str
    source_doc: str
    page: int


class Citation(BaseModel):
    """Links an answer back to a specific chunk."""
    chunk_id: str
    source: str
    page: int


class RetrievedChunk(Chunk):
    """Chunk enriched with a similarity score (returned by retrieve_chunks)."""
    similarity: float


# ─── Part B contracts ─────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    """Describes a student's learning context; drives prompt adaptation."""
    name: str
    level: StudentLevel
    time_budget: int = Field(..., description="Study time in minutes")
    goals: List[str] = Field(default_factory=list)


class SessionLog(BaseModel):
    """Persisted record of one tutoring session."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    student_name: str
    topics_covered: List[str] = Field(default_factory=list)
    weak_topics: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QuizItem(BaseModel):
    """A single multiple-choice question."""
    question: str
    options: List[str] = Field(..., min_length=2)
    correct_index: int
    topic: str
    explanation: str = Field(default="")


class Quiz(BaseModel):
    """A collection of QuizItems on a single topic."""
    topic: str
    items: List[QuizItem] = Field(default_factory=list)
