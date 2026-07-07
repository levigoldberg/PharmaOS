"""Shared Pydantic schemas for PharmaOS workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class WorkflowRun(BaseModel):
    """A minimal workflow run record."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    """A minimal report for a workflow run."""

    run_id: str
    title: str
    content: str
    generated_at: datetime = Field(default_factory=utc_now)
