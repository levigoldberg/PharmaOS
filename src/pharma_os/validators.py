"""Validation helpers for PharmaOS inputs."""

from __future__ import annotations


def validate_workflow_name(workflow: str) -> str:
    """Validate and normalize a workflow name."""

    normalized = workflow.strip()
    if not normalized:
        raise ValueError("workflow name must not be empty")
    return normalized
