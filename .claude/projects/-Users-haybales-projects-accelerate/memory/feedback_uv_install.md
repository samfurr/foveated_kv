---
name: Use uv sync not uv pip install
description: User prefers uv sync with extras over uv pip install for dependency management
type: feedback
---

Use `uv sync --extra <name>` to install dependencies, never `uv pip install`. The project uses uv with pyproject.toml extras.

**Why:** User corrected this directly — they use uv's project-level workflow, not pip-style installs.

**How to apply:** When installing new dependencies, add them to pyproject.toml optional-dependencies first, then `uv sync --extra <name> --extra dev` (always include dev for pytest).
