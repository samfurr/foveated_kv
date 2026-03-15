---
name: Always use uv run for Python commands
description: Use uv run, not bare python3, for all Python commands in this project
type: feedback
---

Always use `uv run python` or `uv run pytest` instead of bare `python3` or `pytest`. The project uses a uv-managed virtualenv.

**Why:** User corrected this — the project's dependencies are only available in the uv environment.

**How to apply:** Prefix all Python/pytest commands with `uv run`.
