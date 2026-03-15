---
name: Never use pip install inside uv
description: Using pip install inside a uv project is an antipattern — use uv sync with extras and build hooks
type: feedback
---

Never use `pip install`, `uv pip install`, or `python -m pip` inside this project. Use `uv sync --extra <name>` for dependencies and `uv build` for building extensions.

**Why:** User explicitly called `pip install` inside uv an antipattern. The project uses uv's project-level workflow exclusively.

**How to apply:** For C++ extensions, use a hatch build hook or `uv run python setup.py build_ext --inplace` rather than `pip install -e .`. All dependency management goes through pyproject.toml extras + `uv sync`.
