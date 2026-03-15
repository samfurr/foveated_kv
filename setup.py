"""Build the foveated_ext C++ extension.

Usage:
    pip install -e ".[ext]"   # builds C++ extension + installs deps
    uv sync --extra ext       # install deps, then: uv run pip install -e .

The extension is optional — the Python fallback path always works.
"""

from setuptools import setup

try:
    from mlx.extension import CMakeExtension, CMakeBuild

    setup(
        name="foveated-ext",
        ext_modules=[CMakeExtension("foveated_ext", sourcedir="csrc")],
        cmdclass={"build_ext": CMakeBuild},
    )
except ImportError:
    print("WARNING: mlx not found, skipping C++ extension build")
    setup(name="foveated-ext")
