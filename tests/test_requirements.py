"""requirements.txt: in sync with pyproject.toml."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_txt_matches_pyproject():
    """requirements.txt lists exactly the dependencies in pyproject.toml."""
    # tomllib is in the standard library from Python 3.11 on.
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    assert sorted(requirements) == sorted(pyproject["project"]["dependencies"])
