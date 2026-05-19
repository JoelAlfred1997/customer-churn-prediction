"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "config.yaml"
