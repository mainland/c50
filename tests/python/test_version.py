"""Package versions follow package tags independently of the upstream import."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from packaging.version import Version
import pytest


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c", "user.name=C50 version test",
            "-c", "user.email=c50-version-test@example.invalid",
            "-c", "commit.gpgsign=false",
            "-c", "tag.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package_version(repository: Path) -> Version:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("SETUPTOOLS_SCM_PRETEND_")
    }
    result = subprocess.run(
        [sys.executable, "-m", "setuptools_scm"],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return Version(result.stdout.strip())


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "pyproject.toml",
        tmp_path / "pyproject.toml",
    )
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "add", "pyproject.toml")
    git(tmp_path, "commit", "--quiet", "-m", "Add package configuration")
    git(tmp_path, "tag", "c5.0-2.07")
    return tmp_path


def test_upstream_tag_does_not_set_package_version(repository: Path) -> None:
    version = package_version(repository)
    assert version.release == (0, 1)
    assert version.is_devrelease


@pytest.mark.parametrize("version", ["0.1.0rc1", "0.1.0"])
def test_package_tag_sets_version(repository: Path, version: str) -> None:
    git(repository, "tag", "v" + version)
    assert package_version(repository) == Version(version)


def test_commits_after_package_tag_get_development_versions(repository: Path) -> None:
    git(repository, "tag", "v0.1.0rc1")
    git(repository, "commit", "--quiet", "--allow-empty", "-m", "Continue development")
    version = package_version(repository)
    assert version > Version("0.1.0rc1")
    assert version.is_devrelease
    assert version.local is not None


def test_dirty_release_checkout_gets_a_development_version(repository: Path) -> None:
    git(repository, "tag", "v0.1.0rc1")
    config = repository / "pyproject.toml"
    config.write_text(config.read_text() + "\n")
    version = package_version(repository)
    assert version != Version("0.1.0rc1")
    assert version.is_devrelease
    assert version.local is not None
