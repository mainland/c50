# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The license notice checker rejects missing and altered notices."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "check_license_notices.py"
SPDX = "/* SPDX-License-Identifier: GPL-3.0-or-later */\n"
RULEQUEST = "/* Copyright 2010 Rulequest Research Pty Ltd. */\n"
MODIFIED = "/* Modified 2026 by Geoffrey Mainland. */\n"


def git(tmp_path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def check(
    tmp_path: Path,
    files: dict[str, str],
    main_files: tuple[str, ...] | None = None,
    reference_ref: str | None = "main",
) -> tuple[int, str]:
    """Check `files`, as tracked after a `main` branch of `main_files`.

    The checker compares its tables with `reference_ref`, as it does with the
    ref in `C50_REFERENCE_REF`. None leaves the variable unset.
    """
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(tmp_path)], check=True
    )
    if main_files is not None:
        for name in main_files:
            (tmp_path / name).write_text("int x;\n")
        git(tmp_path, "add", ".")
        git(tmp_path, "commit", "-q", "-m", "Import.")
        git(tmp_path, "rm", "-q", "-r", "--", ".")
        git(tmp_path, "checkout", "-q", "-b", "work")
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "C50_REFERENCE_REF"
    }
    if reference_ref is not None:
        environment["C50_REFERENCE_REF"] = reference_ref
    result = subprocess.run(
        [sys.executable, "-I", str(CHECKER)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    return result.returncode, result.stderr


def test_complete_imported_file_passes(tmp_path: Path) -> None:
    status, _ = check(
        tmp_path, {"src/c50.cpp": RULEQUEST + SPDX + MODIFIED + "int x;\n"}
    )
    assert status == 0


def test_new_file_without_spdx_line_fails(tmp_path: Path) -> None:
    status, message = check(tmp_path, {"src/new.cpp": "int x;\n"})
    assert status == 1
    assert "missing SPDX-License-Identifier: src/new.cpp" in message


def test_imported_file_without_rulequest_line_fails(tmp_path: Path) -> None:
    status, message = check(
        tmp_path, {"src/c50.cpp": SPDX + MODIFIED + "int x;\n"}
    )
    assert status == 1
    assert "missing RuleQuest copyright line: src/c50.cpp" in message


def test_imported_file_without_modified_line_fails(tmp_path: Path) -> None:
    status, message = check(
        tmp_path, {"src/c50.cpp": RULEQUEST + SPDX + "int x;\n"}
    )
    assert status == 1
    assert "missing Modified line: src/c50.cpp" in message


def test_changed_unmodified_import_fails(tmp_path: Path) -> None:
    status, message = check(
        tmp_path, {"src/text.i": RULEQUEST + "unmodified? no\n"}
    )
    assert status == 1
    assert "differs from its Release 2.07 blob: src/text.i" in message


@pytest.mark.parametrize("credit", [True, False])
def test_train_model_credit(tmp_path: Path, credit: bool) -> None:
    text = SPDX + (RULEQUEST if credit else "") + "int x;\n"
    status, message = check(tmp_path, {"src/c50_model.cpp": text})
    assert (status == 0) == credit
    assert credit or "missing RuleQuest credit: src/c50_model.cpp" in message


def test_table_accepts_a_tracked_counterpart(tmp_path: Path) -> None:
    status, message = check(
        tmp_path,
        {"src/c50.cpp": RULEQUEST + SPDX + MODIFIED + "int x;\n"},
        main_files=("c50.c",),
    )
    assert status == 0, message


def test_table_rejects_a_renamed_import(tmp_path: Path) -> None:
    # The import lost its name and its notices, so no table entry names it.
    status, message = check(
        tmp_path,
        {"src/renamed.cpp": SPDX + "int x;\n"},
        main_files=("c50.c",),
    )
    assert status == 1
    assert "c50.c of main has no tracked file src/c50.cpp" in message


def test_table_rejects_a_source_without_an_entry(tmp_path: Path) -> None:
    status, message = check(
        tmp_path,
        {"src/c50.cpp": RULEQUEST + SPDX + MODIFIED + "int x;\n"},
        main_files=("c50.c", "unlisted.c"),
    )
    assert status == 1
    assert "unlisted.c of main has no entry for src/unlisted.cpp" in message


def test_table_ignores_a_branch_named_main(tmp_path: Path) -> None:
    # Moving `main` must not change the result. Without the variable, the check
    # compares against the import commit, which this repository lacks.
    status, message = check(
        tmp_path,
        {"src/renamed.cpp": SPDX + "int x;\n"},
        main_files=("c50.c",),
        reference_ref=None,
    )
    assert status == 0, message
    assert "no Release 2.07 import commit" in message


@pytest.mark.parametrize(
    "name", ["Makefile", "src/c50_sort.h", "tests/sort_properties.cpp"]
)
def test_credited_files_need_rulequest_credit(
    tmp_path: Path, name: str
) -> None:
    status, message = check(tmp_path, {name: "# " + SPDX.strip() + "\n"})
    assert status == 1
    assert "missing RuleQuest" in message
    assert name in message


def test_makefile_with_credit_passes(tmp_path: Path) -> None:
    text = (
        "# Copyright 2010 Rulequest Research Pty Ltd.\n"
        "# SPDX-License-Identifier: GPL-3.0-or-later\n"
        "# Modified 2026 by Geoffrey Mainland.\n"
    )
    status, message = check(tmp_path, {"Makefile": text})
    assert status == 0, message


def test_repository_passes() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(CHECKER)],
        cwd=CHECKER.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in {0, 77}, result.stderr
