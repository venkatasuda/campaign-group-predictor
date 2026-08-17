"""Refuse any commit that stages the confidential dataset.

Run by pre-commit (see .pre-commit-config.yaml) and safe to run by hand:

    python scripts/check_no_data_committed.py

Why this exists
---------------
`data/*.csv` is already in `.gitignore`. That is advisory: `git add -f` overrides it
without a warning, and an IDE's "stage all" can pick up a file that was moved into the
directory after the ignore rule was written.

Once a file reaches history, removing it is not `git rm`. It means rewriting every commit
that followed, force-pushing, and treating the contents as disclosed regardless - because
anyone who cloned in between still has it. The asymmetry between preventing this and
undoing it is what justifies a hard gate rather than a convention.

The dataset here is marked LP-Internal with redistribution prohibited, so the cost of
getting this wrong is contractual, not merely embarrassing.

Exit codes
----------
0  nothing confidential staged
1  a blocked path is staged; the commit is refused
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: Directories whose contents must never be committed, whatever the extension.
BLOCKED_DIRECTORIES = ("data/",)

#: Extensions that are data by nature wherever they appear. `.gitkeep` and documentation
#: are unaffected because they do not match.
BLOCKED_SUFFIXES = (".csv", ".parquet", ".feather", ".xlsx", ".pkl", ".joblib")

#: Paths that would otherwise match but are legitimate: placeholders that keep an
#: otherwise-empty directory in version control, and sample payloads for the API.
ALLOWED = (
    "data/.gitkeep",
    "artifacts/.gitkeep",
)


def staged_files() -> list[str]:
    """Return the paths git currently has staged, as forward-slash relative paths."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:  # pragma: no cover - not a git repository
        return []
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def is_blocked(path: str) -> bool:
    """Whether a staged path must not be committed."""
    if path in ALLOWED:
        return False
    if path.startswith(BLOCKED_DIRECTORIES):
        return True
    return Path(path).suffix.lower() in BLOCKED_SUFFIXES


def main() -> int:
    offenders = sorted(path for path in staged_files() if is_blocked(path))

    if not offenders:
        return 0

    print("\nCOMMIT REFUSED - confidential or generated data is staged:\n", file=sys.stderr)
    for path in offenders:
        print(f"    {path}", file=sys.stderr)
    print(
        "\nThe campaign dataset is LP-Internal and must not be redistributed; model\n"
        "artifacts are build outputs, not source.\n"
        "\nUnstage them with:\n"
        f"    git restore --staged {' '.join(offenders)}\n"
        "\nIf you are certain this is wrong, fix the rule in\n"
        "scripts/check_no_data_committed.py rather than bypassing the hook - a bypass\n"
        "leaves no record, and this is the failure that cannot be undone by reverting.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
