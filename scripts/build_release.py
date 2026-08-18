"""Build the submission archive from an allowlist.

    python scripts/build_release.py
    python scripts/build_release.py --out dist/submission.zip --dry-run

Why an allowlist and not an exclude list
----------------------------------------
The previous archive was built by copying the working directory and excluding known
offenders. It shipped the confidential dataset, 261 MB of MLflow runs, a 133 MB Terraform
provider binary, caches, a Word lock file and four unsynchronised reports - and it silently
dropped ``.github/``, which made the documentation's CI claim look like a lie.

Both failures have the same cause. **An exclude list is a claim that you have thought of
every bad thing.** You have not: the set of things that should not ship grows every time
someone runs a tool, and the one you forget is the one that matters. An allowlist inverts
the default - nothing ships unless it is named - so a new cache directory appears in the
manifest as *absent* rather than in the archive as a surprise.

The dataset is LP-Internal and its distribution is prohibited. That is not a hygiene
preference; it is the one constraint in this project that cannot be walked back once
breached, which is why the check below is a hard failure rather than a warning.

Exit codes
----------
0  archive written, all guards passed
1  a guard failed - nothing was written
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Directories copied in full. Anything not listed does not ship.
ALLOWED_DIRECTORIES: list[str] = [
    "src",
    "tests",
    "frontend",
    "scripts",
    "docs",
    "terraform",
    "notebooks",
    ".github",
]

#: Individual files at the project root.
ALLOWED_FILES: list[str] = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-serve.txt",
    "requirements-optional.txt",
    "Dockerfile",
    "Dockerfile.frontend",
    "docker-compose.yml",
    "cloudbuild.frontend.yaml",
    "Makefile",
    ".dockerignore",
    ".gcloudignore",
    ".gitignore",
    ".env.example",
    ".pre-commit-config.yaml",
]

#: Reports ship selectively. `reports/` also accumulates scratch output, older conversions
#: and Word lock files, and shipping four unsynchronised versions of one document is worse
#: than shipping none - a reader cannot tell which is authoritative.
ALLOWED_REPORT_FILES: list[str] = [
    "reports/REPORT.md",
    "reports/PEER_REVIEW_SUMMARY.md",
    "reports/findings.json",
    "reports/advanced_experiments.json",
    "reports/latency.json",
    "reports/model_leaderboard.csv",
    "reports/campaign_outcomes.csv",
    "reports/decision_sensitivity.csv",
    "reports/submission.docx",
]

#: The trained model ships; it is the deliverable. metrics.json ships with it because it is
#: the plain-text record of what produced it.
ALLOWED_ARTIFACT_FILES: list[str] = [
    "artifacts/model.pkl",
    "artifacts/metrics.json",
]

#: Excluded even inside an allowed directory. These are generated, not authored.
EXCLUDED_NAMES: set[str] = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".terraform",
    "node_modules",
}

EXCLUDED_SUFFIXES: set[str] = {".pyc", ".pyo", ".log", ".tfstate", ".coverage"}

#: Generated files that carry no suffix distinguishing them, matched by exact name. The
#: first dry run of this script found `terraform/.coverage` - a coverage database written
#: into an infrastructure directory by a stray command, which an exclude-list approach had
#: shipped without anyone noticing.
EXCLUDED_EXACT_NAMES: set[str] = {".coverage", ".DS_Store", "Thumbs.db", "mlflow.db"}

#: Refuse to write an archive containing any of these. The dataset check is the reason this
#: script exists; the others catch the accidents that shipped last time.
FORBIDDEN_SUFFIXES: set[str] = {".csv"}
FORBIDDEN_PREFIXES: tuple[str, ...] = ("~$",)

#: `reports/*.csv` are derived summary tables with no customer data, so they are exempt from
#: the .csv guard. Named explicitly rather than pattern-matched: an exemption that is a
#: pattern will eventually match something it should not.
CSV_EXEMPTIONS: set[str] = {
    "reports/model_leaderboard.csv",
    "reports/campaign_outcomes.csv",
    "reports/decision_sensitivity.csv",
}


def _is_excluded(path: Path) -> bool:
    if any(part in EXCLUDED_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if path.name in EXCLUDED_EXACT_NAMES:
        return True
    return any(path.name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def collect() -> list[Path]:
    """Return every project-relative path that will ship."""
    selected: list[Path] = []

    for directory in ALLOWED_DIRECTORIES:
        root = PROJECT_ROOT / directory
        if not root.is_dir():
            continue
        for item in sorted(root.rglob("*")):
            if item.is_file():
                relative = item.relative_to(PROJECT_ROOT)
                if not _is_excluded(relative):
                    selected.append(relative)

    for name in ALLOWED_FILES + ALLOWED_REPORT_FILES + ALLOWED_ARTIFACT_FILES:
        candidate = PROJECT_ROOT / name
        if candidate.is_file():
            selected.append(Path(name))

    return sorted(set(selected), key=lambda p: p.as_posix())


def guard(paths: list[Path], dry_run: bool = False) -> list[str]:
    """Return a list of reasons the archive must not be written."""
    problems: list[str] = []

    for path in paths:
        posix = path.as_posix()

        if path.suffix.lower() in FORBIDDEN_SUFFIXES and posix not in CSV_EXEMPTIONS:
            problems.append(f"CONFIDENTIAL DATA RISK: {posix}")

        if any(path.name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            problems.append(f"editor lock file: {posix}")

        if posix.startswith("data/") and path.name != ".gitkeep":
            problems.append(f"dataset directory content: {posix}")

    # Positive assertions. Their absence is what made the last archive look dishonest: the
    # documentation described CI, and the workflow file had been dropped by the packaging
    # step rather than being missing from the project.
    required = ["src/api/main.py", "README.md"]
    for name in required:
        if Path(name) not in paths:
            problems.append(f"MISSING required file: {name}")

    # The artifact is required to *ship* and impossible to have in CI.
    #
    # A real build must contain model.pkl - an archive that documents a champion and omits it
    # is the dishonest case this function exists to prevent. But `artifacts/*.pkl` is
    # gitignored, so a CI runner never has one, and demanding it there fails the data guard
    # for the one reason that has nothing to do with data.
    #
    # The two modes ask different questions. A real build asks "is this archive complete and
    # clean?"; --dry-run in CI asks "would this archive leak confidential content?" - a
    # question the model's absence cannot answer either way. Conflating them produced a guard
    # that could only pass on the maintainer's laptop, which is the same class of defect as a
    # test that only passes locally.
    if not dry_run and Path("artifacts/model.pkl") not in paths:
        problems.append(
            "MISSING required file: artifacts/model.pkl - a release archive that documents a "
            "champion model must contain it. Run the training command in the README first."
        )

    if not any(p.as_posix().startswith(".github/workflows/") for p in paths):
        problems.append(
            "MISSING .github/workflows/ - the documentation claims CI; shipping without "
            "the workflow makes that claim look false"
        )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist/campaign-group-predictor.zip")
    parser.add_argument("--dry-run", action="store_true", help="List and check, write nothing.")
    parser.add_argument("--manifest", default="dist/MANIFEST.txt")
    args = parser.parse_args()

    paths = collect()
    problems = guard(paths, dry_run=args.dry_run)

    total_bytes = sum((PROJECT_ROOT / p).stat().st_size for p in paths)
    print(f"{len(paths)} files, {total_bytes / 1_048_576:.1f} MiB uncompressed\n")

    largest = sorted(paths, key=lambda p: (PROJECT_ROOT / p).stat().st_size, reverse=True)[:8]
    print("Largest:")
    for path in largest:
        size = (PROJECT_ROOT / path).stat().st_size / 1_048_576
        print(f"  {size:7.1f} MiB  {path.as_posix()}")

    if problems:
        print("\nREFUSED - the archive was not written:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nGuards passed: no dataset, no lock files, CI workflow present.")

    if args.dry_run:
        print("Dry run - nothing written.")
        return 0

    output = PROJECT_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            arcname = f"campaign-group-predictor/{path.as_posix()}"
            archive.write(PROJECT_ROOT / path, arcname=arcname)

    manifest = PROJECT_ROOT / args.manifest
    manifest.write_text(
        "\n".join(p.as_posix() for p in paths) + "\n",
        encoding="utf-8",
    )

    compressed = output.stat().st_size / 1_048_576
    print(f"\nWrote {output}  ({compressed:.1f} MiB compressed)")
    print(f"Wrote {manifest}  ({len(paths)} entries)")

    if shutil.which("unzip"):
        print("\nVerify with:  unzip -l " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
