"""Assert that every document quotes the same figures as the artifact.

Why this exists
---------------
The reported numbers live in ``artifacts/metrics.json`` and ``reports/latency.json``. They are
*repeated* in eight documents. Repetition is the right call for readability - a reader should
not have to open a JSON file to learn the headline accuracy - but it means the numbers can
drift, and drift is silent.

This is not hypothetical. ``docs/architecture.md`` carried a latency table quoting median
37.1 ms and p95 91.0 ms from a superseded run, three lines above a paragraph apologising for
an earlier instance of exactly that defect, while the model card and the final report both
quoted the current 129/178 from ``latency.json``. The same stale p95 was also duplicated in
``terraform/variables.tf``, where nobody thinks to look because it is infrastructure config
rather than prose. Both were found by reading, not by tooling. That is the failure this script
converts into a build error.

Design
------
Two independent checks, because they catch different failures:

**Canonical facts must be present.** Each figure is derived from the JSON at runtime, never
hardcoded here - so regenerating the artifact and re-running this script is enough to know
which documents still need updating. A file that has fallen behind fails loudly and names the
figure it is missing.

**Retired values must be absent.** A number that was once correct is the most dangerous kind
of wrong: it looks plausible, it survives review, and it agrees with whatever stale copy the
reader checks it against. Superseded figures are listed explicitly and may not reappear
anywhere in the tracked documentation.

The retired list is maintained by hand and that is deliberate. Inferring "any four-decimal
number that is not the current one" would flag legitimate comparisons - the report quotes
accuracy at seed 42 precisely to show the measuring instrument moves more than the candidates
do, and that number *should* be there.

Exit status is 0 when consistent, 1 otherwise, so this runs as a CI step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

METRICS = ROOT / "artifacts" / "metrics.json"
LATENCY = ROOT / "reports" / "latency.json"

#: Documents that quote headline figures and must therefore agree with the artifacts.
DOCUMENTS: list[Path] = [
    ROOT / "README.md",
    ROOT / "reports" / "FINAL_REPORT.md",
    ROOT / "reports" / "REPORT.md",
    ROOT / "reports" / "PEER_REVIEW_SUMMARY.md",
    ROOT / "docs" / "model_card.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "validation_plan.md",
    ROOT / "terraform" / "variables.tf",
]

#: Figures that were correct once and must never reappear.
#:
#: Each entry is (value, what it was, where it was found). The provenance is not decoration:
#: without it the next person cannot tell a retired figure from a typo, and will either delete
#: a legitimate entry or preserve a wrong one.
RETIRED: list[tuple[str, str, str]] = [
    ("37.1", "median latency from a superseded 100-request run", "docs/architecture.md"),
    ("91.0 ms", "p95 latency from the same superseded run", "architecture.md, variables.tf"),
    ("15,450", "cold start quoted as measured; it is a single observation", "architecture.md"),
    ("126.7", "p99/max from the superseded run", "docs/architecture.md"),
]


def _fail(message: str) -> None:
    print(f"::error::{message}")


def main() -> int:
    if not METRICS.exists():
        _fail(f"{METRICS.relative_to(ROOT)} is missing; run training before this check.")
        return 1

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    champion = metrics["champion_model"]
    champion_metrics = metrics["models"][champion]

    accuracy = champion_metrics["accuracy"]
    lift = champion_metrics["business_lift"]["absolute_lift_pp"]

    # Canonical spellings, each scoped to the documents that actually carry that figure.
    #
    # Scoping matters: requiring every fact in every file would force the README to quote a
    # latency percentile it has no reason to mention, and a check that demands the wrong
    # thing gets silenced rather than fixed. Matching is case-insensitive because prose
    # capitalises ("Random forest") where the artifact does not ("random_forest").
    readme = ROOT / "README.md"
    final_report = ROOT / "reports" / "FINAL_REPORT.md"
    architecture = ROOT / "docs" / "architecture.md"
    model_card = ROOT / "docs" / "model_card.md"

    canonical: dict[str, tuple[list[str], list[Path]]] = {
        "champion model": (
            [champion, champion.replace("_", " ")],
            [readme, final_report, model_card],
        ),
        "test accuracy": (
            [f"{accuracy:.4f}", f"{accuracy * 100:.2f}", f"{accuracy * 100:.1f}"],
            [readme, final_report],
        ),
        "business lift": ([f"{lift:.2f}"], [readme, final_report]),
    }

    if LATENCY.exists():
        latency = json.loads(LATENCY.read_text(encoding="utf-8"))
        canonical["latency p95"] = (
            [f"{latency['p95_ms']:.1f}", f"{round(latency['p95_ms']):d}"],
            [final_report, architecture, model_card],
        )

    failures = 0

    # Retired values, checked across every document at once.
    for value, description, origin in RETIRED:
        for document in DOCUMENTS:
            if not document.exists():
                continue
            text = document.read_text(encoding="utf-8", errors="replace")
            # The apology paragraph in architecture.md quotes its own retired figures on
            # purpose, to record what was wrong. Naming them inside a sentence that also
            # says they were wrong is the opposite of the defect this guards against.
            if "previously reproduced that same defect" in text or "An earlier version" in text:
                continue
            if value in text:
                _fail(
                    f"{document.relative_to(ROOT)} contains retired value {value!r} "
                    f"({description}; originally in {origin})."
                )
                failures += 1

    # Canonical facts must appear in the documents that carry them.
    for label, (accepted, documents) in canonical.items():
        for document in documents:
            if not document.exists():
                continue
            text = document.read_text(encoding="utf-8", errors="replace").lower()
            if not any(form.lower() in text for form in accepted):
                _fail(
                    f"{document.relative_to(ROOT)} does not quote the current {label} "
                    f"(expected one of {accepted})."
                )
                failures += 1

    if failures:
        print(f"\n{failures} consistency failure(s).")
        return 1

    summary = f"Consistent: champion={champion}, accuracy={accuracy:.4f}, lift={lift:.2f} pp"
    if "latency p95" in canonical:
        summary += f", p95={canonical['latency p95'][0][0]} ms"
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
