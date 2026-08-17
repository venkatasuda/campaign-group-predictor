"""Combine the project's Markdown documents into one editable Word file.

    python scripts/build_submission_docx.py
    python scripts/build_submission_docx.py --out reports/submission.docx

Why this exists
---------------
The written deliverables are Markdown because that is what lives well beside code and in a
diff. A reviewer reading on a laptop wants one document. This produces that document without
either format becoming the source of truth for the other: Markdown stays canonical, the
Word file is generated, and regenerating it is one command after any edit.

What it deliberately does NOT include
-------------------------------------
`YOUR_TASKS.md` is the assignment brief. It is marked LP-Internal and states that further
distribution and reproduction are prohibited, so it is excluded here and would be excluded
from any artefact leaving this machine. Reproducing a confidential brief inside a document
sent back to its author is a small thing that says something larger about how the rest of
the data was handled.

Requires: python-docx (already in requirements.txt).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Reading order for a reviewer, not filesystem order. The report answers the brief; the
#: model card and validation plan support it; architecture and walkthrough describe the
#: system; the README is reference. The peer-review summary is last because it is a record
#: of how the work was critiqued rather than part of the answer.
DOCUMENTS: list[tuple[str, str]] = [
    ("reports/REPORT.md", "Report"),
    ("docs/model_card.md", "Model Card and ADR-001"),
    ("docs/validation_plan.md", "Business Impact and Validation Plan"),
    ("docs/architecture.md", "Architecture"),
    ("docs/WALKTHROUGH.md", "Code Walkthrough"),
    ("README.md", "README"),
    ("terraform/README.md", "Infrastructure as Code"),
    ("reports/PEER_REVIEW_SUMMARY.md", "Peer Review — Request and Resolutions"),
]

CODE_FONT = "Consolas"
BODY_FONT = "Calibri"


def _add_runs(paragraph, text: str) -> None:
    """Render inline **bold**, *italic* and `code` inside a paragraph."""
    # One pass over the three inline forms, so nesting is not silently mangled.
    for part in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = CODE_FONT
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
        else:
            paragraph.add_run(part)


def _add_table(document: Document, rows: list[str]) -> None:
    """Render a Markdown pipe table, skipping the |---| separator row."""
    parsed = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    parsed = [r for r in parsed if not all(set(c) <= set("-: ") for c in r)]
    if not parsed:
        return

    width = max(len(r) for r in parsed)
    table = document.add_table(rows=len(parsed), cols=width)
    table.style = "Light Grid Accent 1"

    for i, row in enumerate(parsed):
        for j in range(width):
            cell = table.cell(i, j)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            _add_runs(paragraph, row[j] if j < len(row) else "")
            for run in paragraph.runs:
                run.font.size = Pt(9)
                if i == 0:
                    run.bold = True
    document.add_paragraph()


def _add_code_block(document: Document, lines: list[str]) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Pt(18)
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run("\n".join(lines))
    run.font.name = CODE_FONT
    run.font.size = Pt(9)


def convert(markdown: str, document: Document) -> None:
    """Append one Markdown document to an open Word document."""
    lines = markdown.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]

        # Fenced code block
        if line.strip().startswith("```"):
            block: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            _add_code_block(document, block)
            index += 1
            continue

        # Pipe table
        if line.strip().startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            _add_table(document, block)
            continue

        stripped = line.strip()

        if not stripped or stripped in {"---", "***", "___"}:
            index += 1
            continue

        # Headings
        heading = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading:
            level = min(len(heading.group(1)) + 1, 9)
            document.add_heading(heading.group(2).strip(), level=level)
            index += 1
            continue

        # Block quote - used throughout for "Finding" callouts
        if stripped.startswith(">"):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Pt(18)
            _add_runs(paragraph, stripped.lstrip("> ").strip())
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(0x33, 0x44, 0x55)
            index += 1
            continue

        # Lists
        bullet = re.match(r"^[-*+]\s+(.*)", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)", stripped)
        if bullet or numbered:
            style = "List Number" if numbered else "List Bullet"
            content = (numbered or bullet).group(1)
            _add_runs(document.add_paragraph(style=style), content)
            index += 1
            continue

        # Body text: join wrapped lines into one paragraph
        block = []
        while index < len(lines) and lines[index].strip() and not re.match(
            r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||```|---$)", lines[index]
        ):
            block.append(lines[index].strip())
            index += 1
        if block:
            _add_runs(document.add_paragraph(), " ".join(block))

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/submission.docx")
    args = parser.parse_args()

    document = Document()
    style = document.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(10.5)

    title = document.add_heading("Predicting Profitable Customer Groups", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph("Machine Learning and Engineering Challenge")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    notice = document.add_paragraph()
    notice.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = notice.add_run(
        "Confidential — LP-Internal. The dataset and the assignment brief are not "
        "reproduced in this document."
    )
    run.italic = True
    run.font.size = Pt(9)

    included, missing = [], []
    for relative_path, heading in DOCUMENTS:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            missing.append(relative_path)
            continue

        document.add_page_break()
        document.add_heading(heading, level=1)
        source = document.add_paragraph()
        source_run = source.add_run(f"source: {relative_path}")
        source_run.italic = True
        source_run.font.size = Pt(8)
        source_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

        convert(path.read_text(encoding="utf-8"), document)
        included.append(relative_path)

    output = PROJECT_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)

    print(f"Wrote {output}")
    print(f"Included {len(included)} document(s):")
    for item in included:
        print(f"  - {item}")
    if missing:
        print("Not found (skipped):")
        for item in missing:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
