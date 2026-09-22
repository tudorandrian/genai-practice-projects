"""synthetic_docs.py - generate a small private-document set for the RAG demo.

Writes three documents with known, specific facts, so retrieval and grounding can
be checked against expected answers (see ``eval_questions.md``):

  - acme_handbook.pdf    (company policy facts)    - exercises PyPDFLoader + page metadata
  - engineering_notes.md (technical stack facts)
  - support_faq.txt      (support/logistics facts)

The corpus is entirely invented for this repository and is never tracked by git
(``.gitignore``: ``projects/p11_rag_chatbot/data/*``); ``rag_chatbot.demo()`` and the
``rag``-marked tests regenerate it on demand. Only fpdf2 is needed for the PDF
(pure-Python, tiny), and it is imported lazily so this module stays importable
without the ``rag`` dependency group installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

HANDBOOK = [
    "ACME Robotics - Employee Handbook",
    "",
    "ACME Robotics was founded in 2014 in Cluj-Napoca, Romania.",
    "The chief executive officer is Ioana Popescu.",
    "The office is open Monday to Friday, from 9 AM to 6 PM.",
    "Every full-time employee receives 25 days of paid vacation per year.",
    "Remote work is allowed up to 3 days per week.",
    "The health insurance provider for all employees is Regina Maria.",
    "The guest WiFi password is acme-guest-2024.",
    "Expense reports must be submitted by the 5th day of each month.",
    "New employees complete a 90-day onboarding program.",
]

ENGINEERING = """# ACME Engineering Notes

## Backend
The backend service is written in Python using the FastAPI framework.
The primary database is PostgreSQL version 16.

## Infrastructure
We deploy the services using Docker containers orchestrated by Kubernetes.
The continuous integration pipeline runs on GitHub Actions.

## Process
Every pull request requires at least two approvals before it can be merged.
The public API enforces a rate limit of 100 requests per minute per user.
"""

FAQ = """ACME Support FAQ

To reset your password, visit the internal portal at portal.acme.local.
Parking is available in the underground garage on level -2.
The company all-hands meeting takes place on the last Friday of each month.
Support tickets are handled within 48 hours of being opened.
"""


def _write_pdf(lines: list[str], path: Path) -> None:
    from fpdf import FPDF  # lazy import, see module docstring

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in lines:
        # new_x/new_y reset the cursor to the left margin on the next line, else a
        # repeated multi_cell(0, ...) leaves x at the right edge and fpdf2 errors.
        pdf.multi_cell(0, 8, line if line else " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def write_all(folder: str | Path, *, overwrite: bool = False) -> dict[str, Path]:
    """Generate the synthetic document set into ``folder``. Returns {filename: path}.

    ``folder`` is normally ``data/``, which is also where a user's own documents
    go and which Git ignores, so an existing file is never replaced unless
    ``overwrite`` is True: with the default, a collision raises
    ``FileExistsError`` before anything is written. ``demo()`` passes
    ``overwrite=True`` for its own dedicated directory.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "acme_handbook.pdf": folder / "acme_handbook.pdf",
        "engineering_notes.md": folder / "engineering_notes.md",
        "support_faq.txt": folder / "support_faq.txt",
    }
    if not overwrite:
        existing = [name for name, path in paths.items() if path.exists()]
        if existing:
            raise FileExistsError(
                f"{', '.join(existing)} already exist(s) in the target folder; move or "
                "rename your file(s), or pass overwrite=True"
            )
    _write_pdf(HANDBOOK, paths["acme_handbook.pdf"])
    paths["engineering_notes.md"].write_text(ENGINEERING, encoding="utf-8")
    paths["support_faq.txt"].write_text(FAQ, encoding="utf-8")
    for name in paths:  # log the filename only - the path is a local absolute path
        log.info("write_all: wrote %s", name)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    write_all(DATA_DIR)
