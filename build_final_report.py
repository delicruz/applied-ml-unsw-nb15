from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "deliverables" / "final_report"


def _has_modules(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


def _generate_figures() -> None:
    if _has_modules("matplotlib", "numpy"):
        from reporting.report_assets import generate_report_figures

        generate_report_figures()
        return

    project_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not project_python.exists():
        raise RuntimeError(
            "Figure generation requires matplotlib and numpy. Install requirements.txt "
            "or create the project .venv before building the report."
        )
    subprocess.run([str(project_python), str(ROOT / "generate_report_assets.py")], check=True)


def main() -> None:
    _generate_figures()
    if not _has_modules("docx", "reportlab"):
        raise RuntimeError(
            "DOCX/PDF construction requires python-docx and reportlab in the Python "
            "environment used to run build_final_report.py."
        )
    for builder in ("build_final_report_docx.py", "build_final_report_pdf.py"):
        subprocess.run([sys.executable, str(REPORT_DIR / builder)], check=True)


if __name__ == "__main__":
    main()
