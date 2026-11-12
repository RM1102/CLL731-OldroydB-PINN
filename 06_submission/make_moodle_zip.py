"""Assemble the small zip uploaded to Moodle (the code itself stays on GitHub).

  0_README_start_here.txt   from 06_submission/0_README_start_here.txt (Windows line endings)
  1_Summary_2_pages.pdf     from 06_submission/CLL731_term_paper_summary.pdf
  2_Presentation.pptx       from 06_submission/slides/CLL731_OldroydB_PINN_slides.pptx
  3_Full_report.html        from 05_notebook/CLL731_OldroydB_PINN.html

Run:  python make_moodle_zip.py   ->  CLL731_2022CH71476_Rahul_Masand.zip in the project folder
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "CLL731_2022CH71476_Rahul_Masand"
OUT = ROOT / f"{NAME}.zip"
FILES = {
    "1_Summary_2_pages.pdf": ROOT / "06_submission" / "CLL731_term_paper_summary.pdf",
    "2_Presentation.pptx": ROOT / "06_submission" / "slides" / "CLL731_OldroydB_PINN_slides.pptx",
    "3_Full_report.html": ROOT / "05_notebook" / "CLL731_OldroydB_PINN.html",
}
README = ROOT / "06_submission" / "0_README_start_here.txt"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    text = README.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", "\r\n")
    z.writestr(f"{NAME}/0_README_start_here.txt", text.encode("utf-8"))
    for arc, src in FILES.items():
        z.write(src, f"{NAME}/{arc}")
print(f"{OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
for info in zipfile.ZipFile(OUT).infolist():
    print(f"  {info.file_size / 1e6:6.2f} MB  {info.filename}")
