"""Build the final CLL731 submission (professor's three items).

  <name>_submission.zip         1. journal article, 2. report (PDF), 3. codes, README
  <name>_article_report.zip     items 1 and 2 only (for Moodle if the full zip is too big)

The codes are every file git tracks or would track (git ls-files, so .venv,
caches and the old zips are left out), including the trained networks.

Run from anywhere:  python 06_submission/make_final_submission.py
Needs: the report PDF (compile 06_submission/report/*.tex with LaTeX first) and
the article PDF in 01_research/papers/ (see the README there).
"""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "CLL731_2022CH71476_Rahul_Masand"
ARTICLE = next((ROOT / "01_research" / "papers").glob("Boyko_Stone_2022*.pdf"))
REPORT = ROOT / "06_submission" / "report" / f"CLL731_report_Rahul_Masand.pdf"

README = """CLL731 Advanced Transport Phenomena - term paper (ML)
Rahul Arvind Masand, entry number 2022CH71476

Topic: pressure-driven flow of the viscoelastic Oldroyd-B fluid in a narrow
hyperbolic contraction (Boyko & Stone 2022, J. Fluid Mech. 936, A23), solved
again with a physics-informed neural network (PINN).

Contents
  1_Journal_article_Boyko_Stone_2022.pdf   the article (arXiv version, 2109.06928)
  2_Report_2022CH71476_Rahul_Masand.pdf    the report
  3_Codes/                                 all code, trained networks and results
                                           (same as https://github.com/RM1102/CLL731-OldroydB-PINN)

Running the code (needs Python 3.10-3.14, no GPU)
  Windows : double-click 3_Codes/setup_and_run.bat
  Mac     : right-click 3_Codes/setup_and_run_mac.command -> Open
  Linux   : bash 3_Codes/setup_and_run.sh
The first run downloads about 1 GB of packages (5-15 minutes) into a private
folder (3_Codes/.venv); then a menu offers: 1 explorer window, 2 notebook,
3 quick check (~10 s), 4 full retraining (about 4 hours; not needed, the
trained networks are included). Details: 3_Codes/README.md.

The report figures are rebuilt by 3_Codes/06_submission/report/make_report_figures.py
(about 30 s, from the saved results).
"""


def code_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return sorted(p for p in out.split("\0") if p and (ROOT / p).is_file())


def main():
    if not REPORT.exists():
        raise SystemExit(f"compile the report first: {REPORT.relative_to(ROOT)}")
    full = ROOT / f"{NAME}_submission.zip"
    small = ROOT / f"{NAME}_article_report.zip"
    files = code_files()
    with zipfile.ZipFile(full, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(f"{NAME}/0_README.txt", README)
        z.write(ARTICLE, f"{NAME}/1_Journal_article_Boyko_Stone_2022.pdf")
        z.write(REPORT, f"{NAME}/2_Report_2022CH71476_Rahul_Masand.pdf")
        for p in files:
            z.write(ROOT / p, f"{NAME}/3_Codes/{p}")
    with zipfile.ZipFile(small, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(f"{NAME}/0_README.txt", README.replace(
            "all code, trained networks and results",
            "NOT in this file (size limit): see the GitHub link, or the full zip sent by email/Drive"))
        z.write(ARTICLE, f"{NAME}/1_Journal_article_Boyko_Stone_2022.pdf")
        z.write(REPORT, f"{NAME}/2_Report_2022CH71476_Rahul_Masand.pdf")
    for f in (full, small):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")
    print(f"{len(files)} code files")


if __name__ == "__main__":
    main()
