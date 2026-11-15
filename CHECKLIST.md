# Checklist

Companion files: [DECISIONS.md](DECISIONS.md) (what was decided, why, when) · [CHANGELOG.md](CHANGELOG.md) (what changed, when).

Status marks: `[x]` done and verified (how, in brackets) · `[ ]` open · `[~]` in progress.

## Asks

- [x] 2026-11-01 Re-solve Boyko & Stone (2022) JFM 936 A23 with a PINN, as approved by the professor (validated: see 04_results/summary.md)
- [x] 2026-11-01 Go through the project, research it, survey GitHub repositories and tools, make a plan, build it, keep folders organised (01_research, 02_plan, 03_code … 06_submission exist; notebook runs)
- [x] 2026-11-01 A Jupyter notebook the professor can run, with all charts and code, very simple language and everyday examples (executed without errors; HTML copy made)
- [x] 2026-11-01 Keep the Mac awake while working, give status updates, look for speed-ups (caffeinate used; GPU lane added)
- [x] 2026-11-02 Use the laptop only until 8 PM IST; stop processes then (watcher with hard deadline)
- [x] 2026-11-03 Check Moodle readiness and guidelines; 3-line summary; a 1–2 page plain-language submission document with a code walkthrough (06_submission/CLL731_term_paper_summary.pdf)
- [x] 2026-11-03 Put everything on GitHub, runnable with one command on Windows and Mac, tested, with a long-runtime warning (GitHub Actions: Windows + macOS, Python 3.12 and 3.14 passed)
- [x] 2026-11-03 Explain how to use the trained PINN for prediction; a very simple GUI (gui.py, predict.py; self-test screenshot checked)
- [x] 2026-11-03 Repository visibility: public (Rahul chose "Public now")
- [x] 2026-11-03 Test on a free virtual Windows machine with no login (done on GitHub's Windows runner instead)
- [x] 2026-11-03 Storylined slides, detailed but short and easy to read (20 slides, rendered and checked)
- [x] 2026-11-03 Push to GitHub (origin/main up to date)
- [x] 2026-11-12 Moodle zip with a README on deploying from GitHub and using the GUI, a two-page document and the slides (CLL731_2022CH71476_Rahul_Masand.zip, contents listed and opened)
- [x] 2026-11-15 Draft the final submission to the professor's instructions: 1. journal article, 2. report, 3. codes; separate or in one zip (CLL731_2022CH71476_Rahul_Masand_submission.zip, 24.9 MB; unzipped copy passed 7 tests, predictor and figure script)
- [x] 2026-11-15 Report as one PDF (LaTeX) with: Title; Author name and Entry number; Abstract; Introduction (motivation and literature survey, own words, cited); Problem formulation (numbered equations, boundary conditions, geometry sketch); Numerical analysis (methods, codes, all parameter values with units); Results and discussion (only our results, each figure discussed); Path forward; Conclusion; Self-assessment (≥ 120 words, compulsory paragraph ≥ 150 words); References (one consistent format) (06_submission/report/CLL731_report_Rahul_Masand.pdf, 16 pages; every page rendered and checked; self-assessment 374 words, main paragraph 256)
- [x] 2026-11-15 No result or plot from the journal article in the Results section (plagiarism rule) (all 8 figures made by make_report_figures.py from our runs; FE values mentioned in text only, cited)
- [~] 2026-11-15 If the zip is over the size limit: article and report on Moodle, everything by email or Google Drive before the deadline (15 Nov EoD) (fallback CLL731_2022CH71476_Rahul_Masand_article_report.zip, 4.9 MB, ready; the upload and any email/Drive send are Rahul's to do)
- [x] 2026-11-15 "Compact your context" (cannot be triggered by Claude; context compaction happened automatically)
- [ ] 2026-11-15 Commit and push the report and builder to GitHub (waiting for Rahul's go-ahead)

## Feedback

- 2026-11-01 Use very simple, clear language and everyday examples for the professor.
- 2026-11-02 Respect laptop time limits (e.g. stop by 8 PM IST).
- 2026-11-03 Ask for permission more clearly when it is needed, and open things for Rahul.
- 2026-11-03 Use the storylining skill for slides.
- 2026-11-12 Ground rules (CLAUDE.md): keep this checklist, DECISIONS.md and CHANGELOG.md; research before building; short answers; report disk use; one heavy job at a time; verify before "done"; never touch Rahul's own data.
