# Decisions

What was decided, why, and when. The full design-change log for the code is in `02_plan/project_plan.md` (section 9).

| Date | Decision | Why |
|---|---|---|
| 2026-11-01 | Paper: Boyko & Stone (2022), JFM 936 A23; physics kept to confined viscoelastic channel flow | approved by the professor; must not overlap Rahul's MTP (thin films) |
| 2026-11-01 | PINN with hard constraints (stream function on η = Y/H), pressure split P = N_P(Z) + ε²N_P1 | walls, symmetry and flow rate hold exactly; removed a 0.26 % pressure error |
| 2026-11-01 | Independent reference solver: one global Newton solve, not marching | marching down the channel is ill-posed for De > 0 |
| 2026-11-01 | Recover the paper's FE points from the PDF's vector drawing | the numbers were never published; needed for validation |
| 2026-11-03 | Public GitHub repository | Rahul chose "Public now" |
| 2026-11-03 | Windows/Mac testing on GitHub Actions runners | free, no login, clean machines |
| 2026-11-12 | Moodle zip = README + 2-page summary + slides + HTML; code on GitHub | small upload; code stays versioned |
| 2026-11-15 | Report plots show only output of our code: PINN, our reference solver, and our own evaluation of the perturbation formulas we re-derived with SymPy (labelled as such). The paper's FE points are **not** plotted; agreement with them is stated in the text with a citation | professor: results from the article in Results count as plagiarism |
| 2026-11-15 | Report written in LaTeX, compiled with tectonic; figures made by `06_submission/report/make_report_figures.py` from saved results | the professor advised LaTeX; reproducible |
| 2026-11-15 | Submission zip = article PDF + report PDF + codes (a `git archive` of main, with trained networks) + README; a smaller article + report file kept ready in case Moodle rejects the size | follows the instructions' fallback rule |
| 2026-11-15 | Self-assessment claims Level 1 (reproduction) in full; Level 2 contributions described as methodological (new solution method, solver finding, typo, O(ε²) correction), not as new physics | honesty, as the instructions require |
