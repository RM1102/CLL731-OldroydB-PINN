# Storyline of the notebook (headings only; read top to bottom)

Title: **Squeezing a stretchy liquid through a narrowing channel: solving Boyko & Stone (2022) with a physics-informed neural network**

The problem, in brief
Order of the notebook

0. Set-up: run this cell first
1. The physics in plain words
   1.1 A polymer liquid remembers being stretched
   1.2 The channel, and the three numbers that control the flow
   1.3 The equations we solve
2. Three references to check the network against
   2.1 The paper's formulas, re-derived and checked
   2.2 The paper's simulations, recovered from its PDF
   2.3 Our own reference solver, and why marching fails
3. How the PINN works
   3.1 A network graded on the equations, not on answers
   3.2 Boundary conditions built into the network
   3.3 Training: two optimisers and what it costs
4. Results
   4.1 Sanity check: the Newtonian case (De = 0)
   4.2 Pressure drop falls as elasticity grows
   4.3 Where the polymer is stretched: field maps and errors
   4.4 Full 2-D flow against the paper's simulations
   4.5 Why the pressure drops: the shear-stress term does the work
   4.6 One network for every De: explore it yourself
5. What we learned, what is left
Appendix A: how to retrain everything
Appendix B: file map and references

Chain check (each heading follows from the one before):
physics → equations → we need trusted answers → three references → the network →
how it is trained → is it right for the easy case → main result (Δp) → fields →
the paper's harder 2-D cases → physical explanation → a tool → conclusions.
