# Slide storyline (titles only, read top to bottom)

1. *Title:* Squeezing a polymer liquid through a narrowing channel
2. The talk moves from the problem to three trusted references, then the network and its results
3. A polymer liquid remembers being stretched, and that changes the pressure a narrowing channel needs
4. Five balance equations describe the flow, scaled so that every term is of order one
5. The paper's formulas hold: SymPy confirms every one of them, apart from one sign typo
6. Yet the paper's own simulations, recovered from its PDF, show the formula fails above De ≈ 0.15
7. So we built a solver for any De: it solves the whole channel at once and matches within 0.37 %
8. A PINN is a network graded on how well it satisfies the equations, never on known answers
9. The walls, symmetry and flow rate are built into the network, so the loss holds only the equations
10. Two fixes found while building made the network accurate: uniform pressure and fair weighting
11. In the narrow-channel limit the PINN matches our solver within 0.22 % up to De = 0.3
12. It also matches the fields: 0.01–0.5 % error over most of the channel, more near the corners
13. Against the paper's 2-D simulations the PINN stays within 0.68 %; the formula misses by up to 26 %
14. The pressure drops because of the polymer shear-stress gradient, as the paper argued
15. One trained network answers any De instantly, and one command sets it up on Windows or Mac
16. So the PINN reproduces the paper beyond its formula's reach; its limits are the corners and De > 0.3
17. References
18. Thank you
- A1. The full problem: five scaled equations with exact wall, symmetry and inflow conditions
- A2. The full comparison: every network against its reference

Chain check: problem (3–4) → we need trusted answers (5) → but the formula is limited (6) →
so our own solver (7) → the network (8) → how it is built (9) → what made it work (10) →
it matches (11) → also in detail (12) → also in 2-D (13) → why the pressure drops (14) →
use it (15) → conclusions (16).

Parts (tracker): 1 Problem (3–4) · 2 References (5–7) · 3 Network (8–10) · 4 Results (11–14) · 5 Wrap-up (15–16).
Our choice: no separate checkpoint slides (the deck is short); the tracker on every slide shows the part.
