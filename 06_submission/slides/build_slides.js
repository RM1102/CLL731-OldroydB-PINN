// Build the CLL731 term-paper deck (follows the deck-storylining rules).
//
// Run from this folder:  NODE_PATH=<folder with pptxgenjs> node build_slides.js
// Figures come from make_slide_figures.py (assets/) and 04_results/figures/.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..", "..");
const A = (f) => path.join(__dirname, "assets", f);
const OUT = path.join(__dirname, "CLL731_OldroydB_PINN_slides.pptx");

const THEME = {
  name: "CLL731 PINN",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F24", lt1: "FFFFFF", dk2: "0B4F57", lt2: "E8F3F3",
    accent1: "0B6E79", accent2: "2A78D6", accent3: "EB6834", accent4: "1BAF7A",
    accent5: "52514E", accent6: "E34948", hlink: "0B6E79", folHlink: "6250D6",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.author = "Rahul Arvind Masand";
pres.title = "Oldroyd-B flow in a narrowing channel, re-solved with a PINN";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const GREY = C.accent5;

// ---------------------------------------------------------------- layouts
const FOOT = "Rahul Arvind Masand  ·  CLL731 term paper  ·  IIT Delhi";
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.5, y: 0.2, w: 6.5, h: 0.5,
        fontSize: 24, color: C.accent1, valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.74, w: 12.33, h: 1.0,
        fontSize: 30, bold: true, color: C.text1, valign: "top", align: "left", margin: 0,
        lineSpacingMultiple: 1.0 },
        text: "" } },
    { text: { text: FOOT, options: { x: 0.5, y: 7.07, w: 8, h: 0.3, fontSize: 9, color: GREY, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.07, w: 0.55, h: 0.3, fontSize: 9, color: GREY, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: C.text2 } });

// ---------------------------------------------------------------- helpers
const PARTS = ["1. Problem", "2. References", "3. Network", "4. Results", "5. Wrap-up"];

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function tracker(slide, part) {
  const w = 1.08, gap = 0.07, x0 = 12.83 - PARTS.length * w - (PARTS.length - 1) * gap;
  PARTS.forEach((p, i) => {
    const on = i === part;
    slide.addText(p, {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x: x0 + i * (w + gap), y: 0.28,
      w, h: 0.34, fontSize: 10, bold: on, align: "center", valign: "middle", margin: 0,
      color: on ? C.background1 : GREY, fill: { color: on ? C.accent1 : C.background1 },
      line: { color: on ? C.accent1 : "C9CED0", width: 0.75 }, isTextBox: true, objectName: "tracker",
    });
  });
}

function ideaBox(slide, text) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.8, w: 12.33, h: 0.74,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "idea_fill" });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.8, w: 0.07, h: 0.74,
    fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "idea_bar" });
  slide.addText(text, { x: 0.72, y: 1.8, w: 12.0, h: 0.74, fontSize: 18, color: C.text1,
    valign: "middle", margin: 0, isTextBox: true, objectName: "idea_text" });
}

// bullets: strings, or ["sub", text] for an en-dash sub-point under the previous bullet
function bullets(slide, items, opt = {}) {
  const runs = items.map((it, i) => {
    const sub = Array.isArray(it);
    return { text: sub ? it[1] : it, options: {
      bullet: sub ? { code: "2013", indent: 14 } : { indent: 16 }, indentLevel: sub ? 1 : 0,
      fontSize: sub ? 14 : 16, color: sub ? GREY : C.text1,
      paraSpaceAfter: opt.space ?? 12, breakLine: i < items.length - 1 } };
  });
  slide.addText(runs, { x: 0.5, y: opt.y ?? 2.72, w: opt.w ?? 5.3, h: opt.h ?? 4.15,
    valign: "top", margin: 0, isTextBox: true, objectName: "left_text" });
}

function caption(slide, label, what, takeaway, box) {
  slide.addText([
    { text: label + ": ", options: { bold: true } },
    { text: what + " " },
    { text: "Takeaway: ", options: { bold: true, color: C.accent1 } },
    { text: takeaway },
  ], { x: box.x, y: box.y, w: box.w, h: box.h, fontSize: 11, color: GREY, valign: "top",
       margin: 0, isTextBox: true, objectName: "caption" });
}

// picture fitted into a box (keeps its shape), caption underneath
function figure(slide, file, label, what, takeaway, box = { x: 6.15, y: 2.72, w: 6.68, h: 4.15 },
                capH = 0.55) {
  const s = pngSize(file);
  const maxW = box.w, maxH = box.h - capH - 0.06;
  let w = maxW, h = (w * s.h) / s.w;
  if (h > maxH) { h = maxH; w = (h * s.w) / s.h; }
  const x = box.x + (box.w - w) / 2;
  slide.addImage({ path: file, x, y: box.y, w, h, objectName: "figure",
                   altText: `${label}: ${what}` });
  caption(slide, label, what, takeaway, { x: box.x, y: box.y + h + 0.06, w: box.w, h: capH });
  return box.y + h + 0.06 + capH;
}

function table(slide, label, what, takeaway, rows, opt) {
  caption(slide, label, what, takeaway, { x: opt.x, y: opt.y, w: opt.w, h: opt.capH ?? 0.5 });
  const head = rows[0].map((t) => ({ text: t, options: { bold: true, color: C.background1,
                                                        fill: { color: C.accent1 } } }));
  slide.addTable([head, ...rows.slice(1)], {
    x: opt.x, y: opt.y + (opt.capH ?? 0.5) + 0.05, w: opt.w, colW: opt.colW,
    fontSize: opt.fontSize ?? 14, color: C.text1, valign: "middle", margin: 0.05,
    border: { type: "solid", pt: 0.75, color: "C9CED0" }, rowH: opt.rowH ?? 0.36,
    objectName: "table",
  });
}

function content(part, subtitle, title, idea, notes, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(subtitle, { placeholder: "subtitle" });
  s.addText(title, { placeholder: "title" });
  tracker(s, part);
  if (idea) ideaBox(s, idea);
  if (notes) s.addNotes(notes);
  return s;
}

// ================================================================ slides
// ---- 1. title
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  s.addText("Squeezing a polymer liquid through a narrowing channel", { x: 0.8, y: 1.6, w: 11.7,
    h: 1.5, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1,
    valign: "bottom", margin: 0, isTextBox: true });
  s.addText("Re-solving Boyko & Stone (2022, J. Fluid Mech. 936, A23) with a physics-informed neural network",
    { x: 0.8, y: 3.25, w: 11.7, h: 0.9, fontSize: 22, color: "CFE6E8", valign: "top", margin: 0,
      isTextBox: true });
  s.addText([
    { text: "Rahul Arvind Masand (2022CH71476)", options: { bold: true, breakLine: true } },
    { text: "CLL731 Advanced Transport Phenomena · term paper (ML) · IIT Delhi · November 2026",
      options: { breakLine: true } },
    { text: "Code, report and trained networks: github.com/RM1102/CLL731-OldroydB-PINN" },
  ], { x: 0.8, y: 4.9, w: 11.7, h: 1.2, fontSize: 16, color: C.background1, margin: 0,
       paraSpaceAfter: 4, isTextBox: true });
  s.addNotes("This term paper takes a Journal of Fluid Mechanics paper on polymer flow through a " +
    "narrowing channel and solves the same problem with a physics-informed neural network.");
}

// ---- 2. roadmap
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  s.addText("Roadmap", { placeholder: "subtitle" });
  s.addText("The talk moves from the problem to three trusted references, then the network and its results",
    { placeholder: "title" });
  ideaBox(s, "The question is whether a network trained only on the equations can reproduce the paper, " +
    "so we first build answers we can trust.");
  const cards = [
    ["Problem", "Polymer memory", "The equations"],
    ["References", "The paper's formulas", "Its simulations", "Our own solver"],
    ["Network", "The idea", "Built-in conditions", "Two fixes"],
    ["Results", "Narrow channel", "Whole field", "Full 2-D", "The mechanism"],
    ["Wrap-up", "Use it yourself", "Conclusions"],
  ];
  const w = 2.3, gap = 0.2;
  cards.forEach((c, i) => {
    const x = 0.5 + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.95, w, h: 3.0, rectRadius: 0.08,
      fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addText(String(i + 1), { shape: pres.shapes.OVAL, x: x + 0.2, y: 3.15, w: 0.55, h: 0.55,
      fill: { color: C.accent1 }, color: C.background1, fontSize: 18, bold: true, align: "center",
      valign: "middle", margin: 0, isTextBox: true });
    s.addText(c[0], { x: x + 0.2, y: 3.85, w: w - 0.4, h: 0.45, fontSize: 20, bold: true,
      color: C.text2, margin: 0, isTextBox: true });
    s.addText(c.slice(1).map((t, j) => ({ text: t, options: { bullet: { indent: 12 },
      breakLine: j < c.length - 2, paraSpaceAfter: 6 } })),
      { x: x + 0.2, y: 4.4, w: w - 0.35, h: 2.0, fontSize: 14, color: C.text1, valign: "top",
        margin: 0, isTextBox: true });
  });
  s.addNotes("Five parts. The references come before the network on purpose: a network's loss can " +
    "look small while its answer is wrong, so we need trusted answers to compare against.");
}

// ---- 3. problem: polymer memory
pres.addSection({ title: "1. Problem" });
{
  const s = content(0, "Problem (1): polymer memory",
    "A polymer liquid remembers being stretched, and that changes the pressure a narrowing channel needs",
    "Think of honey filled with tiny rubber bands: squeezed through a funnel, they stretch along the flow " +
    "and pull back on it.",
    "Everyday picture first: honey with rubber bands. The Oldroyd-B model is the simplest model with " +
    "constant viscosity and elasticity, good for Boger fluids. The question is the pressure drop.",
    "1. Problem");
  bullets(s, [
    "Boyko & Stone (2022) push a polymer solution at a fixed flow rate through a 4 : 1 contraction.",
    "The Oldroyd-B model treats each molecule as two beads on a spring; A stores their average stretch.",
    "The springs relax over a time λ, so the liquid keeps a memory of earlier stretching.",
    "Question: how does this memory change the pressure drop ΔP across the contraction?",
    "Their answer: elasticity lowers ΔP, and it keeps falling as the memory time grows.",
  ]);
  figure(s, A("s_channel.png"), "Figure 1",
    "The channel in the paper's scaled units, with straight inlet and outlet sections (source: Boyko & Stone 2022).",
    "the half-height falls four-fold, so the flow speeds up and stretches the molecules.");
}

// ---- 4. problem: equations
{
  const s = content(0, "Problem (2): the equations",
    "Five balance equations describe the flow, scaled so that every term is of order one",
    "In the paper's scaled units every term is of order one, so three numbers alone set the flow (Table 1).",
    "The paper scales every quantity by its natural size in a narrow channel, so all terms are order " +
    "one. That is exactly what a neural network needs too.",
    "1. Problem");
  bullets(s, [
    "Recall (slide 3): A is the molecular stretch; U is velocity, P pressure, T = βA polymer stress.",
    "Mass and two force balances fix the velocity and pressure, as in any slow flow.",
    "The stretch equations add memory: A is carried with the flow, stretched, and relaxes back.",
    "Setting ε = 0 gives the narrow-channel limit; keeping ε gives the full 2-D flow.",
    "So the question becomes: how does ΔP change as De grows from 0 to 0.3?",
  ], { space: 16 });
  const yEq = figure(s, A("eq_governing.png"), "Figure 2",
    "The scaled equations; ε² terms shortened (source: Boyko & Stone 2022, eq. 2.14).",
    "pressure is nearly uniform across the channel.", { x: 6.15, y: 2.72, w: 6.68, h: 2.5 }, 0.32);
  table(s, "Table 1", "The three controlling numbers (source: Boyko & Stone 2022).",
    "only De varies along each curve.", [
      ["number", "meaning", "values"],
      ["ε", "outlet half-height ÷ channel length", "0 (limit), 0.02, 0.1"],
      ["De", "memory time ÷ time to cross the contraction", "0 to 0.3"],
      ["β", "polymer share of the viscosity", "0.4"],
    ], { x: 6.15, y: yEq + 0.15, w: 6.68, colW: [0.9, 3.9, 1.88], capH: 0.42, rowH: 0.31 });
}

// ---- 5. references: formulas
pres.addSection({ title: "2. References" });
{
  const s = content(1, "References (1): the formulas",
    "The paper's formulas hold: SymPy confirms every one of them, apart from one sign typo",
    "Before testing a network we need answers we trust; the first is the paper's own low-De series for ΔP.",
    "SymPy is a free computer-algebra library. We re-derived the expansion for a general channel shape " +
    "and computed delta P3 straight from the third-order force balance, without the reciprocal theorem.",
    "2. References");
  bullets(s, [
    "The paper expands the flow in powers of De, like a Taylor series, up to De³.",
    "We re-derived every order with SymPy (a computer-algebra library) for any channel shape.",
    "All fields and pressure terms match, including ΔP₃ = −14.661, computed here independently.",
    "One printed sign in eq. (3.16b) is wrong; the corrected sign reproduces the paper's figure 5.",
    "So the series is right, but it is still a series cut off after De³.",
  ], { space: 18 });
  const yEnd = figure(s, A("eq_series.png"), "Figure 3",
    "The paper's series for this channel (α = 4, β = 0.4) (source: Boyko & Stone 2022, checked in this work).",
    "the coefficients alternate in sign and grow about threefold per order.",
    { x: 6.15, y: 2.85, w: 6.4, h: 1.75 }, 0.5);
  figure(s, A("eq_typo.png"), "Figure 4",
    "The one sign correction, eq. (3.16b) (source: this work).",
    "only the sign changes; every other formula stands.",
    { x: 6.15, y: yEnd + 0.35, w: 5.6, h: 1.75 }, 0.32);
}

// ---- 6. references: simulations
{
  const s = content(1, "References (2): the simulations",
    "Yet the paper's own simulations, recovered from its PDF, show the formula fails above De ≈ 0.15",
    "The series terms alternate in sign and grow about threefold per order, so it can only be trusted for small De.",
    "The finite-element data were never published. The arXiv figures are vector drawings, so the exact " +
    "marker coordinates sit in the file. We calibrated the axes on the formula curves, which we know exactly.",
    "2. References");
  bullets(s, [
    "The paper's finite-element (FE) simulation data were never published as numbers.",
    "But its figures are vector drawings, so each data marker is stored as exact coordinates.",
    "We read them out and calibrated the axes on the paper's own formula curves.",
    ["sub", "calibration error 3 × 10⁻⁶; a straight channel gives ΔP = 2.9999 against an exact 3"],
    "The recovered points follow the formula only up to about De = 0.15.",
    "So beyond that we need a second numerical reference that we control.",
  ], { space: 10 });
  figure(s, A("s_formula_vs_fe.png"), "Figure 5",
    "Pressure drop against De in the narrow channel ε = 0.02 (source: data recovered from Boyko & Stone 2022).",
    "the 3rd-order formula leaves the simulations above De ≈ 0.15.");
}

// ---- 7. references: our solver
{
  const s = content(1, "References (3): our solver",
    "So we built a solver for any De: it solves the whole channel at once and matches within 0.37 %",
    "Stretched molecules change the velocity, and the velocity changes the stretch, in both directions " +
    "along the channel.",
    "Marching downstream station by station looks natural, because molecules carry their stretch " +
    "downstream. It blew up for every De above zero. The fix is a global Newton solve.",
    "2. References");
  bullets(s, [
    "Our solver works in the narrow-channel limit (ε = 0) and at any De.",
    "Marching from inlet to outlet blew up for every De > 0, and sooner on finer grids.",
    ["sub", "the feedback acts like heat conduction along the channel, so the outlet matters too"],
    "So all positions are solved together with Newton's method, in about 5 seconds per De.",
    "Finer grids, rounder corners and a longer outlet change ΔP by less than 0.03 %.",
    "Against the recovered simulations it agrees within 0.37 % for every De up to 0.3.",
  ], { space: 10 });
  figure(s, A("s_ref_vs_fe.png"), "Figure 6",
    "Our solver (orange) against the paper's simulations at ε = 0.02 (source: this work).",
    "two independent calculations agree within 0.37 % over the whole range.");
}

// ---- 8. network: idea
pres.addSection({ title: "3. Network" });
{
  const s = content(2, "Network (1): the idea",
    "A PINN is a network graded on how well it satisfies the equations, never on known answers",
    "Think of a student never shown the answer key, only told how badly each equation is violated at " +
    "random points.",
    "PINN stands for physics-informed neural network (Raissi, Perdikaris and Karniadakis 2019). " +
    "Automatic differentiation applies the chain rule to the network formula itself.",
    "3. Network");
  bullets(s, [
    "A neural network is a flexible formula with tens of thousands of adjustable numbers (weights).",
    "Ours takes a position (Z, Y) and returns stream function, pressure and stretch there.",
    "Automatic differentiation gives every derivative of that formula exactly, without a grid.",
    "The loss is the mean squared error of the equations at thousands of random points.",
    "Training lowers the loss; no simulation result is ever shown to the network.",
  ]);
  // flow chart (native shapes)
  const steps = ["position (Z, Y)", "network: about 64 000 weights",
    "stream function Ψ, pressure P, stretch A", "exact derivatives (automatic differentiation)",
    "loss = mean squared error of the equations"];
  const bx = 6.45, bw = 4.4, bh = 0.5, gap = 0.22;
  steps.forEach((t, i) => {
    const y = 2.75 + i * (bh + gap);
    s.addText(t, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: bx, y, w: bw, h: bh,
      fill: { color: i === 4 ? C.accent1 : C.background2 }, line: { color: C.accent1, width: 1 },
      color: i === 4 ? C.background1 : C.text1, fontSize: 14, align: "center", valign: "middle",
      margin: 0, isTextBox: true, objectName: "figure" });
    if (i < steps.length - 1)
      s.addShape(pres.shapes.LINE, { x: bx + bw / 2, y: y + bh, w: 0, h: gap,
        line: { color: C.accent1, width: 1.5, endArrowType: "triangle" } });
  });
  // loop back: optimiser adjusts the weights
  const yTop = 2.75 + bh + gap + bh / 2, yBot = 2.75 + 4 * (bh + gap) + bh / 2;
  s.addShape(pres.shapes.LINE, { x: bx + bw, y: yBot, w: 0.45, h: 0, line: { color: C.accent3, width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: bx + bw + 0.45, y: yTop, w: 0, h: yBot - yTop,
    line: { color: C.accent3, width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: bx + bw, y: yTop, w: 0.45, h: 0, flipH: true,
    line: { color: C.accent3, width: 1.5, endArrowType: "triangle" } });
  s.addText("optimiser adjusts the weights; repeat", { x: bx + bw + 0.55, y: (yTop + yBot) / 2 - 0.5,
    w: 1.45, h: 1.0, fontSize: 12, color: C.accent3, align: "left", valign: "middle", margin: 0,
    isTextBox: true });
  caption(s, "Figure 7", "How a PINN is trained (source: this work, after Raissi et al. 2019).",
    "the only teacher is the equations.", { x: 6.15, y: 6.35, w: 6.68, h: 0.45 });
}

// ---- 9. network: built-in conditions
{
  const s = content(2, "Network (2): built-in conditions",
    "The walls, symmetry and flow rate are built into the network, so the loss holds only the equations",
    "The network does not output velocity directly; it outputs a stream function Ψ, whose slopes give a " +
    "velocity that conserves mass.",
    "Lines of constant Psi are the paths the liquid follows. Psi = 0 on the centreline and 1 on the wall " +
    "fixes the flow rate at one. Table 1 was computed with deliberately scrambled weights.",
    "3. Network");
  bullets(s, [
    "η = Y/H(Z) runs from 0 on the centreline to 1 on the wall, whatever the channel shape.",
    "The fixed part of Ψ meets no slip and flow rate 1; the network's term vanishes at the wall.",
    "Inputs use η², so every field is automatically mirror-symmetric about the centreline.",
    "Pressure is split so that it is exactly uniform across the channel when ε = 0.",
    "So the loss contains only the equations, and no loss weights need tuning.",
  ], { space: 16 });
  const yEnd = figure(s, A("eq_psi.png"), "Figure 8",
    "How the network outputs N become physical fields (source: this work).",
    "the conditions hold for any weights.", { x: 6.15, y: 2.72, w: 6.4, h: 2.15 }, 0.32);
  table(s, "Table 2", "Checks with random, untrained weights (source: this work).",
    "exact to round-off.", [
      ["position Z", "flow rate", "|U_z| on wall", "|U_y| on centreline"],
      ["0 (inlet corner)", "1.000000000000", "0", "0"],
      ["0.37", "1.000000000000", "0", "0"],
      ["1 (outlet corner)", "1.000000000000", "0", "0"],
    ], { x: 6.15, y: yEnd + 0.15, w: 6.68, colW: [1.7, 1.9, 1.5, 1.58], capH: 0.3, rowH: 0.3 });
}

// ---- 10. network: two fixes
{
  const s = content(2, "Network (3): two fixes, training",
    "Two fixes found while building made the network accurate: uniform pressure and fair weighting",
    "A small training loss is not proof: the network once hid errors by letting the pressure vary across " +
    "the channel, where physics says it cannot.",
    "These came from comparing with the reference solver, not from the loss. Training used Adam, then " +
    "L-BFGS, on one laptop: CPU for most runs and the laptop's GPU for the epsilon 0.1 runs.",
    "3. Network");
  bullets(s, [
    "First test: ΔP was 0.26 % off our solver, although the training loss looked small.",
    "Fix 1 split the pressure so it is uniform across the channel: the error fell to 0.02 %.",
    "In full 2-D the inlet corner dominated the loss, because we divided by its tiny pressure gradient.",
    "Fix 2 capped that weighting and rounded the corners over 3 % of the length (our choice).",
    "Training then took 7–50 minutes per case on a laptop: Adam first, then L-BFGS.",
  ], { space: 24 });
  table(s, "Table 3", "Problems found while building, and their fixes (source: this work).",
    "check against a reference, not just the loss.", [
      ["problem seen", "fix", "effect"],
      ["ΔP 0.26 % off despite a small loss", "pressure split P = P_c(Z) + ε² P₁", "error 0.02 %"],
      ["inlet corner dominated the 2-D loss", "cap weighting, round corners", "loss ~150× lower"],
      ["2-D run over-fitted its points", "L-BFGS restarts on fresh points", "held-out ≈ training loss"],
    ], { x: 6.15, y: 2.8, w: 6.68, colW: [2.45, 2.45, 1.78], capH: 0.5, rowH: 0.8 });
}

// ---- 11. results: narrow channel
pres.addSection({ title: "4. Results" });
{
  const s = content(3, "Results (1): narrow channel",
    "In the narrow-channel limit the PINN matches our solver within 0.22 % up to De = 0.3",
    "Each dot is one trained network; the orange line is our solver and the grey curves are the paper's " +
    "formula at three orders.",
    "Seven networks, each warm-started from the previous De. The momentum route integrates the force " +
    "balance along the centreline: a second, independent way to get delta P from the same network.",
    "4. Results");
  bullets(s, [
    "Recall (slide 7): our solver matches the paper's simulations within 0.37 %.",
    "Seven networks were trained, at De = 0, 0.05, …, 0.3, each starting from the previous one.",
    "All seven lie within 0.22 % of the solver.",
    ["sub", "a second route to ΔP, integrating the force balance, agrees within 0.09 %"],
    "At De = 0.3 the 3rd-order formula is 24 % low, so the network goes where the formula cannot.",
  ], { space: 18 });
  figure(s, A("s_dp_lub.png"), "Figure 9",
    "Pressure drop against De, narrow-channel limit (source: this work).",
    "the network follows the solver where the formula fails.");
}

// ---- 12. results: whole field
{
  const s = content(3, "Results (2): whole field",
    "It also matches the fields: 0.01–0.5 % error over most of the channel, more near the corners",
    "Each line is one field's error at one position along the channel, relative to our solver, at De = 0.3.",
    "The corners are where the hyperbola meets the straight sections: the wall slope changes within " +
    "about one percent of the length, and a smooth network has to bend sharply there.",
    "4. Results");
  bullets(s, [
    "We compare velocity and all three stretch components at every point of the solver's grid.",
    "Across most of the channel the local error is 0.01–0.5 %.",
    "It rises to a few percent near Z = 0 and Z = 1, where the wall slope changes suddenly.",
    ["sub", "and just before the outlet, where the stresses are largest"],
    "So the corners are the main limit of this PINN, not the bulk of the flow.",
  ], { space: 20 });
  figure(s, A("s_err_along.png"), "Figure 10",
    "Local error of each field along the channel, De = 0.3; shaded: corner zones (source: this work).",
    "errors are small in the bulk and peak at the corners.");
}

// ---- 13. results: full 2-D (full-width visual)
{
  const s = content(3, "Results (3): full 2-D flow",
    "Against the paper's 2-D simulations the PINN stays within 0.68 %; the formula misses by up to 26 %",
    "Here the network solves the full 2-D equations, exactly what the paper's simulations solve, for both " +
    "of its channels.",
    "0.68 percent is by the pressure-difference route; by the momentum route the largest error is 1.05 " +
    "percent, at epsilon 0.1, where the training loss is also highest.",
    "4. Results");
  figure(s, A("s_dp_vs_fe.png"), "Figure 11",
    "Pressure drop against De for both channels. Circles: PINN; dashed blue: one network with De as an input; " +
    "triangles: the paper's simulations (source: this work and Boyko & Stone 2022).",
    "the PINN follows the simulations, including 2-D effects at ε = 0.1 that the formula leaves out.",
    { x: 0.9, y: 2.72, w: 11.5, h: 4.2 }, 0.5);
}

// ---- 14. results: mechanism
{
  const s = content(3, "Results (4): the mechanism",
    "The pressure drops because of the polymer shear-stress gradient, as the paper argued",
    "Integrating the force balance along the centreline splits ΔP into four terms; the paper says the fourth " +
    "one causes the drop.",
    "This is the paper's equation 5.3. The PINN points sit on our solver's lines.",
    "4. Results");
  bullets(s, [
    "① is a tiny ε² term (zero here); ②, the solvent-like shear term, barely changes with De.",
    "③, the axial stretch term, is exactly zero on the centreline.",
    "④ is the gradient of the polymer shear stress, and it grows more negative with De.",
    "So the whole drop in ΔP comes from ④, which confirms the paper's explanation.",
    "The 3rd-order formula overstates ④ above De ≈ 0.15, as in the paper's own figure 3.",
  ], { space: 16 });
  figure(s, A("s_contrib.png"), "Figure 12",
    "The four contributions to ΔP against De (source: this work).",
    "only the shear-stress term ④ changes, and it carries the drop.");
}

// ---- 15. wrap-up: use it
pres.addSection({ title: "5. Wrap-up" });
{
  const s = content(4, "Wrap-up (1): use it yourself",
    "One trained network answers any De instantly, and one command sets it up on Windows or Mac",
    "A trained PINN is a formula: give it a position and De, and it returns velocity, pressure and stress " +
    "in a fraction of a second.",
    "The one-command setup was tested on a fresh Windows machine and a fresh Mac. First setup downloads " +
    "about one gigabyte. Retraining everything is optional and takes about four hours.",
    "5. Wrap-up");
  bullets(s, [
    "A single network with De as an extra input covers every De from 0 to 0.3.",
    "The explorer window lets anyone pick a channel, move a De slider and probe any point.",
    "predict.py gives ΔP in pascals, the values at a point, or a whole field as a CSV file.",
    "setup_and_run.bat (Windows) or setup_and_run.sh (Mac) installs everything in one step.",
    ["sub", "tested on fresh Windows and Mac machines; first setup downloads about 1 GB"],
    ["sub", "retraining all networks is optional and takes about 4 hours"],
  ], { space: 9 });
  figure(s, path.join(ROOT, "04_results", "figures", "gui_screenshot.png"), "Figure 13",
    "The explorer window; code at github.com/RM1102/CLL731-OldroydB-PINN (source: this work).",
    "the trained network is a tool, not just a result.");
}

// ---- 16. wrap-up: conclusions
{
  const s = content(4, "Wrap-up (2): conclusions",
    "So the PINN reproduces the paper beyond its formula's reach; its limits are the corners and De > 0.3",
    "All results are for the paper's channel (α = 4, β = 0.4) and De ≤ 0.3; outside that range the " +
    "networks are untested.",
    "Close with the three numbers on the right, then the limits and next steps.",
    "5. Wrap-up");
  bullets(s, [
    "The paper checks out: every formula holds, with one sign typo corrected.",
    "A network trained only on the equations matches two independent references within 0.7 %.",
    "Built-in conditions and checking ΔP two ways mattered more than the size of the network.",
    "Limits: a few percent error at the corners, and minutes of training against seconds for our solver.",
    "Next: higher De with log-conformation, other polymer models (FENE-CR), and inferring λ from a measured ΔP.",
  ], { space: 16 });
  const stats = [["0.22 %", "PINN vs our solver,\nnarrow channel, De ≤ 0.3"],
                 ["0.68 %", "PINN vs the paper's 2-D\nsimulations, both channels"],
                 ["24 %", "paper's 3rd-order formula\nerror at De = 0.3"]];
  stats.forEach(([big, label], i) => {
    const y = 2.8 + i * 1.38;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y, w: 5.4, h: 1.2, rectRadius: 0.08,
      fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "figure" });
    s.addText(big, { x: 7.1, y, w: 2.2, h: 1.2, fontSize: 40, bold: true,
      color: i === 2 ? C.accent3 : C.accent1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(label, { x: 9.4, y, w: 2.8, h: 1.2, fontSize: 14, color: C.text1, valign: "middle",
      margin: 0, isTextBox: true });
  });
}

// ---- 17. references
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "5. Wrap-up" });
  s.addText("References", { placeholder: "title" });
  const refs = [
    "Boyko, E. & Stone, H. A. (2022). Pressure-driven flow of the viscoelastic Oldroyd-B fluid in narrow non-uniform geometries. J. Fluid Mech. 936, A23.",
    "Boyko, E. & Stone, H. A. (2021). Reciprocal theorem for calculating the flow rate–pressure drop relation for complex fluids in narrow geometries. Phys. Rev. Fluids 6, L081301.",
    "Hinch, E. J., Boyko, E. & Stone, H. A. (2024). Fast flow of an Oldroyd-B model fluid through a narrow slowly varying contraction. J. Fluid Mech. 988, A11.",
    "Housiadas, K. D. & Beris, A. N. (2023). Lubrication approximation of pressure-driven viscoelastic flow in a hyperbolic channel. Phys. Fluids 35, 123116.",
    "Tavakol, B., Froehlicher, G., Holmes, D. P. & Stone, H. A. (2017). Extended lubrication theory. Proc. R. Soc. A 473, 20170234.",
    "Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). Physics-informed neural networks. J. Comput. Phys. 378, 686–707.",
    "Wang, S., Sankaran, S., Wang, H. & Perdikaris, P. (2023). An expert's guide to training physics-informed neural networks. arXiv:2308.08468.",
    "Kingma, D. P. & Ba, J. (2015). Adam: a method for stochastic optimization. ICLR.",
  ];
  s.addText(refs.map((r, i) => ({ text: `[${i + 1}]  ${r}`, options: { fontSize: 14,
    paraSpaceAfter: 9, breakLine: i < refs.length - 1 } })),
    { x: 0.5, y: 1.85, w: 12.33, h: 5.0, color: C.text1, valign: "top", margin: 0, isTextBox: true });
}

// ---- 18. thank you
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "5. Wrap-up" });
  s.addText("Thank you", { x: 0.8, y: 2.3, w: 11.7, h: 1.2, fontFace: THEME.headFontFace,
    fontSize: 44, bold: true, color: C.background1, margin: 0, isTextBox: true });
  s.addText([
    { text: "Everything is public and runs on a laptop:", options: { breakLine: true } },
    { text: "github.com/RM1102/CLL731-OldroydB-PINN", options: { bold: true, breakLine: true } },
    { text: "Use of AI: the literature search, code and text were developed with the help of an AI " +
      "assistant (Claude), as stated in the proposal; all results were checked against the paper and " +
      "an independent solver.", options: { fontSize: 12, color: "CFE6E8" } },
  ], { x: 0.8, y: 3.7, w: 11.7, h: 2.0, fontSize: 18, color: C.background1, margin: 0,
       paraSpaceAfter: 8, isTextBox: true });
}

// ---- A1. full equations
pres.addSection({ title: "Appendix" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("Appendix A1: equations", { placeholder: "subtitle" });
  s.addText("The full problem: five scaled equations with exact wall, symmetry and inflow conditions",
    { placeholder: "title" });
  ideaBox(s, "Conditions: no slip on the wall Y = H(Z), symmetry on Y = 0, flow rate 1, fully developed " +
    "flow and stress at the inlet; stress needs no condition at the outlet.");
  figure(s, A("eq_full.png"), "Figure A1",
    "The paper's eq. 2.14 in polymer-stress form, T = βA (source: Boyko & Stone 2022).",
    "ε = 0 drops every ε² term and gives the narrow-channel limit.", { x: 0.9, y: 2.75, w: 11.5, h: 4.15 }, 0.35);
}

// ---- A2. full results table
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("Appendix A2: all results", { placeholder: "subtitle" });
  s.addText("The full comparison: every network against its reference", { placeholder: "title" });
  ideaBox(s, "Errors in percent. The PINN value is the pressure difference; the second route integrates the " +
    "force balance along the centreline.");
  table(s, "Table A1", "Narrow-channel limit: PINN vs our solver (source: this work).",
    "within 0.22 %; formula off by 24 % at 0.3.", [
      ["De", "PINN", "solver", "PINN err", "2nd route", "formula"],
      ["0.00", "0.9965", "0.9961", "+0.05", "+0.03", "+0.0"],
      ["0.05", "0.9181", "0.9188", "−0.08", "+0.06", "−0.0"],
      ["0.10", "0.8599", "0.8609", "−0.11", "+0.03", "−0.4"],
      ["0.15", "0.8151", "0.8165", "−0.17", "−0.06", "−1.8"],
      ["0.20", "0.7802", "0.7820", "−0.22", "−0.09", "−5.2"],
      ["0.25", "0.7534", "0.7547", "−0.17", "−0.00", "−11.9"],
      ["0.30", "0.7313", "0.7328", "−0.21", "−0.02", "−23.5"],
    ], { x: 0.5, y: 2.75, w: 5.9, colW: [0.7, 1.0, 1.0, 1.05, 1.1, 1.05], capH: 0.5, rowH: 0.37 });
  table(s, "Table A2", "Full 2-D: PINN vs the paper's simulations (source: this work).",
    "within 0.68 % (1.05 % by the 2nd route).", [
      ["ε", "De", "PINN", "sims", "PINN err", "2nd route", "formula"],
      ["0.02", "0.0", "0.9968", "0.9975", "−0.07", "−0.09", "−0.1"],
      ["0.02", "0.1", "0.8600", "0.8616", "−0.18", "+0.01", "−0.5"],
      ["0.02", "0.2", "0.7776", "0.7812", "−0.47", "−0.20", "−5.1"],
      ["0.02", "0.3", "0.7252", "0.7302", "−0.68", "−0.16", "−23.3"],
      ["0.1", "0.0", "1.0274", "1.0265", "+0.08", "+0.41", "−3.0"],
      ["0.1", "0.1", "0.8897", "0.8920", "−0.26", "+0.64", "−3.9"],
      ["0.1", "0.2", "0.8154", "0.8124", "+0.37", "+1.04", "−8.7"],
      ["0.1", "0.3", "0.7624", "0.7617", "+0.09", "+1.05", "−26.4"],
    ], { x: 6.7, y: 2.75, w: 6.13, colW: [0.6, 0.55, 0.95, 0.95, 1.0, 1.08, 1.0], capH: 0.5, rowH: 0.37 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
