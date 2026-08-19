# SENTINEL-SplitFed — Pre-Submission Review Checklist

A working checklist of every open **decision** (yours to make) and **task** (mechanical work) before
this Springer submission is ready. Nothing here is done automatically — bring me directions on any item
and I'll execute it. Legend: **[DECIDE]** = your call; **[TASK]** = work to do once decided;
💡 = my suggestion, override freely.

Current draft state: `main.tex` has all sections, 6 tables, 4 figures (per-round AUROC, baseline bars,
sensitivity heatmap, attack-magnitude curve), 16 verified citations. **No system architecture diagram yet.**
Numbers are all n=10. Name standardized to SENTINEL-SplitFed.

---

## A. Core narrative & framing

- [ ] **[DECIDE] The one headline contribution.** What should a reviewer remember in one sentence?
  Candidates: (a) *first SplitFed-native intrusion gate*; (b) *treating staleness as a security-relevant
  ambiguity, not just latency*; (c) *the honest robustness boundary (attack-then-abstain)*.
  💡 (b) as the headline, with (a) as the mechanism and (c) as the honesty hook — matches the docx's thesis.
- [ ] **[DECIDE] How assertive to be.** "To our knowledge, first…" vs. a softer "a SplitFed-native mechanism for…".
  💡 Softer + architectural-specificity framing (the docx's "claims to avoid" warns against overclaiming).
- [ ] **[DECIDE] How prominently to feature the Attack-B vulnerability.** Buried limitation, or foregrounded
  as a deliberate honesty contribution? 💡 Foreground it — reviewers trust papers that surface their own boundary.
- [ ] **[TASK] Re-read the abstract and contributions list** once (A) is set, so both point at the same headline.

## B. Figures & diagrams

- [ ] **[DECIDE + TASK] System architecture diagram (Figure 1) — the biggest gap.** We need to decide:
  what it depicts (client encoders + activation/gradient loop + server modules: async window → staleness
  registry → SNAS gate → robust aggregation), level of detail, and style (TikZ vector vs. drawn in
  draw.io/Excalidraw and imported). 💡 A single left-to-right pipeline: 3 client encoders on the left,
  the activation/gradient split loop in the middle, and the server's 4 modules as a gated pipeline on the
  right with accept/flag/quarantine branches. TikZ if you want it fully reproducible; draw.io if faster.
- [ ] **[DECIDE] Which additional plots (if any) to add.** Candidates NOT yet built:
  - SNAS-trajectory plot: attacker vs. honest client SNAS over rounds with flag/quarantine threshold lines
    (strong "the gate separates them" visual). 💡 worth adding — it's the most direct evidence figure.
  - Attack-B evasion plot: SNAS staying under threshold across the 4 decay functions.
  - Decay-function ablation plot (currently only a table, TABLE 7 in results log).
- [ ] **[DECIDE] Final figure order & count.** 4 now (+architecture = 5, +trajectory = 6…). Springer figures
  should each earn their place; decide the set and sequence.
- [ ] **[TASK] Eyeball the heatmap caption vs. the actual rendered image** (axis labels, legend, marker) and fix wording if they differ.
- [ ] **[DECIDE] Color/style consistency** across all figures (currently Okabe-Ito, serif). Confirm or change.

## C. Section-by-section content review

- [ ] **[TASK] Introduction** — reads well; confirm the contributions bullets match the final headline (A).
- [ ] **[DECIDE] Related Work depth** — 7 subsections now. Confirm APP-SplitFed citation (see F), and decide
  whether to add back FedStrag / FedASMU / PORT / SecureAFL with verified BibTeX or leave them out.
- [ ] **[DECIDE] Threat model scope** — the code defines label-flip, backdoor, and slow-poisoner attacks that
  were never run. Currently the paper lists 6 attack types. Decide: keep only the run ones, or note the others
  as defined-but-not-evaluated. 💡 Only claim what we ran; drop the rest to avoid "where are those results?"
- [ ] **[DECIDE] Metrics to report** — the docx suggests federation-preservation metrics (client-selection
  frequency, contribution entropy) to make the Krum contrast *quantitative*, plus clinical metrics
  (AUPRC, F1, calibration/Brier). Decide which, if any, to add. 💡 Add at least contribution entropy — it
  turns "Krum collapses federation" from a claim into a number, which is the strongest reviewer defense.
- [ ] **[TASK] Method section** — verify every symbol (N, τ, β, γ, θ, thresholds) is defined before use and
  used consistently; confirm the algorithm block matches the prose.
- [ ] **[TASK] Communication-cost section** — re-verify the byte arithmetic against the real tensor shapes.
- [ ] **[TASK] Discussion & limitations** — confirm the four limitations still hold at n=10 (the "single-seed
  sensitivity" one is still true; the "five seeds" one is already fixed).

## D. Theory section (from `sentinel_splitfed_proof.pdf`)

- [ ] **[DECIDE] Include a Theoretical Analysis section or not, for THIS submission.** The proof gives
  Assumptions 1–4, Theorem 1 (accept honest / quarantine malicious / bounded convex aggregate), and
  Proposition 1 (attack-then-abstain evasion — formally explains Attack B). 💡 Include a condensed version:
  it materially strengthens a Springer submission and directly backs the empirical Attack-B finding. But it's
  your call on effort vs. deadline.
- [ ] **[DECIDE] If yes: how heavy** — full lemmas, or just Theorem 1 + Proposition 1 with proofs in an appendix.

## E. Scope: this paper vs. future work

- [ ] **[DECIDE] Client scaling** — stay at 3 clients (with honest low-client framing) or add a 5–6 client
  partition? 💡 Keep 3, frame as realistic cross-silo, list scaling as future work.
- [ ] **[DECIDE] Attack-B mitigation** — discuss only, or implement temporal-SNAS and add a result?
  💡 Discuss-only for this deadline; implementing it is a strong journal-extension story.
- [ ] **[DECIDE] Distributed/latency stress test** — mentioned in docx as strengthening IoMT framing. In or out?

## F. Citations & related work

- [ ] **[DECIDE/CONFIRM] APP-SplitFed citation.** Ask the professor which exact paper she means — the closest
  verified match is a differently-titled ACM TOMM 2024 paper (DOI 10.1145/3695876). See `CITATION_VERIFICATION.md`.
- [ ] **[TASK] Pull exact BibTeX** for the 9 added references from their arXiv/publisher pages (IDs are logged
  in `CITATION_VERIFICATION.md`) and replace the current hand-written `\bibitem` text with verified details.
- [ ] **[DECIDE] Reference style** — Springer `sn-jnl` supports numbered or author-year; confirm which the
  special issue wants, and whether to move from manual `\thebibliography` to a `.bib` + BibTeX.

## G. Springer mechanics & formatting

- [ ] **[TASK] Confirm the `sn-jnl` class option** (`sn-mathphys-num` currently) matches the journal's required style.
- [ ] **[TASK] Check journal limits** — page/word count, figure count, abstract length for the special issue.
- [ ] **[TASK] Table formatting pass** — booktabs consistency, caption placement (Springer wants captions above
  tables, below figures — verify), no vertical rules.
- [ ] **[TASK] Compile once in Overleaf and clear all real warnings** (undefined refs resolve on 2nd pass;
  track down any genuine ones). Report the final warning count to me.

## H. Placeholders that MUST be filled before submission

- [ ] **[TASK] Author names, emails, affiliations** (currently "First/Second/Third Author", "Institution Name").
- [ ] **[TASK] Funding statement** (currently placeholder).
- [ ] **[TASK] Ethics/IRB & MIMIC-IV data-use statement** (PhysioNet credentialing language).
- [ ] **[TASK] Data availability & code availability** statements.
- [ ] **[TASK] Author contributions** statement.

## I. Final pre-submission verification

- [ ] Every figure has a caption + in-text `\ref`, and vice-versa.
- [ ] Every table referenced in text; numbers match across all tables, prose, and abstract.
- [ ] Every `\cite` resolves to a real, verified reference.
- [ ] No leftover "Async-SplitFed-IR", "n=5", "0.0625-floor", or single-seed numbers.
- [ ] Consistent notation and terminology throughout.
- [ ] Compiles cleanly in Overleaf with pdfLaTeX, within the journal's length limit.

---

### Suggested order to tackle it
1. **A (headline)** — everything else hangs off it.
2. **B (architecture diagram + figure set)** — the biggest visible gap.
3. **C + D + E (content, theory, scope decisions)**.
4. **F + G (citations, formatting)**.
5. **H + I (placeholders, final checks)** — last, right before submission.

Bring me any item (or a batch) with your direction and I'll do just that — one step at a time, checking with
you as we go.
