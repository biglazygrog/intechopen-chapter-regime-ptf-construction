# REVISION LOG

Working memory for the look-ahead-bias revision. **Read this file at the start of
every session before touching any code. Update it at the end of every session,
before any commit.** Commit it alongside the code changes it documents — never
on its own.

---

## Context

This repository is the reproduction package for *Regime-Aware Portfolio
Construction: A Gaussian Mixture Model Approach to Multi-Asset Allocation*
(Bevza & Wyse), an InTechOpen book chapter currently in the editor revision
round (not yet published). The package fits a rolling-window Gaussian Mixture
Model with Normal-Wishart MAP shrinkage to six asset-class log-return series,
extracts daily posterior regime probabilities, and uses those probabilities to
drive an AR-based regime forecast (Figures 2-3) and a regime-aware portfolio
backtest (Figure 4). A diagnostic review on 2026-07-26 established that the
daily probability series consumed by every downstream artefact is
**retrospective**: `Optimiser.get_daily_probabilities()` averages the in-sample
E-step responsibilities of ~20 overlapping estimation windows per date, so the
probability labelled date *t* is built from returns dated up to ~1200
observations after *t*. The editor's revision request is to produce a genuinely
one-sided (no-look-ahead) probability series, make it the primary series behind
all forecast and backtest results, and preserve the retrospective series —
clearly labelled — for descriptive use only. This revision was started on
**2026-07-26**.

---

## Decisions on scope

The three questions put to the author after the Phase 1 diagnostic, and the
answers received, verbatim:

**Q1. Is the chapter already published?** The README says the package
"accompanies" it. The fix will change every headline number in Figures 2-4 and
Table A.1. Whether that means an erratum, or a new one-sided appendix alongside
preserved retrospective results, is your call.

> **A1.** The chapter is not yet published — we are in the editor revision
> round. Treat the one-sided series as the primary series. The retrospective
> series stays for descriptive use (Figure 1, Tables 3 and 4) with clear
> labelling. Revised Figures 2-4 and backtest numbers are the corrected results
> for submission.

**Q2. Item F (the observation grid)** — is the `(df != 0).all(axis=1)` filter
deliberate? If not, the fix changes `T` from 3643 to 6605 and invalidates the
look-ahead fix's own baselines. I'd want to settle this *before* implementing,
or explicitly hold it out of scope.

> **A2.** Hold the `(df != 0).all(axis=1)` filter OUT OF SCOPE for this change
> set. Do not change T from 3643 to 6605. Instead: (a) add a comment in
> pipeline.py explaining the filter retains only business days on which all
> assets traded, (b) correct the "5 years of business days" comment in config.py
> and README to say "1250 observations spanning variable calendar time depending
> on data density", and (c) correct ANN_FACTOR: replace the hardcoded 252 with
> the actual mean observations per year computed from the filtered grid. Do not
> change any estimation windows or refit anything.

**Q3. Items B, C, E** are separate bugs. Bundle them with the look-ahead fix, or
keep the change set narrowly scoped to look-ahead?

> **A3.** Bundle B, C, E, and A with the look-ahead fix. Also bundle G and H —
> they are low-risk and low-cost. This gives a single coherent change set that
> fully satisfies the editor's revision request.
>
> Specific decisions on the bundled items:
>
> - **B:** Fix by switching forecast_eval.py and backtest.py to load the new K=2
>   one-sided file. Do not change the K-varying retrospective file.
> - **C:** Fix forecast_ar to be a genuine AR(1) forecast fitted on the expanding
>   history of one-sided probabilities, not p1.shift(1). The random walk baseline
>   should remain p1.shift(1) — that is the correct RW definition. They should now
>   differ.
> - **E:** Add the always-calm baseline hit rate as a column in
>   ar_change_dissertation_table.tsv and as a printed line in forecast_eval.py
>   output. Do not remove any existing columns.
> - **A:** Switch both probs inputs in backtest.py (the AR input and the
>   compute_regime_portfolios_expanding argument) to the one-sided K=2 series.
>   Update the figure note so it correctly states that the Oracle is the only
>   strategy using future regime probabilities.

---

## Change set

Diagnostic item letters (A, B, C, E, F, G, H) refer to the Phase 1 report.

| File | Function | What changes | Status |
|---|---|---|---|
| `models/gmm.py` | `Optimiser.get_daily_probabilities` | Behaviour **unchanged**. Docstring rewritten to state it is retrospective/smoothed, quantify the ~1200-observation look-ahead, and direct forecast/backtest users to `get_filtered_probabilities`. | DONE |
| `models/gmm.py` | `Optimiser.get_filtered_probabilities` | **NEW.** One-sided series. Model *w* (trained on `[s_w, e_w]`) serves dates `e_w+1 … e_{w+1}` via `predict_proba` — parameters strictly from the past, no cross-window averaging, no interior responsibilities. Optional Hungarian label alignment across refits. Inline assertion that each emitted date maps to a model with `e_w < t` and that the date→model map is a strict partition. | DONE |
| `models/gmm.py` | `Optimiser.get_probs_over_time` | Docstring note: same retrospective in-sample averaging, stamped at window end. No behaviour change. | DONE |
| `core/config.py` | module constants | Add `ONESIDED_PROBS_FILE` and `ONESIDED_FORECAST_PROBS_FILE` filenames. | DONE |
| `core/config.py` | `WINDOW_SIZE` (l.24) | Correct comment `# 5 years of business days` → `1250 observations spanning variable calendar time depending on data density` (F-b). | DONE |
| `core/config.py` | `ANN_FACTOR` (l.63) | `252` → **144.40**, mean observations per year computed from the filtered grid, with derivation comment (F-c). | DONE |
| `core/config.py` | `TARGET_REGIME` (l.54) | Correct the comment: it now genuinely refers to the K=2 one-sided fit (B). | DONE |
| `research/analysis/pipeline.py` | `main` | Save two **new** files (`daily_regime_probabilities_onesided.csv`, `daily_regime_probabilities_forecast_onesided.csv`) alongside the two existing ones. Existing files unchanged; their save lines gain an explicit `[RETROSPECTIVE — descriptive use only]` stdout warning. | DONE |
| `research/analysis/pipeline.py` | `main` (l.63) | Add comment explaining `(df != 0).all(axis=1)` retains only business days on which **all** assets traded (F-a). | DONE |
| `research/analysis/pipeline.py` | `_save_dominant_forecast_eval` | Consume the one-sided K=2 series. `forecast_ar` becomes a genuine expanding-window AR(1) forecast on the logit of the one-sided probability; `forecast_rw` stays `p1.shift(1)`. The two series must now differ (C). | DONE |
| `research/analysis/pipeline.py` | imports (l.18) | Import `hard_labels_from_daily_probs` from `core.utils` (the sorted, Series-returning version) instead of `models.regime_analysis` (G). | DONE |
| `research/analysis/forecast_eval.py` | `main` (l.345) | Load `daily_regime_probabilities_forecast_onesided.csv` instead of `daily_regime_probabilities.csv` (B). | DONE |
| `research/analysis/forecast_eval.py` | `main` | Hard guard: refuse to run if the loaded frame has more `p_` columns than `FORECAST_K`, so a K-varying file can never be passed in silently (B, structural). | DONE |
| `research/analysis/forecast_eval.py` | `evaluate_ar_change_multihorizon` | Compute the always-calm baseline hit rate (`1 - mean(z_th)`) per horizon and add it to the summary dict (E). | DONE |
| `research/analysis/forecast_eval.py` | `main` | Add baseline column to `ar_change_dissertation_table.tsv` and a printed line per horizon. No existing columns removed (E). | DONE |
| `research/analysis/backtest.py` | `main` (l.287) | Load one-sided K=2 series. Feeds **both** `generate_ar_forecasts` and the `probs` argument threaded into `compute_regime_portfolios_expanding` (A). | DONE |
| `research/analysis/backtest.py` | `generate_ar_forecasts` (l.264-268) | Rename output column `p_1_filtered` → `p_1_oracle_fwd`; it is a forward average, not a filtered probability. Update the four call sites. | DONE |
| `research/analysis/backtest.py` | `main` (l.448) | Figure note corrected to state the Oracle is the **only** strategy using future regime probabilities — true once the swap lands (A). | DONE |
| `research/analysis/backtest.py` | `compute_metrics` (l.76), `backtest_unconditional_expanding` (l.125-126), `compute_regime_portfolios_expanding` (l.162-163) | Replace hardcoded `252` with `ANN_FACTOR` imported from `core.config` (F-c). | DONE |
| `research/analysis/backtest.py` | `main` (l.320) | Drop the dummy `[0.01]*n` turnover argument; `compute_metrics` ignores it (H). | DONE |
| `models/regime_analysis.py` | `hard_labels_from_daily_probs` (l.40) | Delegate to `core.utils.hard_labels_from_daily_probs` so the unsorted/ndarray variant cannot diverge at `p_10`+. Preserve the ndarray return type for existing callers (`robustness.py`) (G). | DONE |
| `research/analysis/backtest.py` | `backtest_regime_aware_expanding` | **Open question 6 fix.** Build `orig_pos` from the full returns index *before* the `common_dates` restriction and gate `min_train` on it; the expanding slice keeps the restricted-frame position `t`. Moves first rebalance 2010-09-29 → **2008-04-30**. | DONE |
| `research/analysis/backtest.py` | `backtest_regime_aware_expanding` | **Regime-portfolio history fix (Open question 7).** Retain the pre-restriction `returns_full`/`probs_full` and estimate regime-conditional moments from them at `orig_idx + 1`, instead of from the restricted frame at `t`. Raises the first recompute from n=16 (`sum(p_1)`=4.2, fallback) to **n=290 (`sum(p_1)`=67.4, no fallback)**. `+1` includes the current date, matching `backtest_unconditional_expanding`'s `.iloc[:t+1]` and removing a pre-existing inclusive/exclusive inconsistency between the two paths. | DONE |
| `research/analysis/backtest.py` | both `backtest_*` fns, `main` | Return and print `first_rebalance` / `start_date`, so a silent warm-up gate is visible in the summary rather than hidden. | DONE |
| `research/analysis/validate_onesided.py` | **NEW file** | Runnable as `python -m research.analysis.validate_onesided [--t2] [--full]`. Implements T1, T2, T3, T5. | DONE |
| `.gitignore` | **NEW file** | `__pycache__/`, `*.py[cod]`, venvs, build artefacts, editor cruft, `*.log`. Deliberately does **not** ignore `research/output_charts/`. | DONE |
| `README.md` | — | New section "Retrospective vs one-sided regime probabilities": mechanism, look-ahead magnitude, which artefact consumes which file, reduced one-sided sample (2007-02-05 onward, 2393 rows). Add probability-series column to the artefact→producer table. Correct the "5 years" window wording (F-b). | DONE |
| `REVISION_LOG.md` | — | This file. Kept current; committed with the code. | DONE |

**Output files after this change set** — nothing renamed, moved or deleted:

| File | Status | Content | Consumed by |
|---|---|---|---|
| `daily_regime_probabilities.csv` | unchanged | retrospective, K-varying | Figure 1, Tables 3 & 4 |
| `daily_regime_probabilities_forecast.csv` | unchanged | retrospective, K=2 | nothing (retained for reference) |
| `daily_regime_probabilities_onesided.csv` | **new** | one-sided, K-varying | nothing — descriptive/reviewer use only (per resolved Q5) |
| `daily_regime_probabilities_forecast_onesided.csv` | **new** | one-sided, K=2 | Figures 2, 3, 4 |

---

## Session log

*Newest first. One entry per working session, written before every commit and at
the end of every session even if nothing was committed.*

### 2026-07-26 — implementation session 3 (backtest coverage)

**Done.** Investigated backtest coverage options at author request (no code
changes during the analysis). Findings:

- Reducing the AR burn-in 252 → 63 would move the start to 2007-06-06 and add
  189 observations (+8.9%), covering the 2007 crisis onset — but it does **not**
  fix the regime-portfolio fallback, it deepens it (`sum(p_1)` falls 4.2 → 1.4),
  and it thins the AR fit to 63 points.
- An expanding GMM window (250 → 1250) would reach back to 2002-02-20 but
  changes the reviewed estimation methodology: K selection becomes an artefact
  of window length, `Neff_min` collapses early, the κ₀ prior-weight calibration
  (N_k ≈ 568) stops holding, and Tables A.1/A.2 would need redoing.
- The fallback is a **separate constraint** from the AR burn-in:
  `compute_regime_portfolios_expanding` sliced the *restricted* frame, so its
  history always restarted at the backtest start (~16 observations) regardless
  of burn-in.

Author took the recommendation: **leave the burn-in at 252, fix the probs slice
only.** Implemented — `returns_full`/`probs_full` retained pre-restriction and
indexed by `orig_idx + 1`.

Also noted for the record: `backtest.py` does **not** read `AR_MIN_TRAIN`. Its
`main()` sets a local `min_train = 252` and `generate_ar_forecasts` defaults to
252. `AR_MIN_TRAIN = 250` is read only by `forecast_eval.py`. Changing the config
constant alone would move Figure 2 and leave Figure 4 untouched. Left as-is
(consistent with resolved Q3), but flagged as a trap for future edits.

**Tested.** First rebalance 2008-04-30 verified at **n=290, sum(p_0)=222.6,
sum(p_1)=67.4, no fallback for either regime** (was n=16 / 4.2 / r1 fallback).
Checked every recompute through the GFC (2008-04-30, 2008-08-29, 2009-02-27) —
no fallback at any of them, L1 distance between the two regime portfolios
0.45-0.54. Backtest re-run (exit 0); Figure 4 regenerated.

**Not committed.** Awaiting author confirmation of the updated headline numbers.

**Next.** Author confirms → stage and commit everything together. Then Block 2.

### 2026-07-26 — implementation session 2 (Block 1 close-out)

**Done.**
1. **Open question 6 resolved via option (a)** (author instruction).
   `backtest_regime_aware_expanding` now builds `orig_pos` from the full returns
   index *before* the `common_dates` restriction and gates `min_train` on it.
   The expanding slice passed to `compute_regime_portfolios_expanding` still uses
   the restricted-frame position `t` (it must — the frames handed to it are
   restricted), so the regime-moment estimation window is unchanged.
   `first_rebalance` and `start_date` are now returned by both backtest
   functions and printed in the summary, so this class of bug is visible rather
   than silent.
2. Added `.gitignore` (`__pycache__/`, venvs, editor cruft, `*.log`). It
   deliberately does **not** ignore `research/output_charts/` — the reproduction
   package ships its produced figures and tables.
3. Corrected a factual error in the Open question 6 entry (see the correction
   note there).

**Tested.** First rebalance verified at **2008-04-30** (was 2010-09-29).
Backtest re-run (exit 0); Figure 4 regenerated. Max drawdowns now differ across
strategies (−13.9% / −19.5% / −20.0% / −20.1%) instead of being identical to six
decimal places, confirming the strategies are genuinely active through the GFC.
Checked the `weights.sum() < 10` fallback at each recompute through the crash —
result recorded as a caveat under "Headline result changes".

**Not committed.** Awaiting author confirmation of the three requested items
(Figure 4 headline numbers, `.gitignore` contents, final `REVISION_LOG.md`).

**Next.**
1. Author confirms the three items; then stage and commit code + outputs +
   `REVISION_LOG.md` + `.gitignore` together.
2. **Block 2** (queued, not started): Data-section disclosure of the
   `(df != 0).all(axis=1)` filter — see "Queued for Block 2" below.
3. Optional: `--full` validation run (T1 at paper spec) for final sign-off.

### 2026-07-26 — implementation session 1

**Done.** Implemented the full agreed change set: 24 rows in the change-set
table, all marked DONE. New `Optimiser.get_filtered_probabilities()` (one-sided
series); retrospective methods documented, behaviour untouched; bundled items
A, B, C, E, F(a/b/c), G and H; new `research/analysis/validate_onesided.py`;
README rewritten with a "Retrospective vs one-sided" section and a
probability-series column in the artefact table. `ANN_FACTOR` 252 → 144.40.

**Tested.** Full pipeline re-run (exit 0) — one-sided series is 2393 rows,
2007-02-05 to 2026-01-23, exactly `T − WINDOW_SIZE`. Validation suite run with
`--t2`: **T1, T2, T3, T5 all PASS**, T1 and T2 both bitwise identical / exact
zero. T4 investigated in depth (see checklist) — regeneration drift is 3.55e-15
and provably not caused by this change set; both retrospective CSVs restored
with `git checkout` so they are preserved byte-for-byte. T6 reported.
`forecast_eval`, `backtest`, `moments`, `correlations` and Figures 1-3 all
re-run cleanly.

**Not committed.** Awaiting author confirmation of the file list, and a decision
on Open question 6 (Figure 4 comparability) — the regenerated Figure 4 and
backtest numbers should not go to the editor until that is resolved.

**Next.**
1. Author decides Open question 6 (a/b/c). If (a) or (b), re-run
   `research.analysis.backtest` and regenerate Figure 4.
2. Commit code + REVISION_LOG.md together.
3. Optional: `--full` validation run (T1 at paper spec) for final sign-off.

---

## Open questions

*Numbered. Resolution noted next to the item before removal.*

6. **[RESOLVED 2026-07-26 — option (a) implemented, see Session log]**
   `min_train` was double-counted in `backtest_regime_aware_expanding`, and the
   one-sided switch made it material.

   The function reassigns `returns = returns.loc[common_dates]` (l.171) and then
   computes `orig_idx = returns.index.get_loc(date)` (l.189). After the
   reassignment `orig_idx` is *always* equal to `t`, so the
   `orig_idx >= min_train` gate counts 252 observations from the start of the
   **restricted** frame rather than from the start of history. The variable is
   named `orig_idx` precisely because the intent was the unrestricted position;
   the reassignment silently broke it.

   This is pre-existing, but it was benign before: with the retrospective series
   starting 2000-10-31 the restricted frame began ~2002 and the first rebalance
   landed ~2004, comfortably before the GFC. With the one-sided series starting
   2007-02-05:

   - `forecast_df` begins **2008-04-01** (252 min_train + 21 horizon into the series)
   - the restricted frame therefore begins 2008-04-01
   - +252 more observations ⇒ **first actual rebalance 2010-09-29**
   - but the deepest drawdown is **2008-07-14 → 2008-10-24**

   So all four regime strategies sit frozen at the initial equal weight through
   the entire 2008 crash — which is why their max drawdown is identical to six
   decimal places (−0.268542) with identical peak and trough dates, despite
   their signals differing by up to 0.89 at the rebalance dates inside that
   window and their net returns differing on 1866/2120 days. Meanwhile
   `backtest_unconditional_expanding` was **not** subject to the double count —
   it takes `t` directly — so its first rebalance was 2008-02-29 and it traded
   through the crash (max DD −11.4%).

   *(Correction: an earlier draft of this entry said the unconditional
   benchmark's `min_train` counted from 2000-10-02. That was wrong — `main()`
   restricts `returns_df` to the probability dates before calling it, so it
   counts from 2007-02-05. The mechanism and the fix are unaffected; only the
   benchmark's first-rebalance date was misstated.)*

   **Resolution: option (a), per author instruction.** `orig_pos` is now built
   from the full returns index *before* the `common_dates` restriction, and the
   `min_train` gate uses it. The expanding slice passed to
   `compute_regime_portfolios_expanding` continues to use the restricted-frame
   position `t` — it must, since the frames handed to it are restricted — so the
   regime-moment estimation window is unchanged. First rebalance verified at
   **2008-04-30**. Options (b) and (c) were rejected by the author: the
   evaluation window must not start after 2010.

7. **[RESOLVED 2026-07-26 — probs-slice fix implemented, see session 3]**
   Regime portfolios were estimated from the restricted frame, so their history
   restarted at the backtest start (n=16, `sum(p_1)`=4.2) and the high-vol
   portfolio fell back to equal weight for the quarter containing the
   2008-07-14 drawdown peak. Shown to be independent of the AR burn-in — the
   fallback fires at every burn-in choice and worsens as the start moves
   earlier. Fixed by estimating from the full one-sided history at
   `orig_idx + 1`: **n=290, sum(p_1)=67.4, no fallback**, and no fallback at any
   recompute through the GFC.

*Questions 1-5 were raised before implementation and all resolved on
2026-07-26; resolutions recorded below. Questions 6 and 7 arose during
validation and coverage review and were resolved the same day.*

### Resolved

1. **`ANN_FACTOR` value.** → **144.40** (3643 filtered observations / 25.229
   calendar years). Rejected alternatives: 145.00 (complete years only), 134.93
   (all years incl. partial 2000/2026), 159.00 (median). `config.py` carries the
   derivation as a comment. All `* 252` and `/ 252` in `moments.py` and
   `backtest.py` now use the constant. Window sizes and `min_train` unchanged.
2. **`figure3` `window = 252`.** → **Retitle only, do not rescale.** Title is now
   "Rolling 252-Observation Hit Rate"; a footnote states that 252 observations
   span ~1.75 calendar years on this grid.
3. **`min_train` constants.** → **Leave unchanged.** `config.py` now carries a
   comment that `AR_MIN_TRAIN` and `PORT_MIN_TRAIN` are observation counts, not
   calendar days.
4. **One-sided sample start.** → **Accept the shorter sample** (2007-02-05
   onward). `WINDOW_SIZE` unchanged. The reduction is flagged in the Figure 2,
   3 and 4 captions and in the README methodology section.
5. **K-varying one-sided file.** → **Produce it.** Ships with a `#`-prefixed
   provenance header, a README row marked "for descriptive/reviewer use only",
   and a stdout note at write time. Nothing in the pipeline consumes it.

## Queued for Block 2 — Data section disclosure

**Not started. Do not begin until Block 1 is committed and validated.**

Disclose and justify the `(df != 0).all(axis=1)` filter in the chapter's Data
section. Required content, per author instruction:

- **Explicit disclosure:** 6,605 → 3,643 observations, and why.
- **Justification:** synchronous-trading argument — stale prices corrupt the
  cross-asset covariance; on days where any asset did not trade the joint return
  vector is not informative.
- **Supporting analysis, to be done before writing:**
  - (a) Check dropped dates against stress episodes (GFC, European sovereign
    debt crisis, COVID) and confirm stress periods are **not** systematically
    excluded.
  - (b) Bar chart of observations per year for the appendix, making the density
    shift visible to reviewers.
- **Framing:** a motivated methodological choice, not data cleaning.

Note the tension to resolve honestly in (a): the counts already in this log show
2021 with 13 observations and 2013 with 26, against 250 in 2024. Whether the
2008, 2011-12 and 2020 stress windows survive the filter is exactly what (a)
must establish, and the answer is not assumed here.

## Queued for Block 3 — Expanding-window backtest appendix

**Not started. Decision deferred until after Block 2.**

> Expanding window appendix — decision deferred until after Block 2. Before
> implementing, run the expanding window as a standalone analysis and report
> headline numbers (Sharpe, max DD, first rebalance date for all five
> strategies) so the decision on whether to write it up can be made with full
> information. Key question: does the conclusion change over the longer sample?
> If Oracle still clears Unconditional and AR still fails to beat Random Walk,
> the appendix adds robustness. If results differ materially, assess whether the
> inconsistent estimation spec (250 → 1250 expanding then rolling) requires
> additional caveating that would undermine the appendix value.

Reference figures from the session-3 analysis, for scoping: an expanding window
with a 250-observation minimum could emit its first probability at **2002-02-20**
and, with the burn-in left at 252, start the backtest at **2003-08-22** — about
4.5 additional years versus the current 2008-04-01. The estimation-consistency
concerns are recorded in the "Known issues out of scope" table and in the
session-3 log entry.

## Chapter-text follow-up (Block 1 finding, author to action)

Per author instruction on T6: update the chapter conclusion to state that
forecast skill is **short-horizon only** — +1.04pp at h=1, approximately flat at
h=5 (+0.09pp), negative at h=21 (−2.97pp), all measured against the always-calm
baseline. This is a finding, not a failure.

The final Figure 4 result is consistent with it and needs to be reflected too:

- The infeasible Oracle (Sharpe 0.718) beats the unconditional benchmark (0.449)
  decisively on both Sharpe and drawdown, so the regime signal carries real
  economic value.
- No feasible forecast captures it: AR Change 0.396 and AR Baseline 0.373 both
  underperform Unconditional, at 4-5x its turnover.
- **The AR forecasts do not beat a random walk in the backtest** (RW 0.430).
  This follows directly from the Figure 2 result — the backtest rebalances
  monthly against a 21-observation horizon, exactly where AR Change loses to the
  always-calm baseline. The chapter should not claim AR forecasting adds value
  at the allocation horizon; it does not.

---

## Known issues out of scope

| # | Issue | Reason for exclusion |
|---|---|---|
| F | **Observation grid.** `_filter_and_resample` ffills to a daily grid, then `(df != 0).all(axis=1)` drops every row where any asset did not move, taking 12018 raw rows → 6605 → **3643**. Per-year counts collapse to 13 (2021) and 26 (2013) against 250 (2024); the first three rows are month-ends. A 1250-observation window therefore spans **5.96-14.28 calendar years**. | Explicitly excluded by decision A2 — changing `T` would invalidate every baseline the look-ahead fix is measured against. Documentation (F-a, F-b) and `ANN_FACTOR` (F-c) **are** in scope; the filter and `T` are not. |
| — | `models/portfolio.py` (20+ hardcoded `252` sites, unreviewed backtest engine) and `models/forecasting.py` (duplicate AR/DM machinery, own `OUTPUT_DIR`). | Dead code: nothing in the repository imports either (verified by grep), so the `ANN_FACTOR` correction was scoped to live code. Candidates for deletion in a later change set. |
| — | `Optimiser._scale_window(..., "expanding")` (`gmm.py:862-866`) standardises by `X[:end]`; `"rolling"` standardises by whole-window mean/σ. Both are future-looking within the window. | Latent, not live: `SCALE_METHOD = "none"`. `stability_analyzer.py:79` defaults to `"rolling"` but `stability.py` passes `SCALE_METHOD`, so the Table 5 run is safe. Worth a guard in a later change set. |
| — | `gmm.py:1051-1052` computes `mu_raw`/`sigma_raw` from `X[:end]` (expanding, future-looking within window) into `raw_means_`/`raw_stds_`. | Never consumed by any code path. |
| — | Previous-epoch prior is near-degenerate: consecutive windows share 1187 of 1250 observations, so `_build_mu_prior` shrinks each fit ~95 % toward a nearly identical sample. | Causal and therefore not look-ahead. A methodological question for the chapter text, not a code defect. |
| — | Tables 3 and 4 condition on full-sample smoothed labels (`pipeline.py:90`). | Deliberately retained per decision A1 — descriptive statistics. Requires labelling only, not recomputation. |
| — | Tables A.1 and A.2 use the retrospective series. | Correct as-is: they measure *estimation stability* (drift, K-stability, probability autocorrelation), which is a property of the smoothed fit. Note only. |
| — | Regenerating the retrospective CSVs on this machine drifts by 3.55e-15 vs the committed versions (0 label flips). Proven environmental, not caused by this change set. | Files restored with `git checkout`; the committed series is preserved byte-for-byte. Reproducing bit-exactly across BLAS builds is out of scope. |
| — | Extending backtest coverage earlier than 2008-04-01, either by reducing the AR burn-in below 252 or by using an expanding GMM window. | Analysed in session 3 and declined by the author. The burn-in reduction does not fix the fallback and deepens it; the expanding window changes the reviewed estimation methodology and would force Tables A.1/A.2 to be redone. |
| — | `backtest.py` uses a local `min_train = 252` rather than `AR_MIN_TRAIN = 250` from config; the two are independent. | Left as-is, consistent with resolved Q3 (leave training-length constants unchanged). Flagged so a future edit to `AR_MIN_TRAIN` is not assumed to move Figure 4 — it will not. |

---

## Validation checklist

Run 2026-07-26 via `python -m research.analysis.validate_onesided --t2`.

| Test | Description | Status | Result |
|---|---|---|---|
| **T1** | **Truncation invariance (decisive).** Fit on `X[:t]` only, compute the one-sided series, compare against the full-sample fit's one-sided series restricted to dates ≤ the last common window end. Must be exactly equal. Run at three cut points. | **PASS** | Cuts at t=1639 (2008-10-03), t=2550 (2019-08-14), t=3205 (2024-04-18). `max abs diff = 0.000e+00`, **bitwise identical** at all three over 1389 / 2300 / 2955 overlapping dates. |
| **T2** | **Future corruption.** Replace `X[t+1:]` with noise, refit, assert `p_s` unchanged for all `s ≤ t`. Catches leakage via BIC/K-selection and the EWMA/prior paths that T1 could mask. Reduced spec (`K_candidates=[2]`, `window_size=250`). | **PASS** | Future replaced with N(0, 10σ) noise at the same three cut points. `max abs diff` on all past dates = `0.000e+00`. |
| **T3** | **Index-level assertion (also inline in `get_filtered_probabilities`, every call).** For every emitted date *t*, `window_end_indices_[w(t)] < t`. Date→model map is a strict partition. First emitted index > `WINDOW_SIZE - 1`. | **PASS** | Paper spec. 0 dates without a strictly-past model; 0 duplicated dates; first emitted index 1250 (> 1249); model map covers 2393 dates = 2393 emitted. |
| **T4** | **Regression on preserved outputs.** Regenerated `daily_regime_probabilities.csv` and `daily_regime_probabilities_forecast.csv` must be byte-identical to the committed versions. | **PASS (with caveat)** | Not byte-identical on regeneration: `max abs diff = 3.55e-15`, **0 rows differing by >1e-9, 0 hard-label flips**. Proven NOT caused by this change set — HEAD's `models/gmm.py` and the current one produce **bitwise-identical** output on this machine (isolation test, K=2 paper spec, `max abs diff = 0.000e+00`). The drift is environmental (committed CSVs were generated on a different BLAS/library build). **Both files were restored with `git checkout`, so the committed retrospective series is preserved byte-for-byte.** |
| **T5** | **Structural sanity.** Row sums = 1 ± 1e-9; no NaN; monotonic unique index; `len(one_sided) == T - WINDOW_SIZE`. | **PASS** | Paper spec. Row-sum range `[1.000000000000, 1.000000000000]`; no NaN; no negative probabilities; index monotonic and unique; length 2393 = 3643 − 1250. |
| **T6** | **Honest-skill report (output, not pass/fail).** Recompute Figure 2 metrics on the one-sided series and report hit rate against the always-calm baseline. | **PASS (reported)** | See "Headline result changes" below. AR Change now beats the always-calm baseline by **+1.04pp (h=1)** and **+0.09pp (h=5)**, and **loses by −2.97pp (h=21)**. `n_forecasts` 3310 → 2120. |

### Headline result changes (one-sided vs retrospective)

Forecast hit rates, `ar_change_multihorizon_summary.csv`:

| Horizon | Always-calm baseline | AR Change (before) | AR Change (after) | Gain vs baseline (before → after) |
|---|---|---|---|---|
| 1-day | 0.8737 → 0.8520 | 0.8864 | 0.8624 | +1.27pp → **+1.04pp** |
| 5-day | 0.8737 → 0.8530 | 0.8701 | 0.8539 | −0.36pp → **+0.09pp** |
| 21-day | 0.8740 → 0.8577 | 0.8610 | 0.8280 | −1.30pp → **−2.97pp** |

Figure 3 (`dominant_regime_forecast_eval.csv`): AR and RW were bit-identical
(both 0.8377). Now genuinely distinct — **AR 0.8525 vs RW 0.8291**, against an
always-calm baseline of 0.8530.

Figure 4 backtest, **final** (`ANN_FACTOR` = 144.40, one-sided inputs, after both
the `orig_idx` fix and the regime-portfolio history fix). All five evaluated from
2008-04-01:

| Strategy | Sharpe | Ann. return | Max DD | 1st rebalance | Turnover |
|---|---|---|---|---|---|
| Oracle (infeasible) | **0.718** | 2.22% | −10.1% | 2008-04-30 | 0.0022 |
| Unconditional | 0.449 | 1.24% | −11.4% | 2008-02-29 | 0.0007 |
| Random Walk | 0.430 | 1.44% | −14.6% | 2008-04-30 | 0.0067 |
| AR Change | 0.396 | 1.32% | −14.5% | 2008-04-30 | 0.0036 |
| AR Baseline | 0.373 | 1.25% | −14.2% | 2008-04-30 | 0.0026 |

Superseded numbers, retained for the record:

| Stage | Oracle | Uncond. | AR Change | AR Base | RW | Max DD (regime strats) |
|---|---|---|---|---|---|---|
| Before `orig_idx` fix | 0.256 | 0.449 | 0.181 | 0.159 | 0.092 | −26.9% (all identical) |
| After `orig_idx`, before slice fix | 0.545 | 0.449 | 0.297 | 0.271 | 0.233 | −19.5 to −20.1% |
| **Final** | **0.718** | 0.449 | 0.396 | 0.373 | 0.430 | −14.2 to −14.6% |

**Interpretation.** The ordering is coherent: the infeasible Oracle sits well
clear at the top (0.718), as an upper bound should, and every regime-aware
strategy now improves on drawdown relative to the earlier frozen-portfolio runs.

The honest reading is unchanged in substance and sharper in detail:

- The regime signal carries **real economic value** — perfect foresight of it
  (Oracle 0.718) beats the unconditional benchmark (0.449) decisively, on both
  Sharpe and drawdown.
- **No feasible forecast captures it.** AR Change (0.396) and AR Baseline
  (0.373) both fall short of Unconditional, at 4-5x its turnover and ~3pp worse
  drawdown.
- **The AR models do not beat the random walk** (0.430). RW now edges out both,
  which is consistent with Figure 2: forecast skill is short-horizon only, and
  the backtest rebalances monthly at a 21-observation horizon — precisely where
  Figure 2 shows AR Change *losing* to the always-calm baseline (−2.97pp).

This is a coherent story to report: the regimes are real and economically
meaningful, but at the monthly rebalancing horizon the paper's AR forecasts add
nothing over a random walk, and regime-aware allocation does not beat an
unconditional max-Sharpe portfolio after costs.
