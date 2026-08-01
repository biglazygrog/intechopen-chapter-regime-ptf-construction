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
| **BLOCK 2** — grid correction (Open questions 8, 9) | | | |
| `core/utils.py` | `filter_synchronous_trading`, `CASH_LIKE_ASSETS` | **NEW.** Single definition of the synchronous-trading filter, replacing five independent copies. Tests only market indices; cash-like series are exempt because at a pinned policy rate their zero is a rounding artefact of a genuinely near-zero return, not a stale quote. Exempt names absent from a universe are ignored, so it is safe across core and extended tiers. | DONE |
| `research/analysis/pipeline.py` | `main` | Use the shared helper; rewrite the filter comment to the corrected rationale (F-a superseded). | DONE |
| `research/analysis/robustness.py` | `main` | Use the shared helper. | DONE |
| `research/analysis/shrinkage.py` | `main` | Use the shared helper. | DONE |
| `research/analysis/stability_analyzer.py` | `_run_tier` | Use the shared helper (per tier). | DONE |
| `research/analysis/validate_onesided.py` | `_load_returns` | Use the shared helper. | DONE |
| `core/config.py` | `ANN_FACTOR` | 144.40 → **248.48** (6269 / 25.229 yr), with the corrected derivation comment. | DONE |
| `core/config.py` | `WINDOW_SIZE` comment | **Reverts the Block 1 F-b correction.** The grid is now near-regular, so 1250 observations ≈ 5.0 years (range 4.96-5.24) — "approximately five years" is accurate again. | DONE |
| `research/figures/figureA1_observations_per_year.py` | **NEW file** | Appendix Figure A.1 producer — observations per year on the corrected grid. Single hue, hatched partial years, mean line, selective labels. | DONE |
| `README.md` | — | "The observation grid is irregular" section replaced by "The synchronous-trading filter": corrected rationale, cash exemption, the 96.8% attribution, new row chain (12018 → 6605 → 6269), near-regular grid, `ANN_FACTOR` 248.48, window span 4.96-5.24 yr. Look-ahead probe table and one-sided start date recomputed on the new grid. | DONE |
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

### 2026-08-01 — Step 1 close-out: Block 1 documentation sweep

**Done.** Swept the remaining stale Block 1 prose recorded as Open question 10,
per author instruction. Seven sites in four files:

| File | Was | Now |
|---|---|---|
| `moments.py` docstring | ANN_FACTOR 144.40, 3643 obs | 248.48, 6269 obs |
| `moments.py` stdout | "(irregular grid)" | "(near-uniform grid)" |
| `config.py:32` | one-sided starts 2007-02-05 | 2006-01-30 |
| `config.py:67` | 250 obs span ~1.7 calendar years | ~1.0 year |
| `pipeline.py:15` | one-sided starts 2007-02-05 | 2006-01-30 |
| `figure2_forecast_accuracy.py:8` | one-sided starts 2007-02-05 | 2006-01-30 |

**Tested.** `compileall` clean over `core`, `models`, `research`.
`python -m research.analysis.moments` exit 0, and
`table3_regime_moments.{csv,tsv}` are **unchanged on disk** — confirming the
sweep is documentation-only, as scoped. No figure or table was regenerated.

**Committed and pushed.** This closes Step 1. Open question 10 resolved.

**Next.** Step 2 — Point 7 of the Full Chapter Review Report (replace the
correlation bootstrap with a time-series-valid procedure). Diagnostic delivered
2026-08-01; **awaiting author approval before any code changes**. See the new
Open question 11, which records four issues the diagnostic surfaced that need
author rulings before implementation.

### 2026-08-01 — Figure 3 note correction

**Done.** Regenerated Figure 3 with a corrected note. The shipped figure still
carried a Block 1 statement that the Block 2 grid correction had invalidated:
*"window is 252 observations (~1.75 calendar years on this irregular grid)"*.
Both claims are false on the corrected grid — 252 observations span **1.01**
calendar years at `ANN_FACTOR = 248.48`, and the grid is near-uniform, which is
the *opposite* of what Figure A.1 in the same submission demonstrates. This was
the only instance of the stale Block 1 prose rendered into a shipped artefact.

The note now reads:

> Note: 252-observation rolling window (~1.0 calendar year), not 252 calendar
> days. Built from the one-sided K=2 probability series.

Wording supplied by the author. The two trailing clauses were retained on my
recommendation: the observations-vs-days distinction is not carried by the new
phrasing on its own, and the provenance sentence is worth keeping. Also
corrected the module docstring in the same file, which repeated the same stale
facts (`ANN_FACTOR = 144.40`, "irregular grid", 1.75 years).

**Tested.** `python -m research.figures.figure3_rolling_hit_rate`, exit 0. Plot
data **unchanged** — the script re-reported AR 85.1%, RW 82.6%, always-calm
85.7%, identical to the committed values, because
`dominant_regime_forecast_eval.csv` was not touched. Only the note text differs.
The `.png`/`.pdf` binary diffs are larger than the one line of changed text
because matplotlib rewrites embedded metadata on every render.

**Committed** with this entry.

**Next.** Open question 10 — the same Block 1 prose survives in four other
files. None affects a computed number, but three of them misstate the sample
size or the annualisation factor in a reproduction package. Author to decide
whether to sweep them.

### 2026-07-26 — Block 2, Phase 2 (draft text) — BLOCKED, resume here

**Done.** Drafted the Section 3.1 filter-disclosure text. Author approved it
with two corrections I raised against the brief:

1. The brief's asset list read "(equities, government bonds, corporate bonds,
   commodities, cash proxy)". Wrong on two counts — **gold** was missing and the
   cash proxy was wrongly included, which would have contradicted the very next
   clause stating cash is retained unconditionally. The five *tested* assets are
   equities, government bonds, IG credit, commodities and gold.
2. The "3.1-5.0%" drop-rate band holds **from 2002 onward**, not across all
   years: 2001 is 11.1%, and 2000 (95.4%) / 2026 (11.8%) are partial years.

Author left the final-sentence split to my judgement; I took the split (the
unsplit version ran to five clauses).

**APPROVED FINAL TEXT** — insert immediately after the sentence ending
"...and runs to January 2026" in Section 3.1:

> Observations are retained only on days where all five market indices — global
> equities, government bonds, investment-grade corporate credit, commodities and
> gold — recorded a non-zero log return, while the short-duration Treasury
> series used as a cash proxy (LD12TRUU) is retained unconditionally regardless
> of its return. This asymmetry is deliberate: because prices are forward-filled
> onto a common daily grid, a market index that did not update its price yields a
> return of exactly zero, and admitting such a day would insert a spurious zero
> into the cross-asset return vector and so distort the covariance estimates from
> which the mixture model identifies regimes. For the cash proxy the same zeros
> carry the opposite meaning — those that dominate the zero-interest-rate era of
> approximately 2009–2016 and 2021 reflect genuine near-zero accrual at the
> policy-rate floor rather than price staleness — so retaining them is
> economically correct. The filter removes 336 of 6,605 observations, leaving
> 6,269, a drop rate of 5.1% that is distributed near-uniformly at between 3.1%
> and 5.0% in every year from 2002 onward and is therefore consistent with
> genuine cross-market holiday mismatches rather than the systematic exclusion of
> any particular period. Observation counts average approximately 250 per year
> across the sample, and all window and horizon references in this paper are
> stated in observation counts rather than calendar days. The annual distribution
> is shown in Figure A.1.

**BLOCKED — the chapter document does not exist in this environment.** Asked to
write the text into the manuscript as a tracked insertion. Searched the repo,
the `/workspaces` tree, the home directory, the whole filesystem (`find / -xdev`)
and every file ever added in git history across all branches: **no `.docx`,
`.doc`, `.odt`, `.tex` or `.rtf` anywhere**. `manuscript_changes_v2.docx` has
never existed in this codespace. The chapter is presumably local to the author's
machine in Word. Nothing was created, per instruction.

Note for whoever resumes: there is also no "previous tracked change format" to
match — no chapter document has been edited in this project at any point.

Second constraint if the file is supplied: `.docx` tracked changes are `w:ins`
elements in OOXML, and `python-docx` has no API for them. Doing it properly
means hand-editing `word/document.xml` inside the zip, which Word validates
strictly. If we go that route, work on a copy first and confirm it opens and
renders correctly in Word before replacing the original.

**Next — pick one:**
1. Author pastes the approved text into Word with Track Changes on. Fastest,
   correct revision metadata, zero risk to the file. Text is above.
2. Author copies the `.docx` into the workspace and gives the path; I do the
   OOXML insertion on a copy for verification first.
3. Point me at the document via a connector, if one is set up.

### 2026-07-26 — Block 2, session 4 (option (a) implementation)

**Done.** Implemented option (a) — the synchronous-trading filter now tests the
five market indices only; cash (`LD12TRUU`) is retained unconditionally.

1. Consolidated the filter into `core.utils.filter_synchronous_trading`, one
   definition replacing five independent copies. `CASH_LIKE_ASSETS = ("high_yield",)`
   is exempt; names absent from a universe are ignored, so it is safe across the
   core and extended tiers.
2. `ANN_FACTOR` 144.40 → **248.48**. `WINDOW_SIZE` comment reverted to
   "≈ 5.0 years (range 4.96-5.24)" — the Block 1 F-b correction is no longer
   needed, because the corrected grid really is near-regular.
3. Added `research/figures/figureA1_observations_per_year.py` as a permanent
   producer, matching the repo's one-producer-per-figure convention.
4. README's "observation grid is irregular" section replaced by "The
   synchronous-trading filter", with the corrected rationale, the cash
   exemption, the 96.8% attribution, and recomputed look-ahead probes.
5. Regenerated **everything**: pipeline, forecast_eval, backtest, Figures 1-3,
   Figure A.1, moments, correlations, robustness, shrinkage, stability.
6. Open question 9 closed; Block 3 cancelled.

**Tested.** T = 3,643 → **6,269**. One-sided series 2,393 → **5,019 rows**,
starting 2006-01-30. Backtest starts **2007-03-06**. Rolling windows 38 → **80**.
Grid now near-regular: every full year 232-253 observations (mean 250.0), vs
13-250 before.

Validation re-run on the corrected grid: **T1, T2, T3, T5 all PASS** (T1 and T2
both exact zero). **T4 PASS under its redefinition** — all four probability CSVs
byte-identical across two independent full pipeline runs. T6 reported.

**Two Block 1 conclusions changed.** Short-horizon forecast skill does not
survive (AR Change now fails the always-calm baseline at every horizon), and the
AR strategies now beat the random walk in the backtest (0.753 vs 0.648),
reversing Block 1. Both are recorded under "Headline results — FINAL", and the
chapter-text follow-up has been rewritten accordingly. Block 1's numbers were
computed on a sample that excluded most of the zero-rate era and are superseded.

**Not committed.** Awaiting author confirmation of the four requested items.

**Next.** Author confirms → stage and commit. Then Block 2 Phase 2 (the
Section 3.1 draft text), which now needs a different item (d): the grid is
regular, not irregular, so the appendix chart demonstrates regularity.

### 2026-07-26 — Block 2, Phase 1 (data-section analysis)

**Done.** Three read-only analyses on the existing filtered grid. No code, data
or chapter changes. Produced `research/output_charts/figures/
figureA1_observations_per_year.{png,pdf}` (single-series bar chart, house style).

**Findings — the headline result contradicts the expected framing.**

1. **Stress episodes are partially and unevenly excluded.** Baseline drop rate
   across the whole pre-filter grid is 44.8%.

   | Window | dates | dropped | % of all drops | % of window | vs baseline |
   |---|---|---|---|---|---|
   | GFC 2007-06→2009-03 | 478 | 116 | 3.9% | 24.3% | −20.6pp |
   | EU sovereign 2010-05→2012-09 | 630 | 537 | 18.1% | **85.2%** | **+40.4pp** |
   | COVID 2020-02→2020-06 | 107 | 59 | 2.0% | **55.1%** | **+10.3pp** |
   | 2022 tightening | 260 | 101 | 3.4% | 38.8% | −6.0pp |

   The GFC survives well (362 observations retained). The **EU sovereign debt
   crisis does not** — 85.2% of its dates are dropped, leaving 93 observations
   across a 29-month crisis. COVID retains only 48 of 107.

2. **The density shift is NOT concentrated in the early sample.** It is
   concentrated in the middle: 2000-2008 drops 22.4%, **2009-2016 drops 82.8%**,
   2017-2026 drops 31.7%. Worst years: 2021 (95.0%), 2013 (90.0%), 2015 (88.9%),
   2011 (87.7%).

3. **The filter is driven almost entirely by one series.** `high_yield` is zero
   in **96.8%** of all dropped rows (100% within the EU sovereign window). No
   other asset exceeds 8.5%.

**This undermines the intended justification.** `high_yield` is `LD12TRUU Index`
— a Bloomberg US Short Treasury 1-12 Month index, displayed as "Cash" in
`ASSET_DISPLAY`. Its zero-return stretches coincide exactly with zero-interest-
rate policy: 2009-2016 (price pinned in [191.19, 192.23], 105 distinct values in
seven years), 2021 (10 distinct values all year), 2003-2004 (Fed at 1%), versus
4.6% zero days in 2023-2025 once rates normalised.

Those zeros are therefore **not stale quotes from an asset that did not trade**.
They are a rounding artefact: at a near-zero policy rate the true daily accrual
on a T-bill index falls below the stored price precision (~2.6bp on a level of
192 recorded to 2 d.p.). The synchronous-trading argument specified for Phase 2
item (a) is **not accurate for the cause of 96.8% of the exclusions**, and item
(c) cannot "confirm stress periods are not systematically excluded" — one major
episode is.

**Phase 2 held.** Drafting the specified text would put an inaccurate rationale
into the manuscript. Author decision required — see "Open questions" item 8.

**Next.** Author rules on item 8; then draft Section 3.1 text against whatever
rationale is actually correct.

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

10. **[RESOLVED 2026-08-01 — swept, see Session log]** Stale Block 1 prose survived the Block 2 grid
    correction in files other than Figure 3. **No computed value is affected** —
    every live annualisation reads `ANN_FACTOR` from `core.config`, so Table 3
    and the backtest metrics are correct. But a reproduction package that states
    the wrong sample size and the wrong annualisation factor in its own
    documentation is a reviewer-visible defect.

    | Location | Says | Should say |
    |---|---|---|
    | `research/figures/figure3_rolling_hit_rate.py:7-8` | ANN_FACTOR 144.40, irregular grid, 1.75 yr | **FIXED 2026-08-01** |
    | `research/analysis/moments.py:12-14` | ANN_FACTOR 144.40, 3643 observations | 248.48, 6269 |
    | `research/analysis/moments.py:73` | prints "(irregular grid)" | near-uniform grid |
    | `core/config.py:32` | one-sided series starts 2007-02-05 | 2006-01-30 |
    | `core/config.py:67` | 250 observations span ~1.7 calendar years | ~1.0 calendar year |
    | `research/analysis/pipeline.py:15` | one-sided starts 2007-02-05 | 2006-01-30 |
    | `research/figures/figure2_forecast_accuracy.py:8` | one-sided starts 2007-02-05 | 2006-01-30 |

    All are comments, docstrings or one stdout line. A sweep changes no output
    file except `moments.py`'s printed header.

    **Resolution: swept 2026-08-01, per author instruction.** All seven sites
    corrected. Verified doc-only: `research.analysis.moments` re-run exit 0 and
    `table3_regime_moments.{csv,tsv}` are unchanged on disk.

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

9. **[RESOLVED 2026-07-26 — implemented, see session 4]** Block 2 implementation
   questions.

   **9a resolved:** regenerate the retrospective CSVs in place; T4 is redefined
   against the corrected 5-asset grid. The old committed files are superseded,
   not kept as separate artefacts.
   **9b resolved:** Block 3 cancelled — see the Block 3 section.

   Option (a) analysis, 2026-07-26. Effects:

   | | OLD (6-asset filter) | NEW (5 non-cash) |
   |---|---|---|
   | T | 3,643 | **6,269** (+72%) |
   | 2009-2016 observations | 358 | **2,012** |
   | 2020 / 2021 | 77 / 13 | **253 / 252** |
   | Per-year drop rate | 4.6-95.0% | **3.1-5.0%** |
   | `ANN_FACTOR` | 144.40 | **248.48** |
   | 1250-obs window span | 5.96-14.28 yr | **4.96-5.24 yr** |
   | Rolling windows | 38 | **80** |
   | Last window start/end idx | 2331 / 3580 | **4977 / 6226** |
   | One-sided series | 2007-02-05, 2,393 rows | **2006-01-30, 5,019 rows** |
   | Backtest start | 2008-04-01 | **2007-03-06** |

   The grid becomes near-regular; the residual 3-5% drop is genuine
   non-synchronous market holidays across five markets, which is what the
   filter was always meant to catch.

   The filter was replicated at **five** sites (`pipeline.py`, `robustness.py`,
   `shrinkage.py`, `stability_analyzer.py`, `validate_onesided.py`) which could
   silently drift apart. Consolidated into
   `core.utils.filter_synchronous_trading`.

8. **[RESOLVED 2026-07-26 — option (a) approved by author]** The filter's stated
   rationale did not match its actual behaviour, and it removed most of the
   zero-rate era.

   **Resolution: apply `(df != 0).all(axis=1)` to the five non-cash market
   index assets only; Cash (`LD12TRUU`) is retained unconditionally.** The
   synchronous-trading argument is valid for the five market indices, where a
   forward-filled zero really does mean the market did not trade. It was never
   valid for a short-Treasury index at a pinned policy rate, where the zero is a
   rounding artefact of a genuinely near-zero return. Implementation effects and
   remaining decisions are tracked as item 9.

   Original diagnosis retained below for the record.

   Phase 1 established that 96.8% of dropped rows are dropped because
   `high_yield` (`LD12TRUU Index`, a 1-12 month short-Treasury index displayed
   as "Cash") has a zero return, and that its zero stretches coincide with ZIRP
   rather than with non-trading. The consequences:

   - The "stale price / asset did not trade" justification is **wrong for the
     dominant cause**. At a pinned policy rate the true daily accrual is smaller
     than the stored price precision, so the zero is a rounding artefact of a
     genuinely near-zero return — not a missing observation.
   - The filter therefore removes **most of 2009-2016 and 2020-2021** — the
     zero-rate era — and with it 85.2% of the EU sovereign debt crisis and 55.1%
     of the COVID drawdown.
   - The GMM never sees those periods, so the "regimes" it identifies are
     estimated almost entirely from positive-rate environments.

   Options:

   - **(a) Restrict the filter to the five non-cash assets.** Requires the joint
     return vector to be informative where it matters for covariance structure,
     while letting a legitimately-flat cash series through. Recovers most of
     2009-2016 and the EU sovereign window. Changes `T` and re-opens every
     Block 1 baseline — a full re-run and re-validation.
   - **(b) Drop `high_yield`/"Cash" from the universe** and run on five assets.
     Cleanest statistically (a pinned cash series contributes almost nothing to
     a regime covariance structure anyway) but changes the asset universe the
     editor reviewed and affects Tables 1, 3, 4.
   - **(c) Keep the filter, and disclose accurately** — state that the sample is
     conditioned on positive-rate periods and that the regime model is not
     identified over ZIRP. Honest, requires no re-run, but is a substantial
     limitation to put in writing.
   - **(d) Keep the filter and write the originally-specified rationale.** Not
     available — it would put an inaccurate statement in the manuscript.

   No code, data or text changed pending this decision.

*Questions 1-5 were raised before implementation and all resolved on
2026-07-26; resolutions recorded below. Questions 6 and 7 arose during
validation and coverage review and were resolved the same day. Question 8 arose
in Block 2 Phase 1 and is open.*

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

Phase 1 (analysis) is **complete** — see the session-3 and session-4 entries.
Phase 2 (the Section 3.1 draft text) is **outstanding**, and its brief has
changed because Phase 1 overturned the premise it was written against.

Required content, as revised:

- **(a) Justification — corrected.** The synchronous-trading argument applies to
  the **five market indices**: a forward-filled zero means the market did not
  trade, and such a row would corrupt the cross-asset covariance the GMM relies
  on. It does **not** apply to cash (`LD12TRUU`), where a zero at a pinned policy
  rate is an economically genuine near-zero return. Testing cash accounted for
  96.8% of all drops. The original brief's blanket "any asset did not trade"
  wording would be inaccurate.
- **(b) Effect — corrected figures.** 6,605 → **6,269** observations (not 3,643).
- **(c) Stress-episode result.** Under the corrected filter no stress window is
  materially affected. Report the superseded 6-asset behaviour only if the
  chapter needs to explain the correction.
- **(d) Caption — rewritten.** State that the filter achieves **near-uniform
  synchronous trading coverage**. The "13-250 obs/year" line is obsolete and must
  be removed; every full year now retains 232-253 observations (mean 250.0).
  Reference Figure A.1, which now demonstrates regularity rather than
  irregularity.
- **(e) Window span.** "1,250 trading days (approximately five years)" is now
  accurate — the corrected grid gives 4.96-5.24 calendar years. No correction
  needed, contrary to the original brief.
- **Framing:** a motivated methodological choice, not data cleaning.

Draft length: 3-5 sentences integrated into the existing Section 3.1 paragraph.
Author edits and approves before anything is written to the chapter document.

## Block 3 — CANCELLED 2026-07-26

The expanding-window backtest appendix is **cancelled**, by author decision.

**Reason: rendered unnecessary by the option (a) grid correction.** Block 3
existed to extend backtest coverage earlier than 2008-04-01 without an
expanding-window estimation spec being the only route. Correcting the filter to
exempt cash raises T from 3,643 to 6,269 and moves the backtest start to
**2007-03-06** — thirteen months earlier and before the 2008 drawdown window, so
the full 2007 crisis onset is now covered on the paper's own rolling
1250-observation spec. The appendix would add an inconsistent estimation spec
(250 → 1250 expanding, then rolling) for coverage the corrected grid already
provides.

## Chapter-text follow-up (Block 1 finding, author to action)

**Author-approved framing, 2026-07-26.** The Block 1 "short-horizon skill"
framing is withdrawn — it does not survive the corrected grid and must not
appear in the paper. Do not attempt to salvage it.

The conclusion and abstract take these three findings:

1. **Regime information has genuine economic value:** Oracle (Sharpe 1.154)
   decisively clears Unconditional (0.799).

2. **AR forecast portfolios partially capture the signal in portfolio terms** —
   AR Baseline and AR Change (both 0.753) beat Random Walk (0.648) — but fall
   short of the unconditional benchmark (0.799). The regime signal is real but
   the forecast is too noisy to translate into a net improvement.

3. **Forecast accuracy is below the always-calm baseline at every horizon**
   (−0.91pp at h=1, −2.42pp at h=5, −2.89pp at h=21). Regime transitions are not
   predictable in a classification sense, even though the portfolio construction
   around probabilistic forecasts adds modest value over pure random walk.

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

**Re-run 2026-07-26 on the CORRECTED 5-asset grid (T = 6,269)** via
`python -m research.analysis.validate_onesided --t2`. Block 1 results on the
superseded 3,643-row grid are noted in each row for comparison.

| Test | Description | Status | Result (corrected grid) |
|---|---|---|---|
| **T1** | **Truncation invariance (decisive).** Fit on `X[:t]` only, compute the one-sided series, compare against the full-sample fit's one-sided series over the dates both can serve. Must be exactly equal. Three cut points. | **PASS** | Cuts at t=2821 (2012-04-27), t=4388 (2018-07-24), t=5516 (2023-01-20). `max abs diff = 0.000e+00`, **bitwise identical** at all three. (Block 1: also PASS.) |
| **T2** | **Future corruption.** Replace `X[t+1:]` with N(0, 10σ) noise, refit, assert `p_s` unchanged for all `s ≤ t`. Catches leakage via BIC/K-selection and the EWMA/prior paths that T1 could mask. | **PASS** | `max abs diff` on all past dates = `0.000e+00` at all three cut points. (Block 1: also PASS.) |
| **T3** | **Index-level assertion (also inline in `get_filtered_probabilities`, every call).** Every emitted date is served by a model with `e_w < t`; the date→model map is a strict partition; first emitted index > `WINDOW_SIZE - 1`. | **PASS** | 0 dates without a strictly-past model; 0 duplicates; first emitted index 1250; map covers 5019 dates = 5019 emitted. |
| **T4** | **REDEFINED (author decision, Open question 9a).** Formerly byte-identity against the committed 6-asset CSVs — impossible once the input grid changes. Now: **byte-identity across independent pipeline runs on the corrected grid**, i.e. reproducibility. | **PASS** | All four probability CSVs byte-identical across two independent full pipeline runs (`cmp` clean). The superseded 6-asset files are replaced in place, not retained. |
| **T5** | **Structural sanity.** Row sums = 1 ± 1e-9; no NaN; no negative probabilities; monotonic unique index; `len(one_sided) == T - WINDOW_SIZE`. | **PASS** | All checks pass; length 5019 = 6269 − 1250. |
| **T6** | **Honest-skill report (output, not pass/fail).** Figure 2 metrics against the always-calm baseline. | **PASS (reported)** | **Conclusion changed — see below.** AR Change now **fails to beat the baseline at every horizon**: −0.91pp (h=1), −2.42pp (h=5), −2.89pp (h=21). Block 1's short-horizon skill (+1.04pp at h=1) does not survive the grid correction. |

### Headline results — FINAL (corrected 5-asset grid, T = 6,269)

**Both Block 1 conclusions changed when the grid was corrected.** Block 1's
numbers were computed on a sample that excluded most of the zero-rate era; they
are superseded.

Forecast hit rates vs the always-calm baseline, `ar_change_multihorizon_summary.csv`:

| Horizon | Always-calm | AR Change | Gain | AR Baseline | Random Walk |
|---|---|---|---|---|---|
| 1-day | 0.8572 | 0.8481 | **−0.91pp** | 0.8564 | 0.8265 |
| 5-day | 0.8570 | 0.8328 | **−2.42pp** | 0.8540 | 0.8128 |
| 21-day | 0.8572 | 0.8283 | **−2.89pp** | 0.8583 | 0.8075 |

Superseded Block 1 gains (3,643-row grid): +1.04pp / +0.09pp / −2.97pp.
**Change 1: the short-horizon forecast skill does not survive.** AR Change now
fails to beat a constant "always calm" classifier at every horizon. AR Baseline
is essentially at the baseline (−0.08pp, −0.30pp, +0.11pp).

Figure 3 (`dominant_regime_forecast_eval.csv`): AR 0.851 vs RW 0.826, against an
always-calm baseline of 0.857 — the AR model beats the random walk but neither
beats the trivial classifier.

Figure 4 backtest (`ANN_FACTOR` = 248.48), all five from 2007-03-06:

| Strategy | Sharpe | Ann. return | Max DD | 1st rebalance | Turnover |
|---|---|---|---|---|---|
| Oracle (infeasible) | **1.154** | 4.58% | −14.8% | 2007-03-30 | 0.0019 |
| Unconditional | 0.799 | 2.63% | −14.7% | 2007-02-28 | 0.0007 |
| AR Baseline | 0.753 | 3.33% | −18.5% | 2007-03-30 | 0.0011 |
| AR Change | 0.753 | 3.31% | −18.2% | 2007-03-30 | 0.0030 |
| Random Walk | 0.648 | 2.96% | −21.8% | 2007-03-30 | 0.0041 |

Superseded Block 1 Sharpes (3,643-row grid): Oracle 0.718, Unconditional 0.449,
AR Change 0.396, AR Baseline 0.373, RW 0.430.
**Change 2: the AR strategies now beat the random walk** (0.753 vs 0.648),
reversing Block 1. Both still fall short of the unconditional benchmark (0.799),
and both carry ~4pp worse drawdown.

**What survives both grids:** the infeasible Oracle clears the unconditional
benchmark decisively (1.154 vs 0.799), so the regime signal carries real
economic value; and no feasible forecast converts that into a Sharpe improvement
over unconditional max-Sharpe allocation.

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

Sharpe progression across every stage of the revision (each row supersedes the
one above; only the last is current):

| Stage | Grid | Oracle | Uncond. | AR Change | AR Base | RW |
|---|---|---|---|---|---|---|
| Before `orig_idx` fix | 3,643 | 0.256 | 0.449 | 0.181 | 0.159 | 0.092 |
| After `orig_idx`, before slice fix | 3,643 | 0.545 | 0.449 | 0.297 | 0.271 | 0.233 |
| Block 1 final | 3,643 | 0.718 | 0.449 | 0.396 | 0.373 | 0.430 |
| **Block 2 final (current)** | **6,269** | **1.154** | **0.799** | **0.753** | **0.753** | **0.648** |
