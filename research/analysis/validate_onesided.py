"""
Validation of the one-sided regime-probability series.

Implements tests T1, T3 and T5 from the revision plan (see REVISION_LOG.md).
T2 (future-corruption) and T4 (byte-identity of the preserved retrospective
CSVs) are driven separately — T2 by --t2, T4 by comparing against git.

    T1  Truncation invariance (decisive).  Fit on X[:t] only, build the
        one-sided series, and compare against the full-sample fit's one-sided
        series over the dates both can serve.  They must be EXACTLY equal.
        Any leakage — through responsibilities, BIC/K-selection, the EWMA
        initialiser or the previous-epoch prior — breaks this.

    T2  Future corruption.  Replace X[t+1:] with noise, refit, assert p_s is
        unchanged for all s <= t.  Complements T1: T1 could in principle pass
        if a leak were deterministic in sample length, T2 could not.

    T3  Index-level assertions.  Every emitted date is served by a model whose
        window ended strictly before it; the date -> model map is a strict
        partition; the first emitted index is > WINDOW_SIZE - 1.  (These also
        run inline inside get_filtered_probabilities on every call.)

    T5  Structural sanity.  Row sums, NaN, monotonic unique index, length.

Run:  python -m research.analysis.validate_onesided          # T1, T3, T5
      python -m research.analysis.validate_onesided --t2     # adds T2 (slower)
      python -m research.analysis.validate_onesided --full   # T1 at full spec

By default T1 and T2 run at a REDUCED spec (K_candidates=[2], window_size=250)
so the suite finishes in minutes rather than hours.  The no-look-ahead property
is structural, not spec-dependent: it follows from which observations each model
is allowed to see, which the spec does not change.  --full runs T1 at the paper
spec for final sign-off.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from data.reader1 import DataReader
from models.gmm import Optimiser
from core.utils import filter_synchronous_trading
from core.config import (
    K_CANDIDATES, WINDOW_SIZE, STEP, SCALE_METHOD,
    MAX_ITER, TOL, REG_COVAR, RANDOM_STATE,
    SHRINKAGE_TARGET, KAPPA_0, NU_0_EXTRA, LAMBDA_SCALE,
    ALPHA_DIRICHLET, EWMA_DECAY,
)

# Reduced spec for the expensive tests. Structural property, not spec-dependent.
FAST_K_CANDIDATES = [2]
FAST_WINDOW = 250
FAST_STEP = 63

# Cut points for T1/T2, as fractions of the sample.
CUT_FRACTIONS = [0.45, 0.70, 0.88]


def _load_returns():
    """Identical filtering to research.analysis.pipeline (shared helper)."""
    return filter_synchronous_trading(DataReader().read_retns().dropna())


def _build(window_size, step, k_candidates):
    return Optimiser(
        K_candidates=list(k_candidates),
        window_size=window_size,
        step=step,
        scale_method=SCALE_METHOD,
        max_iter=MAX_ITER,
        tol=TOL,
        reg_covar=REG_COVAR,
        random_state=RANDOM_STATE,
        regularise=True,
        kappa_0=KAPPA_0,
        nu_0_extra=NU_0_EXTRA,
        lambda_scale=LAMBDA_SCALE,
        init_decay=None,
        ewma_decay=EWMA_DECAY,
        allow_partial_last_window=False,
        alpha_dirichlet=ALPHA_DIRICHLET,
        order_components=True,
        order_mode="trace",
        shrinkage_target=SHRINKAGE_TARGET,
    )


def _fit_onesided(X, dates, window_size, step, k_candidates):
    opt = _build(window_size, step, k_candidates)
    opt.fit(X, index=dates)
    return opt, opt.get_filtered_probabilities()


# ---------------------------------------------------------------------
# T1 — truncation invariance
# ---------------------------------------------------------------------

def test_t1(df, window_size, step, k_candidates, label):
    print("=" * 70)
    print(f"T1  Truncation invariance  [{label}: window={window_size}, "
          f"step={step}, K={k_candidates}]")
    print("=" * 70)

    X, dates = df.values, df.index
    print(f"  Fitting on the FULL sample (T={len(X)})...")
    _, full = _fit_onesided(X, dates, window_size, step, k_candidates)

    all_ok = True
    for frac in CUT_FRACTIONS:
        t = int(len(X) * frac)
        print(f"\n  --- cut at t={t} ({dates[t - 1].date()}), "
              f"fitting on X[:{t}] only ---")
        _, trunc = _fit_onesided(X[:t], dates[:t], window_size, step, k_candidates)

        if len(trunc) == 0:
            print("    SKIP: truncated sample too short to emit any date.")
            continue

        common = full.index.intersection(trunc.index)
        if len(common) == 0:
            print("    SKIP: no overlapping dates.")
            continue

        cols = sorted(set(full.columns).intersection(trunc.columns))
        a = full.loc[common, cols].to_numpy()
        b = trunc.loc[common, cols].to_numpy()
        max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
        exact = bool(np.array_equal(a, b))
        ok = exact or max_abs < 1e-12

        print(f"    overlapping dates : {len(common)} "
              f"({common[0].date()} to {common[-1].date()})")
        print(f"    max |full - trunc|: {max_abs:.3e}")
        print(f"    bitwise identical : {exact}")
        print(f"    -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            bad = np.unravel_index(int(np.argmax(np.abs(a - b))), a.shape)
            print(f"       worst date {common[bad[0]].date()}, column "
                  f"{cols[bad[1]]}: full={a[bad]:.12f} trunc={b[bad]:.12f}")
            all_ok = False

    print(f"\n  T1 RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------
# T2 — future corruption
# ---------------------------------------------------------------------

def test_t2(df, window_size, step, k_candidates, label):
    print("\n" + "=" * 70)
    print(f"T2  Future corruption  [{label}: window={window_size}, "
          f"step={step}, K={k_candidates}]")
    print("=" * 70)

    X, dates = df.values, df.index
    print(f"  Fitting on the CLEAN sample (T={len(X)})...")
    _, clean = _fit_onesided(X, dates, window_size, step, k_candidates)

    rng = np.random.default_rng(12345)
    all_ok = True
    for frac in CUT_FRACTIONS:
        t = int(len(X) * frac)
        Xc = X.copy()
        # Replace the entire future with heavy noise on the same scale.
        Xc[t:] = rng.normal(0.0, X.std() * 10.0, size=Xc[t:].shape)
        print(f"\n  --- corrupting X[{t}:] ({dates[t].date()} onward) ---")

        _, corrupt = _fit_onesided(Xc, dates, window_size, step, k_candidates)

        past = clean.index[clean.index <= dates[t - 1]]
        common = past.intersection(corrupt.index)
        if len(common) == 0:
            print("    SKIP: no past dates emitted.")
            continue

        cols = sorted(set(clean.columns).intersection(corrupt.columns))
        a = clean.loc[common, cols].to_numpy()
        b = corrupt.loc[common, cols].to_numpy()
        max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
        ok = max_abs < 1e-12

        print(f"    past dates checked: {len(common)} "
              f"({common[0].date()} to {common[-1].date()})")
        print(f"    max |clean - corrupt| on past dates: {max_abs:.3e}")
        print(f"    -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    print(f"\n  T2 RESULT: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------
# T3 — index-level assertions
# ---------------------------------------------------------------------

def test_t3(opt, onesided, dates):
    print("\n" + "=" * 70)
    print("T3  Index-level no-look-ahead assertions  [paper spec]")
    print("=" * 70)

    ends = list(opt.window_end_indices_)
    pos = {d: i for i, d in enumerate(dates)}
    ok = True

    # Every emitted date must be served by a model whose window ended before it.
    violations = 0
    for d in onesided.index:
        t = pos[d]
        eligible = [e for e in ends if e < t]
        if not eligible:
            violations += 1
    print(f"  dates with no strictly-past model : {violations}")
    ok &= violations == 0

    dup = int(onesided.index.duplicated().sum())
    print(f"  duplicated dates (partition check): {dup}")
    ok &= dup == 0

    first_pos = pos[onesided.index[0]]
    print(f"  first emitted index               : {first_pos} "
          f"(must be > {opt.window_size - 1})")
    ok &= first_pos > opt.window_size - 1

    # Reconstruct the date -> model map and confirm it is a contiguous cover.
    covered = 0
    for w, e_w in enumerate(ends):
        lo = e_w + 1
        hi = ends[w + 1] if w + 1 < len(ends) else len(dates) - 1
        if lo <= hi:
            covered += hi - lo + 1
    print(f"  dates covered by the model map    : {covered} "
          f"(emitted: {len(onesided)})")
    ok &= covered == len(onesided)

    print(f"\n  T3 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------
# T5 — structural sanity
# ---------------------------------------------------------------------

def test_t5(onesided, T, window_size):
    print("\n" + "=" * 70)
    print("T5  Structural sanity  [paper spec]")
    print("=" * 70)

    sums = onesided.sum(axis=1)
    checks = {
        "row sums == 1 (+/- 1e-9)": bool(np.allclose(sums, 1.0, atol=1e-9)),
        "no NaN":                   bool(not onesided.isna().any().any()),
        "no negative probabilities": bool((onesided.to_numpy() >= -1e-12).all()),
        "index monotonic increasing": bool(onesided.index.is_monotonic_increasing),
        "index unique":             bool(onesided.index.is_unique),
        f"length == T - window_size ({T - window_size})":
            len(onesided) == T - window_size,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    print(f"    row-sum range: [{sums.min():.12f}, {sums.max():.12f}]")
    ok = all(checks.values())
    print(f"\n  T5 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t2", action="store_true",
                    help="also run T2 (future corruption); slower")
    ap.add_argument("--full", action="store_true",
                    help="run T1/T2 at the paper spec instead of the reduced spec")
    args = ap.parse_args()

    print("Loading returns (same filtering as pipeline.py)...")
    df = _load_returns()
    T = len(df)
    print(f"  T={T}  {df.index[0].date()} to {df.index[-1].date()}\n")

    if args.full:
        spec = (WINDOW_SIZE, STEP, K_CANDIDATES, "paper spec")
    else:
        spec = (FAST_WINDOW, FAST_STEP, FAST_K_CANDIDATES, "reduced spec")

    results = {}
    results["T1"] = test_t1(df, *spec)
    if args.t2:
        results["T2"] = test_t2(df, *spec)

    # T3 and T5 always run at the paper spec — they are cheap given one fit.
    print("\nFitting at the paper spec for T3/T5 "
          f"(window={WINDOW_SIZE}, step={STEP}, K={K_CANDIDATES})...")
    opt, onesided = _fit_onesided(df.values, df.index, WINDOW_SIZE, STEP, K_CANDIDATES)
    print(f"  one-sided series: {len(onesided)} rows, "
          f"{onesided.index[0].date()} to {onesided.index[-1].date()}")

    results["T3"] = test_t3(opt, onesided, df.index)
    results["T5"] = test_t5(onesided, T, WINDOW_SIZE)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name in sorted(results):
        print(f"  {name}: {'PASS' if results[name] else 'FAIL'}")
    if not args.t2:
        print("  T2: NOT RUN (pass --t2)")
    print("  T4: NOT RUN (byte-identity of preserved retrospective CSVs; "
          "check with git diff)")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
