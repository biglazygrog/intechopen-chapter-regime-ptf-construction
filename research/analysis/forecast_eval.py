"""
Multi-horizon AR forecast evaluation.

Compares three forecasters for the high-vol regime probability series:
- AR Change : autoregression on delta-logit (change in logit probability)
- AR Baseline: autoregression on the logit level
- Random Walk

Produces ar_change_multihorizon_summary.csv used by Figure 2 and Figure 3.

Run:  python -m research.analysis.forecast_eval
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from scipy import stats
from sklearn.metrics import roc_auc_score

from core.config import (
    OUTPUT_DIR, AR_HORIZONS, AR_MAX_LAG, AR_MIN_TRAIN, AR_REFIT_FREQ,
    TARGET_REGIME, HV_THRESHOLD,
)


def logit(p, eps=1e-8):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(y):
    return 1 / (1 + np.exp(-y))


def select_ar_order_bic(y: np.ndarray, max_lag: int = 5) -> int:
    """Select optimal AR order using BIC."""
    best_bic = np.inf
    best_order = 1

    for p in range(1, max_lag + 1):
        try:
            model = AutoReg(y, lags=p, old_names=False).fit()
            if model.bic < best_bic:
                best_bic = model.bic
                best_order = p
        except Exception:
            continue

    return best_order


def evaluate_ar_change_multihorizon(
    daily_probs_df: pd.DataFrame,
    target_regime: str = "p_1",
    horizons: list = [1, 5, 21],
    max_ar_lag: int = 5,
    min_train: int = 250,
    refit_frequency: int = 21,
):
    """
    Evaluate AR Change forecast accuracy at multiple horizons.

    AR Change: Forecast delta-logit (change in logit probability) instead of level.
    """

    pcols = sorted([c for c in daily_probs_df.columns if c.startswith("p_")],
                   key=lambda c: int(c.split("_")[1]))
    P = daily_probs_df[pcols].copy()

    if target_regime not in P.columns:
        raise ValueError(f"target_regime={target_regime} not in columns.")

    # Binary outcome: is target regime the argmax?
    hard = P.values.argmax(axis=1)
    dom_k = int(target_regime.split("_")[1])
    z = (hard == dom_k).astype(int)

    eps = 1e-8
    p = P[target_regime].values
    y = logit(p)

    # First difference of logit
    dy = np.diff(y)
    dy = np.concatenate([[0], dy])  # Pad to maintain alignment

    idx = P.index
    max_horizon = max(horizons)

    # Storage for each horizon
    results = {h: {"rows": [], "ar_orders": []} for h in horizons}

    current_ar_order = 1
    current_ar_order_baseline = 1

    print(f"Evaluating AR CHANGE forecasts at horizons: {horizons} days")
    print(f"Total observations: {len(P)}, Min train: {min_train}")
    print()

    for t in range(min_train, len(P) - max_horizon):
        if t % 500 == 0:
            print(f"  Processing t={t}/{len(P) - max_horizon}...")

        y_train = y[:t+1]
        dy_train = dy[1:t+1]  # Exclude the padded zero

        # Re-select AR order periodically
        if (t - min_train) % refit_frequency == 0:
            # AR order for delta-logit
            current_ar_order = select_ar_order_bic(dy_train, max_lag=max_ar_lag)
            # AR order for level (baseline comparison)
            current_ar_order_baseline = select_ar_order_bic(y_train, max_lag=max_ar_lag)

        # === AR CHANGE FORECAST ===
        try:
            fit_change = AutoReg(dy_train, lags=current_ar_order, old_names=False).fit()
            dyhat_all = fit_change.forecast(steps=max_horizon)
            dyhat_all = np.asarray(dyhat_all)

            # Cumulative sum of predicted changes from current level
            y_forecast_change = y[t] + np.cumsum(dyhat_all)
            phat_change = inv_logit(y_forecast_change)
        except Exception:
            # Fallback to random walk
            phat_change = np.full(max_horizon, p[t])

        # === AR BASELINE FORECAST (level) ===
        try:
            fit_baseline = AutoReg(y_train, lags=current_ar_order_baseline, old_names=False).fit()
            yhat_baseline = fit_baseline.forecast(steps=max_horizon)
            phat_baseline = inv_logit(np.asarray(yhat_baseline))
        except Exception:
            phat_baseline = np.full(max_horizon, p[t])

        # === RANDOM WALK FORECAST ===
        phat_rw = np.full(max_horizon, p[t])

        # Evaluate each horizon
        for h in horizons:
            if t + h >= len(P):
                continue

            # Forecasts
            p_change = float(phat_change[h-1])
            p_baseline = float(phat_baseline[h-1])
            p_rw = float(phat_rw[h-1])

            # Realized values
            p_realized = float(p[t+h])
            z_realized = int(z[t+h])

            # Forecast errors (probability scale)
            e_change = p_change - p_realized
            e_baseline = p_baseline - p_realized
            e_rw = p_rw - p_realized

            # Brier scores
            brier_change = (z_realized - p_change)**2
            brier_baseline = (z_realized - p_baseline)**2
            brier_rw = (z_realized - p_rw)**2

            # Log loss
            p_change_c = np.clip(p_change, eps, 1-eps)
            p_baseline_c = np.clip(p_baseline, eps, 1-eps)
            p_rw_c = np.clip(p_rw, eps, 1-eps)

            logloss_change = -(z_realized*np.log(p_change_c) + (1-z_realized)*np.log(1-p_change_c))
            logloss_baseline = -(z_realized*np.log(p_baseline_c) + (1-z_realized)*np.log(1-p_baseline_c))
            logloss_rw = -(z_realized*np.log(p_rw_c) + (1-z_realized)*np.log(1-p_rw_c))

            results[h]["rows"].append({
                "date_t": idx[t],
                "date_th": idx[t+h],
                "horizon": h,
                "ar_order_change": current_ar_order,
                "ar_order_baseline": current_ar_order_baseline,
                "p_t": float(p[t]),
                "p_change_th": p_change,
                "p_baseline_th": p_baseline,
                "p_rw_th": p_rw,
                "p_realized_th": p_realized,
                "z_th": z_realized,
                "error_change": e_change,
                "error_baseline": e_baseline,
                "error_rw": e_rw,
                "brier_change": brier_change,
                "brier_baseline": brier_baseline,
                "brier_rw": brier_rw,
                "logloss_change": logloss_change,
                "logloss_baseline": logloss_baseline,
                "logloss_rw": logloss_rw,
            })
            results[h]["ar_orders"].append(current_ar_order)

    # Compute summary statistics for each horizon
    summaries = {}
    all_details = []

    for h in horizons:
        df_h = pd.DataFrame(results[h]["rows"])
        all_details.append(df_h)

        if len(df_h) == 0:
            continue

        n = len(df_h)

        # Mean metrics
        brier_change = df_h["brier_change"].mean()
        brier_baseline = df_h["brier_baseline"].mean()
        brier_rw = df_h["brier_rw"].mean()

        logloss_change = df_h["logloss_change"].mean()
        logloss_baseline = df_h["logloss_baseline"].mean()
        logloss_rw = df_h["logloss_rw"].mean()

        # MSE
        mse_change = (df_h["error_change"]**2).mean()
        mse_baseline = (df_h["error_baseline"]**2).mean()
        mse_rw = (df_h["error_rw"]**2).mean()

        # AUC-ROC
        z_true = df_h["z_th"].values
        try:
            auc_change = roc_auc_score(z_true, df_h["p_change_th"].values)
            auc_baseline = roc_auc_score(z_true, df_h["p_baseline_th"].values)
            auc_rw = roc_auc_score(z_true, df_h["p_rw_th"].values)
        except:
            auc_change = auc_baseline = auc_rw = np.nan

        # Diebold-Mariano test: AR Change vs Random Walk
        e1 = df_h["error_change"].values
        e2 = df_h["error_rw"].values
        d = e1**2 - e2**2
        d_mean = d.mean()

        # HAC variance
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0.0
        for k in range(1, h):
            if k < len(d):
                gamma_k = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
                gamma_sum += 2 * gamma_k

        var_d = (gamma_0 + gamma_sum) / len(d)
        if var_d <= 0:
            var_d = gamma_0 / len(d)

        dm_stat_vs_rw = d_mean / np.sqrt(var_d)
        dm_pvalue_vs_rw = 2 * (1 - stats.norm.cdf(abs(dm_stat_vs_rw)))

        # DM test: AR Change vs AR Baseline
        e1 = df_h["error_change"].values
        e2 = df_h["error_baseline"].values
        d = e1**2 - e2**2
        d_mean = d.mean()

        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0.0
        for k in range(1, h):
            if k < len(d):
                gamma_k = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
                gamma_sum += 2 * gamma_k

        var_d = (gamma_0 + gamma_sum) / len(d)
        if var_d <= 0:
            var_d = gamma_0 / len(d)

        dm_stat_vs_baseline = d_mean / np.sqrt(var_d)
        dm_pvalue_vs_baseline = 2 * (1 - stats.norm.cdf(abs(dm_stat_vs_baseline)))

        # Hit rates at threshold 0.5
        hit_change = ((df_h["p_change_th"] >= 0.5) == (df_h["z_th"] == 1)).mean()
        hit_baseline = ((df_h["p_baseline_th"] >= 0.5) == (df_h["z_th"] == 1)).mean()
        hit_rw = ((df_h["p_rw_th"] >= 0.5) == (df_h["z_th"] == 1)).mean()

        # High-vol detection at threshold 0.5
        high_vol_mask = df_h["z_th"] == 1
        if high_vol_mask.sum() > 0:
            highvol_change = ((df_h.loc[high_vol_mask, "p_change_th"] >= 0.5)).mean()
            highvol_baseline = ((df_h.loc[high_vol_mask, "p_baseline_th"] >= 0.5)).mean()
            highvol_rw = ((df_h.loc[high_vol_mask, "p_rw_th"] >= 0.5)).mean()
        else:
            highvol_change = highvol_baseline = highvol_rw = np.nan

        # AR order stats
        modal_order = int(pd.Series(results[h]["ar_orders"]).mode().iloc[0])
        mean_order = np.mean(results[h]["ar_orders"])

        summaries[h] = {
            "horizon_days": h,
            "n_forecasts": n,
            "modal_ar_order": modal_order,
            "mean_ar_order": mean_order,
            # Brier
            "brier_change": brier_change,
            "brier_baseline": brier_baseline,
            "brier_rw": brier_rw,
            "brier_skill_change_vs_rw": 1 - brier_change / brier_rw if brier_rw > 0 else np.nan,
            "brier_skill_baseline_vs_rw": 1 - brier_baseline / brier_rw if brier_rw > 0 else np.nan,
            # MSE
            "mse_change": mse_change,
            "mse_baseline": mse_baseline,
            "mse_rw": mse_rw,
            # AUC
            "auc_change": auc_change,
            "auc_baseline": auc_baseline,
            "auc_rw": auc_rw,
            # Log-loss
            "logloss_change": logloss_change,
            "logloss_baseline": logloss_baseline,
            "logloss_rw": logloss_rw,
            # Hit rates
            "hit_rate_change": hit_change,
            "hit_rate_baseline": hit_baseline,
            "hit_rate_rw": hit_rw,
            # High-vol detection
            "highvol_change": highvol_change,
            "highvol_baseline": highvol_baseline,
            "highvol_rw": highvol_rw,
            # DM tests
            "dm_stat_vs_rw": dm_stat_vs_rw,
            "dm_pvalue_vs_rw": dm_pvalue_vs_rw,
            "change_better_than_rw": dm_stat_vs_rw < 0,
            "dm_stat_vs_baseline": dm_stat_vs_baseline,
            "dm_pvalue_vs_baseline": dm_pvalue_vs_baseline,
            "change_better_than_baseline": dm_stat_vs_baseline < 0,
        }

    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    summary_df = pd.DataFrame(summaries).T

    return {
        "summary": summary_df,
        "details": details_df,
    }


def main():
    print("=" * 70)
    print("MULTI-HORIZON AR CHANGE FORECAST EVALUATION")
    print("=" * 70)
    print()

    # Load daily regime probabilities
    probs_path = OUTPUT_DIR / "daily_regime_probabilities.csv"
    if not probs_path.exists():
        print(f"Error: {probs_path} not found. Run pipeline.py first.")
        return

    daily_probs = pd.read_csv(probs_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(daily_probs)} days of regime probabilities")
    print()

    results = evaluate_ar_change_multihorizon(
        daily_probs,
        target_regime=TARGET_REGIME,
        horizons=AR_HORIZONS,
        max_ar_lag=AR_MAX_LAG,
        min_train=AR_MIN_TRAIN,
        refit_frequency=AR_REFIT_FREQ,
    )

    summary = results["summary"]
    details = results["details"]
    horizons = AR_HORIZONS

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: AR CHANGE vs AR BASELINE vs RANDOM WALK")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("Panel A: Brier Score Comparison")
    print("-" * 70)

    brier_table = []
    for h in horizons:
        if h not in summary.index:
            continue
        row = summary.loc[h]
        brier_table.append({
            "Horizon": f"{h}-day",
            "AR Change": f"{row['brier_change']:.4f}",
            "AR Baseline": f"{row['brier_baseline']:.4f}",
            "Random Walk": f"{row['brier_rw']:.4f}",
            "Skill (Change vs RW)": f"{row['brier_skill_change_vs_rw']*100:.1f}%",
            "Skill (Baseline vs RW)": f"{row['brier_skill_baseline_vs_rw']*100:.1f}%",
        })

    brier_df = pd.DataFrame(brier_table)
    print(brier_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("Panel B: AUC-ROC Comparison")
    print("-" * 70)

    auc_table = []
    for h in horizons:
        if h not in summary.index:
            continue
        row = summary.loc[h]
        auc_table.append({
            "Horizon": f"{h}-day",
            "AR Change": f"{row['auc_change']:.4f}",
            "AR Baseline": f"{row['auc_baseline']:.4f}",
            "Random Walk": f"{row['auc_rw']:.4f}",
            "Change - Baseline": f"{(row['auc_change'] - row['auc_baseline'])*100:+.1f}pp",
            "Change - RW": f"{(row['auc_change'] - row['auc_rw'])*100:+.1f}pp",
        })

    auc_df = pd.DataFrame(auc_table)
    print(auc_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("Panel C: High-Volatility Detection (threshold = 0.5)")
    print("-" * 70)

    hv_table = []
    for h in horizons:
        if h not in summary.index:
            continue
        row = summary.loc[h]
        hv_table.append({
            "Horizon": f"{h}-day",
            "AR Change": f"{row['highvol_change']*100:.1f}%",
            "AR Baseline": f"{row['highvol_baseline']*100:.1f}%",
            "Random Walk": f"{row['highvol_rw']*100:.1f}%",
        })

    hv_df = pd.DataFrame(hv_table)
    print(hv_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("Panel D: Diebold-Mariano Tests")
    print("-" * 70)

    dm_table = []
    for h in horizons:
        if h not in summary.index:
            continue
        row = summary.loc[h]

        pval_rw = "<0.0001" if row["dm_pvalue_vs_rw"] < 0.0001 else f"{row['dm_pvalue_vs_rw']:.4f}"
        pval_base = "<0.0001" if row["dm_pvalue_vs_baseline"] < 0.0001 else f"{row['dm_pvalue_vs_baseline']:.4f}"

        sig_rw = "***" if row["dm_pvalue_vs_rw"] < 0.01 else "**" if row["dm_pvalue_vs_rw"] < 0.05 else "*" if row["dm_pvalue_vs_rw"] < 0.10 else ""
        sig_base = "***" if row["dm_pvalue_vs_baseline"] < 0.01 else "**" if row["dm_pvalue_vs_baseline"] < 0.05 else "*" if row["dm_pvalue_vs_baseline"] < 0.10 else ""

        dm_table.append({
            "Horizon": f"{h}-day",
            "DM (Change vs RW)": f"{row['dm_stat_vs_rw']:.3f}",
            "p-value": pval_rw,
            "Change Wins?": ("Yes" if row["change_better_than_rw"] else "No") + sig_rw,
            "DM (Change vs Baseline)": f"{row['dm_stat_vs_baseline']:.3f}",
            "p-value ": pval_base,
            "Change Wins? ": ("Yes" if row["change_better_than_baseline"] else "No") + sig_base,
        })

    dm_df = pd.DataFrame(dm_table)
    print(dm_df.to_string(index=False))

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    summary.to_csv(OUTPUT_DIR / "ar_change_multihorizon_summary.csv")
    print(f"Saved: ar_change_multihorizon_summary.csv")

    details.to_csv(OUTPUT_DIR / "ar_change_multihorizon_details.csv", index=False)
    print(f"Saved: ar_change_multihorizon_details.csv")

    # Dissertation table
    diss_table = pd.DataFrame({
        "Horizon": [f"{h}-day" for h in horizons if h in summary.index],
        "N": [int(summary.loc[h, "n_forecasts"]) for h in horizons if h in summary.index],
        "Brier (Change)": [f"{summary.loc[h, 'brier_change']:.4f}" for h in horizons if h in summary.index],
        "Brier (Baseline)": [f"{summary.loc[h, 'brier_baseline']:.4f}" for h in horizons if h in summary.index],
        "Brier (RW)": [f"{summary.loc[h, 'brier_rw']:.4f}" for h in horizons if h in summary.index],
        "Skill (Change)": [f"{summary.loc[h, 'brier_skill_change_vs_rw']*100:.1f}%" for h in horizons if h in summary.index],
        "AUC (Change)": [f"{summary.loc[h, 'auc_change']:.3f}" for h in horizons if h in summary.index],
        "AUC (Baseline)": [f"{summary.loc[h, 'auc_baseline']:.3f}" for h in horizons if h in summary.index],
        "AUC (RW)": [f"{summary.loc[h, 'auc_rw']:.3f}" for h in horizons if h in summary.index],
        "HighVol Det.": [f"{summary.loc[h, 'highvol_change']*100:.1f}%" for h in horizons if h in summary.index],
    })

    diss_table.to_csv(OUTPUT_DIR / "ar_change_dissertation_table.tsv", sep="\t", index=False)
    print(f"Saved: ar_change_dissertation_table.tsv")

    # Excel
    try:
        with pd.ExcelWriter(OUTPUT_DIR / "ar_change_multihorizon_tables.xlsx", engine="openpyxl") as writer:
            brier_df.to_excel(writer, sheet_name="Brier", index=False)
            auc_df.to_excel(writer, sheet_name="AUC", index=False)
            hv_df.to_excel(writer, sheet_name="HighVol_Detection", index=False)
            dm_df.to_excel(writer, sheet_name="DM_Tests", index=False)
            diss_table.to_excel(writer, sheet_name="Dissertation_Table", index=False)
        print(f"Saved: ar_change_multihorizon_tables.xlsx")
    except Exception as e:
        print(f"Could not save Excel: {e}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for h in horizons:
        if h not in summary.index:
            continue
        row = summary.loc[h]

        auc_improvement = (row['auc_change'] - row['auc_baseline']) * 100
        brier_improvement = (row['brier_baseline'] - row['brier_change']) * 100

        print(f"\n{h}-day horizon:")
        print(f"  AUC: Change={row['auc_change']:.3f} vs Baseline={row['auc_baseline']:.3f} ({auc_improvement:+.1f}pp)")
        print(f"  Brier Skill vs RW: Change={row['brier_skill_change_vs_rw']*100:.1f}% vs Baseline={row['brier_skill_baseline_vs_rw']*100:.1f}%")
        print(f"  High-vol detection: Change={row['highvol_change']*100:.1f}% vs Baseline={row['highvol_baseline']*100:.1f}%")

        if row['dm_pvalue_vs_baseline'] < 0.05 and row['change_better_than_baseline']:
            print(f"  AR Change SIGNIFICANTLY better than AR Baseline (p={row['dm_pvalue_vs_baseline']:.4f})")
        elif row['dm_pvalue_vs_baseline'] < 0.05:
            print(f"  AR Baseline significantly better than AR Change (p={row['dm_pvalue_vs_baseline']:.4f})")
        else:
            print(f"  No significant difference between Change and Baseline (p={row['dm_pvalue_vs_baseline']:.4f})")


if __name__ == "__main__":
    main()
