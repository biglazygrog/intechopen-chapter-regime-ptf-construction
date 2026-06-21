"""
Core GMM classes and model selection.

Contains:
- MyGMM: VVV Gaussian Mixture Model with Normal-Wishart MAP regularization
- Optimiser: Rolling/expanding window GMM optimizer with BIC-based K selection
- Initialization helpers: kmeans_gmm_init, hierarchical_init
- Model selection: select_gmm_by_bic
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Dict

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment

from models.cov_models import (
    get_covariance_model,
    num_params_for_model,
    MCLUST_MODELS,
)


# -------------------------------------------------------------------
# Initialization helpers
# -------------------------------------------------------------------

def kmeans_gmm_init(
    X,
    n_components,
    reg_covar=1e-6,
    random_state=None,
    n_init=10,
    max_iter=300,
):
    """
    Initialize GMM parameters using K-means.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Data matrix.
    n_components : int
        Number of clusters / mixture components.
    reg_covar : float
        Small value added to covariance diagonals for numerical stability.
    random_state : int or None
        Seed for KMeans.
    n_init : int
        KMeans n_init.
    max_iter : int
        KMeans max_iter.

    Returns
    -------
    means : array, shape (K, D)
    covariances : array, shape (K, D, D)
    weights : array, shape (K,)
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    kmeans = KMeans(
        n_clusters=n_components,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)
    means = kmeans.cluster_centers_  # (K, D)

    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        mask = labels == k
        if np.sum(mask) < 2:
            cov_k = np.cov(X, rowvar=False)
        else:
            cov_k = np.cov(X[mask], rowvar=False)
        cov_k += reg_covar * np.eye(n_features)
        covariances[k] = cov_k

    counts = np.bincount(labels, minlength=n_components)
    weights = counts / counts.sum()

    return means, covariances, weights


def hierarchical_init(
    X: np.ndarray,
    max_K: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hierarchical clustering based initialization for GMM.

    - Run Ward hierarchical clustering on the data.
    - Cut the tree into max_K clusters.
    - For each cluster, compute empirical mean, covariance, and weight.

    Returns
    -------
    means0 : (max_K, d)
    covs0  : (max_K, d, d)
    w0     : (max_K,)
    """
    X = np.asarray(X, float)
    n, d = X.shape

    Z = linkage(X, method="ward")
    labels = fcluster(Z, max_K, criterion="maxclust")

    means0 = np.zeros((max_K, d))
    covs0 = np.zeros((max_K, d, d))
    w0 = np.zeros(max_K)

    for k in range(1, max_K + 1):
        mask = (labels == k)
        if mask.sum() < 2:
            means0[k-1] = X.mean(axis=0)
            covs0[k-1] = np.cov(X, rowvar=False) + 1e-6 * np.eye(d)
            w0[k-1] = 1.0 / max_K
        else:
            Xk = X[mask]
            means0[k-1] = Xk.mean(axis=0)
            covs0[k-1] = np.cov(Xk, rowvar=False) + 1e-6 * np.eye(d)
            w0[k-1] = mask.mean()

    if w0.sum() > 0:
        w0 /= w0.sum()
    else:
        w0[:] = 1.0 / max_K

    return means0, covs0, w0


def _num_params_full_cov(k: int, d: int) -> int:
    means = k * d
    covs = k * d * (d + 1) // 2
    weights = k - 1
    return means + covs + weights


# -------------------------------------------------------------------
# MyGMM
# -------------------------------------------------------------------

class MyGMM:
    """
    VVV-only Gaussian Mixture Model with Fraley-Raftery style
    Normal-Wishart MAP regularization.

    - Full covariance Sigma_k per component
    - Responsibilities via standard EM E-step
    - M-step:
        * mu_k shrunk towards mu_0
        * Sigma_k shrunk towards Lambda_0
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
        means_init=None,
        covariances_init=None,
        weights_init=None,
        regularise: bool = True,
        alpha_dirichlet=1.0,
        order_components: bool = True,
        order_mode: str = "trace",
        kappa_0: float = 0.01,
        nu_0_extra: int = 2,
        lambda_scale: float = 1.0,
        mu_prior: Optional[np.ndarray] = None,
        regime_ages_init: Optional[np.ndarray] = None,
    ):
        self.n_components = int(n_components)
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = np.random.RandomState(random_state)
        self.means_init = means_init
        self.covariances_init = covariances_init
        self.weights_init = weights_init
        self.regularise = bool(regularise)

        # kappa_0 may be scalar (legacy, shared shrinkage) or shape (K,) for
        # per-regime strengths (e.g. age-scaled). Validated to a final (K,)
        # vector in _initialise_priors.
        k0_arr = np.asarray(kappa_0, dtype=float)
        self.kappa_0 = float(k0_arr) if k0_arr.ndim == 0 else k0_arr
        self.nu_0_extra = int(nu_0_extra)
        self.lambda_scale = float(lambda_scale)

        # Per-regime prior means (shape K x d) — when None, falls back to
        # grand-mean shrinkage broadcast across all K rows (old behaviour).
        # NaN rows are also filled with the grand mean in _initialise_priors.
        self.mu_prior = None if mu_prior is None else np.asarray(mu_prior, dtype=float)

        # Initial per-regime "age" (windows-since-first-appearance), reordered
        # alongside means_ during EM. Used by the Optimiser to propagate
        # regime persistence across windows.
        if regime_ages_init is None:
            self.regime_ages_ = None
        else:
            self.regime_ages_ = np.asarray(regime_ages_init, dtype=int).copy()

        self.alpha_dirichlet = alpha_dirichlet
        self.order_components = bool(order_components)
        self.order_mode = str(order_mode)

        # Set during fit
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.n_iter_ = 0
        self.converged_ = False

        self._X_mean = None
        self._X_std = None

        # Prior objects
        self.mu_0 = None
        self.kappa_0_ = None
        self.nu_0 = None
        self.Lambda_0 = None

        # Diagnostics
        self.history_ = {
            "log_likelihood": [],
            "weights": [],
            "means": [],
            "resp": [],
        }
        self.lower_bound_ = None
        self.resp_ = None

    def _initialise_priors(self, X: np.ndarray):
        """Initialize Normal-Wishart prior hyperparameters from the data.

        self.mu_0 has shape (K, d) — one prior mean per component.
        Without an explicit mu_prior, every row is the grand mean of X
        (algebraically identical to the legacy single-vector behaviour).
        With mu_prior provided, per-regime rows are used; NaN rows fall
        back to the grand mean.
        """
        n_samples, d = X.shape
        G = self.n_components

        grand_mean = X.mean(axis=0)
        S = np.cov(X, rowvar=False)

        if self.mu_prior is None:
            self.mu_0 = np.tile(grand_mean, (G, 1))
        else:
            mu_p = np.asarray(self.mu_prior, dtype=float).copy()
            if mu_p.shape != (G, d):
                raise ValueError(
                    f"mu_prior has shape {mu_p.shape}, expected ({G}, {d})"
                )
            nan_rows = np.isnan(mu_p).any(axis=1)
            if nan_rows.any():
                mu_p[nan_rows] = grand_mean
            self.mu_0 = mu_p

        # kappa_0 may be either scalar (legacy: shared shrinkage strength)
        # or a (K,) vector (per-regime, e.g. age-scaled). Broadcast to (K,).
        k0 = np.asarray(self.kappa_0, dtype=float)
        if k0.ndim == 0:
            self.kappa_0_ = np.full(G, float(k0))
        elif k0.shape == (G,):
            self.kappa_0_ = k0.copy()
        else:
            raise ValueError(
                f"kappa_0 must be scalar or shape ({G},), got {k0.shape}"
            )

        self.nu_0 = d + self.nu_0_extra
        self.Lambda_0 = self.lambda_scale * S / (G ** (2.0 / d))

    def _initialise_parameters(self, X: np.ndarray):
        n_samples, n_features = X.shape

        if self.means_init is not None:
            self.means_ = np.asarray(self.means_init, dtype=float).copy()
        else:
            idx = self.random_state.choice(n_samples, self.n_components, replace=False)
            self.means_ = X[idx]

        if self.covariances_init is not None:
            self.covariances_ = np.asarray(self.covariances_init, dtype=float).copy()
        else:
            emp_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
            self.covariances_ = np.array(
                [emp_cov.copy() for _ in range(self.n_components)]
            )

        if self.weights_init is not None:
            w = np.asarray(self.weights_init, dtype=float)
            self.weights_ = w / w.sum()
        else:
            self.weights_ = np.ones(self.n_components) / self.n_components

        if self.regularise:
            self._initialise_priors(X)
        else:
            self.mu_0 = None

    def _get_dirichlet_alpha(self):
        """Returns alpha vector of shape (K,) for Dirichlet prior."""
        K = self.n_components

        if np.isscalar(self.alpha_dirichlet):
            alpha = np.full(K, float(self.alpha_dirichlet))
        else:
            alpha = np.asarray(self.alpha_dirichlet, dtype=float)
            if alpha.shape != (K,):
                raise ValueError(
                    f"alpha_dirichlet must be scalar or length-{K} vector"
                )

        if np.any(alpha <= 0):
            raise ValueError("Dirichlet alpha must be strictly positive")

        return alpha

    def _order_components(self, resp: Optional[np.ndarray] = None, mode: str = "trace"):
        """Deterministically reorder components to reduce label switching."""
        K = self.n_components
        if K <= 1:
            return resp

        if mode == "trace":
            key = np.array([np.trace(self.covariances_[k]) for k in range(K)])
            order = np.argsort(key)
        elif mode == "det":
            key = np.array([np.linalg.slogdet(self.covariances_[k])[1] for k in range(K)])
            order = np.argsort(key)
        elif mode == "weight":
            order = np.argsort(-self.weights_)
        else:
            raise ValueError(f"Unknown ordering mode: {mode}")

        self.means_ = self.means_[order]
        self.covariances_ = self.covariances_[order]
        self.weights_ = self.weights_[order]

        # Reorder per-regime prior containers so they remain aligned with the
        # component indexing on subsequent EM iterations. Harmless when mu_0
        # is a grand-mean broadcast or kappa_0_ is constant.
        if self.mu_0 is not None and getattr(self.mu_0, "ndim", 0) == 2:
            self.mu_0 = self.mu_0[order]
        if self.kappa_0_ is not None and getattr(self.kappa_0_, "ndim", 0) == 1:
            self.kappa_0_ = self.kappa_0_[order]
        if self.regime_ages_ is not None and len(self.regime_ages_) == K:
            self.regime_ages_ = self.regime_ages_[order]

        if resp is not None:
            resp = resp[:, order]

        return resp

    def _estimate_log_gaussian_prob(self, X: np.ndarray) -> np.ndarray:
        """log N(x | mu_k, Sigma_k) for each component k."""
        n_samples, n_features = X.shape
        K = self.n_components

        log_prob = np.empty((n_samples, K))

        for k in range(K):
            mean = self.means_[k]
            cov = self.covariances_[k] + self.reg_covar * np.eye(n_features)

            prec = np.linalg.inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Covariance matrix not positive definite.")

            diff = X - mean
            quad = np.sum((diff @ prec) * diff, axis=1)

            log_norm_const = 0.5 * (n_features * np.log(2.0 * np.pi) + log_det)
            log_prob[:, k] = -0.5 * quad - log_norm_const

        return log_prob

    def _e_step(self, X: np.ndarray):
        """Standard E-step."""
        log_gauss = self._estimate_log_gaussian_prob(X)
        log_weights = np.log(self.weights_ + 1e-15)
        log_prob = log_gauss + log_weights

        max_log = np.max(log_prob, axis=1, keepdims=True)
        log_sum = max_log + np.log(
            np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True)
        )

        log_resp = log_prob - log_sum
        resp = np.exp(log_resp)

        log_likelihood = np.mean(log_sum)
        return resp, log_likelihood

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> np.ndarray:
        """MAP M-step for VVV."""
        n_samples, d = X.shape

        Nk = resp.sum(axis=0) + 1e-15

        alpha = self._get_dirichlet_alpha()
        K = self.n_components

        num = Nk + (alpha - 1.0)
        den = Nk.sum() + np.sum(alpha - 1.0)

        num = np.maximum(num, 1e-15)

        self.weights_ = num / den
        self.weights_ /= self.weights_.sum()

        if self.regularise and self.mu_0 is not None:
            means_new = np.zeros((K, d))
            covs_new = np.zeros((K, d, d))

            for k in range(K):
                mu_prior_k = self.mu_0[k]

                if Nk[k] < 1e-10:
                    means_new[k] = mu_prior_k
                    covs_new[k] = self.Lambda_0.copy()
                    continue

                y_bar = (resp[:, k][:, None] * X).sum(axis=0) / Nk[k]

                kappa_k = self.kappa_0_[k]

                mu_k = (Nk[k] * y_bar + kappa_k * mu_prior_k) / (
                    Nk[k] + kappa_k
                )
                means_new[k] = mu_k

                diff = X - y_bar
                Wk = (resp[:, k][:, None] * diff).T @ diff

                delta = (y_bar - mu_prior_k).reshape(-1, 1)
                between = (
                    kappa_k * Nk[k] / (kappa_k + Nk[k])
                ) * (delta @ delta.T)

                num = self.Lambda_0 + Wk + between
                den = self.nu_0 + Nk[k] + d + 2
                cov_k = num / den

                cov_k.flat[:: d + 1] += self.reg_covar
                covs_new[k] = cov_k

            self.means_ = means_new
            self.covariances_ = covs_new
        else:
            self.means_ = (resp.T @ X) / Nk[:, None]
            covs = np.zeros((K, d, d))
            for k in range(K):
                diff = X - self.means_[k]
                weighted = resp[:, k][:, None] * diff
                cov_k = (weighted.T @ diff) / Nk[k]
                cov_k.flat[:: d + 1] += self.reg_covar
                covs[k] = cov_k
            self.covariances_ = covs

        if self.order_components:
            resp = self._order_components(resp=resp, mode=self.order_mode)
        return resp

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self._X_mean = None
        self._X_std = None

        self.history_ = {
            "log_likelihood": [],
            "weights": [],
            "means": [],
            "resp": [],
        }

        self._initialise_parameters(X)

        lower_bound = -np.inf

        for n_iter in range(1, self.max_iter + 1):
            resp, log_lik = self._e_step(X)
            resp = self._m_step(X, resp)

            self.history_["log_likelihood"].append(log_lik)
            self.history_["weights"].append(self.weights_.copy())
            self.history_["means"].append(self.means_.copy())
            self.history_["resp"].append(resp.copy())

            # Verbose EM output disabled for faster execution
            # print(f"\n=== EM Iteration {n_iter} ===")
            # print(f"Log-likelihood: {log_lik:.6f}")
            # print("Weights:", np.round(self.weights_, 6))

            change = log_lik - lower_bound
            if abs(change) < self.tol:
                self.converged_ = True
                self.n_iter_ = n_iter
                lower_bound = log_lik
                break

            lower_bound = log_lik

        if not self.converged_:
            self.n_iter_ = self.max_iter

        self.resp_ = resp
        self.lower_bound_ = lower_bound

        self.history_["log_likelihood"] = np.array(
            self.history_["log_likelihood"]
        )
        self.history_["weights"] = np.array(self.history_["weights"])
        self.history_["means"] = np.array(self.history_["means"])

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        resp, _ = self._e_step(X)
        return resp

    def predict(self, X):
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        log_gauss = self._estimate_log_gaussian_prob(X)
        log_weights = np.log(self.weights_ + 1e-15)
        log_prob = log_gauss + log_weights

        max_log = np.max(log_prob, axis=1, keepdims=True)
        log_sum = max_log + np.log(
            np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True)
        )
        return log_sum.ravel()

    def regime_probabilities_df(self, index=None):
        assert self.resp_ is not None, "Call fit(X) first."
        T, K = self.resp_.shape
        cols = [f"p_{k}" for k in range(K)]
        df = pd.DataFrame(self.resp_, columns=cols)
        if index is not None:
            df.index = index
        return df


# -------------------------------------------------------------------
# Model selection by BIC
# -------------------------------------------------------------------

def select_gmm_by_bic(
    X,
    n_components_list,
    n_starts: int = 5,
    init_method: str = "kmeans",
    **gmm_kwargs,
):
    """
    VVV-only model selection by mclust-style BIC.

    For each K in n_components_list:
      - run EM n_starts times with different initializations
      - keep the run with the highest total log-likelihood
      - compute BIC = 2 logL - p log N  (higher is better)
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    results = []
    best_bic_global = -np.inf
    best_model_global = None

    base_seed = gmm_kwargs.get("random_state", None)
    init_method = init_method.lower()

    for K in n_components_list:
        print(f"\n=== Fitting VVV GMM with K = {K} components ===")
        best_model_K = None
        best_ll_K = -np.inf

        for s in range(n_starts):
            if base_seed is None:
                seed = None
            else:
                seed = base_seed + s

            if init_method == "kmeans":
                km = KMeans(
                    n_clusters=K,
                    n_init=10,
                    max_iter=300,
                    random_state=seed,
                )
                labels = km.fit_predict(X)
                means0 = km.cluster_centers_

                covs0 = np.zeros((K, n_features, n_features))
                weights0 = np.zeros(K)
                for k in range(K):
                    mask = labels == k
                    weights0[k] = mask.sum()
                    if mask.sum() < 2:
                        cov_k = np.cov(X, rowvar=False)
                    else:
                        cov_k = np.cov(X[mask], rowvar=False)
                    cov_k += gmm_kwargs.get("reg_covar", 1e-6) * np.eye(n_features)
                    covs0[k] = cov_k
                weights0 = weights0 / weights0.sum()
            else:
                idx = np.random.default_rng(seed).choice(
                    n_samples, size=K, replace=False
                )
                means0 = X[idx]
                emp_cov = np.cov(X, rowvar=False)
                covs0 = np.array(
                    [
                        emp_cov + gmm_kwargs.get("reg_covar", 1e-6) * np.eye(n_features)
                        for _ in range(K)
                    ]
                )
                weights0 = np.ones(K) / K

            gmm = MyGMM(
                n_components=K,
                means_init=means0,
                covariances_init=covs0,
                weights_init=weights0,
                random_state=seed,
                alpha_dirichlet=gmm_kwargs.get("alpha_dirichlet", 1.0),
                order_components=gmm_kwargs.get("order_components", True),
                order_mode=gmm_kwargs.get("order_mode", "trace"),
                **{k: v for k, v in gmm_kwargs.items() if k != "random_state"},
            )
            gmm.fit(X)

            total_ll = gmm.lower_bound_ * n_samples
            if total_ll > best_ll_K:
                best_ll_K = total_ll
                best_model_K = gmm

        p = _num_params_full_cov(K, n_features)
        bic_K = 2.0 * best_ll_K - p * np.log(n_samples)

        print(f"K = {K}: best logL = {best_ll_K:.2f}, BIC = {bic_K:.2f}")

        results.append(
            {"K": K, "bic": bic_K, "loglik": best_ll_K, "model": best_model_K}
        )

        if bic_K > best_bic_global:
            best_bic_global = bic_K
            best_model_global = best_model_K

    return best_model_global, results


# -------------------------------------------------------------------
# Optimiser: Rolling-window GMM
# -------------------------------------------------------------------

class Optimiser:
    """
    Rolling-window optimizer for MyGMM with:

    - optional scaling per window ("none", "expanding", "rolling")
    - automatic K-selection via BIC
    - hierarchical initialization shared across K
    - optional reuse of previous window parameters (init_decay)
    - optional EWMA smoothing of parameters across *all* past windows (ewma_decay)
    """

    def __init__(
        self,
        K_candidates: Sequence[int] = (2, 3, 4, 5, 6, 7),
        window_size: int = 2500,
        step: int = 2500,
        scale_method: str = "rolling",
        max_iter: int = 200,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = 0,
        regularise: bool = True,
        kappa_0: float = 0.01,
        nu_0_extra: int = 2,
        lambda_scale: float = 1.0,
        init_decay: Optional[float] = None,
        ewma_decay: Optional[float] = None,
        allow_partial_last_window: bool = False,
        alpha_dirichlet=0.5,
        order_components: bool = True,
        order_mode: str = "trace",
        shrinkage_target: str = "grand_mean",
        kappa_age_scale: bool = False,
    ):
        self.K_candidates = sorted(list(K_candidates))
        self.window_size = int(window_size)
        self.step = int(step)

        self.scale_method = scale_method.lower()
        assert self.scale_method in {"none", "expanding", "rolling"}

        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.regularise = regularise
        self.kappa_0 = kappa_0
        self.nu_0_extra = nu_0_extra
        self.lambda_scale = lambda_scale

        if init_decay is not None and not (0.0 < init_decay < 1.0):
            raise ValueError("init_decay must be in (0,1) or None.")
        self.init_decay = init_decay

        self.alpha_dirichlet = alpha_dirichlet
        self.order_components = bool(order_components)
        self.order_mode = str(order_mode)

        if ewma_decay is not None and not (0.0 < ewma_decay < 1.0):
            raise ValueError("ewma_decay must be in (0,1) or None.")
        self.ewma_decay = ewma_decay

        # MAP shrinkage target for mu_k inside EM. Either:
        #   "grand_mean"     — every regime shrunk to the window grand mean (legacy)
        #   "previous_epoch" — regime k shrunk to the matched mean from the
        #                      immediately previous window (per-regime temporal prior)
        if shrinkage_target not in {"grand_mean", "previous_epoch"}:
            raise ValueError(
                "shrinkage_target must be 'grand_mean' or 'previous_epoch'."
            )
        self.shrinkage_target = shrinkage_target

        # When True, scale kappa_0 per regime by the regime's age (number of
        # consecutive windows since first appearance under Hungarian alignment).
        # kappa_{0,k} = kappa_0 * age_k. Persistent regimes get tighter priors;
        # newly-appeared regimes use plain kappa_0. Only effective in combination
        # with shrinkage_target="previous_epoch" (ages are zeroed otherwise).
        self.kappa_age_scale = bool(kappa_age_scale)

        self.index_: Optional[pd.Index] = None

        self.fits: List[Dict] = []
        self.window_ends_: List[int] = []
        self.selected_Ks_: List[int] = []
        self.bic_grid_: List[Dict[int, float]] = []
        self.models_by_K_: Dict[int, MyGMM] = {}

        self.window_start_indices_: List[int] = []
        self.window_end_indices_: List[int] = []
        self.window_end_times_: List = []
        self.window_n_iter_: List[int] = []
        self.window_loglik_: List[float] = []
        self.window_bic_: List[float] = []
        self.window_models_: List[MyGMM] = []

        self.raw_means_: List[np.ndarray] = []
        self.raw_stds_: List[np.ndarray] = []

        self.ewma_params_by_K: Dict[int, Dict[str, np.ndarray]] = {}
        self.allow_partial_last_window = allow_partial_last_window

        self.Neff_by_window_: List[np.ndarray] = []
        self.Neff_min_: List[float] = []

        # Per-window shrinkage diagnostics
        self.window_prior_mode_: List[str] = []
        self.window_prior_coverage_: List[float] = []
        # Ages of the regimes in each window's best-K fit (under Hungarian
        # alignment to the previous best fit). 1 = first appearance.
        self.window_regime_ages_: List[np.ndarray] = []

    def _build_mu_prior(
        self,
        K_current: int,
        fresh_means: np.ndarray,
        d: int,
    ) -> Tuple[Optional[np.ndarray], float, List[Tuple[int, int]]]:
        """Build a per-regime prior mean array of shape (K_current, d) by
        aligning the immediately previous window's fitted means against the
        fresh init means for this window via Hungarian matching on Euclidean
        distance.

        Returns
        -------
        mu_prior : ndarray of shape (K_current, d) or None
            Per-regime prior. NaN-filled rows for components that could not
            be matched (e.g. when previous K differs from current K). MyGMM
            replaces NaN rows with the grand mean.
        coverage : float in [0, 1]
            Fraction of the K_current rows that were filled by a matched
            previous-epoch mean.
        matched_pairs : list of (current_idx, prev_idx) tuples
            Index pairs used for the alignment; consumed by age tracking.
        """
        if not self.window_models_:
            return None, 0.0, []

        prev_model = self.window_models_[-1]
        prev_means = prev_model.means_
        if prev_means is None:
            return None, 0.0, []
        K_prev = prev_means.shape[0]

        diff = fresh_means[:, None, :] - prev_means[None, :, :]
        cost = np.linalg.norm(diff, axis=2)
        rows, cols = linear_sum_assignment(cost)

        mu_prior = np.full((K_current, d), np.nan, dtype=float)
        matched_pairs: List[Tuple[int, int]] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            mu_prior[r] = prev_means[c]
            matched_pairs.append((int(r), int(c)))

        coverage = len(matched_pairs) / float(K_current)
        return mu_prior, coverage, matched_pairs

    def _compute_ages(
        self,
        K_current: int,
        matched_pairs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Per-regime age for the current window. age=1 for newly-appearing
        regimes; matched regimes inherit previous age + 1."""
        ages = np.ones(K_current, dtype=int)
        if not self.window_regime_ages_:
            return ages
        prev_ages = self.window_regime_ages_[-1]
        for r, c in matched_pairs:
            if 0 <= c < len(prev_ages):
                ages[r] = int(prev_ages[c]) + 1
        return ages

    def _scale_window(self, X: np.ndarray, start: int, end: int) -> np.ndarray:
        X_win = X[start:end]

        if self.scale_method == "none":
            return X_win.copy()

        elif self.scale_method == "rolling":
            mu = X_win.mean(axis=0)
            sigma = X_win.std(axis=0, ddof=1)
            sigma[sigma == 0.0] = 1.0
            return (X_win - mu) / sigma

        elif self.scale_method == "expanding":
            mu = X[:end].mean(axis=0)
            sigma = X[:end].std(axis=0, ddof=1)
            sigma[sigma == 0.0] = 1.0
            return (X_win - mu) / sigma

        else:
            raise ValueError(f"Unknown scale_method={self.scale_method}")

    def _bic_for_model(self, model: "MyGMM", X_win: np.ndarray) -> float:
        n, d = X_win.shape
        K = model.n_components

        loglik_total = model.lower_bound_ * n
        n_params = (K - 1) + K * d + K * d * (d + 1) / 2.0
        return float(2.0 * loglik_total - n_params * np.log(n))

    def _update_ewma_params(self, K: int, means: np.ndarray,
                            covs: np.ndarray, weights: np.ndarray):
        if self.ewma_decay is None:
            return

        lam = self.ewma_decay

        if K not in self.ewma_params_by_K:
            self.ewma_params_by_K[K] = {
                "means": means.copy(),
                "covs": covs.copy(),
                "weights": weights.copy(),
            }
        else:
            ew = self.ewma_params_by_K[K]
            ew["means"] = lam * ew["means"] + (1 - lam) * means
            ew["covs"] = lam * ew["covs"] + (1 - lam) * covs
            ew["weights"] = lam * ew["weights"] + (1 - lam) * weights

            s = ew["weights"].sum()
            if s <= 0:
                ew["weights"][:] = 1.0 / K
            else:
                ew["weights"] /= s

    def _blend_initialisation(
        self,
        K: int,
        means0: np.ndarray,
        covs0: np.ndarray,
        w0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        means_init = means0[:K].copy()
        cov_init = covs0[:K].copy()
        w_init = w0[:K].copy()
        s = w_init.sum()
        if s <= 0:
            w_init[:] = 1.0 / K
        else:
            w_init /= s

        if self.ewma_decay is not None and K in self.ewma_params_by_K:
            ew = self.ewma_params_by_K[K]
            means_init = 0.5 * means_init + 0.5 * ew["means"]
            cov_init = 0.5 * cov_init + 0.5 * ew["covs"]
            w_init = 0.5 * w_init + 0.5 * ew["weights"]
            s = w_init.sum()
            if s <= 0:
                w_init[:] = 1.0 / K
            else:
                w_init /= s

        if self.init_decay is not None and K in self.models_by_K_:
            prev = self.models_by_K_[K]
            alpha = self.init_decay

            means_init = alpha * prev.means_ + (1 - alpha) * means_init
            cov_init = alpha * prev.covariances_ + (1 - alpha) * cov_init
            w_init = alpha * prev.weights_ + (1 - alpha) * w_init

            s = w_init.sum()
            if s <= 0:
                w_init[:] = 1.0 / K
            else:
                w_init /= s

        return means_init, cov_init, w_init

    def _bic_select_K(self, X_win: np.ndarray) -> Tuple[int, "MyGMM", Dict[int, float], Dict[int, float], Dict[int, np.ndarray]]:
        n, d = X_win.shape
        maxK = max(self.K_candidates)

        means0, covs0, w0 = hierarchical_init(X_win, maxK)

        best_bic = -np.inf
        best_K = None
        best_model = None
        bic_per_K: Dict[int, float] = {}
        prior_coverage_per_K: Dict[int, float] = {}
        ages_per_K: Dict[int, np.ndarray] = {}

        for K in self.K_candidates:
            means_init, cov_init, w_init = self._blend_initialisation(
                K, means0, covs0, w0
            )

            if self.shrinkage_target == "previous_epoch":
                mu_prior, coverage, matched_pairs = self._build_mu_prior(K, means_init, d)
            else:
                mu_prior, coverage, matched_pairs = None, 0.0, []
            prior_coverage_per_K[K] = coverage

            ages = self._compute_ages(K, matched_pairs)
            ages_per_K[K] = ages

            if self.kappa_age_scale and self.shrinkage_target == "previous_epoch":
                kappa_for_model = (float(self.kappa_0) * ages).astype(float)
            else:
                kappa_for_model = self.kappa_0

            gmm = MyGMM(
                n_components=K,
                max_iter=self.max_iter,
                tol=self.tol,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                means_init=means_init,
                covariances_init=cov_init,
                weights_init=w_init,
                regularise=self.regularise,
                kappa_0=kappa_for_model,
                nu_0_extra=self.nu_0_extra,
                lambda_scale=self.lambda_scale,
                alpha_dirichlet=self.alpha_dirichlet,
                order_components=self.order_components,
                order_mode=self.order_mode,
                mu_prior=mu_prior,
                regime_ages_init=ages,
            )
            gmm.fit(X_win)

            bic_val = self._bic_for_model(gmm, X_win)
            bic_per_K[K] = bic_val

            if bic_val > best_bic:
                best_bic = bic_val
                best_K = K
                best_model = gmm

        assert best_model is not None
        return best_K, best_model, bic_per_K, prior_coverage_per_K, ages_per_K

    def fit(self, X: np.ndarray, index=None):
        """Fit rolling-window GMMs with automatic K-selection."""
        X = np.asarray(X, dtype=float)
        T, d = X.shape

        self.index_ = pd.Index(index) if index is not None else pd.RangeIndex(T)

        self.fits.clear()
        self.window_ends_.clear()
        self.selected_Ks_.clear()
        self.bic_grid_.clear()
        self.models_by_K_.clear()
        self.window_start_indices_.clear()
        self.window_end_indices_.clear()
        self.window_end_times_.clear()
        self.window_n_iter_.clear()
        self.window_loglik_.clear()
        self.window_bic_.clear()
        self.window_models_.clear()
        self.raw_means_.clear()
        self.raw_stds_.clear()
        self.ewma_params_by_K.clear()
        self.Neff_by_window_.clear()
        self.Neff_min_.clear()
        self.window_prior_mode_.clear()
        self.window_prior_coverage_.clear()
        self.window_regime_ages_.clear()

        start = 0

        while start < T:
            if not self.allow_partial_last_window:
                end = start + self.window_size
                if end > T:
                    break
            else:
                end = min(start + self.window_size, T)
                if end <= start:
                    break

            mu_raw = X[:end].mean(axis=0)
            sigma_raw = X[:end].std(axis=0, ddof=1)
            sigma_raw[sigma_raw == 0.0] = 1.0
            self.raw_means_.append(mu_raw)
            self.raw_stds_.append(sigma_raw)

            X_win = self._scale_window(X, start, end)

            best_K, best_model, bic_per_K, prior_coverage_per_K, ages_per_K = self._bic_select_K(X_win)

            self.fits.append(
                {
                    "K": best_K,
                    "model": best_model,
                    "means": best_model.means_.copy(),
                    "covariances": best_model.covariances_.copy(),
                    "weights": best_model.weights_.copy(),
                    "resp": best_model.resp_.copy(),
                }
            )

            resp_win = best_model.resp_
            Neff = resp_win.sum(axis=0)
            self.Neff_by_window_.append(Neff.copy())
            self.Neff_min_.append(float(np.min(Neff)))

            self.fits[-1]["Neff"] = Neff.copy()
            self.fits[-1]["Neff_min"] = float(np.min(Neff))

            end_idx = end - 1

            self.window_start_indices_.append(start)
            self.window_end_indices_.append(end_idx)
            self.window_end_times_.append(self.index_[end_idx])
            self.window_ends_.append(end_idx)
            self.selected_Ks_.append(best_K)
            self.bic_grid_.append(bic_per_K)

            self.window_models_.append(best_model)
            self.window_n_iter_.append(best_model.n_iter_)
            n_win = end - start
            self.window_loglik_.append(best_model.lower_bound_ * n_win)
            self.window_bic_.append(bic_per_K[best_K])

            self.models_by_K_[best_K] = best_model

            self.window_prior_mode_.append(self.shrinkage_target)
            self.window_prior_coverage_.append(prior_coverage_per_K[best_K])
            # best_model.regime_ages_ has been reordered alongside means_ by
            # _order_components, so it's aligned to the final fitted order
            # (the same order that the next window's _build_mu_prior matches).
            final_ages = best_model.regime_ages_
            if final_ages is None:
                final_ages = ages_per_K[best_K]
            self.window_regime_ages_.append(np.asarray(final_ages, dtype=int).copy())

            self._update_ewma_params(
                best_K, best_model.means_, best_model.covariances_, best_model.weights_
            )

            start += self.step

        return self

    def get_probs_over_time(self) -> pd.DataFrame:
        """Per-window regime probabilities."""
        assert len(self.fits) > 0, "Call fit() first."

        maxK = max(f["K"] for f in self.fits)
        cols = [f"p_{k}" for k in range(maxK)]

        rows = []
        for fit in self.fits:
            K = fit["K"]
            resp = fit["resp"]
            avg = resp.mean(axis=0)

            row = np.zeros(maxK)
            row[:K] = avg
            rows.append(row)

        idx = [self.index_[i] for i in self.window_ends_]
        return pd.DataFrame(rows, index=idx, columns=cols)

    def get_daily_probabilities(self) -> pd.DataFrame:
        """Daily regime probabilities."""
        assert len(self.fits) > 0, "Call fit() first."

        maxK = max(f["K"] for f in self.fits)
        cols = [f"p_{k}" for k in range(maxK)]

        frames = []
        for fit, end_idx in zip(self.fits, self.window_ends_):
            resp = fit["resp"]
            K = fit["K"]
            n_win = resp.shape[0]
            start_idx = end_idx - n_win + 1

            arr = np.zeros((n_win, maxK))
            arr[:, :K] = resp

            idx = self.index_[start_idx : end_idx + 1]
            frames.append(pd.DataFrame(arr, index=idx, columns=cols))

        out = pd.concat(frames)
        out = out.groupby(level=0).mean()
        return out.sort_index()

    def get_correlation_matrices(self) -> List[List[np.ndarray]]:
        """For each window, return list of correlation matrices per regime."""
        assert len(self.fits) > 0, "Call fit() first."

        all_corrs: List[List[np.ndarray]] = []

        for fit in self.fits:
            covs = fit["covariances"]
            K, d, _ = covs.shape
            win_corrs: List[np.ndarray] = []

            for k in range(K):
                cov = covs[k]
                std = np.sqrt(np.diag(cov))
                std[std == 0.0] = 1e-12
                corr = cov / np.outer(std, std)
                win_corrs.append(corr)

            all_corrs.append(win_corrs)

        return all_corrs

    def get_selected_K_series(self) -> pd.Series:
        """Time series of selected K per window."""
        idx = [self.index_[i] for i in self.window_ends_]
        return pd.Series(self.selected_Ks_, index=idx, name="K_selected")
