# regimes/models/m_cov_models.py

import numpy as np

MCLUST_MODELS = [
    "EII", "VII",
    "EEI", "VEI", "EVI", "VVI",
    "EEE", "EEV", "VEV", "VVV",
]


class BaseCovarianceModel:
    """Abstract base class for mclust-style covariance parameterisations."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def m_step(self, X, resp, means, reg_covar, priors):
        """
        Update covariance matrices given data X, responsibilities resp,
        and current means.

        Returns
        -------
        covariances : array, shape (K, d, d)
        """
        raise NotImplementedError


# ---------- Full VVV model (unconstrained, like your current implementation) ----------

class VVVModel(BaseCovarianceModel):
    def __init__(self):
        super().__init__("VVV")

    def m_step(self, X, resp, means, reg_covar, priors):
        """
        Same as your current full-covariance update, but factored out here.
        Works for both regularised and unregularised modes;
        'priors' contains (mu_P, kappa_P, nu_P, Lambda_P, regularise_flag).
        """
        n_samples, d = X.shape
        K = resp.shape[1]

        Nk = resp.sum(axis=0) + 1e-15
        covs_new = np.zeros((K, d, d))

        mu_P, kappa_P, nu_P, Lambda_P, regularise = priors

        for k in range(K):
            if Nk[k] < 1e-10:
                # If totally empty, fall back to prior scale
                if regularise and Lambda_P is not None:
                    cov_k = Lambda_P.copy()
                else:
                    cov_k = np.eye(d)
                cov_k.flat[:: d + 1] += reg_covar
                covs_new[k] = cov_k
                continue

            if regularise and (mu_P is not None):
                # MAP update (Fraley–Raftery)
                y_bar = (resp[:, k][:, None] * X).sum(axis=0) / Nk[k]
                diff = X - y_bar
                Wk = (resp[:, k][:, None] * diff).T @ diff

                delta = (y_bar - mu_P).reshape(-1, 1)
                between = (kappa_P * Nk[k] / (kappa_P + Nk[k])) * (delta @ delta.T)

                num = Lambda_P + between + Wk
                den = nu_P + Nk[k] * d + d + 2
                cov_k = num / den
            else:
                # plain ML update
                diff = X - means[k]
                weighted_diff = resp[:, k][:, None] * diff
                cov_k = (weighted_diff.T @ diff) / Nk[k]

            cov_k.flat[:: d + 1] += reg_covar
            covs_new[k] = cov_k

        return covs_new


# ---------- Factory + parameter counting ----------

def get_covariance_model(model_name: str) -> BaseCovarianceModel:
    model_name = model_name.upper()
    if model_name == "VVV":
        return VVVModel()
    # TODO: plug in EII, VII, ..., VEV implementations here
    raise NotImplementedError(f"Covariance model '{model_name}' not yet implemented.")


def num_params_for_model(model_name: str, K: int, d: int) -> int:
    """
    Number of free parameters for a given mclust covariance model.

    Matches the mclust/Fraley–Raftery conventions.
    """
    m = model_name.upper()

    # Means: K * d, Weights: K - 1 for all models
    means = K * d
    weights = K - 1

    if m == "VVV":
        covs = K * d * (d + 1) // 2
    elif m == "EII":
        covs = 1
    elif m == "VII":
        covs = K
    elif m == "EEI":
        covs = d
    elif m == "VEI":
        covs = K + (d - 1)
    elif m == "EVI":
        covs = 1 + K * (d - 1)
    elif m == "VVI":
        covs = K * d
    elif m == "EEE":
        covs = d * (d + 1) // 2
    elif m == "EEV":
        covs = d + K * (d * (d - 1) // 2)
    elif m == "VEV":
        covs = (d - 1) + K * (1 + d * (d - 1) // 2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return means + covs + weights
