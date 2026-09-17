import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from numba import njit

# --- third-party ---
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve_triangular
from scipy.special import gammaln
from sklearn.covariance import GraphicalLassoCV
from sklearn.metrics import f1_score, precision_score, recall_score


@njit(cache=True)
def _cholesky_update_numba(L, v):
    L_new = L.copy()
    v_cur = v.copy()
    N = len(v)
    for k in range(N):
        L_kk = L_new[k, k]
        v_k = v_cur[k]
        r = np.sqrt(L_kk ** 2 + v_k ** 2)
        c = L_kk / r
        s = v_k / r
        L_new[k, k] = r
        if k < N - 1:
            L_k_sub = L_new[k + 1:, k].copy()
            v_sub = v_cur[k + 1:].copy()
            L_new[k + 1:, k] = c * L_k_sub + s * v_sub
            v_cur[k + 1:] = -s * L_k_sub + c * v_sub
    return L_new

@njit(cache=True)
def _cholesky_downdate_numba(L, v):
    L_new = L.copy()
    v_cur = v.copy()
    N = len(v)
    for k in range(N):
        L_kk = L_new[k, k]
        v_k = v_cur[k]
        diff = L_kk ** 2 - v_k ** 2
        if diff <= 0:
            raise ValueError("non-PD downdate")
        r = np.sqrt(diff)
        c = L_kk / r
        s = v_k / r
        L_new[k, k] = r
        if k < N - 1:
            L_k_sub = L_new[k + 1:, k].copy()
            v_sub = v_cur[k + 1:].copy()
            L_new[k + 1:, k] = c * L_k_sub - s * v_sub
            v_cur[k + 1:] = -s * L_k_sub + c * v_sub
    return L_new

"""# class for model"""

class SparseFactorizedPrior:
    """Manages the factorized precision matrix prior and matrix operations.

    K = eps * I + B @ B.T, where each column b_r of B has exactly two
    nonzero entries at positions z_r = (i_r, j_r), with values
    (alpha_r, beta_r) ~ N(0,1) iid.
    """

    def __init__(
        self, N: int, R: int = 0, eps: float = 1e-3,
        lam_poisson: float | None = None, target_d: float = 3.0,
    ):
        self.N = N
        self.eps = eps

        if lam_poisson is None:
            self.lam_poisson = (N * target_d) / 2.0
        else:
            self.lam_poisson = lam_poisson

        # Candidate edge pairs (i, j) with i < j
        self.all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        self.E_total = len(self.all_pairs)  # N*(N-1)/2

        self._R_init = R

        if R > 0:
            selected_indices = np.random.choice(self.E_total, size=R, replace=True)
            self.z = np.array(self.all_pairs)[selected_indices]
            theta_init = np.random.randn(R, 2)
        else:
            self.z = np.empty((0, 2), dtype=int)
            theta_init = np.empty((0, 2))

        self.set_theta(theta_init)

    @property
    def R(self) -> int:
        """Current number of active columns."""
        return len(self.z)

    def set_theta(self, new_theta: np.ndarray):
        """Updates theta and recomputes B, K, and the Cholesky factor L."""
        self.theta = new_theta
        self._update_B_and_K()

    def update_factor(
        self, r: int, new_pair: tuple[int, int], theta_new_r: np.ndarray
    ) -> bool:
        """Rank-1 downdate (remove old column r) followed by rank-1 update
        (add new column r), used by both the swap move (new_pair != old
        pair) and single-column MALA (new_pair == old pair, weights only).
        Returns False (and leaves state untouched by caller's responsibility
        to roll back) if the downdate would be non-PD.
        """
        old_pair = tuple(self.z[r])
        theta_old_r = self.theta[r].copy()

        v_old = np.zeros(self.N)
        v_old[old_pair[0]], v_old[old_pair[1]] = theta_old_r[0], theta_old_r[1]

        v_new = np.zeros(self.N)
        v_new[new_pair[0]], v_new[new_pair[1]] = theta_new_r[0], theta_new_r[1]

        try:
            K_mid = self.K - np.outer(v_old, v_old)
            L_mid = self._cholesky_downdate(self.L, v_old)

            self.K = K_mid + np.outer(v_new, v_new)
            self.L = self._cholesky_update(L_mid, v_new)
        except ValueError:
            return False

        old_i, old_j = self.z[r]
        self.B[old_i, r] = 0.0
        self.B[old_j, r] = 0.0

        self.z[r] = new_pair
        self.theta[r] = theta_new_r

        self.B[new_pair[0], r] = theta_new_r[0]
        self.B[new_pair[1], r] = theta_new_r[1]

        return True

    def _update_B_and_K(self):
        """Full recompute of B, K, and Cholesky factor L from scratch.
        Cost: O(N^3) (dominated by the Cholesky factorization). Used at
        initialization and whenever a full-state rebuild is required.
        """
        self.B = np.zeros((self.N, self.R))
        for r in range(self.R):
            i, j = self.z[r]
            alpha, beta = self.theta[r]
            self.B[i, r] = alpha
            self.B[j, r] = beta

        self.K = self.eps * np.eye(self.N) + self.B @ self.B.T
        self.L = np.linalg.cholesky(self.K)

    def _rebuild_B(self):
        """Rebuilds B from (z, theta) without touching K/L (used after a
        rollback where K/L have already been restored separately)."""
        self.B = np.zeros((self.N, self.R))
        if self.R > 0:
            for r in range(self.R):
                i, j = self.z[r]
                self.B[i, r], self.B[j, r] = self.theta[r]

    def log_prior(self) -> float:
        """Full joint log-prior log P(z, theta), including the Poisson(lam)
        prior on R. NOTE: not called by the fixed-R swap move's acceptance
        ratio (the uniform prior on z contributes a ratio of 1 when R is
        fixed and z is drawn uniformly over pairs, so it cancels; only the
        Gaussian prior on theta matters there -- see Section 2). This
        method remains useful for the Geweke test and for reference.
        """
        R = self.R
        log_p_R = R * np.log(self.lam_poisson) - self.lam_poisson - gammaln(R + 1)

        if R == 0:
            return log_p_R

        log_P_z = -R * np.log(self.E_total)
        log_P_theta = -R * np.log(2 * np.pi) - 0.5 * np.sum(self.theta ** 2)

        return log_p_R + log_P_z + log_P_theta

    def sample_prior(self):
        """Samples (z, theta) from the full Poisson(lam)-R generative prior.
        Used for generating ground truth (Section 5), NOT for fixed-R
        inference (Section 3) or the fixed-R Geweke test (Section 7).
        """
        R_sampled = np.random.poisson(self.lam_poisson)

        if R_sampled > 0:
            selected_indices = np.random.choice(self.E_total, size=R_sampled, replace=True)
            self.z = np.array(self.all_pairs)[selected_indices]
            theta_init = np.random.randn(R_sampled, 2)
        else:
            self.z = np.empty((0, 2), dtype=int)
            theta_init = np.empty((0, 2))

        self.set_theta(theta_init)
        return self.K

    @staticmethod
    def _cholesky_update(L: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rank-1 Cholesky update. Delegates to numba-jitted implementation
        (identical arithmetic/order; only faster). Cost: O(N^2)."""
        return _cholesky_update_numba(L, v)

    @staticmethod
    def _cholesky_downdate(L: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rank-1 Cholesky downdate. Delegates to numba-jitted implementation.
        Raises ValueError if the result would not be PD."""
        try:
            return _cholesky_downdate_numba(L, v)
        except Exception:
            raise ValueError("Downdate results in a non-positive-definite matrix")


class PosteriorEvaluator:
    """Evaluates log-likelihood, log-posterior, and the gradient of the
    log-posterior w.r.t. theta, for a fixed support z."""

    def __init__(self, model: SparseFactorizedPrior, X: np.ndarray):
        self.model = model
        self.X = X
        self.N, self.T = X.shape
        self.S = (X @ X.T) / self.T

    def log_likelihood(self) -> float:
        """log p(X | K) = (T/2) log det K - (T/2) tr(K S) + const."""
        L = self.model.L
        logdet_K = 2.0 * np.sum(np.log(np.diag(L)))
        K = self.model.K
        tr_KS = np.sum(K * self.S)
        return 0.5 * self.T * logdet_K - 0.5 * self.T * tr_KS

    def log_posterior(self) -> float:
        return self.log_likelihood() + self.model.log_prior()

    def grad_theta(self):
        L = self.model.L
        B = self.model.B
        z = self.model.z
        theta = self.model.theta
        T = self.T
        R = self.model.R
        S = self.S

        grad = np.zeros_like(theta)
        if R == 0:
            return grad

        # K^{-1}B を全列まとめて解く(scipyのsolve_triangular呼び出しを R*2回 -> 2回に削減)
        Y = solve_triangular(L, B, lower=True)
        Kinv_B = solve_triangular(L, Y, lower=True, trans='T')

        # S@b_r は非ゼロ位置(i,j)から直接gatherして計算(密行列積 S@B は避ける -> O(RN)のまま)
        S_col_i = S[:, z[:, 0]]  # (N, R)
        S_col_j = S[:, z[:, 1]]  # (N, R)
        SB = theta[:, 0][None, :] * S_col_i + theta[:, 1][None, :] * S_col_j  # (N, R)

        M = Kinv_B - SB

        idx_r = np.arange(R)
        grad[:, 0] = T * M[z[:, 0], idx_r] - theta[:, 0]
        grad[:, 1] = T * M[z[:, 1], idx_r] - theta[:, 1]

        return grad

"""#MALA"""

@dataclass
class MALAConfig:
    c_init: float = 0.1
    alpha: float = -1 / 3
    target_accept: float = 0.574
    gamma: float = 0.2


class AdaptiveMALAStepper:
    def __init__(self, config: MALAConfig = MALAConfig()):
        self.c = config.c_init
        self.alpha = config.alpha
        self.target_accept = config.target_accept
        self.gamma = config.gamma
        self.is_frozen = False

    def get_h(self, N: int, R: int, T: int) -> float:
        """d = R*N (joint embedding dimension of all R columns)."""
        if R == 0:
            return 0.01
        d = R * N
        return float(self.c * (d ** self.alpha) / max(1, T))

    def adapt(self, recent_accept_rate: float):
        if self.is_frozen:
            return
        log_c = np.log(self.c) + self.gamma * (recent_accept_rate - self.target_accept)
        self.c = float(np.exp(log_c))

    def freeze(self):
        self.is_frozen = True


def propose_theta_ula(
    model, evaluator, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Proposes new theta (all R columns jointly) via ULA:
    theta' = theta + (h/2)*grad_theta + sqrt(h)*eta, eta ~ N(0, I).
    """
    theta = model.theta
    grad_at_theta = evaluator.grad_theta()

    if model.R == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    eta = np.random.randn(*theta.shape)
    theta_new = theta + (h / 2.0) * grad_at_theta + np.sqrt(h) * eta
    return theta_new, grad_at_theta


def log_q_density(
    theta_to: np.ndarray, theta_from: np.ndarray,
    grad_at_from: np.ndarray, h: float,
) -> float:
    """log q(theta_to | theta_from) = N(theta_to; theta_from + (h/2)*grad, hI)."""
    if theta_to.size == 0:
        return 0.0
    mean = theta_from + (h / 2.0) * grad_at_from
    diff = theta_to - mean
    return -0.5 / h * np.sum(diff ** 2)


def mala_step(model, evaluator, stepper: AdaptiveMALAStepper) -> tuple[bool, float, float]:
    """Full-theta (all R columns jointly) MALA step. Requires a full
    Cholesky rebuild via model.set_theta on every proposal -- O(N^3).

    Returns: (accepted, log_accept_ratio, h)
    """
    T = evaluator.T
    R = model.R
    N = model.N
    if R == 0:
        return True, 0.0, stepper.get_h(N, 0, T)

    h = stepper.get_h(N, R, T)

    theta_old = model.theta.copy()
    log_p_old = evaluator.log_posterior()

    theta_new, grad_old = propose_theta_ula(model, evaluator, h)

    model.set_theta(theta_new)
    log_p_new = evaluator.log_posterior()
    grad_new = evaluator.grad_theta()

    log_q_forward = log_q_density(theta_new, theta_old, grad_old, h)
    log_q_backward = log_q_density(theta_old, theta_new, grad_new, h)

    log_accept_ratio = (log_p_new - log_p_old) + (log_q_backward - log_q_forward)

    if np.log(np.random.rand()) < log_accept_ratio:
        return True, log_accept_ratio, h
    else:
        model.set_theta(theta_old)
        return False, log_accept_ratio, h

"""# swap mpve"""

def build_cov_bias_weights(S: np.ndarray, all_pairs: list[tuple[int, int]]) -> np.ndarray:
    weights = np.array([abs(S[i, j]) for (i, j) in all_pairs])
    return weights / weights.sum()


def swap_move_step_cov_local(model: SparseFactorizedPrior,
                              evaluator: PosteriorEvaluator,
                              pair_probs: np.ndarray,
                              pair_to_idx: dict[tuple[int, int], int],
                              sigma: float = 0.01) -> tuple[bool, float]:
    """
    R fixed swap move。
    z: coviarance
    theta: theta_new = theta_old + sigma * eta
    """
    R = model.R
    if R == 0:
      return False, -np.inf

    all_pairs = model.all_pairs
    z_old = model.z.copy()
    theta_old = model.theta.copy()
    K_old, L_old = model.K.copy(), model.L.copy()

    log_lik_before = evaluator.log_likelihood()

    r = np.random.randint(R)
    old_pair = tuple(model.z[r])
    old_pair_idx = pair_to_idx[old_pair]

    new_pair_idx = np.random.choice(len(all_pairs), p=pair_probs)
    new_pair = tuple(all_pairs[new_pair_idx])

    theta_old_r = model.theta[r].copy()
    theta_new_r = theta_old_r + sigma * np.random.randn(2)

    success = model.update_factor(r, new_pair, theta_new_r)
    if not success:
        model.z, model.theta, model.K, model.L = z_old, theta_old, K_old, L_old
        model._rebuild_B()
        return False, -np.inf

    log_lik_after = evaluator.log_likelihood()

    log_q_correction = np.log(pair_probs[old_pair_idx]) - np.log(pair_probs[new_pair_idx])

    # prior ratio (theta)
    log_prior_ratio = -0.5 * (np.sum(theta_new_r**2) - np.sum(theta_old_r**2))

    log_ratio = (log_lik_after - log_lik_before) + log_q_correction + log_prior_ratio

    if np.log(np.random.rand()) < log_ratio:
        return True, log_ratio
    else:
        model.z, model.theta, model.K, model.L = z_old, theta_old, K_old, L_old
        model._rebuild_B()
        return False, log_ratio

"""# run mcmc function"""

def run_mcmc(
    model,
    evaluator,
    stepper,
    X: np.ndarray,
    n_steps: int = 10000,
    burn_in: int = 5000,
    thinning: int = 1,
    n_swaps_per_mala: int = 10,
    check_interval: int = 50,
) -> dict:
    """Runs a fixed-R MCMC chain: a composition kernel of
    (swap x n_swaps_per_mala) -> (MALA x 1), executed every iteration.

    Args:
        model: SparseFactorizedPrior instance (R fixed at construction)
        evaluator: PosteriorEvaluator bound to model and data X
        stepper: AdaptiveMALAStepper (File A or File B variant)
        X: observed data, shape (N, T)
        n_steps: total MCMC iterations
        burn_in: iterations discarded as burn-in
        thinning: subsampling interval for saved samples
        n_swaps_per_mala: number of swap attempts per MALA update
        check_interval: how often to adapt the MALA step size during burn-in

    Returns:
        Dict with sampled histories (z, theta, K, R, log_posterior) and
        overall accept rates.
    """
    N, T = X.shape
    S = (X @ X.T) / T

    all_pairs = model.all_pairs
    pair_probs = build_cov_bias_weights(S, all_pairs)
    pair_to_idx = {tuple(p): i for i, p in enumerate(all_pairs)}
    sigma_swap = 0.01

    samples = []
    r_history = []
    log_p_history = []

    counts = {
        "mala": {"attempts": 0, "accepts": 0},
        "swap": {"attempts": 0, "accepts": 0},
    }

    recent_mala_accepts = []

    for step in range(n_steps):
        if step == burn_in:
            stepper.freeze()

        for _ in range(n_swaps_per_mala):
            counts["swap"]["attempts"] += 1
            swap_accepted, _ = swap_move_step_cov_local(
                model=model,
                evaluator=evaluator,
                pair_probs=pair_probs,
                pair_to_idx=pair_to_idx,
                sigma=sigma_swap,
            )
            if swap_accepted:
                counts["swap"]["accepts"] += 1

        counts["mala"]["attempts"] += 1
        mala_accepted, _, _ = mala_step(model, evaluator, stepper)
        if mala_accepted:
            counts["mala"]["accepts"] += 1

        recent_mala_accepts.append(1 if mala_accepted else 0)

        if step < burn_in and (step + 1) % check_interval == 0 and len(recent_mala_accepts) >= check_interval:
            recent_rate = np.mean(recent_mala_accepts[-check_interval:])
            stepper.adapt(recent_rate)

        r_history.append(model.R)
        log_p_history.append(evaluator.log_posterior())

        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples.append({
                "z": model.z.copy(),
                "theta": model.theta.copy(),
                "K": model.K.copy(),
                "R": model.R,
            })

        total_attempts = sum(c["attempts"] for c in counts.values())
        total_accepts = sum(c["accepts"] for c in counts.values())

        accept_rates = {
            "mala": counts["mala"]["accepts"] / counts["mala"]["attempts"] if counts["mala"]["attempts"] > 0 else 0.0,
            "swap": counts["swap"]["accepts"] / counts["swap"]["attempts"] if counts["swap"]["attempts"] > 0 else 0.0,
            "overall": total_accepts / total_attempts if total_attempts > 0 else 0.0,
        }

        if step % 10000 == 0:
            print(f"{step}steps---------\n")
            print(f"mala accept rate:{accept_rates['mala']}\n")
            print(f"swap accept rate:{accept_rates['swap']}\n")
            print(f"R:{model.R}\n")

    return {
        "samples": samples,
        "r_history": r_history,
        "log_p_history": log_p_history,
        "accept_rates": accept_rates,
        "final_c": stepper.c,
    }

"""# initialize section"""

def initialize_from_glasso(
    N: int, R_target: int, K_glasso: np.ndarray,
    all_pairs: list[tuple[int, int]],
    eps: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> "SparseFactorizedPrior":
    """Initializes z via weighted sampling WITHOUT replacement, weights
    proportional to |K_glasso_ij| -- guided by GLasso evidence, but
    stochastic across chains/seeds (required for R-hat to be meaningful;
    see module docstring). theta is seeded from K_glasso's values via the
    alpha*beta = K_glasso_ij convention (alpha=sqrt(|q|), beta=sign(q)*sqrt(|q|)).
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = np.array([abs(K_glasso[i, j]) for (i, j) in all_pairs])
    weights = weights + 1e-8  # avoid zero-weight pairs being un-selectable
    probs = weights / weights.sum()

    chosen_idx = rng.choice(len(all_pairs), size=R_target, replace=False, p=probs)
    top_pairs = [all_pairs[i] for i in chosen_idx]

    model = SparseFactorizedPrior(N=N, R=0, eps=eps)
    model.z = np.array(top_pairs, dtype=int)
    theta_init = np.zeros((R_target, 2))
    for r, (i, j) in enumerate(top_pairs):
        q = K_glasso[i, j]
        theta_init[r, 0] = np.sqrt(abs(q))
        theta_init[r, 1] = np.sign(q) * np.sqrt(abs(q))
    model.set_theta(theta_init)
    return model


def initialize_random_uniform(
    N: int, R_target: int, all_pairs: list[tuple[int, int]],
    eps: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> "SparseFactorizedPrior":
    """Purely random initialization, no GLasso information -- control
    condition for the convergence check (Section 7)."""
    if rng is None:
        rng = np.random.default_rng()

    chosen_idx = rng.choice(len(all_pairs), size=R_target, replace=False)
    top_pairs = [all_pairs[i] for i in chosen_idx]

    model = SparseFactorizedPrior(N=N, R=0, eps=eps)
    model.z = np.array(top_pairs, dtype=int)
    theta_init = rng.standard_normal((R_target, 2))
    model.set_theta(theta_init)
    return model

"""# data generation"""

def sample_X_given_K(K: np.ndarray, T: int) -> np.ndarray:
    """Samples T iid observations x_t ~ N(0, K^{-1}), returned as (N, T)."""
    N = K.shape[0]
    cov = np.linalg.inv(K)
    X = np.random.multivariate_normal(mean=np.zeros(N), cov=cov, size=T).T
    return X


def generate_misspecified_K(
    N: int, graph_type: str = "erdos_renyi", eps: float = 1e-3
) -> np.ndarray:
    """Generates ground truth K NOT from the model's own prior, for the
    misspecified-recovery experiments (brief Section 6)."""
    if graph_type == "erdos_renyi":
        G = nx.erdos_renyi_graph(N, p=2.0 / N)
        A = nx.to_numpy_array(G)
    elif graph_type == "grid":
        side = int(np.ceil(np.sqrt(N)))
        G = nx.grid_2d_graph(side, side)
        A = nx.to_numpy_array(G)[:N, :N]
    elif graph_type == "scale_free":
        G = nx.barabasi_albert_graph(N, m=2)
        A = nx.to_numpy_array(G)
    elif graph_type == "ar1":
        rho = 0.5
        K = np.zeros((N, N))
        scale = 1.0 / (1 - rho**2)
        for i in range(N):
            K[i, i] = scale * (1 + rho**2) if 0 < i < N - 1 else scale
            if i > 0:
                K[i, i-1] = K[i-1, i] = -rho * scale
        return K
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    weight_matrix = A * 0.5
    deg_sum = np.sum(np.abs(weight_matrix), axis=1)
    return weight_matrix + np.diag(deg_sum + eps)


def generate_well_specified_K(
    N: int, target_d: float = 4.0, eps: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Generates ground truth (K_true, z_true) from the model's own
    Poisson-R prior (SparseFactorizedPrior.sample_prior(), Section 1).

    Returns z_true alongside K_true (unlike generate_misspecified_K, which
    returns K only) since z_true is needed for the R-selection sanity
    check (distinct active edge count vs. true edge count).
    """
    model = SparseFactorizedPrior(N=N, R=0, eps=eps, target_d=target_d)
    K_true = model.sample_prior()
    z_true = model.z.copy()
    return K_true, z_true


def true_edge_count_from_K(K_true: np.ndarray, threshold: float = 1e-5) -> int:
    """Counts distinct active edges (upper triangle, off-diagonal) in a
    ground-truth K. Works for both well-specified and misspecified K_true."""
    N = K_true.shape[0]
    iu = np.triu_indices(N, k=1)
    return int(np.sum(np.abs(K_true[iu]) > threshold))

"""# select R"""

def held_out_log_likelihood(K_hat: np.ndarray, X_test: np.ndarray) -> float:
    """log p(X_test | K_hat) up to the same normalizing constant used
    elsewhere: (T/2) logdet(K_hat) - (T/2) tr(K_hat @ S_test)."""
    T_test = X_test.shape[1]
    S_test = (X_test @ X_test.T) / T_test
    sign, logdet = np.linalg.slogdet(K_hat)
    return 0.5 * T_test * logdet - 0.5 * T_test * np.trace(K_hat @ S_test)


def select_R_by_held_out_likelihood(
    X: np.ndarray, N: int, R_grid: list[int],
    n_steps: int, burn_in: int, thinning: int,
    train_frac: float = 0.5,
    n_swaps_per_mala: int = 10,
) -> tuple[int, dict]:
    """Splits X into train/test along T, fits one chain per candidate R
    on X_train, scores each by held-out log-likelihood on X_test, and
    returns the best R.

    Returns:
        best_R: the R with highest held-out log-likelihood
        results: dict mapping R -> held-out log-likelihood
    """
    T = X.shape[1]
    T_train = int(T * train_frac)
    X_train, X_test = X[:, :T_train], X[:, T_train:]

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    glasso = GraphicalLassoCV().fit(X_train.T)
    K_glasso = glasso.precision_

    results = {}
    for R in R_grid:
        rng = np.random.default_rng(R)  # deterministic per-R seed for reproducibility
        model = initialize_from_glasso(N, R, K_glasso, all_pairs, rng=rng)
        evaluator = PosteriorEvaluator(model, X_train)
        stepper = AdaptiveMALAStepper()

        mcmc_res = run_mcmc(
            model=model, evaluator=evaluator, stepper=stepper,
            X=X_train, n_steps=n_steps, burn_in=burn_in, thinning=thinning,
            n_swaps_per_mala=n_swaps_per_mala,
        )

        K_hat = np.mean([s["K"] for s in mcmc_res["samples"]], axis=0)
        held_out_ll = held_out_log_likelihood(K_hat, X_test)
        results[R] = held_out_ll
        print(f"R={R:4d}  held-out log-lik={held_out_ll:.2f}")

    best_R = max(results, key=results.get)
    return best_R, results

"""# R_hat, ESS"""

def compute_rhat(chains_data: np.ndarray) -> float:
    """Gelman-Rubin R-hat. chains_data: shape (n_chains, n_samples),
    a scalar functional (e.g. one K_ij entry, or log-posterior) tracked
    across chains and post-burn-in samples."""
    n_chains, n_samples = chains_data.shape
    if n_samples < 2 or n_chains < 2:
        return 1.0

    chain_means = np.mean(chains_data, axis=1)
    chain_vars = np.var(chains_data, axis=1, ddof=1)

    W = np.mean(chain_vars)
    B = n_samples * np.var(chain_means, ddof=1)

    if W == 0:
        return 1.0 if B == 0 else np.inf

    var_hat = ((n_samples - 1) / n_samples) * W + B / n_samples
    return float(np.sqrt(var_hat / W))

def compute_ess(chains_data: np.ndarray) -> float:
    """Geyerの initial monotone sequence estimator。
    max_lagで打ち切らず、自己相関のペア和が単調減少しなくなるまで
    適応的に追跡することで、以前のmax_lag=200による過大評価を解消する。
    """
    n_chains, n_samples = chains_data.shape
    total_n = n_chains * n_samples
    if n_samples < 4:
        return float(total_n)

    centered = chains_data - chains_data.mean(axis=1, keepdims=True)
    pooled = centered.flatten()
    var = np.var(pooled)
    if var == 0:
        return float(total_n)

    max_lag = n_samples - 1
    autocorrs = []
    for lag in range(1, max_lag):
        c = np.mean(pooled[:-lag] * pooled[lag:]) / var
        autocorrs.append(c)
        if len(autocorrs) > 1000:  # 実務上の安全な上限
            break

    autocorrs = np.array(autocorrs)

    # Geyerの手法: ラグを2つずつペアにして、ペア和が単調減少しなくなった
    # 時点(あるいは負になった時点)で打ち切る
    autocorr_sum = 0.0
    for i in range(0, len(autocorrs) - 1, 2):
        pair_sum = autocorrs[i] + autocorrs[i + 1]
        if pair_sum < 0:
            break
        autocorr_sum += pair_sum

    ess = total_n / (1 + 2 * autocorr_sum)
    return float(max(1.0, ess))
"""# R chain_check"""

def run_convergence_check(
    N: int = 20, T: int = 200, target_d: float = 4.0,
    n_chains_glasso: int = 2,
    n_chains_random: int = 1,
    n_steps_glasso: int = 20000,
    n_steps_random: int = 100000,
    burn_in_frac: float = 0.25,
    thinning: int = 5,
    n_swaps_per_mala: int = 10,
    graph_type: str = "sparse_factorized",
    seed: int = 0,
) -> tuple[dict, np.ndarray]:
    """Runs n_chains_glasso chains from GLasso-guided init (at the
    PRODUCTION n_steps_glasso -- matching the real experiments, so R-hat
    computed among THESE chains doubles as the actual production
    diagnostic) and n_chains_random chain(s) from random init (at a much
    larger n_steps_random).

    n_chains_random defaults to 1 (not matched to n_chains_glasso): the
    random-init group's role is NOT to have its own R-hat computed -- it
    exists purely to provide one independent, unbiased estimate of the
    true posterior, uncontaminated by GLasso's shared informational bias.
    R-hat among GLasso-init chains alone cannot detect a shared blind spot
    they all inherit from GLasso (all chains would agree with each other
    while all missing the same region) -- only comparison against a
    GLasso-independent source can catch that. A single well-run random
    chain is sufficient for that cross-check; more would mostly add cost
    without adding diagnostic power for this specific purpose.
    """
    np.random.seed(seed)
    if graph_type == "sparse_factorized":
        K_true, _ = generate_well_specified_K(N=N, target_d=target_d)
    else:
        K_true = generate_misspecified_K(N=N, graph_type=graph_type)
    X = sample_X_given_K(K_true, T=T)

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    glasso = GraphicalLassoCV().fit(X.T)
    K_glasso = glasso.precision_
    R_target = int(round(target_d * N / 2))

    results = {"glasso_init": [], "random_init": []}

    group_configs = [
        ("glasso_init", n_chains_glasso, n_steps_glasso,
         lambda rng: initialize_from_glasso(N, R_target, K_glasso, all_pairs, rng=rng)),
        ("random_init", n_chains_random, n_steps_random,
         lambda rng: initialize_random_uniform(N, R_target, all_pairs, rng=rng)),
    ]

    for label, n_chains, n_steps, init_fn in group_configs:
        burn_in = int(n_steps * burn_in_frac)
        print(f"\n--- {label}  (n_chains={n_chains}, n_steps={n_steps}, burn_in={burn_in}) ---")
        for c in range(n_chains):
            chain_seed = 10_000 * (0 if label == "glasso_init" else 1) + c + 1
            np.random.seed(chain_seed)
            rng = np.random.default_rng(chain_seed)

            model = init_fn(rng)
            evaluator = PosteriorEvaluator(model, X)
            stepper = AdaptiveMALAStepper()

            mcmc_res = run_mcmc(
                model=model, evaluator=evaluator, stepper=stepper, X=X,
                n_steps=n_steps, burn_in=burn_in, thinning=thinning,
                n_swaps_per_mala=n_swaps_per_mala,
            )
            z_samples = [s["z"] for s in mcmc_res["samples"]]
            edge_probs = compute_posterior_edge_probs(z_samples, N)
            results[label].append(edge_probs)
            print(f"  chain {c}: swap_accept={mcmc_res['accept_rates']['swap']:.3f}  "
                  f"mala_accept={mcmc_res['accept_rates']['mala']:.3f}  "
                  f"final R={model.R}")

    return results, K_true


def compare_convergence(results: dict) -> tuple[float, float]:
    """Compares GLasso-init vs random-init posterior edge probabilities.
    Returns (max_abs_diff, mean_abs_diff) and shows a diagonal scatter plot.
    """
    glasso_probs = np.stack(results["glasso_init"])
    random_probs = np.stack(results["random_init"])

    glasso_mean = glasso_probs.mean(axis=0)
    random_mean = random_probs.mean(axis=0)

    N = glasso_mean.shape[0]
    iu = np.triu_indices(N, k=1)
    g = glasso_mean[iu]
    r = random_mean[iu]

    max_diff = np.max(np.abs(g - r))
    mean_diff = np.mean(np.abs(g - r))
    print(f"\nMax |glasso_mean - random_mean| across all pairs: {max_diff:.4f}")
    print(f"Mean |glasso_mean - random_mean| across all pairs: {mean_diff:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g, r, alpha=0.6)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect agreement")
    ax.set_xlabel("Edge prob (GLasso-initialized chains)")
    ax.set_ylabel("Edge prob (Randomly-initialized chains)")
    ax.set_title("Convergence check: does init matter?")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return max_diff, mean_diff

"""# evaluate function"""

def compute_posterior_edge_probs(z_history: list, N: int) -> np.ndarray:
    """Posterior edge-inclusion probability matrix: for each pair (i,j),
    the fraction of pooled posterior samples in which some column selects
    that pair. z_history: list of z arrays (one per pooled sample, each
    shape (R, 2))."""
    counts = np.zeros((N, N))
    n_samples = len(z_history)
    if n_samples == 0:
        return counts

    for z in z_history:
        seen_pairs = set()
        for (i, j) in z:
            pair = (i, j) if i < j else (j, i)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                counts[i, j] += 1
                counts[j, i] += 1

    return counts / n_samples


def relative_frobenius_error(K_est: np.ndarray, K_true: np.ndarray) -> float:
    return float(np.linalg.norm(K_est - K_true, ord="fro") / np.linalg.norm(K_true, ord="fro"))


def bf_threshold(N: int, R: float, bf: float = 3.0) -> float:
    """Posterior edge-inclusion probability threshold corresponding to a
    target Bayes factor `bf`, given N nodes and R (posterior mean active
    columns). Baseline p0 = P(a specific pair selected at random, R draws
    with replacement from M=C(N,2) candidates), matching
    SparseFactorizedPrior's sampling scheme.
    """
    M = N * (N - 1) / 2
    p0 = 1 - (1 - 1 / M) ** R
    prior_odds = p0 / (1 - p0)
    post_odds = bf * prior_odds
    return post_odds / (1 + post_odds)


def evaluate_edge_recovery(
    edge_probs: np.ndarray, K_true: np.ndarray, threshold: float = 0.5
) -> dict:
    """F1/precision/recall of the thresholded posterior edge-inclusion
    matrix against the true edge support, on the upper-triangle
    (off-diagonal) entries only."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    N = K_true.shape[0]
    iu = np.triu_indices(N, k=1)

    true_labels = (np.abs(K_true[iu]) > 1e-5).astype(int)
    pred_labels = (edge_probs[iu] >= threshold).astype(int)

    return {
        "f1": f1_score(true_labels, pred_labels, zero_division=0),
        "precision": precision_score(true_labels, pred_labels, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, zero_division=0),
    }


def evaluate_glasso_baseline(
    K_glasso: np.ndarray, K_true: np.ndarray, glasso_edge_threshold: float = 1e-3
) -> dict:
    """Evaluates the Graphical Lasso point estimate with the same metrics
    used for the Bayesian posterior estimate, for baseline comparison
    (brief Section 6). GLasso gives a point estimate (not probabilities),
    so its "edge_probs" equivalent is a hard 0/1 mask at
    glasso_edge_threshold.
    """
    N = K_true.shape[0]
    glasso_edge_mask = (np.abs(K_glasso) > glasso_edge_threshold).astype(float)

    recovery = evaluate_edge_recovery(glasso_edge_mask, K_true, threshold=0.5)
    frob_err = relative_frobenius_error(K_glasso, K_true)

    return {
        "glasso_f1": recovery["f1"],
        "glasso_precision": recovery["precision"],
        "glasso_recall": recovery["recall"],
        "glasso_frobenius_error": frob_err,
    }

"""# calibration curve"""

def compute_calibration_curve(edge_probs_list, K_true_list, n_bins=10):
    """Pools (edge_probs, K_true) pairs across multiple seeds/replicates
    and computes the calibration curve.

    Args:
        edge_probs_list: list of (N,N) posterior edge-probability matrices
        K_true_list: list of (N,N) ground-truth precision matrices,
            same length, index-aligned with edge_probs_list
        n_bins: number of probability bins

    Returns:
        bin_centers, observed_freq, bin_counts (all np.ndarray)
    """
    all_probs = []
    all_true = []
    for edge_probs, K_true in zip(edge_probs_list, K_true_list):
        N = K_true.shape[0]
        iu = np.triu_indices(N, k=1)
        all_probs.append(edge_probs[iu])
        all_true.append((np.abs(K_true[iu]) > 1e-5).astype(int))

    probs = np.concatenate(all_probs)
    true_mask = np.concatenate(all_true)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, observed_freq, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        is_last_bin = (hi == bin_edges[-1])
        if is_last_bin:
            mask = (probs >= lo) & (probs <= hi)   # BUG FIX: inclusive upper edge
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            observed_freq.append(true_mask[mask].mean())
            bin_counts.append(int(mask.sum()))

    return np.array(bin_centers), np.array(observed_freq), np.array(bin_counts)


def plot_calibration_curve(bin_centers, observed_freq, bin_counts,
                            title_suffix="", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

    sizes = 20 + 200 * (bin_counts / bin_counts.max())
    ax.scatter(bin_centers, observed_freq, s=sizes, color="steelblue",
               edgecolor="black", zorder=3, label="observed")
    ax.plot(bin_centers, observed_freq, color="steelblue", alpha=0.5, zorder=2)

    for x, y, n in zip(bin_centers, observed_freq, bin_counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")

    ax.set_xlabel("Predicted posterior edge-inclusion probability")
    ax.set_ylabel("Observed fraction of true edges")
    ax.set_title(f"Calibration curve{title_suffix}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig

"""# run"""

def parse_args():
    parser = argparse.ArgumentParser(description="MCMC Graph Recovery Pipeline")
    parser.add_argument("--N", type=int, default=6, help="nodes")
    parser.add_argument("--T", type=int, default=200, help="numbers of X's samples")
    parser.add_argument(
        "--graph_type", type=str, default="erdos_renyi",
        choices=["erdos_renyi", "grid", "scale_free", "ar1", "sparse_factorized"],
        help="types of graphs",
    )
    parser.add_argument("--target_d", type=float, default=4.0,
                        help="target avg degree (used for R sizing and for "
                             "sparse_factorized ground truth generation)")
    parser.add_argument("--n_steps", type=int, default=20000, help="MCMC steps")
    parser.add_argument("--burn_in", type=int, default=5000, help="burn-in steps")
    parser.add_argument("--num_seeds", type=int, default=5, help="dataset replicates")
    parser.add_argument("--n_chains", type=int, default=2, help="chains per dataset")
    parser.add_argument("--save_dir", type=str, default="./results", help="save directory")
    parser.add_argument("--thinning", type=int, default=5, help="thinning")
    parser.add_argument("--n_swaps_per_mala", type=int, default=10,
                        help="swap attempts per MALA update")

    args, _ = parser.parse_known_args()
    return args


def run_multi_seed_experiment(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.save_dir, f"{args.graph_type}_N{args.N}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    print(
        f"=== start: type of graphs={args.graph_type}, N={args.N}, T={args.T},"
        f" loops={args.num_seeds}, chains={args.n_chains} ==="
    )
    print(f"save directory: {exp_dir}\n")

    all_seed_results = []
    all_edge_probs = []
    all_K_true = []

    for seed in range(args.num_seeds):
        print(f"--- [Dataset Seed {seed+1}/{args.num_seeds}] computing ---")
        np.random.seed(seed)
        if args.graph_type == "sparse_factorized":
            K_true, z_true = generate_well_specified_K(N=args.N, target_d=args.target_d)
        else:
            K_true = generate_misspecified_K(N=args.N, graph_type=args.graph_type)
        X = sample_X_given_K(K_true, T=args.T)

        # --- Graphical Lasso: used for (a) R selection, (b) chain
        # initialization, (c) baseline comparison metrics ---
        glasso = GraphicalLassoCV().fit(X.T)
        K_glasso = glasso.precision_
        print("NaN in K_glasso:", np.isnan(K_glasso).any())
        all_pairs = [(i, j) for i in range(args.N) for j in range(i + 1, args.N)]

        # --- R selection via held-out likelihood (Section 6) ---
        n_glasso_edges = int(np.sum(np.abs(np.triu(K_glasso, k=1)) > 1e-3))
        center = max(n_glasso_edges, 5)
        max_R = args.N * (args.N - 1) // 2
        R_grid = sorted(set(
            min(int(round(center * mult)), max_R) for mult in [1.0, 1.2, 1.5]
            if int(round(center * mult)) > 0
        ))
        best_R, r_search_results = select_R_by_held_out_likelihood(
            X=X, N=args.N, R_grid=R_grid,
            n_steps=args.n_steps // 3, burn_in=args.burn_in // 3,
            thinning=args.thinning, n_swaps_per_mala=args.n_swaps_per_mala,
        )
        R_target = best_R
        print(f"  -> selected R={R_target} via held-out likelihood: {r_search_results}")

        # --- run mcmc for the same data, across n_chains chains ---
        chains_log_post = []
        chains_K_history = []
        chains_z_pooled = []
        chains_K_est = []
        mala_accepts, swap_accepts = [], []

        for c in range(args.n_chains):
            chain_seed = seed * 1000 + c + 1
            np.random.seed(chain_seed)
            rng = np.random.default_rng(chain_seed)  # per-chain rng (Section 4 fix)

            model = initialize_from_glasso(args.N, R_target, K_glasso, all_pairs, rng=rng)
            evaluator = PosteriorEvaluator(model, X)
            stepper = AdaptiveMALAStepper()

            mcmc_res = run_mcmc(
                model=model, evaluator=evaluator, stepper=stepper, X=X,
                n_steps=args.n_steps, burn_in=args.burn_in, thinning=args.thinning,
                n_swaps_per_mala=args.n_swaps_per_mala,
            )

            chains_log_post.append(mcmc_res["log_p_history"])

            K_samples = [s["K"] for s in mcmc_res["samples"]]
            z_samples = [s["z"] for s in mcmc_res["samples"]]

            chains_K_history.append(K_samples)
            chains_z_pooled.extend(z_samples)
            chains_K_est.append(np.mean(K_samples, axis=0))

            mala_accepts.append(mcmc_res["accept_rates"]["mala"])
            swap_accepts.append(mcmc_res["accept_rates"]["swap"])

        # --- R-hat and ESS (Log Posterior + per-entry K) ---
        log_post_array = np.array(chains_log_post)
        K_history_array = np.array(chains_K_history)  # (n_chains, n_samples, N, N)

        rhat_log_post = compute_rhat(log_post_array)
        ess_log_post = compute_ess(log_post_array)

        rhat_K_matrix = np.zeros((args.N, args.N))
        ess_K_matrix = np.zeros((args.N, args.N))
        for i in range(args.N):
            for j in range(args.N):
                rhat_K_matrix[i, j] = compute_rhat(K_history_array[:, :, i, j])
                ess_K_matrix[i, j] = compute_ess(K_history_array[:, :, i, j])

        max_rhat_K = float(np.max(rhat_K_matrix))
        min_ess_K = float(np.min(ess_K_matrix))

        # --- recovery metrics (Bayesian estimate) ---
        K_est_mean = np.mean(chains_K_est, axis=0)
        edge_probs = compute_posterior_edge_probs(chains_z_pooled, args.N)
        frob_err = relative_frobenius_error(K_est_mean, K_true)
        R_bar = float(np.mean([len(z) for z in chains_z_pooled]))
        threshold = bf_threshold(args.N, R_bar, bf=3.0)
        recovery_metrics = evaluate_edge_recovery(edge_probs, K_true, threshold=threshold)
        active_edge_count = int(np.sum(edge_probs[np.triu_indices(args.N, k=1)] > threshold))
        true_edge_count = true_edge_count_from_K(K_true)

        # --- Graphical Lasso baseline metrics (NEW, Section 8) ---
        glasso_metrics = evaluate_glasso_baseline(K_glasso, K_true)

        all_edge_probs.append(edge_probs)
        all_K_true.append(K_true)

        # --- save matrix data ---
        seed_dir = os.path.join(exp_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        np.save(os.path.join(seed_dir, "K_true.npy"), K_true)
        np.save(os.path.join(seed_dir, "K_est.npy"), K_est_mean)
        np.save(os.path.join(seed_dir, "edge_probs.npy"), edge_probs)

        seed_data = {
            "seed": seed,
            "R_target": R_target,
            "r_search_results": r_search_results,
            "R_bar": R_bar,
            "threshold_used": threshold,
            "frobenius_error": frob_err,
            "f1": recovery_metrics["f1"],
            "precision": recovery_metrics["precision"],
            "recall": recovery_metrics["recall"],
            "active_edge_count": active_edge_count,
            "true_edge_count": true_edge_count,
            **glasso_metrics,
            "diagnostics": {
                "rhat_log_posterior": float(rhat_log_post),
                "ess_log_posterior": float(ess_log_post),
                "max_rhat_K": max_rhat_K,
                "min_ess_K": min_ess_K,
                "mala_accept_rate": float(np.mean(mala_accepts)),
                "swap_accept_rate": float(np.mean(swap_accepts)),
            },
        }
        all_seed_results.append(seed_data)

        print(f"  -> R-hat (LogPost): {rhat_log_post:.4f} | Max R-hat(K): {max_rhat_K:.4f}")
        print(f"  -> ESS (LogPost): {ess_log_post:.1f} | Min ESS(K): {min_ess_K:.1f}")
        print(f"  -> F1: {recovery_metrics['f1']:.4f} | Frobenius error: {frob_err:.4f}")
        print(f"  -> GLasso F1: {glasso_metrics['glasso_f1']:.4f} | "
              f"GLasso Frobenius error: {glasso_metrics['glasso_frobenius_error']:.4f}\n")

    # --- summary and save ---
    summary = {
        "config": vars(args),
        "timestamp": timestamp,
        "metrics_summary": {
            "frobenius_error_mean": float(np.mean([r["frobenius_error"] for r in all_seed_results])),
            "frobenius_error_std": float(np.std([r["frobenius_error"] for r in all_seed_results])),
            "f1_mean": float(np.mean([r["f1"] for r in all_seed_results])),
            "f1_std": float(np.std([r["f1"] for r in all_seed_results])),
            "glasso_f1_mean": float(np.mean([r["glasso_f1"] for r in all_seed_results])),
            "glasso_frobenius_error_mean": float(np.mean([r["glasso_frobenius_error"] for r in all_seed_results])),
            "rhat_log_post_mean": float(np.mean([r["diagnostics"]["rhat_log_posterior"] for r in all_seed_results])),
            "min_ess_K_mean": float(np.mean([r["diagnostics"]["min_ess_K"] for r in all_seed_results])),
        },
        "per_seed_results": all_seed_results,
    }

    json_path = os.path.join(exp_dir, "summary_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # --- calibration curve (Section 9, bug-fixed), pooled across all seeds ---
    bin_centers, observed_freq, bin_counts = compute_calibration_curve(
        all_edge_probs, all_K_true, n_bins=10
    )
    plot_calibration_curve(
        bin_centers, observed_freq, bin_counts,
        title_suffix=f" ({args.graph_type}, N={args.N}, T={args.T})",
        save_path=os.path.join(exp_dir, "calibration_curve.png"),
    )

    print("=== complete ===")
    print(f"average Frobenius error: {summary['metrics_summary']['frobenius_error_mean']:.4f}")
    print(f"average GLasso Frobenius error: {summary['metrics_summary']['glasso_frobenius_error_mean']:.4f}")
    print(f"average R-hat (LogPost): {summary['metrics_summary']['rhat_log_post_mean']:.4f}")
    print(f"path: {json_path}")

    return summary


GRAPH_TYPES = ["sparse_factorized", "erdos_renyi", "grid", "scale_free", "ar1"]
T_OVER_N_GRID = [0.5, 2, 5, 10, 20]
STEPS_BY_RATIO = {
    0.5: {"n_steps": 15000, "burn_in": 4000, "n_swaps_per_mala": 10},
    2:   {"n_steps": 30000, "burn_in": 7500, "n_swaps_per_mala": 10},
    5:   {"n_steps": 30000, "burn_in": 7500, "n_swaps_per_mala": 15},
    10:  {"n_steps": 30000, "burn_in": 8000, "n_swaps_per_mala": 20},
    20:  {"n_steps": 40000, "burn_in": 10000, "n_swaps_per_mala": 25},
}

def run_full_grid(
    base_args,
    graph_types=GRAPH_TYPES,
    T_over_N_grid=T_OVER_N_GRID,
    steps_by_ratio=STEPS_BY_RATIO,
    consolidated_save_path=None,
):
    if consolidated_save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        consolidated_save_path = os.path.join(
            base_args.save_dir, f"CONSOLIDATED_N{base_args.N}_{timestamp}.json"
        )

    all_results = {}

    for graph_type in graph_types:
        all_results[graph_type] = {}
        for ratio in T_over_N_grid:
            args = copy.deepcopy(base_args)
            args.graph_type = graph_type
            args.T = max(1, round(ratio * args.N))

            step_config = steps_by_ratio.get(ratio, {"n_steps": args.n_steps, "burn_in": args.burn_in})
            args.n_steps = step_config["n_steps"]
            args.burn_in = step_config["burn_in"]

            print(f"\n{'='*70}")
            print(f"### graph_type={graph_type}  T/N={ratio}  (T={args.T}, "
                  f"n_steps={args.n_steps}, burn_in={args.burn_in}) ###")
            print(f"{'='*70}")

            try:
                summary = run_multi_seed_experiment(args)
                all_results[graph_type][ratio] = summary
            except Exception as e:
                print(f"  !! ERROR on graph_type={graph_type}, T/N={ratio}: {e}")
                print(f"  !! Skipping this combination, continuing with the rest.")
                all_results[graph_type][ratio] = {"error": str(e)}

            with open(consolidated_save_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  -> consolidated results saved to {consolidated_save_path}")

    return all_results, consolidated_save_path


def run_single_combination(base_args, graph_type: str, ratio: float,
                            steps_by_ratio=STEPS_BY_RATIO):
    args = copy.deepcopy(base_args)
    args.graph_type = graph_type
    args.T = max(1, round(ratio * args.N))

    step_config = steps_by_ratio.get(ratio, {
        "n_steps": args.n_steps, "burn_in": args.burn_in,
        "n_swaps_per_mala": args.n_swaps_per_mala,
    })
    args.n_steps = step_config["n_steps"]
    args.burn_in = step_config["burn_in"]
    args.n_swaps_per_mala = step_config["n_swaps_per_mala"]

    print(f"{'='*70}")
    print(f"### graph_type={graph_type}  T/N={ratio}  (T={args.T}, "
          f"n_steps={args.n_steps}, burn_in={args.burn_in}) ###")
    print(f"{'='*70}")

    summary = run_multi_seed_experiment(args)
    return summary


def load_or_init_consolidated(save_dir: str, consolidated_save_path: str = None):
    if consolidated_save_path and os.path.exists(consolidated_save_path):
        with open(consolidated_save_path, "r") as f:
            return json.load(f)
    return {}


def save_to_consolidated(all_results: dict, graph_type: str, ratio: float,
                          summary: dict, consolidated_save_path: str):
    os.makedirs(os.path.dirname(consolidated_save_path), exist_ok=True)
    if graph_type not in all_results:
        all_results[graph_type] = {}
    all_results[graph_type][str(ratio)] = summary
    with open(consolidated_save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  -> saved to {consolidated_save_path}")
    return all_results



GRAPH_TYPE = "ar1"
N = 50
RATIO = 2
# ================================

base_args = parse_args()
base_args.N = N
base_args.target_d = 4.0
base_args.num_seeds = 20
base_args.n_chains = 4
base_args.save_dir = "./results_ar1912n50tn2"   

consolidated_path = os.path.join(base_args.save_dir, f"CONSOLIDATED_{GRAPH_TYPE}_N{N}_ratio{RATIO}.json")
all_results = load_or_init_consolidated(base_args.save_dir, consolidated_path)

summary = run_single_combination(base_args, graph_type=GRAPH_TYPE, ratio=RATIO)
all_results = save_to_consolidated(all_results, f"{GRAPH_TYPE}_N{N}", RATIO, summary, consolidated_path)