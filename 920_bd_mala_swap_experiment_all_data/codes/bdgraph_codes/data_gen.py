import numpy as np
import networkx as nx


def sample_X_given_K(K: np.ndarray, T: int) -> np.ndarray:
    """Samples T iid observations x_t ~ N(0, K^{-1}), returned as (N, T)."""
    N = K.shape[0]
    cov = np.linalg.inv(K)
    X = np.random.multivariate_normal(mean=np.zeros(N), cov=cov, size=T).T
    return X


def generate_misspecified_K(
    N: int, graph_type: str = "erdos_renyi", eps: float = 1e-3
) -> np.ndarray:
    """Generates ground truth K NOT from the model's own prior."""
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
        rho = np.random.uniform(0.3, 0.7)
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


class SparseFactorizedPrior:
    """generate_well_specified_K が sample_prior() を呼ぶために必要な、
    最小限のモデルクラス。メイン実験ファイルのSection 1と同じ内容。
    """
    def __init__(self, N, R=0, eps=1e-3, lam_poisson=None, target_d=3.0):
        self.N = N
        self.eps = eps
        self.lam_poisson = lam_poisson if lam_poisson is not None else (N * target_d) / 2.0
        self.all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        self.E_total = len(self.all_pairs)

        if R > 0:
            selected_indices = np.random.choice(self.E_total, size=R, replace=True)
            self.z = np.array(self.all_pairs)[selected_indices]
            theta_init = np.random.randn(R, 2)
        else:
            self.z = np.empty((0, 2), dtype=int)
            theta_init = np.empty((0, 2))
        self.set_theta(theta_init)

    @property
    def R(self):
        return len(self.z)

    def set_theta(self, new_theta):
        self.theta = new_theta
        self.B = np.zeros((self.N, self.R))
        for r in range(self.R):
            i, j = self.z[r]
            alpha, beta = self.theta[r]
            self.B[i, r] = alpha
            self.B[j, r] = beta
        self.K = self.eps * np.eye(self.N) + self.B @ self.B.T

    def sample_prior(self):
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


def generate_well_specified_K(N: int, target_d: float = 4.0, eps: float = 1e-3):
    """Generates ground truth (K_true, z_true) from the model's own
    Poisson-R prior."""
    model = SparseFactorizedPrior(N=N, R=0, eps=eps, target_d=target_d)
    K_true = model.sample_prior()
    z_true = model.z.copy()
    return K_true, z_true