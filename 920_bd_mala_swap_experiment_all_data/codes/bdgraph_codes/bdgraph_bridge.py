"""
=============================================================================
BDgraph baseline bridge (Python side)
=============================================================================
Calls run_bdgraph.R via subprocess (file-based bridge, per brief: "no need
for a fancy bridge"). Uses the SAME (K_true, X) pairs as the well-specified/
misspecified recovery experiments, for a fair comparison.

Requires: R + BDgraph installed (apt install r-base r-cran-bdgraph works on
Ubuntu/Colab -- avoids needing CRAN access, which may be blocked in some
sandboxed environments).
=============================================================================
"""

import os
import subprocess
import tempfile
import time

import numpy as np

# 変更後（bdgraph_bridge.py自身の場所を基準にする）
import os as _os
BDGRAPH_R_SCRIPT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "run_bdgraph.R")

def evaluate_edge_recovery(edge_probs, K_true, threshold=0.5):
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


def relative_frobenius_error(K_est, K_true):
    return float(np.linalg.norm(K_est - K_true, ord="fro") / np.linalg.norm(K_true, ord="fro"))

def run_bdgraph_baseline(
    X: np.ndarray, K_true: np.ndarray,
    iter: int = 5000, burnin: int = 1000,
    r_script_path: str = BDGRAPH_R_SCRIPT,
    return_edge_probs: bool = False,   # 追加
) -> dict:
    """... (docstring既存のまま) ...
    return_edge_probs=True の場合、戻り値の辞書に "_edge_probs"(NxN array)
    と "_K_est"(NxN array) も含める(JSONにそのまま突っ込まないよう、
    呼び出し側でpopしてnpyとして保存することを想定したprivateなキー)。
    """
    N, T = X.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        X_csv_path = os.path.join(tmpdir, "X.csv")
        # BDgraph expects (n_samples, n_variables) = X.T relative to our (N,T)
        np.savetxt(X_csv_path, X.T, delimiter=",")

        result = subprocess.run(
            ["Rscript", r_script_path, X_csv_path, tmpdir, str(iter), str(burnin)],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(f"BDgraph R script failed:\n{result.stderr}")

        edge_probs = np.loadtxt(
            os.path.join(tmpdir, "bdgraph_edge_probs.csv"),
            delimiter=",", skiprows=1,
        )
        K_est_bd = np.loadtxt(
            os.path.join(tmpdir, "bdgraph_K_est.csv"),
            delimiter=",", skiprows=1,
        )
        timing = np.genfromtxt(
            os.path.join(tmpdir, "bdgraph_timing.csv"),
            delimiter=",", skip_header=1,
        )
        samples_per_sec = float(timing[-1])  # last column: samples_per_sec

    # Reuse the SAME evaluation functions as the Bayesian method (Section 8)
    recovery = evaluate_edge_recovery(edge_probs, K_true, threshold=0.5)
    frob_err = relative_frobenius_error(K_est_bd, K_true)

    result = {
        "bdgraph_f1": recovery["f1"],
        "bdgraph_precision": recovery["precision"],
        "bdgraph_recall": recovery["recall"],
        "bdgraph_frobenius_error": frob_err,
        "bdgraph_samples_per_sec": samples_per_sec,
    }
    if return_edge_probs:
        result["_edge_probs"] = edge_probs
        result["_K_est"] = K_est_bd
    return result
