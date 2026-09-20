#!/usr/bin/env python3
"""compute_auprc.py

複数の手法（例: sparse_factorized vs BDgraph）のedge_probを、
PR曲線・AUPRC（閾値フリー）で比較するツール。

想定ディレクトリ構造（1手法あたり）:
    dataset_dir/
        seed_0/
            edge_probs.npy   # (N,N)対称行列 or (M,)上三角ベクトル
            K_true.npy       # (N,N)真の精度行列
        seed_1/
            ...

複数の手法を比較する場合は、--dataset ラベル パス を手法の数だけ繰り返す。

Usage:
    # 1手法だけの場合(自分の手法のAUPRCを見るだけ)
    python compute_auprc.py --dataset sparse_factorized path/to/dataset_dir

    # 2手法比較(本来の目的)
    python compute_auprc.py \\
        --dataset sparse_factorized path/to/our_dataset_dir \\
        --dataset BDgraph path/to/bdgraph_dataset_dir \\
        -o auprc_comparison

出力:
    <output>/auprc_comparison.json  -- 数値結果
    <output>/pr_curve.png           -- PR曲線のプロット(手法ごとに1本、プールした全ペア版)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def true_edges_from_K(K_true: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    N = K_true.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    return np.abs(K_true[iu, ju]) > eps


def edge_probs_to_vector(edge_probs: np.ndarray, N: int) -> np.ndarray:
    if edge_probs.ndim == 1:
        M_expected = N * (N - 1) // 2
        if edge_probs.shape[0] != M_expected:
            raise ValueError(
                f"edge_probs is 1D with shape {edge_probs.shape}, "
                f"but expected length {M_expected} (=N*(N-1)/2 for N={N})"
            )
        return edge_probs
    elif edge_probs.ndim == 2:
        if edge_probs.shape != (N, N):
            raise ValueError(
                f"edge_probs is 2D with shape {edge_probs.shape}, "
                f"but K_true has shape ({N}, {N}) -- mismatch"
            )
        iu, ju = np.triu_indices(N, k=1)
        return edge_probs[iu, ju]
    else:
        raise ValueError(f"Unexpected edge_probs.ndim={edge_probs.ndim}")


def find_seed_dirs(dataset_dir: Path) -> list[Path]:
    seed_dirs = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    return seed_dirs


def load_dataset(dataset_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """各seedのtrue_edgesベクトルとedge_probsベクトルのリストを返す。"""
    seed_dirs = find_seed_dirs(dataset_dir)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {dataset_dir}")

    all_true_edges = []
    all_edge_probs = []
    seeds = []

    for seed_dir in seed_dirs:
        edge_probs_path = seed_dir / "edge_probs.npy"
        k_true_path = seed_dir / "K_true.npy"
        if not edge_probs_path.exists() or not k_true_path.exists():
            print(f"  [skip] {seed_dir.name}: missing edge_probs.npy or K_true.npy")
            continue

        K_true = np.load(k_true_path)
        edge_probs_raw = np.load(edge_probs_path)
        N = K_true.shape[0]

        true_edges = true_edges_from_K(K_true)
        edge_probs = edge_probs_to_vector(edge_probs_raw, N)

        all_true_edges.append(true_edges)
        all_edge_probs.append(edge_probs)
        seeds.append(int(seed_dir.name.split("_")[-1]))

    return all_true_edges, all_edge_probs, seeds


def compute_method_auprc(dataset_dir: Path) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    all_true_edges, all_edge_probs, seeds = load_dataset(dataset_dir)

    # --- per-seedのAUPRC ---
    per_seed_auprc = []
    for true_edges, edge_probs, seed in zip(all_true_edges, all_edge_probs, seeds):
        if true_edges.sum() == 0 or true_edges.sum() == len(true_edges):
            # 全部true/全部falseだとAUPRCが定義できないのでスキップ
            continue
        ap = average_precision_score(true_edges, edge_probs)
        per_seed_auprc.append({"seed": seed, "auprc": float(ap)})

    auprc_values = [r["auprc"] for r in per_seed_auprc]

    # --- 全seedをプールした版(1本のPR曲線) ---
    pooled_true = np.concatenate(all_true_edges)
    pooled_probs = np.concatenate(all_edge_probs)
    pooled_ap = float(average_precision_score(pooled_true, pooled_probs))
    precision, recall, thresholds = precision_recall_curve(pooled_true, pooled_probs)

    return {
        "n_seeds": len(all_true_edges),
        "per_seed_auprc_mean": float(np.mean(auprc_values)) if auprc_values else None,
        "per_seed_auprc_std": float(np.std(auprc_values, ddof=1)) if len(auprc_values) > 1 else 0.0,
        "per_seed_auprc_sem": (
            float(np.std(auprc_values, ddof=1) / np.sqrt(len(auprc_values)))
            if len(auprc_values) > 1 else 0.0
        ),
        "per_seed": per_seed_auprc,
        "pooled_auprc": pooled_ap,
        "pooled_n_true_edges": int(pooled_true.sum()),
        "pooled_n_candidate_pairs": int(len(pooled_true)),
        "pooled_pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
    }


def plot_comparison(results: dict[str, dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, r in results.items():
        pr = r["pooled_pr_curve"]
        ax.plot(pr["recall"], pr["precision"], label=f"{label} (AUPRC={r['pooled_auprc']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (pooled over seeds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", nargs=2, action="append", metavar=("LABEL", "PATH"),
        required=True,
        help="ラベルとデータセットディレクトリのパスの組。複数指定可能。",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="出力先ディレクトリ (省略時: 1つ目のデータセットディレクトリの中に "
             "auprc_comparison/ を自動作成して保存)",
    )
    args = parser.parse_args()

    if args.output is not None:
        output_dir = args.output
    else:
        first_dataset_dir = Path(args.dataset[0][1])
        output_dir = first_dataset_dir / "auprc_comparison"
    args.output = output_dir

    args.output.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    for label, path_str in args.dataset:
        dataset_dir = Path(path_str)
        print(f"=== {label} ({dataset_dir}) ===")
        r = compute_method_auprc(dataset_dir)
        all_results[label] = r
        print(f"  pooled AUPRC = {r['pooled_auprc']:.4f} "
              f"(n_true_edges={r['pooled_n_true_edges']}, n_pairs={r['pooled_n_candidate_pairs']})")
        print(f"  per-seed AUPRC = {r['per_seed_auprc_mean']:.4f} +/- {r['per_seed_auprc_sem']:.4f} "
              f"(SEM, n_seeds={r['n_seeds']})")

    # JSON保存(PR曲線の生データは大きいので別に軽量版も残す)
    json_path = args.output / "auprc_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote results to: {json_path}")

    # プロット
    try:
        plot_path = args.output / "pr_curve.png"
        plot_comparison(all_results, plot_path)
    except ImportError:
        print("[warn] matplotlib not available -- skipping plot")


if __name__ == "__main__":
    main()
