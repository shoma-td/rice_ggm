#!/usr/bin/env python3
"""recompute_f1_threshold.py

あるデータセットのディレクトリ（seed_0/, seed_1/, ... のサブフォルダを持ち、
各フォルダに edge_probs.npy, K_est.npy, K_true.npy がある形式）から、
指定した閾値（デフォルト0.5）でedgeを判定し直してF1スコアを再計算するツール。

想定ディレクトリ構造:
    dataset_dir/
        seed_0/
            edge_probs.npy   # (N, N) 対称行列 or (M,) 上三角ベクトル
            K_est.npy
            K_true.npy       # (N, N) 真の精度行列
        seed_1/
            ...

BDgraphのデフォルト閾値(0.5)と揃えて比較するための、閾値固定版F1の算出が目的。
BF(ベイズファクター)ベースの独自閾値を使った既存のF1とは別に、
「両手法とも同じ0.5で切った場合」の公平な比較用の数値を出す。

Usage:
    python recompute_f1_threshold.py dataset_dir -o f1_at_threshold.json
    python recompute_f1_threshold.py dataset_dir --threshold 0.5
    python recompute_f1_threshold.py dataset_dir --threshold 0.5 0.3481701881451697  # 複数閾値を一度に比較
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def true_edges_from_K(K_true: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """K_true (N,N) の上三角(i<j)から、真のedgeの有無を表すブール配列(M,)を作る。

    M = N(N-1)/2, 順序は itertools.combinations(range(N), 2) と同じ
    (i=0,j=1), (i=0,j=2), ..., (i=0,j=N-1), (i=1,j=2), ...
    """
    N = K_true.shape[0]
    iu, ju = np.triu_indices(N, k=1)
    return np.abs(K_true[iu, ju]) > eps


def edge_probs_to_vector(edge_probs: np.ndarray, N: int) -> np.ndarray:
    """edge_probs.npy を (M,) の上三角ベクトルに正規化する。

    - 既に1次元 (M,) ならそのまま返す
    - 2次元 (N,N) の対称行列なら、上三角(i<j)を取り出す
    """
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


def compute_f1(true_edges: np.ndarray, pred_edges: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(true_edges & pred_edges))
    fp = int(np.sum(~true_edges & pred_edges))
    fn = int(np.sum(true_edges & ~pred_edges))
    tn = int(np.sum(~true_edges & ~pred_edges))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "true_edge_count": int(np.sum(true_edges)),
        "pred_edge_count": int(np.sum(pred_edges)),
    }


def find_seed_dirs(dataset_dir: Path) -> list[Path]:
    seed_dirs = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    return seed_dirs


def process_dataset(dataset_dir: Path, thresholds: list[float]) -> dict[str, Any]:
    seed_dirs = find_seed_dirs(dataset_dir)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {dataset_dir}")

    results: dict[str, Any] = {"dataset_dir": str(dataset_dir), "thresholds": {}}

    for threshold in thresholds:
        per_seed_results = []
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
            pred_edges = edge_probs >= threshold

            metrics = compute_f1(true_edges, pred_edges)
            metrics["seed"] = int(seed_dir.name.split("_")[-1])
            per_seed_results.append(metrics)

        f1_values = [r["f1"] for r in per_seed_results]
        precision_values = [r["precision"] for r in per_seed_results]
        recall_values = [r["recall"] for r in per_seed_results]

        results["thresholds"][str(threshold)] = {
            "f1_mean": float(np.mean(f1_values)) if f1_values else None,
            "f1_std": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0,
            "f1_sem": (
                float(np.std(f1_values, ddof=1) / np.sqrt(len(f1_values)))
                if len(f1_values) > 1
                else 0.0
            ),
            "precision_mean": float(np.mean(precision_values)) if precision_values else None,
            "recall_mean": float(np.mean(recall_values)) if recall_values else None,
            "n_seeds": len(per_seed_results),
            "per_seed": per_seed_results,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="seed_0/, seed_1/, ... を含むデータセットディレクトリ")
    parser.add_argument(
        "--threshold", type=float, nargs="+", default=[0.5],
        help="閾値(複数指定可, default: 0.5)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="出力JSONファイルパス (省略時: データセットディレクトリ内に f1_at_threshold.json として保存)",
    )
    args = parser.parse_args()

    # -o を省略したら、データセットディレクトリの中に保存する
    output_path = args.output if args.output is not None else args.dataset_dir / "f1_at_threshold.json"

    results = process_dataset(args.dataset_dir, args.threshold)

    print(f"=== {args.dataset_dir} ===")
    for threshold_str, r in results["thresholds"].items():
        print(f"  threshold={threshold_str}: "
              f"F1 = {r['f1_mean']:.4f} +/- {r['f1_sem']:.4f} (SEM, n={r['n_seeds']}), "
              f"precision = {r['precision_mean']:.4f}, recall = {r['recall_mean']:.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote results to: {output_path}")


if __name__ == "__main__":
    main()
