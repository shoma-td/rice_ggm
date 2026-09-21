"""
AUPRC vs T/N 用のデータ集計モジュール（役割: 集計のみ、プロットはしない）。

aggregate_results.py の find_auprc_jsons / load_auprc_comparison を使って
auprc_comparison.json を探し、build_fig1_results と同じ形のネスト辞書
{graph_type: {"edge_factor": {T_over_N: (mean, sem)},
              "bdgraph": {T_over_N: (mean, sem)}}}
を返す build_auprc_vs_tn(...) だけを提供する。

グラフを描く役割は plot_auprc.py、実行をまとめる役割は run_auprc.py が
担当する（このファイルはプロット・実行のことは一切知らない）。
"""

import os
import json

import aggregate_results as agg


def build_auprc_vs_tn(root_dirs, N=20, ours_key="sparse_factorized",
                       bdgraph_key="BDgraph", use_pooled=False):
    """
    Scans root_dirs for every auprc_comparison.json (at any T/N, filtered
    to the given N), and builds the same nested-dict shape
    aggregate_results.build_fig1_results returns -- {graph_type:
    {"edge_factor": {T_over_N: (mean, sem)}, "bdgraph": {T_over_N: (mean,
    sem)}}} -- but with AUPRC instead of F1, so it can be passed straight
    into plot_auprc.plot_auprc_separate.

    There is no "glasso" entry (Table 1's own reasoning: graphical lasso's
    single cross-validated point estimate isn't a probabilistic score, so
    no AUPRC is computed for it) -- the plotting code already skips
    whichever of edge_factor/bdgraph is absent.

    use_pooled: False (default) uses per_seed_auprc_mean/_sem (consistent
        with how Table 1's F1/Frobenius are mean+/-SEM across seeds);
        True uses the single pooled-PR-curve AUPRC instead, with SEM left
        at 0 (a pooled statistic has no seed-to-seed SEM of its own).

    Finding a dataset's graph_type/N/T_over_N reuses the same
    sibling-summary_results.json-search as
    aggregate_results.auprc_by_generator_from_root (auprc_comparison.json
    itself carries no such metadata).

    Prints which (graph_type, T/N) combinations were found.
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    results = {}
    found = []

    for root_dir in root_dirs:
        for auprc_path in agg.find_auprc_jsons(root_dir):
            d = os.path.dirname(auprc_path)
            sibling = None
            for _ in range(3):
                candidate = os.path.join(d, "summary_results.json")
                if os.path.exists(candidate):
                    sibling = candidate
                    break
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
            if sibling is None:
                print(f"  [skip] {auprc_path}: no summary_results.json found "
                      f"in ancestor directories to get graph_type/N/T from")
                continue
            try:
                with open(sibling) as f:
                    config = json.load(f)["config"]
                gen = config["graph_type"]
                N_here = config["N"]
                tn_here = round(config["T"] / config["N"], 4)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  [skip] {auprc_path}: couldn't read sibling config ({e})")
                continue
            if N_here != N:
                continue

            try:
                rec = agg.load_auprc_comparison(auprc_path, ours_key=ours_key, bdgraph_key=bdgraph_key)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  [skip] {auprc_path}: {e} (check ours_key/bdgraph_key)")
                continue

            if use_pooled:
                ours_val = (rec["auprc_ours_pooled"], 0.0)
                bdg_val = (rec["auprc_bdgraph_pooled"], 0.0)
            else:
                ours_val = rec["auprc_ours_mean_sem"]
                bdg_val = rec["auprc_bdgraph_mean_sem"]

            results.setdefault(gen, {}).setdefault("edge_factor", {})[tn_here] = ours_val
            results.setdefault(gen, {}).setdefault("bdgraph", {})[tn_here] = bdg_val
            found.append((gen, tn_here))

    print("Found AUPRC (graph_type, T/N) combinations:")
    for gen, tn in sorted(set(found)):
        print(f"  {gen:<18} T/N={tn}")

    return results


if __name__ == "__main__":
    print(__doc__)
