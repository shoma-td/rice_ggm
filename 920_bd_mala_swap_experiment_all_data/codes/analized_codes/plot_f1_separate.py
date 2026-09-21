"""
F1(/Frobenius) vs T/N 用の「別々ファイル出力」専用プロットモジュール。

plot_auprc.py と同じ考え方: make_paper_figures.plot_fig1 は5つの
graph_typeを1枚のグリッド画像(grid_f1_frobenius.png)にまとめるが、
このファイルは graph_type ごとに独立したPNGを書き出す版を提供する。

データはそのまま aggregate_results.build_fig1_results(...) の出力を使う
(集計側は変更不要 -- 既存の aggregate_results.py を再利用するだけ)。
"""

import os
import matplotlib.pyplot as plt

# Fig.1 と見た目・色分け・並び順を揃えるため、既存の定義をそのまま再利用する
from make_paper_figures import METHOD_STYLE, MISSPECIFIED_ORDER, MISSPECIFIED_DISPLAY

_GEN_ORDER = ["sparse_factorized"] + MISSPECIFIED_ORDER
_GEN_TITLE = {"sparse_factorized": "Factor model (matched)", **MISSPECIFIED_DISPLAY}


def plot_one_f1_panel(gen_results, title, figsize=(4.2, 3.4)):
    """gen_results: {"edge_factor": {...}, "glasso": {...}, "bdgraph": {...}}
    (どれか欠けていてもOK)。1つのgraph_type分のF1-vs-T/N図を1枚作って返す。"""
    fig, ax = plt.subplots(figsize=figsize)

    any_data = False
    for method in ["edge_factor", "glasso", "bdgraph"]:
        if method not in gen_results or not gen_results[method]:
            continue
        any_data = True
        tn_vals = sorted(gen_results[method].keys())
        means = [gen_results[method][tn][0] for tn in tn_vals]
        sems = [gen_results[method][tn][1] for tn in tn_vals]
        style = METHOD_STYLE[method]
        ax.errorbar(tn_vals, means, yerr=sems, capsize=2, linewidth=1.2,
                    markersize=4, **style)

    if not any_data:
        ax.text(0.5, 0.5, "[RESULT: pending]", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")

    ax.set_xlabel("$T/N$")
    ax.set_ylabel("Edge F1")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if any_data:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_f1_separate(results, output_dir=".", prefix="f1_vs_tn"):
    """
    results: aggregate_results.build_fig1_results(...) の出力。

    graph_type ごとに独立したファイル
        <output_dir>/<prefix>_sparse_factorized.png
        <output_dir>/<prefix>_erdos_renyi.png
        <output_dir>/<prefix>_grid.png
        <output_dir>/<prefix>_scale_free.png
        <output_dir>/<prefix>_ar1.png
    を書き出す。書き出したファイルパスのリストを返す。
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for gen in _GEN_ORDER:
        gen_results = results.get(gen, {})
        fig = plot_one_f1_panel(gen_results, _GEN_TITLE[gen])
        save_path = os.path.join(output_dir, f"{prefix}_{gen}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote F1 panel to: {save_path}")
        written.append(save_path)
    return written
