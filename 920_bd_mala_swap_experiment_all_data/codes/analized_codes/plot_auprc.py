"""
AUPRC vs T/N 用のプロット専用モジュール（役割: 描画のみ、データ集計はしない）。

aggregate_auprc.build_auprc_vs_tn(...) が返す辞書
    results[graph_type]["edge_factor" or "bdgraph"][T_over_N] = (mean, sem)
を受け取り、Fig.1 (grid_f1_frobenius.png) とは別に、graph_type ごとに
1枚ずつ独立したPNGファイルとして出力する（1枚にまとめたグリッド画像では
ない点が make_paper_figures.plot_fig1 との違い）。
"""

import os
import matplotlib.pyplot as plt

# Fig.1 と見た目・色分けを揃えるため、既存の定義をそのまま再利用する
from make_paper_figures import METHOD_STYLE, MISSPECIFIED_ORDER, MISSPECIFIED_DISPLAY

# AUPRCは "edge_factor"(Ours) と "bdgraph" のみ（glassoは確率的スコアを
# 出さないのでAUPRCの対象外 -- Table 1と同じ理由）
_AUPRC_METHODS = ["edge_factor", "bdgraph"]

_GEN_ORDER = ["sparse_factorized"] + MISSPECIFIED_ORDER
_GEN_TITLE = {"sparse_factorized": "Factor model (matched)", **MISSPECIFIED_DISPLAY}


def plot_one_auprc_panel(gen_results, title, figsize=(4.2, 3.4)):
    """gen_results: {"edge_factor": {T_over_N: (mean, sem)}, "bdgraph": {...}}
    (一方だけでもOK)。1つのgraph_type分のAUPRC-vs-T/N図を1枚作って返す。"""
    fig, ax = plt.subplots(figsize=figsize)

    any_data = False
    for method in _AUPRC_METHODS:
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
    ax.set_ylabel("AUPRC")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    if any_data:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_auprc_separate(results, output_dir=".", prefix="auprc_vs_tn"):
    """
    results: aggregate_auprc.build_auprc_vs_tn(...) の出力。

    graph_type ごとに独立したファイル
        <output_dir>/<prefix>_sparse_factorized.png
        <output_dir>/<prefix>_erdos_renyi.png
        <output_dir>/<prefix>_grid.png
        <output_dir>/<prefix>_scale_free.png
        <output_dir>/<prefix>_ar1.png
    を書き出す（データが無いgraph_typeは "[RESULT: pending]" のプレース
    ホルダー付きの図になる）。書き出したファイルパスのリストを返す。
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for gen in _GEN_ORDER:
        gen_results = results.get(gen, {})
        fig = plot_one_auprc_panel(gen_results, _GEN_TITLE[gen])
        save_path = os.path.join(output_dir, f"{prefix}_{gen}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote AUPRC panel to: {save_path}")
        written.append(save_path)
    return written
