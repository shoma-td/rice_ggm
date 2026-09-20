"""
Fig. 1 (recovery vs T/N, all 5 generators, ours/GLasso/BDgraph) and
Fig. 2(a) (edge-probability calibration, matched vs misspecified
generators, ours vs BDgraph), built directly from the structures that
aggregate_results.py's build_fig1_results / build_fig2a_calibration_data
produce.

This is the file aggregate_results.py's docstrings refer to when they say
"the nested dict plot_fig1() expects" / "the nested dict
plot_fig2a_calibration() expects" -- it didn't exist yet in this workspace,
so it's written now specifically to wire in the BDgraph data that's
already loadable via load_bdgraph_raw / load_bdgraph_calibration_pairs.

Usage (run wherever your real result files live -- this session doesn't
have them):

    import aggregate_results as agg
    import make_paper_figures as figs

    ALL_EXPERIMENTS_DIR = "C:/Users/mech-user/rice_gnn/rice_ggm/experiment2_bd_mala_swap"
    BDGRAPH_DIR = ALL_EXPERIMENTS_DIR  # or wherever bdgraph_*.json live

    # ---- Fig. 1 ----
    bdgraph_raw = agg.load_bdgraph_raw([BDGRAPH_DIR])
    bdgraph_for_fig1 = agg.bdgraph_results_for_fig1(bdgraph_raw, N=20)
    results = agg.build_fig1_results([ALL_EXPERIMENTS_DIR], bdgraph_results=bdgraph_for_fig1)
    figs.plot_fig1(results, save_path="fig1_recovery.pdf")

    # ---- Fig. 2(a), ALL 5 graph_types (recommended) ----
    calibration_data = agg.build_fig2a_calibration_data_all_generators(
        [ALL_EXPERIMENTS_DIR], bdgraph_root_dirs=[BDGRAPH_DIR], N=20, T_over_N=0.5)
    figs.plot_fig2a_calibration_all(calibration_data, save_path="fig2a_calibration.pdf")

    # ---- Fig. 2(a), single representative misspecified generator instead ----
    # (only if you specifically want the narrower 2-panel version)
    seed_dirs_matched = agg.find_seed_dirs([ALL_EXPERIMENTS_DIR], "sparse_factorized", N=20, T_over_N=0.5)
    seed_dirs_alt = agg.find_seed_dirs([ALL_EXPERIMENTS_DIR], "erdos_renyi", N=20, T_over_N=0.5)
    bdg_matched = agg.load_bdgraph_calibration_pairs([BDGRAPH_DIR], "sparse_factorized", N=20, T_over_N=0.5)
    bdg_alt = agg.load_bdgraph_calibration_pairs([BDGRAPH_DIR], "erdos_renyi", N=20, T_over_N=0.5)
    calibration_data_2panel = agg.build_fig2a_calibration_data(
        seed_dirs_matched, seed_dirs_alt,
        bdgraph_edge_probs_matched=bdg_matched,
        bdgraph_edge_probs_alternative=bdg_alt,
    )
    figs.plot_fig2a_calibration(calibration_data_2panel, save_path="fig2a_calibration_2panel.pdf")

Both plot_fig1/plot_fig2a_calibration raise clearly (rather than silently
plotting an empty panel) if the matched (sparse_factorized) entry is
missing entirely; a missing misspecified generator instead renders as a
"[RESULT: pending]" placeholder panel so a partially-collected run still
produces a figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Fig. 1: recovery vs T/N, all 5 generators
# ---------------------------------------------------------------------

MISSPECIFIED_ORDER = ["erdos_renyi", "grid", "scale_free", "ar1"]
MISSPECIFIED_DISPLAY = {
    "erdos_renyi": "Erd\u0151s\u2013R\u00e9nyi",
    "grid": "Grid",
    "scale_free": "Scale-free",
    "ar1": "AR(1)",
}
METHOD_STYLE = {
    "edge_factor": dict(color="tab:blue", marker="o", label="Ours"),
    "glasso": dict(color="tab:orange", marker="s", label="GLasso"),
    "bdgraph": dict(color="tab:green", marker="^", label="BDgraph"),
}


def _plot_one_panel(ax, gen_results, title):
    """Plots F1 vs T/N for whichever of edge_factor/glasso/bdgraph are
    present in gen_results (each a dict {T_over_N: (mean, sem)})."""
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
                ha="center", va="center", fontsize=8, color="gray")
    ax.set_xlabel("$T/N$")
    ax.set_ylabel("Edge F1")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)


def plot_fig1(results, save_path=None, figsize=None):
    """
    results: output of aggregate_results.build_fig1_results(...), i.e.
        results[graph_type][method][T_over_N] = (mean_f1, sem_f1)
        method in {"edge_factor", "glasso", "bdgraph"}

    Layout matches the caption: one panel for the matched factor model
    (which may extend to T/N=20), and a 2x2 grid of panels for the four
    misspecified generators (erdos_renyi, grid, scale_free, ar1), each up
    to T/N=10. A single shared legend is drawn once.

    Raises KeyError if "sparse_factorized" is entirely missing from
    results (that panel is required by the caption) -- everything else
    degrades gracefully to a "[RESULT: pending]" placeholder panel so a
    partially-collected run still produces a figure you can look at.
    """
    if "sparse_factorized" not in results:
        raise KeyError(
            "results has no 'sparse_factorized' entry -- the matched factor "
            "model panel is required. Did build_fig1_results actually find "
            "any sparse_factorized summary_results.json files under the "
            "root_dirs you passed it?")

    if figsize is None:
        figsize = (3.6, 6.4)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])

    ax_matched = fig.add_subplot(gs[0, :])
    _plot_one_panel(ax_matched, results["sparse_factorized"], "Factor model (matched)")

    misspecified_axes = []
    for i, gen in enumerate(MISSPECIFIED_ORDER):
        row, col = 1 + i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        gen_results = results.get(gen, {})
        _plot_one_panel(ax, gen_results, MISSPECIFIED_DISPLAY[gen])
        misspecified_axes.append(ax)

    # Shared legend from whichever panel actually has plotted lines.
    handles, labels = [], []
    for ax in [ax_matched] + misspecified_axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Wrote Fig. 1 to: {save_path}")

    return fig


# ---------------------------------------------------------------------
# Fig. 2(a): edge-probability calibration
# ---------------------------------------------------------------------

def _plot_calibration_panel(ax, panel_data, title):
    """panel_data: dict {"edge_factor": (bin_centers, mean_freq, sem_freq,
    counts), "bdgraph": (...)} -- whichever methods are present."""
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Identity")
    any_data = False
    for method, style in [("edge_factor", METHOD_STYLE["edge_factor"]),
                           ("bdgraph", METHOD_STYLE["bdgraph"])]:
        if method not in panel_data:
            continue
        any_data = True
        bin_centers, mean_freq, sem_freq, counts = panel_data[method]
        ax.errorbar(bin_centers, mean_freq, yerr=sem_freq, capsize=2,
                    linewidth=1.2, markersize=4, **style)
    if not any_data:
        ax.text(0.5, 0.5, "[RESULT: pending]", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted inclusion probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)


def plot_fig2a_calibration(calibration_data, save_path=None, figsize=(6.4, 3.0),
                            matched_title="Factor model (matched)",
                            alternative_title="Misspecified generator"):
    """
    calibration_data: output of aggregate_results.build_fig2a_calibration_data(...):
        calibration_data["sparse_factorized"]["edge_factor"/"bdgraph"] = 4-tuple
        calibration_data["alternative"]["edge_factor"/"bdgraph"] = 4-tuple

    Two side-by-side panels: matched (sparse_factorized) and ONE
    misspecified generator (alternative). A single shared legend is drawn
    once. Use plot_fig2a_calibration_all() instead if you want all 4
    misspecified generators shown rather than a single hand-picked one.
    """
    if "sparse_factorized" not in calibration_data:
        raise KeyError(
            "calibration_data has no 'sparse_factorized' entry -- the "
            "matched-model calibration panel is required.")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    _plot_calibration_panel(axes[0], calibration_data["sparse_factorized"], matched_title)
    _plot_calibration_panel(axes[1], calibration_data.get("alternative", {}), alternative_title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Wrote Fig. 2(a) to: {save_path}")

    return fig


def plot_fig2a_calibration_all(calibration_data, save_path=None, figsize=None):
    """
    Same idea as plot_fig1's layout, but for calibration curves: one wide
    panel for the matched factor model, plus a 2x2 grid of panels for all
    four misspecified generators (erdos_renyi, grid, scale_free, ar1),
    instead of picking just one as a representative "alternative".

    calibration_data: output of
        aggregate_results.build_fig2a_calibration_data_all_generators(...),
        i.e. {graph_type: {"edge_factor": 4-tuple, "bdgraph": 4-tuple}}
        keyed by every graph_type found (missing ones render as a
        "[RESULT: pending]" placeholder panel rather than being skipped).
    """
    if "sparse_factorized" not in calibration_data:
        raise KeyError(
            "calibration_data has no 'sparse_factorized' entry -- the "
            "matched-model calibration panel is required.")

    if figsize is None:
        figsize = (6.4, 7.6)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])

    ax_matched = fig.add_subplot(gs[0, :])
    _plot_calibration_panel(ax_matched, calibration_data["sparse_factorized"],
                             "Factor model (matched)")

    misspecified_axes = []
    for i, gen in enumerate(MISSPECIFIED_ORDER):
        row, col = 1 + i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        _plot_calibration_panel(ax, calibration_data.get(gen, {}), MISSPECIFIED_DISPLAY[gen])
        misspecified_axes.append(ax)

    handles, labels = [], []
    for ax in [ax_matched] + misspecified_axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Wrote Fig. 2(a) (all generators) to: {save_path}")

    return fig


if __name__ == "__main__":
    print(__doc__)
