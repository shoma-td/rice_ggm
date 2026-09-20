"""
Data aggregation for Fig. 1 / Table 1 / Fig. 2(a)(b)(c).

This file reads your actual result files (summary_results.json,
fig2b_2c_*.json, fig2c_*_edge_count_history.npy, per-seed edge_probs.npy /
K_true.npy) and reshapes them into the exact structures that
make_paper_figures.py's plotting functions expect. Every step is a plain
function you can call individually and inspect -- nothing happens inside
a black box.

BDgraph loading IS now included (find_bdgraph_jsons / load_bdgraph_raw /
bdgraph_results_for_fig1 / bdgraph_results_for_table1 /
load_bdgraph_calibration_pairs below), and AUPRC loading too
(find_auprc_jsons / load_auprc_comparison / auprc_by_generator_from_root).
make_paper_figures.py (plot_fig1 / plot_fig2a_calibration) is the file
that actually turns this module's output into the Fig. 1 / Fig. 2(a)
images -- see its module docstring for the full end-to-end usage example
including BDgraph.
"""

import os
import glob
import json
import numpy as np


# ---------------------------------------------------------------------
# Step 1: find and load summary_results.json files
# ---------------------------------------------------------------------

def find_summary_jsons(root_dir):
    """Recursively finds every summary_results.json under root_dir.
    Matches how run_multi_seed_experiment saves them: one per
    (graph_type, N, timestamp) directory."""
    return sorted(glob.glob(os.path.join(root_dir, "**", "summary_results.json"), recursive=True))


def load_summary_json(path):
    """Loads one summary_results.json and returns a flat record with
    just what the figures need:

        {
            "graph_type": str,
            "N": int,
            "T_over_N": float,          # T / N, rounded to avoid float noise
            "edge_factor_f1": [list of per-seed F1 values],
            "edge_factor_frob": [list of per-seed relative Frobenius error],
            "glasso_f1": [list of per-seed values],
            "glasso_frob": [list of per-seed values],
        }
    """
    with open(path) as f:
        data = json.load(f)

    config = data["config"]
    N = config["N"]
    T = config["T"]
    T_over_N = round(T / N, 4)

    per_seed = data["per_seed_results"]
    return {
        "graph_type": config["graph_type"],
        "N": N,
        "T_over_N": T_over_N,
        "edge_factor_f1": [r["f1"] for r in per_seed],
        "edge_factor_frob": [r["frobenius_error"] for r in per_seed],
        "glasso_f1": [r["glasso_f1"] for r in per_seed],
        "glasso_frob": [r["glasso_frobenius_error"] for r in per_seed],
    }


def find_bdgraph_edge_prob_dirs(root_dir, graph_type, N=20, T_over_N=0.5):
    """
    Finds BDgraph's saved per-seed edge_probs/K_true directories, matching
    the layout from the bdgraph_bridge.py fix:

        {root_dir}/.../{graph_type}_N{N}_T{T}/seed_{seed}/
            edge_probs.npy, bdgraph_K_est.npy, K_true.npy

    T is computed from N and T_over_N (round(T_over_N * N)), same
    convention the pipeline itself uses for T.
    """
    T = max(1, round(T_over_N * N))
    pattern = os.path.join(root_dir, "**", f"{graph_type}_N{N}_T{T}", "seed_*")
    return sorted(glob.glob(pattern, recursive=True))


def load_bdgraph_calibration_pairs(root_dir, graph_type, N=20, T_over_N=0.5):
    """Loads (edge_probs, K_true) pairs for BDgraph, for one graph_type/N/T-N,
    from the directories found by find_bdgraph_edge_prob_dirs. Returns a
    list of (edge_probs, K_true) tuples, ready to pass into
    build_fig2a_calibration_data's bdgraph_edge_probs_* arguments."""
    dirs = find_bdgraph_edge_prob_dirs(root_dir, graph_type, N=N, T_over_N=T_over_N)
    pairs = []
    for d in dirs:
        edge_probs = np.load(os.path.join(d, "edge_probs.npy"))
        K_true = np.load(os.path.join(d, "K_true.npy"))
        pairs.append((edge_probs, K_true))
    return pairs


def find_seed_dirs(root_dir, graph_type, N=20, T_over_N=0.5):
    """
    Finds every seed_{seed} directory (saved by run_multi_seed_experiment)
    matching a specific graph_type, N, and T/N. Used to build the
    seed_dirs_matched / seed_dirs_alternative lists that
    build_fig2a_calibration_data expects.

    Filters on ALL THREE of graph_type, N, and T/N -- N defaults to 20
    since that's the paper's primary scope; pass N=50 explicitly if you
    specifically want the N=50 data (e.g. for a figure that, like Fig.
    2(c), is scoped to N=50).
    """
    dirs = []
    for path in find_summary_jsons(root_dir):
        with open(path) as f:
            config = json.load(f)["config"]
        if (config["graph_type"] == graph_type
                and config["N"] == N
                and round(config["T"] / config["N"], 4) == T_over_N):
            exp_dir = os.path.dirname(path)
            dirs.extend(sorted(glob.glob(os.path.join(exp_dir, "seed_*"))))
    return dirs


def load_summary_json_diagnostics(path):
    """Loads one summary_results.json and extracts per-seed diagnostics
    (R-hat, ESS, acceptance rates), alongside graph_type/N/T_over_N.
    Separate from load_summary_json since the [DIAGNOSTICS:] paragraph
    needs different fields than the recovery figures do.

    NOTE (fixed): the real per-seed diagnostics dict has PLAIN and
    RANK-NORMALIZED R-hat as separate suffixed keys
    (rhat_log_posterior_plain/_rank, max_rhat_K_plain/_rank), not the
    unsuffixed rhat_log_posterior/max_rhat_K this used to assume. Since
    the paper reports both variants for the log-posterior, both are
    pulled out for that one.

    NOTE (decided 9/19): max_rhat_K_rank is NOT used/reported -- for most
    candidate pairs (inactive, near-constant across a chain), the
    rank-normalized R-hat is undefined (0/0), and since max_rhat_K_rank
    is a max over ~N(N-1)/2 pairs, a single undefined pair poisons the
    whole max to NaN. Only max_rhat_K_plain (which doesn't have this
    degeneracy) is used for the max-over-K diagnostic."""
    with open(path) as f:
        data = json.load(f)
    config = data["config"]
    N = config["N"]
    T = config["T"]
    T_over_N = round(T / N, 4)
    per_seed = data["per_seed_results"]

    return {
        "graph_type": config["graph_type"],
        "N": N,
        "T_over_N": T_over_N,
        "rhat_log_posterior_plain": [r["diagnostics"]["rhat_log_posterior_plain"] for r in per_seed],
        "rhat_log_posterior_rank": [r["diagnostics"]["rhat_log_posterior_rank"] for r in per_seed],
        "max_rhat_K_plain": [r["diagnostics"]["max_rhat_K_plain"] for r in per_seed],
        "ess_log_posterior": [r["diagnostics"]["ess_log_posterior"] for r in per_seed],
        "min_ess_K": [r["diagnostics"]["min_ess_K"] for r in per_seed],
        "mala_accept_rate": [r["diagnostics"]["mala_accept_rate"] for r in per_seed],
        "swap_accept_rate": [r["diagnostics"]["swap_accept_rate"] for r in per_seed],
        "bd_accept_rate": [r["diagnostics"]["bd_accept_rate"] for r in per_seed],
    }


def per_seed_diagnostics_table(root_dirs, graph_type=None, csv_path=None):
    """
    Unlike build_diagnostics_summary (which pools everything into
    min/max/mean per (graph_type, T/N) cell, losing which seed each value
    came from), this returns ONE ROW PER (file, seed) with every
    diagnostic field side by side -- so you can check whether, e.g., a
    high max_rhat_K_plain and a low swap_accept_rate actually occur on
    the SAME seed (real correlation) or on different seeds (coincidence
    of two separate marginal tails).

    Pass graph_type="sparse_factorized" (or leave None for everything) to
    scope it.

    Returns a list of dicts, one per (file, seed), with:
        path, graph_type, N, T_over_N, seed,
        rhat_log_posterior_plain, rhat_log_posterior_rank,
        max_rhat_K_plain, ess_log_posterior, min_ess_K,
        mala_accept_rate, swap_accept_rate, bd_accept_rate

    Prints the table sorted by (T_over_N, seed) and optionally writes it
    to csv_path.
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    rows = []
    for root_dir in root_dirs:
        for path in find_summary_jsons(root_dir):
            try:
                with open(path) as f:
                    data = json.load(f)
                config = data["config"]
                if graph_type and config["graph_type"] != graph_type:
                    continue
                N = config["N"]
                T = config["T"]
                T_over_N = round(T / N, 4)
                for r in data["per_seed_results"]:
                    d = r["diagnostics"]
                    rows.append({
                        "path": os.path.abspath(path),
                        "graph_type": config["graph_type"],
                        "N": N,
                        "T_over_N": T_over_N,
                        "seed": r["seed"],
                        "rhat_log_posterior_plain": d["rhat_log_posterior_plain"],
                        "rhat_log_posterior_rank": d["rhat_log_posterior_rank"],
                        "max_rhat_K_plain": d["max_rhat_K_plain"],
                        "ess_log_posterior": d["ess_log_posterior"],
                        "min_ess_K": d["min_ess_K"],
                        "mala_accept_rate": d["mala_accept_rate"],
                        "swap_accept_rate": d["swap_accept_rate"],
                        "bd_accept_rate": d["bd_accept_rate"],
                    })
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  [skip] {path}: {e}")
                continue

    rows.sort(key=lambda r: (r["graph_type"], r["T_over_N"], r["seed"]))

    if not rows:
        print("No per-seed diagnostics found.")
        return rows

    header = (f"{'graph_type':<18} {'T/N':>6} {'seed':>4} {'rhat_K':>8} "
              f"{'rhat_logpost':>12} {'swap_acc':>9} {'mala_acc':>9} {'bd_acc':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['graph_type']:<18} {r['T_over_N']:>6} {r['seed']:>4} "
              f"{r['max_rhat_K_plain']:>8.4f} {r['rhat_log_posterior_plain']:>12.4f} "
              f"{r['swap_accept_rate']:>9.4f} {r['mala_accept_rate']:>9.4f} "
              f"{r['bd_accept_rate']:>7.4f}")

    if csv_path:
        import csv
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV to: {csv_path}")

    return rows


def check_rhat_swap_correlation(root_dirs, graph_type="sparse_factorized", output_path=None):
    """
    Uses per_seed_diagnostics_table to check, WITHIN each T/N separately
    (correlating across T/N would just reflect the shared T/N trend, not
    a same-dataset relationship), whether max_rhat_K_plain and
    swap_accept_rate are correlated across the 20 seeds -- i.e. whether
    the datasets with poor R-hat are the SAME datasets with low swap
    acceptance, which would support the "swap acceptance decay causes
    genuine multimodal mixing failure" hypothesis, as opposed to the
    R-hat degradation and the swap acceptance drop being unrelated
    marginal trends that both happen to worsen with T/N.

    Prints, for each T/N found, the Pearson correlation coefficient
    between max_rhat_K_plain and swap_accept_rate across seeds, plus n.

    output_path: if given, also writes the printed report to this file.
    """
    rows = per_seed_diagnostics_table(root_dirs, graph_type=graph_type)
    by_tn = {}
    for r in rows:
        by_tn.setdefault(r["T_over_N"], []).append(r)

    report_lines = [f"Within-{graph_type}, per-T/N correlation between "
                     f"max_rhat_K_plain and swap_accept_rate (across seeds):"]
    for tn in sorted(by_tn):
        recs = by_tn[tn]
        rhat_vals = np.array([r["max_rhat_K_plain"] for r in recs])
        swap_vals = np.array([r["swap_accept_rate"] for r in recs])
        if len(recs) < 3 or np.std(rhat_vals) == 0 or np.std(swap_vals) == 0:
            report_lines.append(f"  T/N={tn}: n={len(recs)}, not enough variation to correlate")
            continue
        corr = float(np.corrcoef(rhat_vals, swap_vals)[0, 1])
        report_lines.append(f"  T/N={tn}: n={len(recs)}, corr(max_rhat_K_plain, swap_accept_rate) = {corr:.3f}")

    print("\n".join(report_lines))

    if output_path:
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\nWrote correlation report to: {output_path}")


def build_diagnostics_summary(root_dirs, output_path=None):
    """
    Scans root_dirs for every summary_results.json and pools ALL per-seed
    diagnostics across every (graph_type, N, T/N) combination found, for
    the paper's [DIAGNOSTICS: report R-hat, ESS, acceptance rates]
    paragraph. No new runs needed -- this just aggregates what's already
    in the saved JSON files.

    Prints a summary report (worst-case R-hat, worst-case ESS, and
    acceptance-rate ranges across the whole grid) and returns the raw
    pooled arrays in a dict, in case you want to slice them differently
    (e.g. per graph_type or per T/N) yourself.

    output_path: if given, also writes the same printed report text to
        this file.
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    all_rhat_log_post_plain, all_rhat_log_post_rank = [], []
    all_max_rhat_K_plain = []
    all_ess_log_post, all_min_ess_K = [], []
    all_mala_accept, all_swap_accept, all_bd_accept = [], [], []
    found = []
    skipped = []

    for root_dir in root_dirs:
        for path in find_summary_jsons(root_dir):
            try:
                rec = load_summary_json_diagnostics(path)
            except (KeyError, json.JSONDecodeError) as e:
                skipped.append((path, str(e)))
                continue
            found.append((rec["graph_type"], rec["N"], rec["T_over_N"]))
            all_rhat_log_post_plain += rec["rhat_log_posterior_plain"]
            all_rhat_log_post_rank += rec["rhat_log_posterior_rank"]
            all_max_rhat_K_plain += rec["max_rhat_K_plain"]
            all_ess_log_post += rec["ess_log_posterior"]
            all_min_ess_K += rec["min_ess_K"]
            all_mala_accept += rec["mala_accept_rate"]
            all_swap_accept += rec["swap_accept_rate"]
            all_bd_accept += rec["bd_accept_rate"]

    for path, err in skipped:
        print(f"  [skip] {path}: {err}")

    def _stats(vals):
        vals = np.asarray(vals, dtype=float)
        return {"min": float(vals.min()), "max": float(vals.max()),
                "mean": float(vals.mean()), "n": len(vals)}

    summary = {
        "n_combinations": len(set(found)),
        "n_runs_pooled": len(all_rhat_log_post_plain),
        "rhat_log_posterior_plain": _stats(all_rhat_log_post_plain),
        "rhat_log_posterior_rank": _stats(all_rhat_log_post_rank),
        "max_rhat_K_plain": _stats(all_max_rhat_K_plain),
        "ess_log_posterior": _stats(all_ess_log_post),
        "min_ess_K": _stats(all_min_ess_K),
        "mala_accept_rate": _stats(all_mala_accept),
        "swap_accept_rate": _stats(all_swap_accept),
        "bd_accept_rate": _stats(all_bd_accept),
    }

    report_lines = [
        f"Pooled diagnostics across {summary['n_combinations']} (graph_type, N, T/N) "
        f"combinations, {summary['n_runs_pooled']} individual chains:",
        f"  R-hat (log-post, plain):  min={summary['rhat_log_posterior_plain']['min']:.4f}  "
        f"max={summary['rhat_log_posterior_plain']['max']:.4f}  mean={summary['rhat_log_posterior_plain']['mean']:.4f}",
        f"  R-hat (log-post, rank):   min={summary['rhat_log_posterior_rank']['min']:.4f}  "
        f"max={summary['rhat_log_posterior_rank']['max']:.4f}  mean={summary['rhat_log_posterior_rank']['mean']:.4f}",
        f"  R-hat (max over K, plain): min={summary['max_rhat_K_plain']['min']:.4f}  "
        f"max={summary['max_rhat_K_plain']['max']:.4f}  mean={summary['max_rhat_K_plain']['mean']:.4f}",
        f"  ESS (log-posterior):    min={summary['ess_log_posterior']['min']:.1f}  "
        f"max={summary['ess_log_posterior']['max']:.1f}  mean={summary['ess_log_posterior']['mean']:.1f}",
        f"  ESS (min over K):       min={summary['min_ess_K']['min']:.1f}  "
        f"max={summary['min_ess_K']['max']:.1f}  mean={summary['min_ess_K']['mean']:.1f}",
        f"  MALA accept rate:       min={summary['mala_accept_rate']['min']:.3f}  "
        f"max={summary['mala_accept_rate']['max']:.3f}  mean={summary['mala_accept_rate']['mean']:.3f}",
        f"  Swap accept rate:       min={summary['swap_accept_rate']['min']:.3f}  "
        f"max={summary['swap_accept_rate']['max']:.3f}  mean={summary['swap_accept_rate']['mean']:.3f}",
        f"  Birth-death accept rate: min={summary['bd_accept_rate']['min']:.3f}  "
        f"max={summary['bd_accept_rate']['max']:.3f}  mean={summary['bd_accept_rate']['mean']:.3f}",
    ]
    print("\n".join(report_lines))

    if output_path:
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"\nWrote diagnostics summary to: {output_path}")

    return summary


def mean_sem(values):
    """Mean and standard error of the mean across seeds."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return None
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem


# ---------------------------------------------------------------------
# NEW: F1-only extraction / search helper
# ---------------------------------------------------------------------
#
# Two problems this solves at once:
#   1. "summary_results.json ってどこにあるの？" -- this walks every path
#      you give it (pass as many root_dirs as you want -- e.g. every
#      Drive folder you can think of -- and it will report every file it
#      actually found, with its full path, so you can see where your
#      results really live).
#   2. You don't need the full build_fig1_results/build_table1_results
#      machinery (which also wants a bdgraph_results argument) just to
#      see what F1 numbers you already have. This gives you a flat,
#      easy-to-read table (and an optional CSV) of F1 only.

def extract_f1_only(root_dirs, csv_path=None, glasso=True):
    """
    Recursively scans root_dirs for every summary_results.json, and
    returns a flat list of records:

        {
            "path": str,              # full path to the summary_results.json found
            "graph_type": str,
            "N": int,
            "T_over_N": float,
            "n_seeds": int,
            "edge_factor_f1_mean": float,
            "edge_factor_f1_sem": float,
            "glasso_f1_mean": float,   # omitted if glasso=False
            "glasso_f1_sem": float,    # omitted if glasso=False
        }

    sorted by (graph_type, N, T_over_N). Prints the same table to stdout
    (so this also doubles as a "where are my results" search: every
    summary_results.json this call actually found is printed with its
    full path).

    Pass root_dirs=["/"] (or ["/content"], ["/content/drive"], etc.) to
    search broadly if you're not sure where your results ended up --
    it's a plain recursive glob, so it's safe to point it at a large
    directory, just possibly slow.

    If csv_path is given, also writes the flat table to that path as CSV
    (no pandas dependency -- written by hand with the csv module).
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    records = []
    for root_dir in root_dirs:
        for path in find_summary_jsons(root_dir):
            try:
                rec = load_summary_json(path)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  [skip] {path}: {e}")
                continue

            ef_mean, ef_sem = mean_sem(rec["edge_factor_f1"])
            row = {
                "path": os.path.abspath(path),
                "graph_type": rec["graph_type"],
                "N": rec["N"],
                "T_over_N": rec["T_over_N"],
                "n_seeds": len(rec["edge_factor_f1"]),
                "edge_factor_f1_mean": ef_mean,
                "edge_factor_f1_sem": ef_sem,
            }
            if glasso:
                gl_mean, gl_sem = mean_sem(rec["glasso_f1"])
                row["glasso_f1_mean"] = gl_mean
                row["glasso_f1_sem"] = gl_sem
            records.append(row)

    records.sort(key=lambda r: (r["graph_type"], r["N"], r["T_over_N"]))

    if not records:
        print("No summary_results.json found under:", root_dirs)
        return records

    header = f"{'graph_type':<18} {'N':>4} {'T/N':>6} {'n':>3} {'edge-factor F1':>18}"
    if glasso:
        header += f" {'glasso F1':>16}"
    header += "   path"
    print(header)
    print("-" * len(header.rstrip("   path")) + " " + "-" * 40)
    for r in records:
        line = (f"{r['graph_type']:<18} {r['N']:>4} {r['T_over_N']:>6} {r['n_seeds']:>3} "
                f"{r['edge_factor_f1_mean']:>7.4f} +/- {r['edge_factor_f1_sem']:<6.4f}")
        if glasso:
            line += f"  {r['glasso_f1_mean']:>7.4f} +/- {r['glasso_f1_sem']:<6.4f}"
        line += f"   {r['path']}"
        print(line)

    if csv_path:
        import csv
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nWrote CSV to: {csv_path}")

    return records


def extract_recovery_summary(root_dirs, csv_path=None, glasso=True):
    """extract_f1_onlyのF1+Frobenius版。graph_type/N/T_over_N/n_seedsに加え、
    edge_factor_f1/frob と glasso_f1/frob の平均±SEMを一覧化する。
    フォーマットが古い/欠けているファイルはスキップし、理由を表示する。"""
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    records = []
    for root_dir in root_dirs:
        for path in find_summary_jsons(root_dir):
            try:
                rec = load_summary_json(path)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  [skip] {path}: missing/unexpected field {e}")
                continue

            try:
                ef_f1_mean, ef_f1_sem = mean_sem(rec["edge_factor_f1"])
                ef_frob_mean, ef_frob_sem = mean_sem(rec["edge_factor_frob"])
            except KeyError as e:
                print(f"  [skip] {path}: rec missing key {e} (keys present: {list(rec.keys())})")
                continue

            row = {
                "path": os.path.abspath(path),
                "graph_type": rec["graph_type"],
                "N": rec["N"],
                "T_over_N": rec["T_over_N"],
                "n_seeds": len(rec["edge_factor_f1"]),
                "edge_factor_f1_mean": ef_f1_mean,
                "edge_factor_f1_sem": ef_f1_sem,
                "edge_factor_frob_mean": ef_frob_mean,
                "edge_factor_frob_sem": ef_frob_sem,
            }
            if glasso:
                try:
                    gl_f1_mean, gl_f1_sem = mean_sem(rec["glasso_f1"])
                    gl_frob_mean, gl_frob_sem = mean_sem(rec["glasso_frob"])
                    row["glasso_f1_mean"] = gl_f1_mean
                    row["glasso_f1_sem"] = gl_f1_sem
                    row["glasso_frob_mean"] = gl_frob_mean
                    row["glasso_frob_sem"] = gl_frob_sem
                except KeyError as e:
                    print(f"  [warn] {path}: no glasso field {e}, leaving glasso_* blank for this row")
                    row["glasso_f1_mean"] = row["glasso_f1_sem"] = None
                    row["glasso_frob_mean"] = row["glasso_frob_sem"] = None
            records.append(row)

    records.sort(key=lambda r: (r["graph_type"], r["N"], r["T_over_N"]))

    if csv_path and records:
        import csv
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote CSV to: {csv_path}")

    return records


# ---------------------------------------------------------------------
# Step 2: build the Fig. 1 input structure
# ---------------------------------------------------------------------

def build_fig1_results(root_dirs, bdgraph_results=None):
    """
    Scans root_dirs (a list of directories -- e.g. wherever your N=20 and
    N=50 experiment folders live) for every summary_results.json, and
    builds the nested dict plot_fig1() expects:

        results[graph_type]["edge_factor"][T_over_N] = (mean_f1, sem_f1)
        results[graph_type]["glasso"][T_over_N] = (mean_f1, sem_f1)
        results[graph_type]["bdgraph"][T_over_N] = (mean_f1, sem_f1)  # if provided

    `bdgraph_results`, if given, should already be in the shape:
        {graph_type: {T_over_N: (mean_f1, sem_f1)}}
    (i.e. call load_bdgraph_results(...) yourself first, then pass its
    output here -- kept separate since the BDgraph file format isn't
    wired up yet).

    Prints a short report of which (graph_type, T/N) combinations were
    found, so missing data is obvious rather than silently absent.
    """
    results = {}
    found = []

    for root_dir in root_dirs:
        for path in find_summary_jsons(root_dir):
            rec = load_summary_json(path)
            gen = rec["graph_type"]
            tn = rec["T_over_N"]
            found.append((gen, rec["N"], tn))

            results.setdefault(gen, {}).setdefault("edge_factor", {})[tn] = mean_sem(rec["edge_factor_f1"])
            results.setdefault(gen, {}).setdefault("glasso", {})[tn] = mean_sem(rec["glasso_f1"])

    if bdgraph_results:
        for gen, tn_dict in bdgraph_results.items():
            results.setdefault(gen, {}).setdefault("bdgraph", {}).update(tn_dict)

    print("Found (graph_type, N, T/N) combinations:")
    for gen, N, tn in sorted(found):
        print(f"  {gen:<18} N={N:<4} T/N={tn}")

    return results


# ---------------------------------------------------------------------
# Step 3: build the Table 1 input structure (N=50, T/N=0.5)
# ---------------------------------------------------------------------

def build_table1_results(root_dirs, bdgraph_results=None, N=50, T_over_N=0.5):
    """
    Same scan as build_fig1_results, but filtered to a single (N, T/N)
    and reshaped for make_table1():

        results[graph_type]["glasso"] = {"f1": mean, "frob": mean}
        results[graph_type]["edge_factor"] = {"f1": mean, "frob": mean}
        results[graph_type]["bdgraph"] = {"f1": mean, "frob": mean}  # if provided

    `bdgraph_results` here should be shaped:
        {graph_type: {"f1": mean, "frob": mean}}
    """
    results = {}
    found = []

    for root_dir in root_dirs:
        for path in find_summary_jsons(root_dir):
            rec = load_summary_json(path)
            if rec["N"] != N or rec["T_over_N"] != T_over_N:
                continue
            gen = rec["graph_type"]
            found.append(gen)

            ef_f1 = mean_sem(rec["edge_factor_f1"])
            ef_frob = mean_sem(rec["edge_factor_frob"])
            gl_f1 = mean_sem(rec["glasso_f1"])
            gl_frob = mean_sem(rec["glasso_frob"])

            results.setdefault(gen, {})["edge_factor"] = {"f1": ef_f1[0], "frob": ef_frob[0]}
            results.setdefault(gen, {})["glasso"] = {"f1": gl_f1[0], "frob": gl_frob[0]}

    if bdgraph_results:
        for gen, vals in bdgraph_results.items():
            results.setdefault(gen, {})["bdgraph"] = vals

    print(f"Found generators at N={N}, T/N={T_over_N}: {found}")
    return results


# ---------------------------------------------------------------------
# Step 4: BDgraph loader
# ---------------------------------------------------------------------
#
# Expected file format (one file per graph_type/N/T combination, e.g.
# "bdgraph_scale_free_N20_ratio0_5.json"):
#
#   {
#     "scale_free_N20_T10": {
#       "0": {"bdgraph_f1": ..., "bdgraph_precision": ..., "bdgraph_recall": ...,
#             "bdgraph_frobenius_error": ..., "bdgraph_samples_per_sec": ...,
#             "graph_type": "scale_free", "N": 20, "T": 10, "seed": 0},
#       "1": { ... },
#       ...
#     }
#   }
#
# i.e. a single top-level key (its exact text doesn't matter -- we read
# graph_type/N/T straight from each seed's own record) mapping to a dict
# of per-seed records keyed by seed index as a string.

def find_bdgraph_jsons(root_dir):
    """Recursively finds every bdgraph_*.json under root_dir."""
    return sorted(glob.glob(os.path.join(root_dir, "**", "bdgraph_*.json"), recursive=True))


def load_bdgraph_raw(root_dirs):
    """
    Scans root_dirs for bdgraph_*.json files and pools every per-seed
    record into a flat structure:

        raw[graph_type][N][T_over_N] = {
            "f1": [list of per-seed bdgraph_f1],
            "frob": [list of per-seed bdgraph_frobenius_error],
            "samples_per_sec": [list of per-seed bdgraph_samples_per_sec],
        }

    NOTE: keyed by N as well as T_over_N -- an earlier version of this
    function only keyed by (graph_type, T_over_N), which silently pooled
    N=20 and N=50 results together whenever both existed at the same T/N
    (a real bug, caught when Table 1's BDgraph numbers looked suspiciously
    close to the N=20 figures).

    Prints which (graph_type, N, T/N) combinations were found, same
    convention as build_fig1_results, so gaps are visible.
    """
    raw = {}
    found = []

    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    for root_dir in root_dirs:
        for path in find_bdgraph_jsons(root_dir):
            with open(path) as f:
                data = json.load(f)
            for _, seed_records in data.items():
                for _, rec in seed_records.items():
                    gen = rec["graph_type"]
                    N = rec["N"]
                    T = rec["T"]
                    tn = round(T / N, 4)
                    found.append((gen, N, tn))

                    bucket = raw.setdefault(gen, {}).setdefault(N, {}).setdefault(
                        tn, {"f1": [], "frob": [], "samples_per_sec": []})
                    bucket["f1"].append(rec["bdgraph_f1"])
                    bucket["frob"].append(rec["bdgraph_frobenius_error"])
                    if "bdgraph_samples_per_sec" in rec:
                        bucket["samples_per_sec"].append(rec["bdgraph_samples_per_sec"])

    print("Found BDgraph (graph_type, N, T/N) combinations:")
    for gen, N, tn in sorted(set(found)):
        n_seeds = sum(1 for g, n, t in found if g == gen and n == N and t == tn)
        print(f"  {gen:<18} N={N:<4} T/N={tn:<6} ({n_seeds} seeds)")

    return raw


def bdgraph_results_for_fig1(bdgraph_raw, N=20):
    """Reshapes load_bdgraph_raw's output into what build_fig1_results'
    `bdgraph_results` argument expects:
        {graph_type: {T_over_N: (mean_f1, sem_f1)}}
    N defaults to 20 (the paper's primary scope for Fig. 1) -- pass N=50
    explicitly if that's what you actually want plotted.
    """
    out = {}
    for gen, n_dict in bdgraph_raw.items():
        if N not in n_dict:
            continue
        out[gen] = {tn: mean_sem(vals["f1"]) for tn, vals in n_dict[N].items()}
    return out


def bdgraph_results_for_table1(bdgraph_raw, N=50, T_over_N=0.5):
    """Reshapes load_bdgraph_raw's output into what build_table1_results'
    `bdgraph_results` argument expects, for one N/T-N slice:
        {graph_type: {"f1": mean, "frob": mean}}
    """
    out = {}
    for gen, n_dict in bdgraph_raw.items():
        if N not in n_dict or T_over_N not in n_dict[N]:
            continue
        vals = n_dict[N][T_over_N]
        out[gen] = {"f1": mean_sem(vals["f1"])[0], "frob": mean_sem(vals["frob"])[0]}
    return out


# ---------------------------------------------------------------------
# Step 5: Fig. 2(b)/(c) loaders (these formats ARE known, from
# run_fig2b_fig2c_diagnostics)
# ---------------------------------------------------------------------

def load_fig2b_per_seed_results(json_path):
    """Loads fig2b_2c_N*_TN*.json as-is -- it's already a list of
    per-seed dicts in exactly the shape plot_fig2b_recovery_vs_R() wants
    (R_grid, best_R, r_grid_results with f1 per candidate R)."""
    with open(json_path) as f:
        return json.load(f)


def load_fig2c_data(json_path, npy_path, seed=0):
    """Loads the seed=0 entry's true_edge_count/R_sel from the JSON, and
    the edge-count history array from the .npy file. Returns a tuple
    ready to pass into plot_fig2c_edge_count_posterior:

        (edge_count_history, true_edge_count, R_sel)
    """
    with open(json_path) as f:
        all_seeds = json.load(f)

    seed_entry = next(r for r in all_seeds if r["seed"] == seed)
    true_edge_count = seed_entry["true_edge_count"]
    R_sel = seed_entry["fig2c_R_sel_used"]

    edge_count_history = np.load(npy_path)
    return edge_count_history, true_edge_count, R_sel


# ---------------------------------------------------------------------
# Step 6: calibration data for Fig. 2(a)
# ---------------------------------------------------------------------

def compute_calibration_curve_per_seed(edge_probs, K_true, n_bins=10):
    """
    Calibration curve for a SINGLE dataset (one seed), evaluated over a
    FIXED set of bins (0..n_bins-1, same edges every call) so results
    from different seeds line up bin-for-bin. Bins with zero pairs in
    this dataset get NaN (rare at N>=20, but handled).

    Returns: (observed_freq, bin_counts), both length n_bins arrays.
    bin_centers can be recovered as (bin_edges[:-1]+bin_edges[1:])/2 for
    the same n_bins, or just call get_bin_centers(n_bins).
    """
    N = edge_probs.shape[0]
    iu = np.triu_indices(N, k=1)
    probs = edge_probs[iu]
    true_mask = (np.abs(K_true[iu]) > 1e-5).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    observed_freq = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        is_last_bin = (b == n_bins - 1)
        mask = (probs >= lo) & (probs <= hi) if is_last_bin else (probs >= lo) & (probs < hi)
        count = mask.sum()
        bin_counts[b] = count
        if count > 0:
            observed_freq[b] = true_mask[mask].mean()

    return observed_freq, bin_counts


def get_bin_centers(n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def compute_calibration_curve_with_uncertainty(edge_probs_list, K_true_list, n_bins=10):
    """
    Computes a calibration curve PER DATASET (seed) and aggregates across
    datasets bin-by-bin (mean and standard error of the mean), so the
    result carries cross-dataset uncertainty rather than pooling every
    pair from every seed into one giant bin count.

    This is the "include uncertainty across datasets" version -- use this
    instead of compute_calibration_curve() when you want error bars.

    Returns: (bin_centers, mean_freq, sem_freq, total_bin_counts)
        bin_centers: length n_bins
        mean_freq: length n_bins, mean observed frequency across datasets
                   that had at least one pair in that bin (NaN if none did)
        sem_freq: length n_bins, standard error of that mean across datasets
                   (0 if only one dataset contributed to that bin)
        total_bin_counts: length n_bins, total pooled pair count in that
                   bin across all datasets (for marker sizing only)
    """
    n_seeds = len(edge_probs_list)
    per_seed_freq = np.full((n_seeds, n_bins), np.nan)
    per_seed_counts = np.zeros((n_seeds, n_bins), dtype=int)

    for s, (edge_probs, K_true) in enumerate(zip(edge_probs_list, K_true_list)):
        freq, counts = compute_calibration_curve_per_seed(edge_probs, K_true, n_bins=n_bins)
        per_seed_freq[s] = freq
        per_seed_counts[s] = counts

    bin_centers = get_bin_centers(n_bins)
    mean_freq = np.full(n_bins, np.nan)
    sem_freq = np.zeros(n_bins)
    total_counts = per_seed_counts.sum(axis=0)

    for b in range(n_bins):
        vals = per_seed_freq[:, b]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        mean_freq[b] = vals.mean()
        sem_freq[b] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

    # drop bins with no data at all from any dataset
    keep = ~np.isnan(mean_freq)
    return bin_centers[keep], mean_freq[keep], sem_freq[keep], total_counts[keep]


def compute_calibration_curve(edge_probs_list, K_true_list, n_bins=10):
    """Pools (edge_probs, K_true) pairs across multiple seeds/replicates
    and computes ONE calibration curve with no cross-dataset uncertainty
    (matches the pipeline's own compute_calibration_curve exactly, bug
    fix for the inclusive last-bin edge included). Prefer
    compute_calibration_curve_with_uncertainty() for the paper figure,
    which reports error bars across datasets -- this pooled version is
    kept for parity/sanity-checking against the pipeline's saved PNGs.

    Returns: (bin_centers, observed_freq, bin_counts)
    """
    all_probs = []
    all_true = []
    for edge_probs, K_true in zip(edge_probs_list, K_true_list):
        N = edge_probs.shape[0]
        iu = np.triu_indices(N, k=1)
        all_probs.append(edge_probs[iu])
        all_true.append((np.abs(K_true[iu]) > 1e-5).astype(float))

    all_probs = np.concatenate(all_probs)
    all_true = np.concatenate(all_true)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, observed_freq, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        is_last_bin = (hi == bin_edges[-1])
        if is_last_bin:
            # matches the pipeline's own compute_calibration_curve: the last
            # bin's upper edge must be inclusive, or edge_probs of exactly
            # 1.0 get dropped from every bin entirely
            mask = (all_probs >= lo) & (all_probs <= hi)
        else:
            mask = (all_probs >= lo) & (all_probs < hi)
        count = mask.sum()
        if count == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        observed_freq.append(all_true[mask].mean())
        bin_counts.append(count)

    return np.array(bin_centers), np.array(observed_freq), np.array(bin_counts)


def build_fig2a_calibration_data_all_generators(root_dirs, bdgraph_root_dirs=None,
                                                 N=20, T_over_N=0.5, n_bins=10,
                                                 generators=None):
    """
    Like build_fig2a_calibration_data, but builds the calibration curve for
    EVERY graph_type at once (all 5, not just one matched + one
    hand-picked "alternative"), so Fig. 2(a) doesn't have to leave out
    3 of the 4 misspecified generators.

    root_dirs: passed to find_seed_dirs for each graph_type.
    bdgraph_root_dirs: passed to load_bdgraph_calibration_pairs for each
        graph_type, or None to skip BDgraph curves entirely.
    generators: which graph_types to build curves for; defaults to all 5
        (sparse_factorized + the 4 misspecified generators).

    Returns: {graph_type: {"edge_factor": 4-tuple, "bdgraph": 4-tuple}}
    -- same per-method shape as build_fig2a_calibration_data, just keyed
    by every graph_type instead of only "sparse_factorized"/"alternative".
    Pass this straight into plot_fig2a_calibration_all() in
    make_paper_figures.py.

    Prints which graph_types actually had seed dirs (with edge_probs.npy/
    K_true.npy) found, so a graph_type quietly missing that data is
    visible rather than silently absent from the figure.
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    if generators is None:
        generators = ["sparse_factorized", "erdos_renyi", "grid", "scale_free", "ar1"]

    calibration_data = {}
    for gen in generators:
        seed_dirs = []
        for root_dir in root_dirs:
            seed_dirs.extend(find_seed_dirs(root_dir, gen, N=N, T_over_N=T_over_N))
        if not seed_dirs:
            print(f"  [skip] {gen}: no seed_* dirs found at N={N}, T/N={T_over_N}")
            continue

        edge_probs_list, K_true_list = [], []
        for d in seed_dirs:
            try:
                edge_probs_list.append(np.load(os.path.join(d, "edge_probs.npy")))
                K_true_list.append(np.load(os.path.join(d, "K_true.npy")))
            except FileNotFoundError as e:
                print(f"  [skip seed dir] {d}: {e}")
                continue

        if not edge_probs_list:
            print(f"  [skip] {gen}: seed dirs found but no usable edge_probs.npy/K_true.npy")
            continue

        entry = {
            "edge_factor": compute_calibration_curve_with_uncertainty(
                edge_probs_list, K_true_list, n_bins=n_bins)
        }
        print(f"  found {gen}: {len(edge_probs_list)} seeds (ours)")

        if bdgraph_root_dirs:
            bdg_root_list = [bdgraph_root_dirs] if isinstance(bdgraph_root_dirs, str) else bdgraph_root_dirs
            bdg_pairs = []
            for bdg_root in bdg_root_list:
                bdg_pairs.extend(load_bdgraph_calibration_pairs(bdg_root, gen, N=N, T_over_N=T_over_N))
            if bdg_pairs:
                ep, kt = zip(*bdg_pairs)
                entry["bdgraph"] = compute_calibration_curve_with_uncertainty(
                    list(ep), list(kt), n_bins=n_bins)
                print(f"  found {gen}: {len(bdg_pairs)} seeds (BDgraph)")
            else:
                print(f"  [skip] {gen}: no BDgraph edge_probs/K_true pairs found")

        calibration_data[gen] = entry

    return calibration_data


def build_fig2a_calibration_data(seed_dirs_matched, seed_dirs_alternative,
                                  bdgraph_edge_probs_matched=None,
                                  bdgraph_edge_probs_alternative=None,
                                  n_bins=10):
    """
    seed_dirs_matched / seed_dirs_alternative: lists of per-seed directory
    paths (the `seed_{seed}` folders saved by run_multi_seed_experiment),
    each expected to contain edge_probs.npy and K_true.npy.

    bdgraph_edge_probs_*: optional, pre-computed lists of (edge_probs,
    K_true) NxN array pairs for BDgraph, if you have them (format
    depends on your BDgraph bridge -- adjust as needed once that's wired
    up).

    Uses compute_calibration_curve_with_uncertainty (per-seed curves,
    aggregated with mean/SEM across seeds) -- so the returned data
    carries cross-dataset error bars, per the draft's "include
    uncertainty across datasets" requirement.

    Returns the nested dict plot_fig2a_calibration() expects, now with
    4-tuples (bin_centers, mean_freq, sem_freq, bin_counts) per method.
    """
    def _load_pairs(seed_dirs):
        edge_probs_list, K_true_list = [], []
        for d in seed_dirs:
            edge_probs_list.append(np.load(os.path.join(d, "edge_probs.npy")))
            K_true_list.append(np.load(os.path.join(d, "K_true.npy")))
        return edge_probs_list, K_true_list

    calibration_data = {}

    ep, kt = _load_pairs(seed_dirs_matched)
    calibration_data["sparse_factorized"] = {
        "edge_factor": compute_calibration_curve_with_uncertainty(ep, kt, n_bins=n_bins)
    }

    ep, kt = _load_pairs(seed_dirs_alternative)
    calibration_data["alternative"] = {
        "edge_factor": compute_calibration_curve_with_uncertainty(ep, kt, n_bins=n_bins)
    }

    if bdgraph_edge_probs_matched:
        ep, kt = zip(*bdgraph_edge_probs_matched)
        calibration_data["sparse_factorized"]["bdgraph"] = compute_calibration_curve_with_uncertainty(list(ep), list(kt), n_bins=n_bins)

    if bdgraph_edge_probs_alternative:
        ep, kt = zip(*bdgraph_edge_probs_alternative)
        calibration_data["alternative"]["bdgraph"] = compute_calibration_curve_with_uncertainty(list(ep), list(kt), n_bins=n_bins)

    return calibration_data


# ---------------------------------------------------------------------
# Step 7: generate Table 1's LaTeX rows directly from real data
# ---------------------------------------------------------------------

def generate_table1_latex_rows(root_dirs, bdgraph_raw=None, N=20, T_over_N=0.5,
                                auprc_by_generator=None, output_path=None):
    """
    Generates the exact LaTeX rows for Table 1 (\\label{tab:recovery} in
    main.tex) computed directly from real data -- no manual transcription
    needed. Prints them ready to paste into the tabular environment, and
    returns them as a list of strings.

    root_dirs: passed to find_summary_jsons for edge-factor/glasso data
        (same argument you'd pass to extract_recovery_summary).
    bdgraph_raw: output of load_bdgraph_raw(...), or None to leave the
        BDg component of each cell as "[BDg]".
    auprc_by_generator: optional dict {graph_type: (auprc_ours, auprc_bdgraph)}
        -- exact key/shape TBD once auprc_comparison.json's schema is
        confirmed; leave None to leave the AUPRC column as [RESULT].
    N, T_over_N: which slice of the grid to use (default matches Table 1:
        N=20, T/N=0.5).
    output_path: if given, also writes the generated rows (one per line,
        exactly as printed) to this path -- so the table doesn't need to
        be copy-pasted out of the console. A good target is a .tex file
        that main.tex \\input{}s directly, or a plain .txt to paste by
        hand; either works, this just writes the raw lines.

    Row order and display names match main.tex's Table 1 exactly.
    """
    display_names = {
        "sparse_factorized": "Factor model",
        "erdos_renyi": "Erd\\H{o}s--R\\'enyi",
        "grid": "Grid",
        "scale_free": "Scale-free",
        "ar1": "AR(1)",
    }
    order = ["sparse_factorized", "erdos_renyi", "grid", "scale_free", "ar1"]

    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    per_gen = {}
    for root_dir in root_dirs:
        for path in find_summary_jsons(root_dir):
            try:
                rec = load_summary_json(path)
            except (KeyError, json.JSONDecodeError):
                continue
            if rec["N"] != N or rec["T_over_N"] != T_over_N:
                continue
            per_gen[rec["graph_type"]] = {
                "ef_f1": mean_sem(rec["edge_factor_f1"]),
                "ef_frob": mean_sem(rec["edge_factor_frob"]),
                "gl_f1": mean_sem(rec["glasso_f1"]),
                "gl_frob": mean_sem(rec["glasso_frob"]),
            }

    bdg_by_gen = {}
    if bdgraph_raw:
        for gen, n_dict in bdgraph_raw.items():
            if N in n_dict and T_over_N in n_dict[N]:
                vals = n_dict[N][T_over_N]
                bdg_by_gen[gen] = {
                    "f1": mean_sem(vals["f1"]),
                    "frob": mean_sem(vals["frob"]),
                }

    def _fmt_pm(ms):
        if ms is None:
            return "?"
        m, s = ms
        return f"${m:.3f}\\pm{s:.3f}$"

    lines = []
    for gen in order:
        name = display_names[gen]
        stats = per_gen.get(gen)
        if stats is None:
            lines.append(f"{name:<22} & [RESULT] & [RESULT] & [RESULT] \\\\")
            continue
        bdg = bdg_by_gen.get(gen)
        bdg_f1_str = _fmt_pm(bdg["f1"]) if bdg else "[BDg]"
        bdg_frob_str = _fmt_pm(bdg["frob"]) if bdg else "[BDg]"
        f1_cell = f"{_fmt_pm(stats['ef_f1'])}/{_fmt_pm(stats['gl_f1'])}/{bdg_f1_str}"
        frob_cell = f"{_fmt_pm(stats['ef_frob'])}/{_fmt_pm(stats['gl_frob'])}/{bdg_frob_str}"
        auprc_cell = "[RESULT]"
        if auprc_by_generator and gen in auprc_by_generator:
            a_ours, a_bdg = auprc_by_generator[gen]
            auprc_cell = f"${a_ours:.3f}$/${a_bdg:.3f}$"
        lines.append(f"{name:<22} & {f1_cell} & {frob_cell} & {auprc_cell} \\\\")

    print("\n".join(lines))

    if output_path:
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote LaTeX rows to: {output_path}")

    return lines


# ---------------------------------------------------------------------
# Step 8: AUPRC loader (auprc_comparison.json)
# ---------------------------------------------------------------------
#
# Confirmed real schema (one file per dataset, i.e. one per (graph_type,
# N, T/N) combination, sitting at the same directory level as that
# dataset's seed_* folders -- so it's a SIBLING of that dataset's
# summary_results.json):
#
#   {
#     "sparse_factorized": {
#       "n_seeds": 20,
#       "per_seed_auprc_mean": ..., "per_seed_auprc_std": ..., "per_seed_auprc_sem": ...,
#       "per_seed": [{"seed": 0, "auprc": ...}, ...],
#       "pooled_auprc": ...,
#       "pooled_n_true_edges": ..., "pooled_n_candidate_pairs": ...,
#       "pooled_pr_curve": {"precision": [...], "recall": [...]}
#     },
#     "BDgraph": { ... same shape ... }
#   }
#
# NOTE (unverified assumption -- flagged, not silently baked in): the
# top-level key "sparse_factorized" looks like a FIXED label for "our
# method" (the edge-factor/factor-model prior), not something that varies
# by the dataset's true graph_type -- i.e. even a file living under an
# erdos_renyi/grid/scale_free/ar1 dataset folder should still use the key
# "sparse_factorized" for our method's numbers, since that's our model's
# internal name, not the true generator's name. This matches how the
# calibration code above uses "sparse_factorized" as a fixed dict key
# for the matched-model curve regardless of the underlying graph type.
# ours_key/bdgraph_key are exposed as parameters below specifically so
# that if this assumption turns out to be wrong (e.g. the key actually
# says "erdos_renyi" when that's the true generator), nothing needs to be
# rewritten -- just pass the right key.

def find_auprc_jsons(root_dir):
    """Recursively finds every auprc_comparison.json under root_dir."""
    return sorted(glob.glob(os.path.join(root_dir, "**", "auprc_comparison.json"), recursive=True))


def load_auprc_comparison(path, ours_key="sparse_factorized", bdgraph_key="BDgraph"):
    """Loads one auprc_comparison.json and returns:
        {"auprc_ours_pooled": float, "auprc_bdgraph_pooled": float,
         "auprc_ours_mean_sem": (mean, sem), "auprc_bdgraph_mean_sem": (mean, sem)}
    Uses pooled_auprc as the headline number (matches how F1/Frobenius in
    Table 1 are otherwise reported as per-seed mean+/-SEM, but AUPRC over
    a pooled PR curve is the more standard way to report it -- both are
    returned so you can pick).
    """
    with open(path) as f:
        data = json.load(f)
    ours = data[ours_key]
    bdg = data[bdgraph_key]
    return {
        "auprc_ours_pooled": ours["pooled_auprc"],
        "auprc_bdgraph_pooled": bdg["pooled_auprc"],
        "auprc_ours_mean_sem": (ours["per_seed_auprc_mean"], ours["per_seed_auprc_sem"]),
        "auprc_bdgraph_mean_sem": (bdg["per_seed_auprc_mean"], bdg["per_seed_auprc_sem"]),
    }


def auprc_by_generator_from_root(root_dirs, N=20, T_over_N=0.5, use_pooled=True,
                                  ours_key="sparse_factorized", bdgraph_key="BDgraph"):
    """
    Scans root_dirs for every auprc_comparison.json, and for each one,
    finds the graph_type by looking at the sibling summary_results.json
    in the SAME directory (auprc_comparison.json itself carries no
    graph_type/N/T metadata). Filters to the requested (N, T_over_N) and
    returns:

        {graph_type: (auprc_ours, auprc_bdgraph)}

    ready to pass straight into generate_table1_latex_rows's
    auprc_by_generator argument.

    use_pooled=True uses pooled_auprc (pooled PR curve over all seeds'
    pairs); False uses the mean of per-seed AUPRC instead.

    Prints which (graph_type, T/N) combinations were found/skipped, same
    convention as the other loaders here.
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    out = {}
    for root_dir in root_dirs:
        for auprc_path in find_auprc_jsons(root_dir):
            # The real layout has auprc_comparison.json inside its own
            # "auprc_comparison/" subfolder rather than directly beside
            # summary_results.json (confirmed from actual paths like
            # .../ar1_N20_tn05_20260917_131128/auprc_comparison/auprc_comparison.json,
            # with summary_results.json one level up, in
            # .../ar1_N20_tn05_20260917_131128/). So walk up a few ancestor
            # directories (not just the immediate parent) looking for it.
            sibling = None
            d = os.path.dirname(auprc_path)
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
                print(f"  [skip] {auprc_path}: no summary_results.json found in "
                      f"{os.path.dirname(auprc_path)} or its parent directories "
                      f"to get graph_type/N/T from")
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

            if N_here != N or tn_here != T_over_N:
                continue

            try:
                rec = load_auprc_comparison(auprc_path, ours_key=ours_key, bdgraph_key=bdgraph_key)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  [skip] {auprc_path}: {e} (check ours_key/bdgraph_key against this "
                      f"file's actual top-level keys)")
                continue

            if use_pooled:
                out[gen] = (rec["auprc_ours_pooled"], rec["auprc_bdgraph_pooled"])
            else:
                out[gen] = (rec["auprc_ours_mean_sem"][0], rec["auprc_bdgraph_mean_sem"][0])
            print(f"  found AUPRC for {gen:<18} N={N_here} T/N={tn_here}: "
                  f"ours={out[gen][0]:.4f}  BDgraph={out[gen][1]:.4f}")

    return out


# ---------------------------------------------------------------------
# Step 9: one-call Table 1 generator (F1/Frobenius + BDgraph + AUPRC)
# ---------------------------------------------------------------------

def generate_table1_full(root_dirs, bdgraph_root_dirs=None, N=20, T_over_N=0.5,
                          auprc_ours_key="sparse_factorized", auprc_bdgraph_key="BDgraph",
                          output_path=None):
    """
    Convenience wrapper doing the full pipeline in one call:
      1. load_bdgraph_raw + bdgraph_results_for_table1-style lookup (via
         generate_table1_latex_rows's own bdgraph_raw argument)
      2. auprc_by_generator_from_root for the AUPRC column
      3. generate_table1_latex_rows to print/return/save the LaTeX rows

    bdgraph_root_dirs: root(s) to search for bdgraph_*.json (pass None to
        leave the BDg component of F1/Frobenius as "[BDg]").
    output_path: if given, writes the generated LaTeX rows to this file
        (e.g. "table1_rows.tex") in addition to printing/returning them --
        so results end up saved on disk, not just in the console.
    Everything else matches generate_table1_latex_rows.

    Returns the same list of LaTeX row strings.
    """
    bdgraph_raw = load_bdgraph_raw(bdgraph_root_dirs) if bdgraph_root_dirs else None
    auprc_by_generator = auprc_by_generator_from_root(
        root_dirs, N=N, T_over_N=T_over_N,
        ours_key=auprc_ours_key, bdgraph_key=auprc_bdgraph_key)
    return generate_table1_latex_rows(
        root_dirs, bdgraph_raw=bdgraph_raw, N=N, T_over_N=T_over_N,
        auprc_by_generator=auprc_by_generator, output_path=output_path)


# ---------------------------------------------------------------------
# Step 10: Remark 1 -- representability check (diagonal dominance after
# a positive diagonal rescaling)
# ---------------------------------------------------------------------
#
# Remark 1 in main.tex says: a matrix K=eps*I + BB^T with factor width <=2
# (i.e. representable by our prior for SOME choice of R, allocations, and
# weights) if and only if K-eps*I is diagonally dominant after a positive
# diagonal rescaling (Boman et al. 2005). The remark's own worked example
# is a 3x3 all-pairs-correlated triangle (unit diagonal, K_ij=rho on every
# off-diagonal pair), which is positive definite for rho<1 but only
# representable for rho<=1/2 -- this is the exact number the LP below
# reproduces, which is how it's checked below.
#
# "Diagonally dominant after a positive diagonal rescaling" (a matrix M
# with positive diagonal is called an H-matrix in this generalized sense)
# is equivalent to: there exists a positive vector d such that, for every
# row i,
#     |M_ii| d_i - sum_{j != i} |M_ij| d_j >= 0.
# This is a linear feasibility problem in d, which is solved here as the
# linear program
#     maximize t  s.t.  |M_ii| d_i - sum_{j!=i} |M_ij| d_j >= t  for all i,
#                        sum_i d_i = N,  d_i >= 0
# (the equality constraint just fixes the scale, since the problem is
# homogeneous of degree 1 in d and t together). The optimal t* is called
# the "comparison margin" below (matching the main.tex placeholder's own
# wording, "fraction with negative comparison margin"): t* >= 0 means K is
# representable (up to the finite-R rank/support constraints Remark 1
# also states), t* < 0 means K lies in the excluded class (conflicting-
# sign strong correlations on a cycle, as in the triangle example).

def diagonal_dominance_margin(M):
    """
    Computes the comparison margin t* for a symmetric matrix M (see the
    module-level note above for the exact LP). Requires scipy
    (pip install scipy if you don't have it already).

    Returns a float: >= 0 (up to solver tolerance) means M can be made
    diagonally dominant by a positive diagonal rescaling; < 0 means it
    cannot (M is in the class Remark 1 excludes).

    A node with M_ii == 0 and an all-zero row (e.g. a graph-isolated node,
    per Remark 1's own $K_{ii}=\\varepsilon$ almost surely statement) is
    handled correctly without special-casing: its row constraint reduces
    to "0 >= t", which just caps the achievable margin at 0 (a boundary,
    not a violation) rather than forcing it negative.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as e:
        raise ImportError(
            "diagonal_dominance_margin needs scipy (pip install scipy "
            "--break-system-packages, or plain pip install scipy)") from e

    M = np.asarray(M, dtype=float)
    N = M.shape[0]
    if M.shape != (N, N):
        raise ValueError(f"M must be square, got shape {M.shape}")

    diag_abs = np.abs(np.diag(M))
    offdiag_abs = np.abs(M).copy()
    np.fill_diagonal(offdiag_abs, 0.0)

    # variables: d_1..d_N, t  (N+1 total)
    c = np.zeros(N + 1)
    c[-1] = -1.0  # linprog minimizes; minimize -t == maximize t

    # row i: -diag_abs[i]*d_i + sum_j offdiag_abs[i,j]*d_j + t <= 0
    A_ub = np.zeros((N, N + 1))
    for i in range(N):
        A_ub[i, :N] = offdiag_abs[i, :]
        A_ub[i, i] = -diag_abs[i]
        A_ub[i, N] = 1.0
    b_ub = np.zeros(N)

    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = [N]

    bounds = [(0, None)] * N + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed to solve for representability margin: {res.message}")
    return -res.fun


EPS_BY_GRAPH_TYPE = {
    "sparse_factorized": 1e-3,
    "erdos_renyi": 0.1,
    "grid": 0.1,
    "scale_free": 0.1,
    "ar1": 0.0,  # AR(1) sets its diagonal directly, no separate floor (see main.tex Sec. IV setup)
}


def compute_representability_summary(root_dirs, N=20, T_over_N=0.5,
                                      eps_by_graph_type=None, generators=None,
                                      tol=1e-6, csv_path=None):
    """
    For every K_true.npy found (via find_seed_dirs, same per-seed
    directories used for Fig. 2(a) calibration) at the given (N, T_over_N),
    computes the Remark 1 comparison margin of K_true - eps*I (eps taken
    per graph_type from eps_by_graph_type, defaulting to
    EPS_BY_GRAPH_TYPE, which matches the epsilon values stated in
    main.tex's SETUP paragraph). Negative margin means that dataset's true
    K is outside the class our prior can represent exactly (Remark 1's
    excluded class of cycles with strong, conflicting-sign partial
    correlations) -- this is a statement about representability, distinct
    from "generator mismatch" (the softer, statistical fact that a
    misspecified generator is simply harder to estimate well from finite
    T/N, which Fig. 1's recovery curves already speak to) -- a dataset can
    be fully representable (margin >= 0) and still be hard to recover at
    low T/N, and conversely a dataset that IS in the excluded class isn't
    necessarily poorly recovered in practice (the model still fits its
    best representable approximation).

    Returns a dict {graph_type: {"n": int, "n_negative": int,
    "fraction_negative": float, "margins": [float, ...]}}, and prints a
    summary table (n checked, fraction with negative margin, mean/min
    margin) per graph_type plus an overall total. Optionally writes the
    per-dataset margins to csv_path.
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    if eps_by_graph_type is None:
        eps_by_graph_type = EPS_BY_GRAPH_TYPE
    if generators is None:
        generators = list(eps_by_graph_type.keys())

    summary = {}
    csv_rows = []

    for gen in generators:
        eps = eps_by_graph_type.get(gen, 0.0)
        seed_dirs = []
        for root_dir in root_dirs:
            seed_dirs.extend(find_seed_dirs(root_dir, gen, N=N, T_over_N=T_over_N))

        margins = []
        for d in seed_dirs:
            k_path = os.path.join(d, "K_true.npy")
            if not os.path.exists(k_path):
                print(f"  [skip] {d}: no K_true.npy")
                continue
            K_true = np.load(k_path)
            M = K_true - eps * np.eye(K_true.shape[0])
            try:
                margin = diagonal_dominance_margin(M)
            except RuntimeError as e:
                print(f"  [skip] {d}: {e}")
                continue
            margins.append(margin)
            csv_rows.append({"graph_type": gen, "seed_dir": d, "eps": eps, "margin": margin,
                              "representable": margin >= -tol})

        if not margins:
            print(f"  [skip] {gen}: no K_true.npy found at N={N}, T/N={T_over_N}")
            continue

        margins_arr = np.asarray(margins)
        n_negative = int((margins_arr < -tol).sum())
        summary[gen] = {
            "n": len(margins),
            "n_negative": n_negative,
            "fraction_negative": n_negative / len(margins),
            "margins": margins,
        }

    if not summary:
        print("No representability data found -- check root_dirs/N/T_over_N.")
        return summary

    header = f"{'graph_type':<18} {'n':>4} {'n_negative':>11} {'frac_negative':>14} {'mean_margin':>12} {'min_margin':>11}"
    print(header)
    print("-" * len(header))
    total_n, total_neg = 0, 0
    for gen, s in summary.items():
        m = np.asarray(s["margins"])
        print(f"{gen:<18} {s['n']:>4} {s['n_negative']:>11} {s['fraction_negative']:>14.3f} "
              f"{m.mean():>12.4f} {m.min():>11.4f}")
        total_n += s["n"]
        total_neg += s["n_negative"]
    print("-" * len(header))
    print(f"{'TOTAL':<18} {total_n:>4} {total_neg:>11} {total_neg/total_n if total_n else float('nan'):>14.3f}")

    if csv_path and csv_rows:
        import csv
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote per-dataset margins to: {csv_path}")

    return summary


if __name__ == "__main__":
    print(__doc__)
