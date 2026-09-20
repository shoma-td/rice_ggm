import aggregate_results as agg
import make_paper_figures as fig

ALL_EXPERIMENTS_DIR = "C:/Users/mech-user/rice_gnn/rice_ggm/experiment2_bd_mala_swap"  # 実際のパスに合わせて調整
BDGRAPH_DIR = "C:/Users/mech-user/rice_gnn/rice_ggm/bdgraph_eps01"


# --- BDgraphを読み込む ---
bdgraph_raw = agg.load_bdgraph_raw(BDGRAPH_DIR)

# --- Fig. 1 ---
fig1_data = agg.build_fig1_results(
    [ALL_EXPERIMENTS_DIR],
    bdgraph_results=agg.bdgraph_results_for_fig1(bdgraph_raw),
)
fig.plot_fig1_per_generator(fig1_data, save_dir="fig1_output")
fig.plot_fig1(fig1_data, save_path="fig1_output.png")

recovery_records = agg.extract_recovery_summary([ALL_EXPERIMENTS_DIR], csv_path="recovery_summary.csv")

diag_summary = agg.build_diagnostics_summary([ALL_EXPERIMENTS_DIR])

agg.check_rhat_swap_correlation([ALL_EXPERIMENTS_DIR], graph_type="sparse_factorized")

rows = agg.per_seed_diagnostics_table([ALL_EXPERIMENTS_DIR], graph_type="sparse_factorized", csv_path="sparse_factorized_diagnostics_per_seed.csv")

rows = agg.generate_table1_full(
    [ALL_EXPERIMENTS_DIR],
    bdgraph_root_dirs=[BDGRAPH_DIR],
    output_path="table1_rows.tex",
)

agg.build_diagnostics_summary([ALL_EXPERIMENTS_DIR], output_path="diagnostics_summary.txt")
agg.check_rhat_swap_correlation([ALL_EXPERIMENTS_DIR], output_path="rhat_swap_corr.txt")
"""
# --- BDgraphを読み込む ---
bdgraph_raw = agg.load_bdgraph_raw(BDGRAPH_DIR)

# --- Table 1 ---
table1_data = agg.build_table1_results(
    [ALL_EXPERIMENTS_DIR],
    bdgraph_results=agg.bdgraph_results_for_table1(bdgraph_raw, T_over_N=0.5),
    N=50, T_over_N=0.5,
)
table1_markdown = fig.make_table1(table1_data)

# ファイルにも保存しておく
with open("table1_output.md", "w") as f:
    f.write(table1_markdown)


FIG2BC_DIR = "C:/Users/mech-user/rice_gnn/rice_ggm/results_rtest_grid913n50tn05"  # fig2b_2c_N50_TN0.5.json がある場所

per_seed = agg.load_fig2b_per_seed_results(f"{FIG2BC_DIR}/fig2b_2c_N50_TN0.5.json")
fig.plot_fig2b_recovery_vs_R(per_seed, save_path="fig2b_output.png")

ec_hist, true_ec, r_sel = agg.load_fig2c_data(
    f"{FIG2BC_DIR}/fig2b_2c_N50_TN0.5.json",
    f"{FIG2BC_DIR}/fig2c_seed0_edge_count_history.npy",
)
fig.plot_fig2c_edge_count_posterior(ec_hist, true_ec, r_sel, save_path="fig2c_output.png")




BDGRAPH_EDGE_PROBS_DIR = "C:/Users/mech-user/rice_gnn/rice_ggm/graph/bd_cv"  # edge_probs.npyがある場所

# --- edge-factor prior側 (N=20, T/N=0.5) ---
seed_dirs_matched = agg.find_seed_dirs(ALL_EXPERIMENTS_DIR, "sparse_factorized")
print(f"matched: {len(seed_dirs_matched)} seed dirs found")

seed_dirs_alternative = []
for gt in ["erdos_renyi", "grid", "scale_free", "ar1"]:
    dirs = agg.find_seed_dirs(ALL_EXPERIMENTS_DIR, gt)
    print(f"  {gt}: {len(dirs)} seed dirs found")
    seed_dirs_alternative += dirs

# --- BDgraph側 (N=20, T/N=0.5) ---
bdgraph_pairs_matched = agg.load_bdgraph_calibration_pairs(
    BDGRAPH_EDGE_PROBS_DIR, "sparse_factorized", N=20, T_over_N=0.5)
print(f"bdgraph matched: {len(bdgraph_pairs_matched)} pairs found")

bdgraph_pairs_alternative = []
for gt in ["erdos_renyi", "grid", "scale_free", "ar1"]:
    pairs = agg.load_bdgraph_calibration_pairs(BDGRAPH_EDGE_PROBS_DIR, gt, N=20, T_over_N=0.5)
    print(f"  bdgraph {gt}: {len(pairs)} pairs found")
    bdgraph_pairs_alternative += pairs

# --- calibration計算 + プロット ---
cal_data = agg.build_fig2a_calibration_data(
    seed_dirs_matched, seed_dirs_alternative,
    bdgraph_edge_probs_matched=bdgraph_pairs_matched,
    bdgraph_edge_probs_alternative=bdgraph_pairs_alternative,
)
fig.plot_fig2a_calibration(cal_data, save_path="output_fig2a.png")
"""