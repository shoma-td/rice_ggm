"""
F1(/Frobenius) vs T/N の図を「graph_typeごとに別々のファイル」として出す
実行スクリプト（run_figures.py の grid_f1_frobenius.png は1枚のグリッド
画像なので、それとは別にこちらを使う）。

役割ごとにファイルを分けている:
  - aggregate_results.py    : データ集計（既存のbuild_fig1_resultsをそのまま使う）
  - plot_f1_separate.py     : 描画（graph_typeごとに別々のPNGを書き出す）
  - run_f1_separate.py (このファイル) : 上の2つをつないで実行するだけ

前提:
  - aggregate_results.py, make_paper_figures.py, plot_f1_separate.py が、
    このファイルと同じフォルダに置いてあること。
  - run_figures.py と同じ ALL_EXPERIMENTS_DIR / BDGRAPH_DIRS の構成。

実行方法:
  1. 下の ALL_EXPERIMENTS_DIR / BDGRAPH_DIRS を自分の環境のパスに直す
     (run_figures.py のものと同じ値でOK)。
  2. python run_f1_separate.py
  3. 実行が終わると、このファイルと同じフォルダに
         f1_vs_tn_sparse_factorized.png
         f1_vs_tn_erdos_renyi.png
         f1_vs_tn_grid.png
         f1_vs_tn_scale_free.png
         f1_vs_tn_ar1.png
     の5枚が出力される。
"""

import aggregate_results as agg
import plot_f1_separate as f1_plot

# ---- ここを自分の環境に合わせて直す（run_figures.py と同じ値でOK） ----
ALL_EXPERIMENTS_DIR = r"C:\Users\mech-user\rice_gnn\rice_ggm\920_bd_mala_swap_experiment_all_data\experimental_datas_first"
BDGRAPH_DIRS = [
    r"C:\Users\mech-user\rice_gnn\rice_ggm\920_bd_mala_swap_experiment_all_data\experimental_datas_first",
]
N = 20
# ----------------------------------------------------------------------


def main():
    print("=== F1 vs T/N (all graph types, separate files) ===")
    bdgraph_raw = agg.load_bdgraph_raw(BDGRAPH_DIRS)
    bdgraph_for_fig1 = agg.bdgraph_results_for_fig1(bdgraph_raw, N=N)
    results = agg.build_fig1_results([ALL_EXPERIMENTS_DIR], bdgraph_results=bdgraph_for_fig1)
    f1_plot.plot_f1_separate(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
