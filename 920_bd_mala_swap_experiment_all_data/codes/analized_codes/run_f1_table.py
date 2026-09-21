"""
F1 vs T/N の折れ線グラフと同じデータを、全graph_typeまとめた1つの
貼り付け用Markdown表として出力する実行スクリプト。

役割ごとにファイルを分けている:
  - aggregate_results.py : データ集計（既存のbuild_fig1_resultsをそのまま使う）
  - table_utils.py       : 表の整形（Markdown表を組み立てるだけ）
  - run_f1_table.py (このファイル) : 上の2つをつないで実行するだけ

前提:
  - aggregate_results.py, make_paper_figures.py, table_utils.py が、
    このファイルと同じフォルダに置いてあること。
  - run_figures.py と同じ ALL_EXPERIMENTS_DIR / BDGRAPH_DIRS の構成。

実行方法:
  1. 下の ALL_EXPERIMENTS_DIR / BDGRAPH_DIRS を自分の環境のパスに直す
     (run_figures.py のものと同じ値でOK)。
  2. python run_f1_table.py
  3. コンソールにMarkdown表が表示され、同じフォルダに f1_table.md として
     も保存される（行: graph_type x T/N、列: Ours/GLasso/BDgraph、
     セルは "mean ± sem" 形式）。
"""

import aggregate_results as agg
import table_utils as tbl

# ---- ここを自分の環境に合わせて直す（run_figures.py と同じ値でOK） ----
ALL_EXPERIMENTS_DIR = r"C:\Users\vayar\OneDrive\Desktop\rice_ggm\rice_ggm\920_bd_mala_swap_experiment_all_data\experimental_datas_first"
BDGRAPH_DIRS = [
    r"C:\Users\vayar\OneDrive\Desktop\rice_ggm\rice_ggm\920_bd_mala_swap_experiment_all_data\experimental_datas_first",
]
N = 20
# ----------------------------------------------------------------------

METHODS = ["edge_factor", "glasso", "bdgraph"]
METHOD_LABELS = {"edge_factor": "Ours", "glasso": "GLasso", "bdgraph": "BDgraph"}


def main():
    print("=== F1 vs T/N table (all graph types) ===")
    bdgraph_raw = agg.load_bdgraph_raw(BDGRAPH_DIRS)
    bdgraph_for_fig1 = agg.bdgraph_results_for_fig1(bdgraph_raw, N=N)
    results = agg.build_fig1_results([ALL_EXPERIMENTS_DIR], bdgraph_results=bdgraph_for_fig1)
    tbl.write_markdown_table(results, METHODS, METHOD_LABELS, save_path="f1_table.md")


if __name__ == "__main__":
    main()
