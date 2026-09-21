"""
AUPRC vs T/N の折れ線グラフと同じデータを、全graph_typeまとめた1つの
貼り付け用Markdown表として出力する実行スクリプト。

役割ごとにファイルを分けている:
  - aggregate_auprc.py : データ集計（既存のbuild_auprc_vs_tnをそのまま使う）
  - table_utils.py     : 表の整形（Markdown表を組み立てるだけ）
  - run_auprc_table.py (このファイル) : 上の2つをつないで実行するだけ

前提:
  - aggregate_results.py, make_paper_figures.py, aggregate_auprc.py,
    table_utils.py が、このファイルと同じフォルダに置いてあること。
  - run_auprc.py と同じ ALL_EXPERIMENTS_DIR の構成。

実行方法:
  1. 下の ALL_EXPERIMENTS_DIR を自分の環境のパスに直す
     (run_auprc.py のものと同じ値でOK)。
  2. python run_auprc_table.py
  3. コンソールにMarkdown表が表示され、同じフォルダに auprc_table.md
     としても保存される（行: graph_type x T/N、列: Ours/BDgraph
     [glassoはAUPRC非対応のため列なし]、セルは "mean ± sem" 形式）。
"""

import aggregate_auprc as auprc_agg
import table_utils as tbl

# ---- ここを自分の環境に合わせて直す（run_auprc.py と同じ値でOK） ----
ALL_EXPERIMENTS_DIR = r"C:\Users\vayar\OneDrive\Desktop\rice_ggm\rice_ggm\920_bd_mala_swap_experiment_all_data\experimental_datas_first"
N = 20
# ----------------------------------------------------------------------

METHODS = ["edge_factor", "bdgraph"]
METHOD_LABELS = {"edge_factor": "Ours", "bdgraph": "BDgraph"}


def main():
    print("=== AUPRC vs T/N table (all graph types) ===")
    results = auprc_agg.build_auprc_vs_tn([ALL_EXPERIMENTS_DIR], N=N)
    if not results:
        print("\nauprc_comparison.json が見つかりませんでした。"
              "ALL_EXPERIMENTS_DIR を確認してください。")
        return
    tbl.write_markdown_table(results, METHODS, METHOD_LABELS, save_path="auprc_table.md")


if __name__ == "__main__":
    main()
