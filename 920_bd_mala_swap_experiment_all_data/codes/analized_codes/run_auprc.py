"""
AUPRC vs T/N の図を作る実行スクリプト（run_figures.py とは別ファイル）。

役割ごとにファイルを分けている:
  - aggregate_auprc.py : データ集計（auprc_comparison.json を探して集計）
  - plot_auprc.py      : 描画（graph_typeごとに別々のPNGを書き出す）
  - run_auprc.py (このファイル) : 上の2つをつないで実行するだけ

前提:
  - aggregate_results.py, make_paper_figures.py, aggregate_auprc.py,
    plot_auprc.py が、このファイルと同じフォルダに置いてあること。
  - run_figures.py の ALL_EXPERIMENTS_DIR と同じフォルダ構成
    (experiment3 以下に5種類のgraph_typeの結果がまとまっている想定)。

実行方法:
  1. 下の ALL_EXPERIMENTS_DIR を自分の環境のパスに直す
     (run_figures.py のものと同じ値でOK)。
  2. python run_auprc.py
  3. 実行が終わると、このファイルと同じフォルダに
         auprc_vs_tn_sparse_factorized.png
         auprc_vs_tn_erdos_renyi.png
         auprc_vs_tn_grid.png
         auprc_vs_tn_scale_free.png
         auprc_vs_tn_ar1.png
     の5枚が出力される（Fig.1のように1枚のグリッド画像にまとめず、
     graph_typeごとに独立したファイルになる）。auprc_comparison.json が
     まだ無いgraph_typeは "[RESULT: pending]" のプレースホルダー付きの
     図になるだけで、他のgraph_typeやエラーには影響しない。
"""

import aggregate_auprc as auprc_agg
import plot_auprc as auprc_plot

# ---- ここを自分の環境に合わせて直す（run_figures.py と同じ値でOK） ----
ALL_EXPERIMENTS_DIR = r"C:\Users\mech-user\rice_gnn\rice_ggm\920_bd_mala_swap_experiment_all_data\experimental_datas_first"
N = 20
# ----------------------------------------------------------------------


def main():
    print("=== AUPRC vs T/N (all graph types, separate files) ===")
    results = auprc_agg.build_auprc_vs_tn([ALL_EXPERIMENTS_DIR], N=N)
    if not results:
        print("\nauprc_comparison.json が見つかりませんでした。"
              "ALL_EXPERIMENTS_DIR を確認してください。")
        return
    auprc_plot.plot_auprc_separate(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
