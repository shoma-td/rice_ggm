"""
Fig. 1 / Fig. 2(a) / Remark 1（表現可能性チェック）を実データから生成する
実行スクリプト。

前提:
  - aggregate_results.py と make_paper_figures.py が、このファイルと
    同じフォルダに置いてあること（プロジェクトの paper/aggregate_results.py
    / paper/make_paper_figures.py をダウンロードして同じフォルダへ）。
  - matplotlib, numpy, scipy がインストールされていること
    (無ければ: pip install matplotlib numpy scipy)

実行方法:
  1. 下の ALL_EXPERIMENTS_DIR / BDGRAPH_DIRS を自分の環境のパスに直す。
     BDGRAPH_DIRS はリストで、Table1/Fig.1用の bdgraph_*.json サマリーが
     ある場所と、Fig.2(a)用の生データ(edge_probs.npy)がある bdgraph_eps01
     フォルダの両方を含める（両方同じ場所ならリストは1要素でOK）。
  2. コマンドプロンプト/PowerShellでこのファイルのあるフォルダに移動し、
         python run_figures.py
     を実行する。
  3. 実行が終わると、このファイルと同じフォルダに
         grid_f1_frobenius.png       (Fig. 1: 5 graph_type全部)
         calibration_curve.png       (Fig. 2(a): 5 graph_type全部)
         representability_margins.csv (Remark 1: データセットごとの生マージン)
     が出力される。途中の [skip] 行は既知の旧フォーマットファイル
     （sparse_factorized_N20_tn2_...）のスキップや、edge_probs.npy/
     K_true.npyがまだ保存されていないgraph_typeのスキップなので、
     見つかった/見つからなかったgraph_typeを確認してから気にすれば良い。
  4. できたファイルをそのままこちらに送ってもらえれば、main.texの
     includegraphics/[RESULT]プレースホルダに反映してコンパイル確認する。
"""

import aggregate_results as agg
import make_paper_figures as figs

# ---- ここを自分の環境に合わせて直す --------------------------------
ALL_EXPERIMENTS_DIR = r"C:\Users\mech-user\rice_gnn\rice_ggm\experiment3"
BDGRAPH_DIRS = [
    # bdgraph_ar1_cv フォルダという名前だが、実際は5種類のgraph_type全部の
    # bdgraph_*.json (Table1/Fig.1用) と raw_edge_probs/ (Fig.2(a)用) が
    # 両方ここに入っている -- 1つのフォルダでOK
    r"C:\Users\mech-user\rice_gnn\rice_ggm\bdgraph_920\bdgraph_results_ar1_cv",
]
N = 20
T_OVER_N = 0.5          # Fig. 2(a) / Remark 1 はこのT/Nでのデータを使う
# ----------------------------------------------------------------------


def main():
    # ---------------- Fig. 1 (5 graph_type全部) ----------------
    print("=== Fig. 1: recovery vs T/N (all graph types) ===")
    bdgraph_raw = agg.load_bdgraph_raw(BDGRAPH_DIRS)
    bdgraph_for_fig1 = agg.bdgraph_results_for_fig1(bdgraph_raw, N=N)
    results = agg.build_fig1_results([ALL_EXPERIMENTS_DIR], bdgraph_results=bdgraph_for_fig1)
    figs.plot_fig1(results, save_path="grid_f1_frobenius.png")

    # ---------------- Fig. 2(a) (5 graph_type全部) ----------------
    print("\n=== Fig. 2(a): edge-probability calibration (all graph types) ===")
    calibration_data = agg.build_fig2a_calibration_data_all_generators(
        [ALL_EXPERIMENTS_DIR], bdgraph_root_dirs=BDGRAPH_DIRS, N=N, T_over_N=T_OVER_N)
    figs.plot_fig2a_calibration_all(calibration_data, save_path="calibration_curve.png")

    # ---------------- Remark 1: representability check ----------------
    print("\n=== Remark 1: representability (diagonal-dominance) check ===")
    agg.compute_representability_summary(
        [ALL_EXPERIMENTS_DIR], N=N, T_over_N=T_OVER_N,
        csv_path="representability_margins.csv")

    # ---------------- Table 1 (F1/Frobenius + BDgraph + AUPRC) ----------------
    print("\n=== Table 1: LaTeX rows (edge-factor / glasso / BDgraph, N=20, T/N=0.5) ===")
    agg.generate_table1_full(
        [ALL_EXPERIMENTS_DIR], bdgraph_root_dirs=BDGRAPH_DIRS, N=N, T_over_N=T_OVER_N,
        output_path="table1_rows.txt")

    print("\nDone. grid_f1_frobenius.png / calibration_curve.png / "
          "representability_margins.csv / table1_rows.txt ができているはずです。")


if __name__ == "__main__":
    main()
