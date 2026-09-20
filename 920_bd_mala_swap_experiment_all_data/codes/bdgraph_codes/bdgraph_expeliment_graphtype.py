import os
import json
import numpy as np
from data_gen import generate_well_specified_K, generate_misspecified_K, sample_X_given_K
from bdgraph_bridge import run_bdgraph_baseline

# --- ここだけ指定すればよい ---
GRAPH_TYPE = "ar1"   # 実行したいグラフタイプ
N =20
T_OVER_N_GRID = [0.5, 2, 5, 10, 20]
NUM_SEEDS = 20
# -----------------------------

SAVE_DIR = "bdgraph919/bdgraph_results_ar1_cv_new920"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_or_init_results(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_result(all_results, key, seed, result, path):
    if key not in all_results:
        all_results[key] = {}
    all_results[key][str(seed)] = result
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results

# --- 変更後 ---
EDGE_PROBS_SAVE_DIR = os.path.join(SAVE_DIR, "raw_edge_probs")  # 追加

def run_one(graph_type, N, T, seed, iter_=5000, burnin=1000):
    np.random.seed(seed)
    if graph_type == "sparse_factorized":
        K_true, _ = generate_well_specified_K(N=N, target_d=4.0)
    else:
        K_true = generate_misspecified_K(N=N, graph_type=graph_type, eps = 0.1)
    X = sample_X_given_K(K_true, T=T)

    result = run_bdgraph_baseline(X, K_true, iter=iter_, burnin=burnin, return_edge_probs=True)

    # aggregate_results.py の build_fig2a_calibration_data と同じ規約
    # (edge_probs.npy, K_true.npy をseedごとのフォルダに保存)で永続化
    edge_probs = result.pop("_edge_probs")
    K_est = result.pop("_K_est")
    seed_dir = os.path.join(EDGE_PROBS_SAVE_DIR, f"{graph_type}_N{N}_T{T}", f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    np.save(os.path.join(seed_dir, "edge_probs.npy"), edge_probs)
    np.save(os.path.join(seed_dir, "bdgraph_K_est.npy"), K_est)
    np.save(os.path.join(seed_dir, "K_true.npy"), K_true)

    result["graph_type"] = graph_type
    result["N"] = N
    result["T"] = T
    result["seed"] = seed
    return result

# --- メイン実行 ---
if __name__ == "__main__":
    for ratio in T_OVER_N_GRID:
        T = max(1, round(ratio * N))

        # T/Nごとに別々のファイルに保存
        consolidated_path = os.path.join(
            SAVE_DIR, f"bdgraph_{GRAPH_TYPE}_N{N}_ratio{ratio}.json"
        )
        all_results = load_or_init_results(consolidated_path)

        key = f"{GRAPH_TYPE}_N{N}_T{T}"
        print(f"\n{'='*60}")
        print(f"### graph_type={GRAPH_TYPE}  T/N={ratio}  (T={T}) ###")
        print(f"{'='*60}")

        for seed in range(NUM_SEEDS):
            if key in all_results and str(seed) in all_results[key]:
                print(f"スキップ(実行済み): seed={seed}")
                continue

            print(f"実行中: seed={seed}")
            result = run_one(GRAPH_TYPE, N, T, seed)
            all_results = save_result(all_results, key, seed, result, consolidated_path)
            print(f"  F1={result['bdgraph_f1']:.4f}, RelFrob={result['bdgraph_frobenius_error']:.4f}, "
                  f"samples/sec={result['bdgraph_samples_per_sec']:.1f}")

        print(f"  -> saved to {consolidated_path}")

    print("\n=== 全T/N完了 ===")