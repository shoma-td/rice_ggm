#!/usr/bin/env python3
"""check_rhat_across_datasets.py

実験結果ディレクトリ（複数可）以下を再帰的に走査し、見つかった
summary_results.json すべて（= 1ファイル = 1つの (graph_type, N, T) 実験
データセット）について、R-hat が閾値(デフォルト1.05)を超えたseedが
いくつあるかを集計する。

aggregate_results.py の find_summary_jsons / load_summary_json_diagnostics
と同じやり方でファイルを探すが、あちらは全データセットを1つにプール
した集計（[DIAGNOSTICS]パラグラフ用）だったのに対し、これは
「データセットごとに」内訳を出す点が異なる。

R-hatのフィールド名は実験の時期によって変わっている可能性がある
(例: "rhat_log_posterior" だけの古いスキーマ、"rhat_log_posterior_plain"
/ "rhat_log_posterior_rank" のように plain/rank 両方を持つ新しいスキーマ)。
決め打ちせず、各ファイルの診断値辞書から "rhat" を含むキーを自動検出する
ことで、どちらのスキーマにも対応する。

Usage:
    python check_rhat_across_datasets.py root_dir [root_dir2 ...]
    python check_rhat_across_datasets.py root_dir --threshold 1.05
    python check_rhat_across_datasets.py root_dir -o rhat_by_dataset.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any


def find_summary_jsons(root_dir: str) -> list[str]:
    """root_dir以下を再帰的に探索し、summary_results.jsonのパス一覧を返す。

    aggregate_results.py の find_summary_jsons と同じ考え方:
    run_multi_seed_experiment が (graph_type, N, timestamp) ごとの
    ディレクトリに1つずつ保存する形式を想定している。
    """
    return sorted(glob.glob(os.path.join(root_dir, "**", "summary_results.json"), recursive=True))


def detect_rhat_fields(per_seed_results: list[dict[str, Any]]) -> list[str]:
    """per_seed_results の diagnostics 辞書から、名前に'rhat'を含む
    フィールドをすべて拾う(大文字小文字は区別しない)。

    1つのファイル内でも seed によってキーが微妙に欠けているケースに
    備え、全seedのdiagnosticsキーの和集合を取ってから'rhat'を含むもの
    を残す(先頭seedだけ見て決め打ちしない)。
    """
    fields: set[str] = set()
    for r in per_seed_results:
        diagnostics = r.get("diagnostics", {})
        for key in diagnostics.keys():
            if "rhat" in key.lower():
                fields.add(key)
    return sorted(fields)


def load_dataset_rhat_summary(path: str, threshold: float) -> dict[str, Any]:
    """1つのsummary_results.jsonを読み、configと
    「R-hatフィールドごとの超過seed数」をまとめて返す。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data.get("config", {})
    per_seed = data.get("per_seed_results", [])
    N = config.get("N")
    T = config.get("T")
    T_over_N = round(T / N, 4) if (N and T) else None

    rhat_fields = detect_rhat_fields(per_seed)

    per_field: dict[str, Any] = {}
    # このデータセット内で「いずれかのR-hatフィールドが閾値超え」のseed
    any_field_flagged_seeds: set[int] = set()

    for field in rhat_fields:
        seed_values = []
        flagged_seeds = []
        for r in per_seed:
            diagnostics = r.get("diagnostics", {})
            if field not in diagnostics:
                continue
            seed = r.get("seed")
            value = diagnostics[field]
            seed_values.append(value)
            if value >= threshold:
                flagged_seeds.append(seed)
                any_field_flagged_seeds.add(seed)

        if not seed_values:
            continue

        per_field[field] = {
            "n_seeds": len(seed_values),
            "n_flagged": len(flagged_seeds),
            "flagged_seeds": sorted(flagged_seeds),
            "max": max(seed_values),
            "mean": sum(seed_values) / len(seed_values),
        }

    return {
        "path": path,
        "graph_type": config.get("graph_type"),
        "N": N,
        "T": T,
        "T_over_N": T_over_N,
        "n_seeds_total": len(per_seed),
        "rhat_fields_found": rhat_fields,
        "per_field": per_field,
        "any_field_flagged": {
            "n_flagged": len(any_field_flagged_seeds),
            "flagged_seeds": sorted(any_field_flagged_seeds),
        },
    }


def field_category(field: str) -> str:
    """フィールド名から "_plain" / "_rank" サフィックスを取り除いた
    カテゴリ名を返す。

    例: "rhat_log_posterior_plain" -> "rhat_log_posterior"
        "max_rhat_K_rank"          -> "max_rhat_K"
        "rhat_log_posterior"（旧スキーマ、サフィックスなし） -> そのまま

    これで「log posteriorのR-hat」と「Kの要素のR-hat」を、plain/rank
    のバリエーションをまとめた1つの表として扱える。
    """
    for suffix in ("_plain", "_rank"):
        if field.endswith(suffix):
            return field[: -len(suffix)]
    return field


def build_category_tables(datasets: list[dict[str, Any]], threshold: float) -> dict[str, list[dict[str, Any]]]:
    """データセットのリストから、カテゴリ(log posterior系 / K系など)ごとの
    表(行のリスト)を作る。

    各行は1つのデータセット(graph_type, N, T/N)に対応し、そのカテゴリに
    属する各フィールド(plain, rankなど)について
    「n_flagged/n_seeds」「max」「mean」の列を持つ。
    """
    # まず全データセットを見て、カテゴリ -> そのカテゴリに属するフィールド名一覧、を確定する
    category_fields: dict[str, set[str]] = {}
    for d in datasets:
        for field in d["rhat_fields_found"]:
            cat = field_category(field)
            category_fields.setdefault(cat, set()).add(field)

    tables: dict[str, list[dict[str, Any]]] = {}
    for cat, fields in category_fields.items():
        fields_sorted = sorted(fields)  # 例: [..._plain, ..._rank] の順に揃う
        rows = []
        for d in datasets:
            row: dict[str, Any] = {
                "graph_type": d["graph_type"],
                "N": d["N"],
                "T_over_N": d["T_over_N"],
                "n_seeds": d["n_seeds_total"],
            }
            for field in fields_sorted:
                s = d["per_field"].get(field)
                if s is None:
                    row[f"{field}__n_flagged"] = ""
                    row[f"{field}__max"] = ""
                    row[f"{field}__mean"] = ""
                else:
                    row[f"{field}__n_flagged"] = f"{s['n_flagged']}/{s['n_seeds']}"
                    row[f"{field}__max"] = round(s["max"], 4)
                    row[f"{field}__mean"] = round(s["mean"], 4)
            rows.append(row)
        tables[cat] = rows

    return tables


def write_csv_tables(category_tables: dict[str, list[dict[str, Any]]], output_dir: Path) -> list[Path]:
    """build_category_tablesの出力を、カテゴリごとに1つのCSVファイルとして
    書き出す。ファイル名は rhat_table_<カテゴリ名>.csv。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cat, rows in category_tables.items():
        if not rows:
            continue
        safe_name = cat.replace("/", "_")
        path = output_dir / f"rhat_table_{safe_name}.csv"
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "root_dirs", nargs="+", type=str,
        help="summary_results.json を探索するルートディレクトリ(複数指定可)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.05,
        help="R-hatの警告閾値 (default: 1.05)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="出力JSONパス (省略時: 1つ目のroot_dirの中に rhat_by_dataset.json として保存)",
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=None,
        help="表形式(CSV)の出力先ディレクトリ (省略時: 1つ目のroot_dirの中に "
             "rhat_tables/ を自動作成して保存)",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="CSV表の出力をスキップする(JSONだけ欲しい場合)",
    )
    args = parser.parse_args()

    output_path = args.output if args.output is not None else Path(args.root_dirs[0]) / "rhat_by_dataset.json"
    csv_dir = args.csv_dir if args.csv_dir is not None else Path(args.root_dirs[0]) / "rhat_tables"

    datasets = []
    for root_dir in args.root_dirs:
        for path in find_summary_jsons(root_dir):
            datasets.append(load_dataset_rhat_summary(path, args.threshold))

    # graph_type -> N -> T/N の順で見やすくソート
    datasets.sort(key=lambda d: (str(d["graph_type"]), d["N"] or 0, d["T_over_N"] or 0))

    print(f"{len(datasets)} 個のデータセット(summary_results.json)が見つかりました。\n")

    header = f"{'graph_type':<14} {'N':>4} {'T/N':>6} {'n_seeds':>8}"
    print(header)
    print("-" * len(header))

    grand_total_flagged = 0
    grand_total_seeds = 0

    for d in datasets:
        print(f"{str(d['graph_type']):<14} {d['N']!s:>4} {d['T_over_N']!s:>6} {d['n_seeds_total']:>8}")
        if not d["rhat_fields_found"]:
            print("    (R-hatフィールドが見つかりませんでした)")
            continue
        for field, s in d["per_field"].items():
            print(f"    {field:<28} {s['n_flagged']:>3}/{s['n_seeds']:<3} seeds >= {args.threshold}  "
                  f"(max={s['max']:.4f}, mean={s['mean']:.4f})  flagged={s['flagged_seeds']}")
        any_f = d["any_field_flagged"]
        print(f"    {'ANY field flagged':<28} {any_f['n_flagged']:>3}/{d['n_seeds_total']:<3} seeds  "
              f"flagged={any_f['flagged_seeds']}")
        grand_total_flagged += any_f["n_flagged"]
        grand_total_seeds += d["n_seeds_total"]
        print()

    print(f"=== 全データセット合計 ===")
    print(f"  {grand_total_flagged}/{grand_total_seeds} seeds が"
          f"いずれかのR-hatフィールドで閾値({args.threshold})を超過")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": args.threshold,
                "n_datasets": len(datasets),
                "grand_total_flagged_seeds": grand_total_flagged,
                "grand_total_seeds": grand_total_seeds,
                "datasets": datasets,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nWrote results to: {output_path}")

    if not args.no_csv:
        category_tables = build_category_tables(datasets, args.threshold)
        written = write_csv_tables(category_tables, csv_dir)
        print(f"\nWrote {len(written)} table(s) to {csv_dir}:")
        for p in written:
            print(f"  {p}")


if __name__ == "__main__":
    main()
