"""
折れ線グラフ（F1 vs T/N, AUPRC vs T/N）と同じデータを、貼り付け用の
Markdown表としてまとめるための共通ヘルパー（役割: 表の整形のみ）。

build_fig1_results / aggregate_auprc.build_auprc_vs_tn がどちらも返す
{graph_type: {method: {T_over_N: (mean, sem)}}} という形の辞書を受け取り、
全graph_typeをまとめた1つのMarkdown表（文字列）を作る。

使う側（run_f1_table.py / run_auprc_table.py）がデータ集計を担当し、
このファイルは整形だけを担当する。
"""

from make_paper_figures import MISSPECIFIED_ORDER, MISSPECIFIED_DISPLAY

GEN_ORDER = ["sparse_factorized"] + MISSPECIFIED_ORDER
GEN_DISPLAY = {"sparse_factorized": "Factor model (matched)", **MISSPECIFIED_DISPLAY}


def _fmt_cell(gen_results, method, tn):
    """gen_results[method] に tn のデータがあれば 'mean ± sem' 形式の文字列、
    無ければ '—' を返す。"""
    vals = gen_results.get(method, {})
    if tn not in vals:
        return "—"
    mean, sem = vals[tn]
    return f"{mean:.3f} ± {sem:.3f}"


def build_markdown_table(results, methods, method_labels):
    """
    results: {graph_type: {method: {T_over_N: (mean, sem)}}}
        (build_fig1_results / aggregate_auprc.build_auprc_vs_tn の出力そのまま)
    methods: 列にする method のキーのリスト（例: ["edge_factor", "glasso",
        "bdgraph"] または ["edge_factor", "bdgraph"]）。存在しない列は
        自動でスキップされない -- 呼び出し側が要る列だけ渡す。
    method_labels: {method_key: 表示名} （例: {"edge_factor": "Ours",
        "glasso": "GLasso", "bdgraph": "BDgraph"}）。

    全graph_typeをまとめた1つのMarkdown表を文字列で返す。行は
    GEN_ORDER の順（sparse_factorized -> erdos_renyi -> grid ->
    scale_free -> ar1）、各graph_type内はT/Nの昇順。

    データが全く無いgraph_typeの行は出力しない。
    """
    header = ["Graph type", "T/N"] + [method_labels[m] for m in methods]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    for gen in GEN_ORDER:
        gen_results = results.get(gen, {})
        # このgen_resultsに出てくる全T/Nの和集合（どのmethodにあってもOK）
        all_tns = sorted({tn for m in methods for tn in gen_results.get(m, {}).keys()})
        if not all_tns:
            continue
        for i, tn in enumerate(all_tns):
            gen_label = GEN_DISPLAY.get(gen, gen) if i == 0 else ""
            row = [gen_label, f"{tn:g}"]
            row += [_fmt_cell(gen_results, m, tn) for m in methods]
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def write_markdown_table(results, methods, method_labels, save_path):
    """build_markdown_table(...) の結果をファイルに書き出す。"""
    text = build_markdown_table(results, methods, method_labels)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Wrote table to: {save_path}")
    print()
    print(text)
    return text
