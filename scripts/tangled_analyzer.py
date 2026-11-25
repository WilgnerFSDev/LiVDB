import os
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.panel import Panel
from rich.table import Table

from scripts.ui import UI, console

REPORTS_DIR = "reports"


def compute_tangled_distribution(csv_path: str, max_bins: int = 20) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "label" not in df.columns or "cve_id" not in df.columns:
        raise ValueError("Dataset CSV must contain 'cve_id' and 'label' columns.")

    df_v = df[df["label"] == 1]

    if df_v.empty:
        return {
            "total_cves": 0,
            "total_functions_changed": 0,
            "single_func_count": 0,
            "single_func_pct": 0.0,
            "multi_func_count": 0,
            "multi_func_pct": 0.0,
            "mean_funcs": 0.0,
            "median_funcs": 0.0,
            "max_funcs": 0,
            "counts_by_num": {},
            "percentages": {},
            "bins": [],
            "more_than_max": {"count": 0, "percentage": 0.0},
        }

    cve_counts = df_v.groupby("cve_id").size()
    total_cves = len(cve_counts)
    total_functions_changed = int(cve_counts.sum())

    counts_by_num = dict(Counter(cve_counts))
    percentages = {k: (v / total_cves) * 100 for k, v in counts_by_num.items()}

    single_func_count = counts_by_num.get(1, 0)
    single_func_pct = percentages.get(1, 0.0)
    multi_func_count = sum(v for k, v in counts_by_num.items() if k > 1)
    multi_func_pct = (multi_func_count / total_cves) * 100 if total_cves > 0 else 0.0

    mean_funcs = float(cve_counts.mean())
    median_funcs = float(cve_counts.median())
    max_funcs = int(cve_counts.max())

    bins: List[Dict[str, Any]] = []
    for k in range(1, max_bins + 1):
        cnt = counts_by_num.get(k, 0)
        pct = percentages.get(k, 0.0)
        bins.append({"functions": k, "count": cnt, "percentage": pct})

    gt_max_count = sum(v for k, v in counts_by_num.items() if k > max_bins)
    gt_max_pct = (gt_max_count / total_cves) * 100 if total_cves > 0 else 0.0

    return {
        "total_cves": total_cves,
        "total_functions_changed": total_functions_changed,
        "single_func_count": single_func_count,
        "single_func_pct": single_func_pct,
        "multi_func_count": multi_func_count,
        "multi_func_pct": multi_func_pct,
        "mean_funcs": mean_funcs,
        "median_funcs": median_funcs,
        "max_funcs": max_funcs,
        "counts_by_num": counts_by_num,
        "percentages": percentages,
        "bins": bins,
        "more_than_max": {"count": gt_max_count, "percentage": gt_max_pct},
    }


def print_terminal_histogram(data: Dict[str, Any], dataset_name: str) -> None:
    if data["total_cves"] == 0:
        UI.warning(f"No vulnerable functions (label=1) found in dataset '{dataset_name}'.")
        return

    table = Table(
        title=f"Tangled Commits Distribution - {dataset_name}",
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Functions Changed", justify="center", style="bold yellow")
    table.add_column("CVE Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="magenta")
    table.add_column("Terminal Histogram", justify="left", style="blue")

    max_bar_width = 32
    for item in data["bins"][:10]:
        k = item["functions"]
        cnt = item["count"]
        pct = item["percentage"]
        bar_len = int((pct / 100) * max_bar_width)
        bar = "█" * max(1, bar_len) if cnt > 0 else ""
        label_fn = f"{k} function" if k == 1 else f"{k} functions"
        table.add_row(label_fn, f"{cnt:,}", f"{pct:5.2f}%", f"{bar} {pct:.1f}%")

    gt10_count = sum(v for k, v in data["counts_by_num"].items() if k > 10)
    gt10_pct = (gt10_count / data["total_cves"]) * 100 if data["total_cves"] > 0 else 0.0
    gt10_bar_len = int((gt10_pct / 100) * max_bar_width)
    gt10_bar = "█" * max(1, gt10_bar_len) if gt10_count > 0 else ""
    table.add_row("> 10 functions", f"{gt10_count:,}", f"{gt10_pct:5.2f}%", f"{gt10_bar} {gt10_pct:.1f}%")

    console.print()
    console.print(table)
    console.print()

    summary_text = (
        f"[bold]Total CVEs Analyzed:[/bold] {data['total_cves']:,}  |  "
        f"[bold]Total Functions Changed:[/bold] {data['total_functions_changed']:,}\n"
        f"[bold green]Non-Tangled (1 function):[/bold green] {data['single_func_count']:,} ({data['single_func_pct']:.2f}%)  |  "
        f"[bold red]Tangled (>1 functions):[/bold red] {data['multi_func_count']:,} ({data['multi_func_pct']:.2f}%)\n"
        f"[bold]Mean:[/bold] {data['mean_funcs']:.2f} functions/CVE  |  "
        f"[bold]Median:[/bold] {data['median_funcs']:.1f}  |  "
        f"[bold]Max:[/bold] {data['max_funcs']} functions in single CVE"
    )
    console.print(Panel(summary_text, title="Summary Statistics", border_style="cyan"))
    console.print()


def generate_tangled_histogram_plot(
    data: Dict[str, Any],
    output_png_path: str,
    dataset_name: str,
    max_x: int = 20,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)

    x_vals = list(range(1, max_x + 1))
    percentages = [data["percentages"].get(i, 0.0) for i in x_vals]

    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)

    bars = ax.bar(
        x_vals,
        percentages,
        width=0.82,
        color="#1f77b4",
        edgecolor="none",
    )

    ax.set_xlabel("Number of Functions Changed", fontsize=11, fontweight="medium", labelpad=6)
    ax.set_ylabel("Percentage of Changes (%)", fontsize=11, fontweight="medium", labelpad=6)

    ticks = [1, 4, 7, 10, 13, 16, 19]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=10)

    max_y = max(percentages) if percentages else 40
    ax.set_ylim(0, max_y * 1.12 if max_y > 0 else 50)
    ax.tick_params(axis="both", which="major", labelsize=10)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)

    clean_title = dataset_name.replace(".csv", "").replace("_", " ")
    ax.set_title(f"Tangled Commits Distribution\n({clean_title})", fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_png_path


def run_tangled_analysis(csv_path: str, output_dir: str = REPORTS_DIR) -> Optional[str]:
    dataset_name = os.path.basename(csv_path)
    UI.info(f"Analyzing tangled commits distribution for '{dataset_name}'...")

    try:
        data = compute_tangled_distribution(csv_path)
    except Exception as e:
        UI.error(f"Failed to analyze dataset: {e}")
        return None

    print_terminal_histogram(data, dataset_name)

    base_name = dataset_name.replace(".csv", "")
    output_png_name = f"tangled_histogram_{base_name}.png"
    output_png_path = os.path.join(output_dir, output_png_name)

    try:
        png_path = generate_tangled_histogram_plot(data, output_png_path, dataset_name)
        UI.success(f"High-resolution histogram image saved to: [bold]{png_path}[/bold]")
        return png_path
    except Exception as e:
        UI.error(f"Failed to generate histogram plot: {e}")
        return None
