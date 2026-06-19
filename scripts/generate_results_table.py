#!/usr/bin/env python3
"""
scripts/generate_results_table.py
──────────────────────────────────
Read the MOTA results CSV and generate:
  - A pretty terminal table (rich)
  - A Markdown table for the README
  - Per-class breakdown if per_class CSV is present

Usage
─────
python scripts/generate_results_table.py --csv outputs/mota_results.csv
python scripts/generate_results_table.py --csv outputs/mota_results.csv --markdown
"""

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def build_rich_table(df: pd.DataFrame, title: str = "MOTA Results") -> Table:
    table = Table(title=title, show_header=True, header_style="bold cyan",
                  show_footer=True)

    column_map = {
        "seq_name":            ("Sequence",  "right"),
        "n_frames":            ("Frames",    "right"),
        "mota":                ("MOTA ↑",    "right"),
        "motp":                ("MOTP ↑",    "right"),
        "idf1":                ("IDF1 ↑",    "right"),
        "num_switches":        ("IDS ↓",     "right"),
        "num_false_positives": ("FP ↓",      "right"),
        "num_misses":          ("FN ↓",      "right"),
        "recall":              ("Recall ↑",  "right"),
        "precision":           ("Prec ↑",    "right"),
        "fps":                 ("FPS",       "right"),
    }

    present = [k for k in column_map if k in df.columns]

    for k in present:
        header, justify = column_map[k]
        table.add_column(header, justify=justify)

    for _, row in df.iterrows():
        cells = []
        for k in present:
            val = row.get(k, "—")
            if isinstance(val, float):
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        table.add_row(*cells)

    # Mean summary
    numeric = [k for k in present if k not in ("seq_name",)]
    means   = df[numeric].mean()
    mean_cells = []
    for k in present:
        if k == "seq_name":
            mean_cells.append("[bold]MEAN[/bold]")
        elif k in means:
            mean_cells.append(f"[bold]{means[k]:.2f}[/bold]")
        else:
            mean_cells.append("—")
    table.add_row(*mean_cells, style="bold green")

    return table


def to_markdown_table(df: pd.DataFrame) -> str:
    """Generate GitHub-flavoured Markdown table."""
    column_map = {
        "seq_name":            "Sequence",
        "n_frames":            "Frames",
        "mota":                "MOTA ↑",
        "motp":                "MOTP ↑",
        "idf1":                "IDF1 ↑",
        "num_switches":        "IDS ↓",
        "num_false_positives": "FP ↓",
        "num_misses":          "FN ↓",
        "recall":              "Recall ↑",
        "precision":           "Prec ↑",
    }

    present = [k for k in column_map if k in df.columns]
    headers = [column_map[k] for k in present]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---:" if k != "seq_name" else "---"
                                     for k in present]) + " |")

    for _, row in df.iterrows():
        cells = []
        for k in present:
            val = row.get(k, "—")
            if isinstance(val, float):
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    # Mean row
    numeric = [k for k in present if k not in ("seq_name",)]
    means   = df[numeric].mean()
    mean_cells = []
    for k in present:
        if k == "seq_name":
            mean_cells.append("**MEAN**")
        elif k in means:
            mean_cells.append(f"**{means[k]:.2f}**")
        else:
            mean_cells.append("—")
    lines.append("| " + " | ".join(mean_cells) + " |")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate MOTA results table")
    p.add_argument("--csv",      default="outputs/mota_results.csv",
                   help="Path to CSV from run_tracker.py")
    p.add_argument("--markdown", action="store_true",
                   help="Print Markdown table for README")
    p.add_argument("--save-md",  type=Path, default=None,
                   help="Save Markdown table to this file")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        console.print(f"[red]CSV not found: {csv_path}[/]")
        console.print("Run the tracker first: python run_tracker.py")
        return

    df = pd.read_csv(csv_path)

    # Rich table
    table = build_rich_table(df)
    console.print(table)

    # Markdown
    if args.markdown or args.save_md:
        md = to_markdown_table(df)
        if args.markdown:
            console.print("\n[bold]Markdown table:[/]\n")
            console.print(md)
        if args.save_md:
            args.save_md.write_text(md)
            console.print(f"\n[green]Markdown saved to {args.save_md}[/]")


if __name__ == "__main__":
    main()
