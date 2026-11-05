#!/usr/bin/env python3
"""Generate interactive figures from saved descriptive or causal CSV data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import causal_figures, descriptive_figures, descriptive_summary, write_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Descriptive table directory or per-window metrics CSV")
    parser.add_argument("output", type=Path, help="Directory for interactive HTML figures")
    parser.add_argument("--causal", action="store_true", help="Read a per-window metrics CSV")
    parser.add_argument("--paired", type=Path, help="Paired contrast CSV for causal plots")
    args = parser.parse_args()
    figures = causal_figures(args.input, args.paired) if args.causal else descriptive_figures(args.input)
    write_figures(figures, args.output)
    if not args.causal:
        (args.output / "descriptive_summary.md").write_text(descriptive_summary(args.input))


if __name__ == "__main__":
    main()
