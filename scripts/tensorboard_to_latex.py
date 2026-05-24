from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

COMMAND_NAME = "plot-loss"


@dataclass
class PlotStyle:
    figsize: tuple = (7.0, 4.0)
    dpi: int = 300

    background_color: str = "#ffffff"
    line_color: str = "#1f77b4"
    grid_color: str = "#eaeaea"
    text_color: str = "#222222"

    line_width: float = 1.0
    alpha: float = 0.95

    title_size: int = 18
    label_size: int = 14
    tick_size: int = 11

    smooth_window: int = 25


def setup_latex_style(style: PlotStyle) -> None:
    """
    Configure matplotlib for a clean LaTeX/distill.pub inspired aesthetic.
    """

    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": style.grid_color,
            "axes.linewidth": 1.0,
            "axes.facecolor": style.background_color,
            "figure.facecolor": style.background_color,
            "grid.color": style.grid_color,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "xtick.color": style.text_color,
            "ytick.color": style.text_color,
            "axes.labelcolor": style.text_color,
            "text.color": style.text_color,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        },
    )


def load_tensorboard_csv(csv_path: str) -> pd.DataFrame:
    """
    Load TensorBoard CSV export.

    Expected columns:
        - Step
        - Value
        - RunName
    """

    df = pd.read_csv(csv_path)

    required_columns = {"Step", "Value", "RunName"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """
    Smooth the loss curve using a centered moving average.
    """

    return series.rolling(window=window, center=True, min_periods=1).mean()


def prepare_run_dataframe(
    df: pd.DataFrame,
    run_name: str,
    smooth_window: int,
) -> pd.DataFrame:
    """
    Extract and smooth a single run.
    """

    run_df = df[df["RunName"] == run_name].sort_values("Step").copy()

    run_df["SmoothedValue"] = smooth_series(
        run_df["Value"],
        smooth_window,
    )

    return run_df


def create_loss_plot(
    df: pd.DataFrame,
    output_path: str,
    style: PlotStyle,
    title: str | None = None,
    grid_title: str | None = "Loss",
) -> None:
    """
    Create a clean publication-style loss curve.
    """

    setup_latex_style(style)

    fig, ax = plt.subplots(
        figsize=style.figsize,
        dpi=style.dpi,
    )

    run_names = sorted(df["RunName"].unique())

    for run_name in run_names:
        run_df = prepare_run_dataframe(
            df=df,
            run_name=run_name,
            smooth_window=style.smooth_window,
        )

        # Draw smoothed curve first and retrieve its color
        (smooth_line,) = ax.plot(
            run_df["Step"],
            run_df["SmoothedValue"],
            linewidth=style.line_width,
            alpha=style.alpha,
            label=run_name,
        )

        line_color = smooth_line.get_color()

        # Raw curve with the same color but lighter
        ax.plot(
            run_df["Step"],
            run_df["Value"],
            color=line_color,
            linewidth=0.8,
            alpha=0.18,
        )

    ax.grid(True)

    ax.set_xlabel(r"\textbf{Training Step}", fontsize=style.label_size)
    ax.set_ylabel(r"\textbf{" + grid_title + "}", fontsize=style.label_size)

    if title is not None:
        ax.set_title(
            rf"\textbf{{{title}}}",
            fontsize=style.title_size,
            pad=18,
        )

    ax.tick_params(axis="both", labelsize=style.tick_size)

    if len(run_names) > 1:
        ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path)
    plt.close(fig)


def process_tensorboard_csv(
    input_csv: str,
    output_path: str,
    title: str | None,
    grid_title: str | None = "Loss",
    smooth_window: int = 25,
) -> None:
    """
    Full pipeline for generating the plot.
    """

    style = PlotStyle(smooth_window=smooth_window)

    df = load_tensorboard_csv(input_csv)

    create_loss_plot(
        df=df,
        output_path=output_path,
        style=style,
        title=title,
        grid_title=grid_title,
    )

    print(f"[INFO] Saved plot to: {output_path}")


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """
    Register this subcommand to the main CLI parser.

    Args:
        subparsers:
            Subparsers object from argparse.ArgumentParser().
    """

    parser = subparsers.add_parser(
        COMMAND_NAME,
        help="Generate a beautiful LaTeX-style loss plot from TensorBoard CSV",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to TensorBoard CSV file",
    )

    parser.add_argument(
        "--output",
        default=os.path.join("data", "loss_plot.pdf"),
        help="Output path for the generated figure",
    )

    parser.add_argument(
        "--title",
        default="Training Loss",
        help="Plot title",
    )

    parser.add_argument(
        "--grid-title",
        default="Loss",
        help="Plot grid title",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Moving average smoothing window",
    )

    parser.set_defaults(func=main)

    return parser


def main(args: argparse.Namespace) -> None:
    """
    Entry point for the CLI command.

    Args:
        args:
            Parsed command-line arguments.
    """

    process_tensorboard_csv(
        input_csv=args.input,
        output_path=args.output,
        title=args.title,
        grid_title=args.grid_title,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    register_subparser(subparsers)
    args = parser.parse_args()
    args.func(args)
