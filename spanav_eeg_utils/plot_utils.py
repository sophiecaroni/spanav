"""
********************************************************************************
    Title: Plotting utilities

    Author: Sophie Caroni
    Date of creation: 18.02.2026

    Description:
    This script contains helper functions for plotting.
********************************************************************************
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import spanav_eeg_utils.io_utils as io

from contextlib import contextmanager
from matplotlib.figure import Figure
from pathlib import Path


@contextmanager
def plot_context(
):
    increase = 0
    params = {
        'figure.dpi': 300,
        'axes.grid': False,
        'font.size': 7 + increase,              # General fontsize
        'axes.titlesize': 7 + increase,         # Subplot titles
        'figure.titlesize': 9 + increase,       # Overall figure title
        'axes.labelsize': 7 + increase,         # Axis labels (x and y)
        'xtick.labelsize': 6 + increase,        # X-axis tick labels
        'ytick.labelsize': 6 + increase,        # Y-axis tick labels
        'legend.fontsize': 5 + increase,        # Legend text
        'legend.title_fontsize': 5 + increase,       # Legend title

        # Line widths
        'axes.linewidth': 1,         # Border (spines) width
        'xtick.major.width': 1,      # X tick line width
        'ytick.major.width': 1,      # Y tick line width

        # Tick appearance
        'xtick.major.size': 2,       # Tick length
        'ytick.major.size': 2
    }

    # Other specific customizations are stored in the despine.mplstyle file
    style_path = (Path(__file__).resolve().parent / ".." / "viz" / "despine.mplstyle").resolve()

    # Create figure applying the custom context
    with plt.rc_context(rc=params), plt.style.context(str(style_path)):
        yield


def layout_subplots_grid(
        n: int,
        max_cols: int = 6,
) -> tuple[int, int]:
    """

    :param n:
    :param max_cols:
    :return:
    """
    ncols = int(np.ceil(np.sqrt(n)))
    ncols = min(max(ncols, 1), max_cols)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def save_figure(
        save_dir: str | None,
        fname: str,
        fig: Figure | None = None,
        sid: str | None = None,
        dpi: int = 900,
        prevent_overwrite: bool = False,
        group_parent_dir: str | None = None,
        check_path_sid_strings: bool = True,
        **kwargs,
) -> None:
    outputs_path = io.get_outputs_path(sid, group_parent_dir=group_parent_dir)
    if save_dir is not None:
        outputs_path /= save_dir

    save_path = io.set_for_save(outputs_path, check_sid_strings=check_path_sid_strings)

    full_save_path = save_path / fname

    if prevent_overwrite and os.path.exists(full_save_path):
        prefix = 'NEW_'
        full_save_path = save_path / f'{prefix}_{fname}'

    if fig is None:
        plt.savefig(full_save_path, dpi=dpi, **kwargs)
    else:
        fig.savefig(full_save_path, dpi=dpi, **kwargs)


def get_nrows_ncols(
        subplots_elements: list
) -> tuple[int, int]:
    n = int(len(subplots_elements))
    nrows, ncols = (2, int(np.ceil(n/2))) if n > 2 else (1, n)
    return nrows, ncols

