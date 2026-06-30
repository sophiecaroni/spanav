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
import matplotlib.colors as mcolors
from spanav_eeg_utils.spanav_utils import get_task_epo_types
from pathlib import Path
from importlib.resources import files
from contextlib import contextmanager
from matplotlib.figure import Figure
from mne.io import BaseRaw


@contextmanager
def plot_context(
):
    increase = 0
    params = {
        'figure.dpi': 400,
        'axes.grid': False,
        'font.size': 7 + increase,              # General fontsize
        'axes.titlesize': 7 + increase,         # Subplot titles
        'figure.titlesize': 9 + increase,       # Overall figure title
        'axes.labelsize': 7 + increase,         # Axis labels (x and y)
        'xtick.labelsize': 6 + increase,        # X-axis tick labels
        'ytick.labelsize': 6 + increase,        # Y-axis tick labels
        'legend.fontsize': 6 + increase,        # Legend text
        'legend.title_fontsize': 7 + increase,  # Legend title

        # Line widths
        'axes.linewidth': 1,         # Border (spines) width
        'xtick.major.width': 1,      # X tick line width
        'ytick.major.width': 1,      # Y tick line width

        # Tick appearance
        'xtick.major.size': 2,       # Tick length
        'ytick.major.size': 2
    }

    # Other specific customizations are stored in the despine.mplstyle file
    style_path = files("spanav_tbi") / "mplstyles" / "despine.mplstyle"

    # Create figure applying the custom context
    with plt.rc_context(rc=params), plt.style.context(str(style_path)):
        yield


def get_cont_rec_plot_kwargs(raw_rec: BaseRaw) -> dict:
    return {
        # 'duration': 40,
        'duration': raw_rec.times[-1] if raw_rec.times[-1] < 230 else raw_rec.times[-1] // 2,
        'n_channels': len(raw_rec.ch_names),
        'clipping': None,
        'scalings': {'eeg': 10e-5},
        }


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
        save_dir: Path | str | None ,
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


def get_epo_palette(
) -> dict:
    return {
        'ContMov': 'tab:green', 'blContMov': 'tab:green',
        'Stasis': 'tab:blue', 'blStasis': 'tab:blue',
        'MovOn': 'OrangeRed', 'blMovOn': 'OrangeRed',
        'ObjPres': 'orange', 'blObjPres': 'orange',
        'Raw': 'purple', 'blRaw': 'purple',
    }


def get_cond_palette(
) -> dict:
    palette = {
        # blinded
        'A': '#0d2f8a',  # dark navy blue
        'B': '#4293f5',  # medium blue
        'C': '#b8d9ff',  # light blue

        # unblinded
        'HF': '#0d2f8a',    # dark navy blue
        'cTBS': '#4293f5',  # medium blue
        'iTBS': '#b8d9ff',  # light blue
    }

    # Add to the palette colors for conditions merged with epoch-types (useful in cbpt results for cond × epo_type)
    epo_types = get_task_epo_types() + [f'bl{et}' for et in get_task_epo_types()]

    cmap = plt.colormaps['tab20c']  # organized in groups of 4 similar colors
    cmap_hue_groups = 5
    cmap_shade_per_hue = 4
    max_colors = cmap_hue_groups*cmap_shade_per_hue
    conds = list(palette.keys())  # wrap in a list to create a new object and not just a view
    for i, cond in enumerate(conds):
        # Jump by cmap_shade_per_hue indices for each cond to grab a new hue-family
        cond_index = (i * cmap_shade_per_hue) % max_colors  # modulo assures no error is raised if the last color is reached, restart from beginning
        for j, et in enumerate(epo_types):
            label = f"{cond}_{et}"
            color_idx = (cond_index + j) % max_colors  # select the j-th shade within that hue-family
            palette[label] = mcolors.rgb2hex(cmap(color_idx))
    return palette


def add_higher_title_text(
        fig,
        axes,
        title,
) -> None:

    # Get bounding box spanning the axes (in figure coordinates)
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    top = max(ax.get_position().y1 for ax in axes)

    # Define title position
    x = (left + right) / 2  # center
    fig_height = fig.get_size_inches()[1]
    offset = 0.2 / fig_height  # 0.2 inches above the top of the axes
    y = top + offset  # a bit higher than the axis titles

    # Get title fontsize
    title_fontsize = axes[0].title.get_fontsize()

    # Add title
    fig.text(
        x,
        y,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold"
    )
