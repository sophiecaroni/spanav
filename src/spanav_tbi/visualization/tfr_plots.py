"""
********************************************************************************
    Title: Time-frequency representations (TFR) plots

    Author: Sophie Caroni
    Date of creation: 23.02.2026

    Description:
    This script contains functions to plot TFR objects.
********************************************************************************
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import EpochsTFR, AverageTFR
from pandas.errors import EmptyDataError

from spanav_eeg_utils.plot_utils import plot_context
from spanav_eeg_utils.spanav_utils import map_epo_type_labels, get_epo_types
from spanav_tbi.visualization.iter_plots import all_sid_plots, all_group_plots
from typing import Iterable

TFR = EpochsTFR | AverageTFR


def _compute_tfr_vlim(tfr_array: Iterable[TFR], pkind: str) -> tuple[float, float]:
    if pkind == 'topomap':
        axis = (1, 2)  # average across frequency and timepoints
    elif pkind in ('spectrum', 'heatmap'):  # average across channels in spectrograms and power spectra
        axis = 0
    else:
        raise ValueError(f'Accepted plot kinds are "heatmap", "topomap" or "spectrum"; got {pkind = }')
    # vmin = -np.nanmax([np.nanmax(np.nanmean(t.data, axis=axis)) for t in tfr_array])
    vmax = np.nanmax([np.nanmax(np.nanmean(t.data, axis=axis)) for t in tfr_array])
    vmin = -vmax
    return vmin, vmax


def plot_tfr_subplots(
        tfr_df: pd.DataFrame,
        pkind: str,
        axes: plt.Axes | np.ndarray,
        vlim: tuple[float, float],
        subplot_col: str = 'epo_type',
        show_ax_titles: bool = True,
        show_xlabel: bool = True,
        **kwargs,
) -> None:
    if subplot_col == 'epo_type':
        base_order = get_epo_types()
        epo_type_order = base_order + [f'bl{e}' for e in base_order]
        epo_type_order += [f'{e}_wide' for e in epo_type_order]
        subplot_cats = sorted(tfr_df['epo_type'].unique(), key=epo_type_order.index)
    else:
        subplot_cats = tfr_df[subplot_col].unique()
    with plot_context():
        for i, (ax, subplot_cat) in enumerate(zip(axes, subplot_cats)):
            subplot_df = tfr_df[tfr_df[subplot_col] == subplot_cat].reset_index()

            if pkind == 'spectrum':  # plot spectrum
                # Iterate so in case there are multple TFRs, each of them is plotted as a different line
                for ir, row in subplot_df.iterrows():
                    tfr = row['tfr']

                    # Average TFR across time (-1) to get a spectrum, then average across channels (0) to get (n_freqs,)
                    avg_psd = tfr.data.mean(axis=(-1, 0))  # it gets to (n_chan, n_freqs)
                    ax.plot(tfr.freqs, avg_psd, **kwargs)
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power (log)')
                    ax.set_ylim(vlim)

            else:
                # In case there are multple TFRs, compute their average to plot as heatmap/topomap
                if len(subplot_df) > 1:
                    tfr = subplot_df.loc[:, 'tfr'].mean()
                else:
                    tfr = subplot_df.loc[0, 'tfr']
                if pkind == 'heatmap':
                    tfr.plot(
                        combine='mean',  # averages across channels in case there are multiples
                        axes=ax,
                        cmap='RdBu_r',  # set because default switches between two
                        colorbar=int(i) + 1 == len(axes),  # only add colorbar to right-most subplots
                        show=False,
                        vlim=vlim,
                        **kwargs
                    )

                elif pkind == 'topomap':  # plot topo maps
                    tfr.plot_topomap(
                        axes=ax,
                        cmap='RdBu_r',  # set because default switches between two
                        colorbar=int(i) + 1 == len(axes),  # only add colorbar to right-most subplots
                        show=False,
                        vlim=vlim,
                        **kwargs,
                    )

            # Further axis customization
            if not i == 0:
                # Remove automatically set y-axis label if not left-most plot
                ax.set_ylabel('')

            if show_ax_titles:
                title = map_epo_type_labels().get(subplot_cat, subplot_cat)
                ax.set_title(title)
            if not show_xlabel:
                # Remove automatically set x-axis label
                ax.set_xlabel('')

        if pkind == 'spectrum':
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.figure.legend(by_label.values(), by_label.keys())


def _filter_tfr_plot_df(tfr_df: pd.DataFrame) -> pd.DataFrame:
    """Filter TFR plots dataframe for epochs that were stasis-corrected and extracted from wide windows"""
    plot_df = tfr_df[(tfr_df['epo_type'].str.endswith('wide')) & (tfr_df['epo_type'].str.startswith('bl'))]
    if plot_df.empty:
        raise EmptyDataError('No data of the needed epo-types found for this plot.')
    return plot_df


def all_sid_tfr_plots(
        tfr_df: pd.DataFrame,
        pkind: str = 'heatmap',
        show: bool = True,
        save: bool = False,
) -> None:
    plot_df = _filter_tfr_plot_df(tfr_df)
    all_sid_plots(plot_df, 'tfr', plot_tfr_subplots, _compute_tfr_vlim, 'TFR', pkind, show, save)


def all_group_tfr_plots(
        tfr_df: pd.DataFrame,
        pkind: str = 'heatmap',
        show: bool = True,
        save: bool = False,
) -> None:
    plot_df = _filter_tfr_plot_df(tfr_df)
    all_group_plots(
        plot_df, 'tfr', plot_tfr_subplots, vlim_fn=_compute_tfr_vlim, plots_subdir='TFR', pkind=pkind,
        show=show, save=save
    )
