"""
********************************************************************************
    Title: Power spectral density (PSD) plots

    Author: Sophie Caroni
    Date of creation: 21.04.2026

    Description:
    This script contains functions to plot PSD objects.
********************************************************************************
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import EpochsSpectrum, Spectrum
from spanav_eeg_utils.plot_utils import plot_context
from spanav_eeg_utils.spanav_utils import map_epo_type_labels, get_epo_types
from spanav_tbi.visualization.iter_plots import all_sid_plots, all_group_plots
from spanav_tbi.processing.psd import average_psd_channels
from typing import Iterable
from pandas.errors import EmptyDataError

PSD = EpochsSpectrum | Spectrum


def _compute_psd_ylim(psd_array: Iterable[PSD], pkind: str) -> tuple[float, float]:
    if pkind not in ['topomap', 'spectrum']:
        raise (ValueError, f'Accepted plot kinds are "topomap" and "spectrum"; got {pkind = }')
    if pkind == 'spectrum':
        axis = 0  # average across channels
    elif pkind == 'topomap':
        axis = -1  # average across frequencies
    else:
        raise ValueError(f'Accepted plot kinds are "topomap" and "spectrum"; got {pkind = }')
    vmax = np.nanmax([np.nanmax(np.nanmean(t.data, axis=axis)) for t in psd_array])
    vmin = - vmax
    return vmin, vmax


def _order_epo_types(df: pd.DataFrame) -> list[str]:
    base_order = get_epo_types()
    epo_type_order = base_order + [f'bl{e}' for e in base_order]
    try:
        return sorted(df['epo_type'].unique(), key=epo_type_order.index)
    except ValueError:  # if some epo_type values are not present in epo_type_order
        return sorted(df['epo_type'].unique())


def plot_psd_subplots(
        psd_df: pd.DataFrame,
        pkind: str,
        axes: plt.Axes | np.ndarray | list,
        vlim: tuple[float, float],
        subplot_col: str = 'epo_type',
        show_ax_titles: bool = True,
        show_xlabel: bool = True,
        show_legend: bool = True,
        **kwargs,
) -> None:
    if subplot_col == 'epo_type':
        subplot_cats = _order_epo_types(psd_df)
    else:
        subplot_cats = psd_df[subplot_col].unique()
    with plot_context():
        for i, (ax, subplot_cat) in enumerate(zip(axes, subplot_cats)):
            subplot_df = psd_df[psd_df[subplot_col] == subplot_cat].reset_index()

            if pkind == 'topomap':
                # In case there are multple spectra, plot the topo of their average
                if len(subplot_df) > 1:
                    psd = subplot_df.loc[:, 'psd'].mean()
                else:
                    psd = subplot_df.loc[0, 'psd']
                fmin, fmax = psd.freqs.min(), psd.freqs.max()
                psd.plot_topomap(
                    bands={f'All ({fmin}-{fmax})': (fmin, fmax)},
                    axes=ax,
                    cmap='RdBu_r',
                    colorbar=int(i) + 1 == len(axes),
                    show=False,
                    vlim=vlim,
                    **kwargs,
                )
                # Remove frequency interval title
                ax.set_title('')

            else:  # plot spectrum
                # Iterate so in case there are multple spectra, each of them is plotted as a different line
                for ir, row in subplot_df.iterrows():
                    psd = row['psd']

                    if psd.info['nchan'] > 1:
                        # psd = psd.mean(axis=0)  # mean across channels, it becomes (n_freqs, )
                        psd = average_psd_channels(psd)

                    ax.plot(psd.freqs, psd.data.flatten(), **kwargs)
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Power (log)')
                    ax.set_ylim(vlim)

            if not i == 0:
                ax.set_ylabel('')
            if show_ax_titles:
                if subplot_col == 'epo_type':
                    title = map_epo_type_labels().get(subplot_cat, subplot_cat)
                else:
                    title = subplot_cat
                ax.set_title(title)
            if not show_xlabel:
                ax.set_xlabel('')

        if pkind == 'spectrum' and show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.figure.legend(by_label.values(), by_label.keys())


def _filter_psd_plot_df(psd_df: pd.DataFrame) -> pd.DataFrame:
    """Filter PSD plots dataframe for epochs that were stasis-corrected"""
    plot_df = psd_df[psd_df['epo_type'].str.startswith('bl')]
    if plot_df.empty:
        raise EmptyDataError('No data of the needed epo-types found for this plot.')
    return plot_df


def all_sid_psd_plots(
        psd_df: pd.DataFrame,
        pkind: str = 'spectrum',
        show: bool = True,
        save: bool = False,
) -> None:
    plot_df = _filter_psd_plot_df(psd_df)
    all_sid_plots(plot_df, 'psd', plot_psd_subplots, _compute_psd_ylim, 'PSD', pkind, show, save)


def all_group_psd_plots(
        psd_df: pd.DataFrame,
        pkind: str = 'spectrum',
        show: bool = True,
        save: bool = False,
) -> None:
    plot_df = _filter_psd_plot_df(psd_df)
    all_group_plots(
        plot_df, 'psd', plot_psd_subplots, vlim_fn=_compute_psd_ylim, plots_subdir='CBPT/PSD', pkind=pkind,
        show=show, save=save
    )
