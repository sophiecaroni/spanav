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


def plot_tfr_by_epo(
        tfr_df: pd.DataFrame,
        pkind: str,
        epo_axes: plt.Axes | np.ndarray,
        vlim: tuple[float, float],
        epo_title: bool = True,
        show_xlabel: bool = True,
        **kwargs,
) -> None:
    base_order = get_epo_types()
    epo_type_order = base_order + [f'bl{e}' for e in base_order]
    epo_type_order += [f'{e}_wide' for e in epo_type_order]
    epo_types_sorted = sorted(tfr_df['epo_type'].unique(), key=epo_type_order.index)

    with plot_context():
        for i, epo_type in enumerate(epo_types_sorted):
            epo_type_df = tfr_df[tfr_df['epo_type'] == epo_type]

            assert len(epo_type_df) == 1, f'This plot is made for one TFR per epoch-type! Here got {len(epo_type_df)} for {epo_type} type'
            ax = epo_axes[i]
            epo_type_df.reset_index(inplace=True)
            tfr_plot = epo_type_df.loc[0, 'tfr']

            # Drop nan-padded channels before plotting, otherwise MNE produces blank figures
            valid_chs = [ch for ch, d in zip(tfr_plot.ch_names, tfr_plot.data) if not np.all(np.isnan(d))]
            tfr_plot = tfr_plot.copy().pick(valid_chs)

            if pkind == 'heatmap':
                tfr_plot.plot(
                    combine='mean',  # averages across channels in case there are multiples
                    axes=ax,
                    cmap='RdBu_r',  # set because default switches between two
                    colorbar=int(i) + 1 == len(epo_axes),  # only add colorbar to right-most subplots
                    show=False,
                    vlim=vlim,
                    **kwargs
                )

            elif pkind == 'topomap':  # plot topo maps
                tfr_plot.plot_topomap(
                    axes=ax,
                    cmap='RdBu_r',  # set because default switches between two
                    colorbar=int(i) + 1 == len(epo_axes),  # only add colorbar to right-most subplots
                    show=False,
                    vlim=vlim,
                    **kwargs,
                )

            else:  # plot spectrum
                # Average TFR across time to get a specturm, then average across channels
                chs_psd = tfr_plot.data.mean(axis=-1)  # it gets to (n_chan, n_freqs)
                avg_psd = chs_psd.mean(axis=0)  # it gets to (n_freqs,)

                # Plot
                ax.plot(
                    tfr_plot.freqs,
                    avg_psd,
                    **kwargs
                )
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power (log)')
                ax.set_ylim(vlim)

            # Further axis customization
            if not i == 0:
                # Remove automatically set y-axis label if not left-most plot
                ax.set_ylabel('')

            if epo_title:
                # Put epoch-type as title
                epo_type_lbl = map_epo_type_labels().get(epo_type, epo_type)
                ax.set_title(epo_type_lbl)
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
    all_sid_plots(plot_df, 'tfr', plot_tfr_by_epo, _compute_tfr_vlim, 'TFR', pkind, show, save)


def all_group_tfr_plots(
        tfr_df: pd.DataFrame,
        pkind: str = 'heatmap',
        show: bool = True,
        save: bool = False,
) -> None:
    plot_df = _filter_tfr_plot_df(tfr_df)
    all_group_plots(plot_df, 'tfr', plot_tfr_by_epo, _compute_tfr_vlim, 'TFR', pkind, show, save)
