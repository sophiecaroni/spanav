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
from typing import Iterable

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
    vmax = max(p.data.mean(axis=axis).max() for p in psd_array)
    vmin = -vmax
    return vmin, vmax


def plot_psd_by_epo(
        psd_df: pd.DataFrame,
        pkind: str,
        epo_axes: plt.Axes | np.ndarray,
        vlim: tuple[float, float],
        epo_title: bool = True,
        show_xlabel: bool = True,
        **kwargs,
) -> None:
    base_order = get_epo_types()
    epo_type_order = base_order + [f'bl{e}' for e in base_order]
    epo_types_sorted = sorted(psd_df['epo_type'].unique(), key=epo_type_order.index)

    with plot_context():
        for i, epo_type in enumerate(epo_types_sorted):
            epo_type_df = psd_df[psd_df['epo_type'] == epo_type]

            assert len(epo_type_df) == 1, f'This plot is made for one PSD per epoch-type! Here got {len(epo_type_df)} for {epo_type} type'
            ax = epo_axes[i]
            epo_type_df.reset_index(inplace=True)
            psd_plot = epo_type_df.loc[0, 'psd']

            if pkind == 'topomap':
                fmin, fmax = psd_plot.freqs.min(), psd_plot.freqs.max()
                psd_plot.plot_topomap(
                    bands={f'All ({fmin}-{fmax})': (fmin, fmax)},
                    axes=ax,
                    cmap='RdBu_r',
                    colorbar=int(i) + 1 == len(epo_axes),
                    show=False,
                    vlim=vlim,
                    **kwargs,
                )
                # Remove frequency interval title
                ax.set_title('')

            else:  # plot spectrum
                avg_psd = psd_plot.data.mean(axis=0)  # mean across channels, it becomes (n_freqs,)
                ax.plot(
                    psd_plot.freqs,
                    avg_psd,
                    **kwargs
                )
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power (log)')
                ax.set_ylim(vlim)

            if not i == 0:
                ax.set_ylabel('')
            if epo_title:
                epo_type_lbl = map_epo_type_labels().get(epo_type, epo_type)
                ax.set_title(epo_type_lbl)
            if not show_xlabel:
                ax.set_xlabel('')

        if pkind == 'spectrum':
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.figure.legend(by_label.values(), by_label.keys())


def all_sid_psd_plots(
        psd_df: pd.DataFrame,
        pkind: str = 'spectrum',
        show: bool = True,
        save: bool = False,
) -> None:
    all_sid_plots(psd_df, 'psd', plot_psd_by_epo, _compute_psd_ylim, 'PSD', pkind, show, save)


def all_group_psd_plots(
        psd_df: pd.DataFrame,
        pkind: str = 'spectrum',
        show: bool = True,
        save: bool = False,
) -> None:
    all_group_plots(psd_df, 'psd', plot_psd_by_epo, _compute_psd_ylim, 'PSD', pkind, show, save)
