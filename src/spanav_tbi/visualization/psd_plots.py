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
from spanav_eeg_utils.plot_utils import plot_context, save_figure, add_higher_title_text, get_cond_palette
from spanav_eeg_utils.spanav_utils import map_epo_type_labels, get_epo_types
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


def iter_plot_sid_psd(
        psd_df: pd.DataFrame,
        pkind: str = 'spectrum',
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():
        sup_cond = pkind == 'spectrum'  # define whether conditions will be superimposed (for spectra) or on rows (for topomaps)

        # Create one figure per subject
        for sid, sid_df in psd_df.groupby('sid'):
            n_conds = len(sid_df['cond'].unique())
            n_epo_types = len(sid_df['epo_type'].unique())
            nrows = 1 if sup_cond else n_conds
            n_cols = n_epo_types
            fig, axes = plt.subplots(
                nrows, n_cols, sharey=True, sharex=True,
                figsize=(n_epo_types * 4.0, nrows * 3.5), squeeze=False
            )
            axes = axes.flatten()

            # Compute vlim across all subject conditions
            vlim = _compute_psd_ylim(sid_df['psd'].values, pkind=pkind)

            for i, (cond, cond_df) in enumerate(sid_df.groupby('cond')):
                epo_axes = axes if sup_cond else axes[i*n_cols : i*n_cols+n_cols]

                show_xlabel = True if sup_cond else i == n_conds - 1
                plot_kwargs: dict = dict(epo_title=i == 0, vlim=vlim)
                if sup_cond:
                    plot_kwargs.update(label=cond, color=get_cond_palette().get(cond))

                plot_psd_by_epo(cond_df, pkind, epo_axes=epo_axes, show_xlabel=show_xlabel, **plot_kwargs)

                if not sup_cond and n_conds > 1:
                    add_higher_title_text(fig, epo_axes, f"Cond {cond}")

            if save:
                fname = f'{sid}_etypes_{pkind}.png'
                save_figure(save_dir=str(sid), group_parent_dir='plots/PSD', fname=fname, fig=fig, sid=str(sid))
            if show:
                plt.show()
            plt.close()


def iter_plot_group_psd(
        psd_df: pd.DataFrame,
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():

        # Create one separate figure for each group
        for i_g, (group, group_df) in enumerate(psd_df.groupby('group')):
            n_epo_types = len(group_df['epo_type'].unique())
            n_cols = n_epo_types
            fig_width = n_cols * 4.0
            fig, axes = plt.subplots(
                1, n_cols, sharey=True, sharex=True, figsize=(fig_width, 3.5),
                squeeze=False
            )
            axes = axes.flatten()
            vlim = _compute_psd_ylim(group_df['psd'].values, 'spectrum')

            # For each condition, plot spectrum in the same axes (epoch-type subplots) so that they overlap
            for i_c, (cond, cond_df) in enumerate(group_df.groupby('cond')):
                plot_kwargs: dict = dict(
                    epo_title=i_c == 0,
                    vlim=vlim,
                    label=cond,
                    color=get_cond_palette().get(cond),
                )
                plot_psd_by_epo(cond_df, 'spectrum', epo_axes=axes, **plot_kwargs)

            if save:
                fname = f'group{group}_etypes_spectrum.png'
                save_figure(group_parent_dir='plots/PSD', fname=fname, fig=fig, save_dir=f'WP73{group}')
            if show:
                plt.show()
            plt.close()
