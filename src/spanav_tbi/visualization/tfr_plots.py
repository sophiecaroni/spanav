"""
********************************************************************************
    Title: Spectrogram plots of time-frequency representations (TFR)

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
from spanav_eeg_utils.plot_utils import plot_context, save_figure, add_higher_title_text, get_cond_palette
from spanav_eeg_utils.spanav_utils import map_epo_type_labels, get_epo_types
from typing import Iterable

TFR = EpochsTFR | AverageTFR


def _compute_tfr_vlim(tfr_array: Iterable[TFR], pkind: str) -> tuple[float, float]:
    if pkind not in ['tfr', 'topomap', 'spectrum']:
        raise (ValueError, f'Accepted plot kinds are "tfr" and "topomap"; got {pkind = }')
    axis = (1, 2) if pkind == 'topomap' else 0  # average across frequency and timepoints in topomapmaps; across channels in spectrograms and power spectra
    # vmin = min(t.data.mean(axis=axis).min() for t in tfr_array)
    vmax = max(t.data.mean(axis=axis).max() for t in tfr_array)
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

            if pkind == 'tfr':
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
                ax.set_ylabel('Power')
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


def iter_plot_sid_tfr(
        tfr_df: pd.DataFrame,
        pkind: str = 'tfr',
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():
        sup_cond = pkind == 'spectrum'  # superimpose power spectra of different conditions, use different subplots for topomaps/TFRs

        # Create one figure per subject
        for sid, sid_df in tfr_df.groupby('sid'):

            # Create one subplot per condition and per epoch-type fpr topomaps and TFR, superimpose condition for spectra
            n_conds = len(sid_df['cond'].unique())
            n_rows = 1 if sup_cond else n_conds
            n_epo_types = len(sid_df['epo_type'].unique())
            n_cols = n_epo_types
            fig_height = n_rows * 3.5
            fig_width = n_cols * 4.0
            fig, axes = plt.subplots(
                n_rows, n_cols, sharey=True, sharex=True, figsize=(fig_width, fig_height),
                squeeze=False  # does not flatten automatically if 1D
            )
            axes = axes.flatten()

            # Define limits of powe0 commonly across conds/epoch-types, for easier comparison within the figure
            vlim = _compute_tfr_vlim(sid_df['tfr'].values, pkind=pkind)

            # Each stimulating condition has a row (of subplots)
            for i, (cond, cond_df) in enumerate(sid_df.groupby('cond')):

                # Define axes where to plot each epoch_types
                start_ax_idx = int(i) * n_cols
                end_ax_idx = start_ax_idx + n_cols
                epo_axes = axes if sup_cond else axes[start_ax_idx:end_ax_idx]

                # Plot
                show_xlabel = True if sup_cond else i == n_rows-1
                plot_kwargs: dict = dict(  # define other plot args
                    epo_title=i == 0,
                    vlim=vlim,
                )
                if sup_cond:
                    # Add label
                    plot_kwargs.update(label=cond, color=get_cond_palette().get(cond))

                plot_tfr_by_epo(cond_df, pkind, epo_axes=epo_axes, show_xlabel=show_xlabel, **plot_kwargs)

                # Customize axes for additional title
                if not sup_cond:  # just leave existing titles (epoch tpes)
                    if n_epo_types == 1:
                        # Replace current title with condition and add epoch type as upper title
                        add_title = epo_axes[0].get_title()
                        epo_axes[0].set_title(f"Cond {cond}")
                    else:
                        # If there are multiple epoch-type, add condition as additional title for each row (if not superimposed)
                        add_title = f"Cond {cond}"
                    add_higher_title_text(fig, epo_axes, add_title)

            if save:
                fname = f'{sid}_etypes_{pkind}.png'
                save_figure(save_dir=str(sid), group_parent_dir='plots/TFR', fname=fname, fig=fig, sid=str(sid))
            if show:
                plt.show()
            plt.close()


def iter_plot_group_tfr(
        tfr_df: pd.DataFrame,
        pkind: str = 'tfr',
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():
        sup_cond = pkind == 'spectrum'  # superimpose power spectra of different conditions, use different subplots for topomaps/TFRs

        n_conds = len(tfr_df['cond'].unique())
        n_epo_types = len(tfr_df['epo_type'].unique())
        n_groups = len(tfr_df['group'].unique())

        ncols = n_epo_types
        nrows = n_groups if sup_cond else n_conds * n_groups
        fig_height = nrows * 3.5
        fig_width = n_epo_types * 4.0
        plots_per_group = n_epo_types if sup_cond else n_conds * n_epo_types

        fig, axes = plt.subplots(
            nrows, ncols, sharey=True, sharex=True, figsize=(fig_width, fig_height),
            squeeze=False  # does not flatten automatically if 1D
        )
        axes = axes.flatten()

        for i_g, (group, group_df) in enumerate(tfr_df.groupby('group')):

            # Define limits colorbar commonly across conds/epoch-types, for easier comparison within the figure
            vlim = _compute_tfr_vlim(group_df['tfr'].values, pkind)

            # Each stimulating condition has a row (of subplots); superimposed on the same axes for spectra
            for i_c, (cond, cond_df) in enumerate(group_df.groupby('cond')):
                start_ax_idx = int(i_g) * plots_per_group if sup_cond else (int(i_c) * n_epo_types) + (int(i_g) * plots_per_group)
                end_ax_idx = start_ax_idx + n_epo_types
                epo_axes = axes[start_ax_idx:end_ax_idx]

                show_xlabel = True if sup_cond else i_c == n_conds - 1
                plot_kwargs: dict = dict(
                    epo_title=i_c == 0,
                    vlim=vlim,
                )
                if sup_cond:
                    plot_kwargs.update(label=cond, color=get_cond_palette().get(cond))

                plot_tfr_by_epo(cond_df, pkind, epo_axes=epo_axes, show_xlabel=show_xlabel, **plot_kwargs)

                if not sup_cond:
                    title = f"Cond {cond}"
                    add_higher_title_text(fig, epo_axes, title)

            if save:
                fname = f'group{group}_etypes_{pkind}.png'
                save_figure(group_parent_dir='plots/TFR', fname=fname, fig=fig, save_dir=f'WP73{group}')
            if show:
                plt.show()
            plt.close()
