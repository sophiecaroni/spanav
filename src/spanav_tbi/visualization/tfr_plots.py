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
from spanav_eeg_utils.plot_utils import plot_context, save_figure, add_higher_title_text
from spanav_eeg_utils.spanav_utils import map_epo_type_labels
from typing import Iterable

TFR = EpochsTFR | AverageTFR


def _compute_tfr_vlim(tfr_array: Iterable[TFR], pkind: str) -> tuple[float, float]:
    if pkind not in ['tfr', 'topomap']:
        raise(ValueError, f'Accepted plot kinds are "tfr"and "topomap"; got {pkind = }')
    axis = 0 if pkind == 'tfr' else (1, 2)  # avergae across channels (if there are mutliple) in TFR heatmap plots; across frequency and timepoints (if there are mutliple) in topomapmap plots
    vmin = min(t.data.mean(axis=axis).min() for t in tfr_array)
    vmax = max(t.data.mean(axis=axis).max() for t in tfr_array)
    return vmin, vmax


def plot_tfr_by_epo(
        tfr_df: pd.DataFrame,
        pkind: str,
        epo_axes: plt.Axes | np.ndarray,
        epo_title: bool = True,
        xaxis_label: bool = True,
        **kwargs,
) -> None:
    with plot_context():
        for i, (epo_type, epo_type_df) in enumerate(tfr_df.groupby('epo_type')):

            assert len(epo_type_df) == 1, f'This plot is made for one TFR per epoch-type! Here got {len(epo_type_df)} for {epo_type} type'
            ax = epo_axes[i]
            epo_type_df.reset_index(inplace=True)
            tfr_plot = epo_type_df.loc[0, 'tfr']

            colorbar = int(i) + 1 == len(epo_axes)

            if pkind == 'tfr':
                # Only add colorbar to right-most subplots
                tfr_plot.plot(
                    combine='mean',  # averages across channels in case there are multiples
                    axes=ax,
                    cmap='RdBu_r',  # set because default switches between two
                    colorbar=colorbar,
                    **kwargs
                )

            else:  # plot topo maps
                tfr_plot.plot_topomap(
                    axes=ax,
                    cmap='RdBu_r',  # set because default switches between two
                    colorbar=colorbar,
                    **kwargs,
                )
                # print(
                #     f"\t{fig.axes[0].collections[0].get_clim() = }"
                # )

            # Further axis customization
            if not i == 0:
                # Remove automatically set y-axis label if not left-most plot
                ax.set_ylabel('')

            if epo_title:
                # Put epoch-type as title
                epo_type_lbl = map_epo_type_labels().get(epo_type, epo_type)
                ax.set_title(epo_type_lbl)
            if not xaxis_label:
                # Remove automatically set x-axis label
                ax.set_xlabel('')


def iter_plot_sid_tfr(
        tfr_df: pd.DataFrame,
        pkind: str = 'tfr',
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():

        # Create one figure per subject
        for sid, sid_df in tfr_df.groupby('sid'):
            n_conds = len(sid_df['cond'].unique())
            n_epo_types = len(sid_df['epo_type'].unique())
            fig_height = n_conds * 3.5
            fig_width = n_epo_types * 4.0
            fig, axes = plt.subplots(
                n_conds, n_epo_types, sharey=True, sharex=True, figsize=(fig_width, fig_height),
                squeeze=False  # does not flatten automatically if 1D
            )
            axes = axes.flatten()

            # Define limits colorbar commonly across conds/epoch-types, for easier comparison within the figure
            vlim = _compute_tfr_vlim(sid_df['tfr'].values, pkind=pkind)

            # Each stimulating condition has a row (of subplots)
            for i, (cond, cond_df) in enumerate(sid_df.groupby('cond')):

                # Define axes where to plot each epoch_types
                start_ax_idx = int(i) * n_epo_types
                end_ax_idx = start_ax_idx + n_epo_types
                epo_axes = axes[start_ax_idx:end_ax_idx]

                # Plot
                plot_kwargs = dict(  # define other plot args
                    epo_title=i == 0,
                    xaxis_label=i == n_conds-1,
                )

                plot_tfr_by_epo(cond_df, pkind, epo_axes=epo_axes, vlim=vlim, show=False, **plot_kwargs)

                # Customize axes for additional title
                if n_epo_types == 1:
                    # If there is only one epoch-type, replace current title with condition and add epoch type as upper title
                    add_title = epo_axes[0].get_title()
                    epo_axes[0].set_title(f"Cond {cond}")
                else:
                    # If there are multiple epoch-type, add condition as additional title
                    add_title = f"Cond {cond}"
                add_higher_title_text(fig, epo_axes, add_title)

            if save:
                fname_suff = 'TFR' if pkind == 'tfr' else 'topomap'
                fname = f'{sid}_etypes_{fname_suff}.png'
                save_figure(group_parent_dir='plots/TFR', fname=fname, fig=fig, sid=str(sid), save_dir=None)
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
        n_conds = len(tfr_df['cond'].unique())
        n_epo_types = len(tfr_df['epo_type'].unique())
        n_groups = len(tfr_df['group'].unique())

        ncols = n_epo_types
        nrows = n_conds * n_groups
        fig_height = n_conds * 3.5
        fig_width = n_epo_types * 4.0

        fig, axes = plt.subplots(
            nrows, ncols, sharey=True, sharex=True, figsize=(fig_width, fig_height),
            squeeze=False  # does not flatten automatically if 1D
        )
        axes = axes.flatten()
        plots_per_group = nrows * ncols // 2

        for i_g, (group, group_df) in enumerate(tfr_df.groupby('group')):

            # Define limits colorbar commonly across conds/epoch-types, for easier comparison within the figure
            vlim = _compute_tfr_vlim(group_df['tfr'].values, pkind)

            # Each stimulating condition has a row (of subplots)
            for i_c, (cond, cond_df) in enumerate(group_df.groupby('cond')):
                # Define axes where to plot each epoch_types
                start_ax_idx = (int(i_c) * n_epo_types) + (int(i_g) * plots_per_group)
                end_ax_idx = start_ax_idx + n_epo_types
                epo_axes = axes[start_ax_idx:end_ax_idx]

                # Call plot
                plot_kwargs = dict(
                    epo_title=i_c == 0,
                    xaxis_label=i_c == n_conds-1,
                )
                plot_tfr_by_epo(cond_df, pkind, epo_axes=epo_axes, vlim=vlim, show=False, **plot_kwargs)
                title = f"Cond {cond}"
                add_higher_title_text(fig, epo_axes, title)

            if save:
                fname_suff = 'TFR' if pkind == 'tfr' else 'topomap'
                fname = f'group{group}_etypes_{fname_suff}.png'
                save_figure(group_parent_dir='plots/TFR', fname=fname, fig=fig, save_dir=f'WP73{group}')
            if show:
                plt.show()
            plt.close()
