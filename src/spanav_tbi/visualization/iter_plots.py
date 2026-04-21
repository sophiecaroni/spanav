"""
    Title: Shared iteration helpers for spectral plots

    Author: Sophie Caroni
    Date of creation: 21.04.2026

    Description:
    Shared figure-iteration logic for subject-level and group-level spectral
    plots (PSD and TFR). Data-type-specific rendering is injected via plot_fn
    and vlim_fn callables.
"""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable
from spanav_eeg_utils.plot_utils import plot_context, save_figure, add_higher_title_text, get_cond_palette


def all_sid_plots(
        df: pd.DataFrame,
        data_col: str,
        plot_fn: Callable,
        vlim_fn: Callable,
        plots_subdir: str,
        pkind: str = 'spectrum',
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():
        sup_cond = pkind == 'spectrum'

        for sid, sid_df in df.groupby('sid'):
            n_conds = len(sid_df['cond'].unique())
            n_epo_types = len(sid_df['epo_type'].unique())
            n_rows = 1 if sup_cond else n_conds
            n_cols = n_epo_types
            fig, axes = plt.subplots(
                n_rows, n_cols, sharey=True, sharex=True,
                figsize=(n_cols * 4.0, n_rows * 3.5), squeeze=False
            )
            axes = axes.flatten()
            vlim = vlim_fn(sid_df[data_col].values, pkind=pkind)

            for i, (cond, cond_df) in enumerate(sid_df.groupby('cond')):
                epo_axes = axes if sup_cond else axes[i * n_cols: i * n_cols + n_cols]
                show_xlabel = True if sup_cond else i == n_rows - 1
                plot_kwargs: dict = dict(epo_title=i == 0, vlim=vlim)
                if sup_cond:
                    plot_kwargs.update(label=cond, color=get_cond_palette().get(cond))
                plot_fn(cond_df, pkind, epo_axes=epo_axes, show_xlabel=show_xlabel, **plot_kwargs)

                if not sup_cond and n_conds > 1:
                    if n_epo_types == 1:
                        add_title = epo_axes[0].get_title()
                        epo_axes[0].set_title(f"Cond {cond}")
                    else:
                        add_title = f"Cond {cond}"
                    add_higher_title_text(fig, epo_axes, add_title)

            if save:
                fname = f'{sid}_etypes_{pkind}.png'
                save_figure(save_dir=str(sid), group_parent_dir=f'plots/{plots_subdir}', fname=fname, fig=fig, sid=str(sid))
            if show:
                plt.show()
            plt.close()


def all_group_plots(
        df: pd.DataFrame,
        data_col: str,
        plot_fn: Callable,
        vlim_fn: Callable,
        plots_subdir: str,
        pkind: str = 'spectrum',
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():
        sup_cond = pkind == 'spectrum'

        for group, group_df in df.groupby('group'):
            n_conds = len(group_df['cond'].unique())
            n_epo_types = len(group_df['epo_type'].unique())
            n_rows = 1 if sup_cond else n_conds
            n_cols = n_epo_types
            fig, axes = plt.subplots(
                n_rows, n_cols, sharey=True, sharex=True,
                figsize=(n_cols * 4.0, n_rows * 3.5), squeeze=False
            )
            axes = axes.flatten()
            vlim = vlim_fn(group_df[data_col].values, pkind=pkind)

            for i_c, (cond, cond_df) in enumerate(group_df.groupby('cond')):
                start_ax_idx = 0 if sup_cond else i_c * n_epo_types
                epo_axes = axes[start_ax_idx: start_ax_idx + n_epo_types]
                show_xlabel = True if sup_cond else i_c == n_conds - 1
                plot_kwargs: dict = dict(epo_title=i_c == 0, vlim=vlim)
                if sup_cond:
                    plot_kwargs.update(label=cond, color=get_cond_palette().get(cond))
                plot_fn(cond_df, pkind, epo_axes=epo_axes, show_xlabel=show_xlabel, **plot_kwargs)

                if not sup_cond:
                    add_higher_title_text(fig, epo_axes, f"Cond {cond}")

            if save:
                fname = f'group{group}_etypes_{pkind}.png'
                save_figure(group_parent_dir=f'plots/{plots_subdir}', fname=fname, fig=fig, save_dir=f'WP73{group}')
            if show:
                plt.show()
            plt.close()