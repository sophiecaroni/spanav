import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spanav_tbi.analysis.stat_utils import get_psd_epo_types
from spanav_tbi.processing.psd import get_group_level_psd_df
from spanav_tbi.visualization.psd_plots import plot_psd_subplots
from spanav_eeg_utils.plot_utils import plot_context, get_cond_palette, get_epo_palette, save_figure


def _prepare_df_for_psd_cluster_plot(df: pd.DataFrame, effect: str) -> pd.DataFrame:
    df = df.copy()
    df = df[df['epo_type'].isin(get_psd_epo_types())]
    df['effect'] = effect
    if '×' in effect:  # epoch-type by condition interaction
        df['cond'] = df.apply(lambda r: f"{r['cond']}_{r['epo_type']}", axis=1)
    return df


def plot_psd_cluster_freqs(test_results, show: bool = True, save: bool = False):
    group_df = get_group_level_psd_df(load=True, test=False, save=False, ch_avg=True)
    for effect, effect_results in test_results.items():
        plot_df = _prepare_df_for_psd_cluster_plot(group_df, effect)
        all_psd_cluster_plots(
            plot_df, effect=effect, pkind='spectrum', sign_mask=effect_results['significant'],
            plots_subdir='PSD', show=show, save=save
        )


def all_psd_cluster_plots(
        df: pd.DataFrame,
        plots_subdir: str,
        pkind: str,
        effect: str,
        sign_mask: np.ndarray | None = None,
        show: bool = True,
        save: bool = False,
) -> None:
    with plot_context():
        for group, group_df in df.groupby('group'):
            fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(4.0, 3.5))
            superimp_col = 'cond' if effect.startswith('cond') else 'epo_type'
            for i, (superimp_cat, superimp_df) in enumerate(group_df.groupby(superimp_col)):
                color = get_cond_palette().get(superimp_cat) if superimp_col == 'cond' else get_epo_palette().get(superimp_cat)
                plot_kwargs = dict(show_ax_titles=False, vlim=None, label=superimp_cat, color=color, show_legend=False)
                plot_psd_subplots(superimp_df, pkind, subplot_col='effect', axes=[ax], **plot_kwargs, linewidth=1)

            ax.fill_between(
                group_df.copy().reset_index().loc[0, 'psd'].freqs, *ax.get_ylim(), where=sign_mask.flatten(), color='grey', alpha=0.2,
                label='Clusters', edgecolor=None)

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.figure.legend(by_label.values(), by_label.keys())
            if save:
                test_type = 'freqs' if pkind == 'spectrum' else 'sensors'
                fname = f'group{group}_cbpt_{effect}_{test_type}_TEST.png'
                save_figure(group_parent_dir=f'plots/{plots_subdir}', fname=fname, fig=fig, save_dir=f'WP73{group}')
            if show:
                plt.show()
            plt.close()
