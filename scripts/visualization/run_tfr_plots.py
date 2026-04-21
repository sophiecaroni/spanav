"""
    Title: Time-frequency Representations (TFR) plots

    Author: Sophie Caroni
    Date of creation: 16.03.2026

    Description:
    This script generates various TFR plot types for different levels aggregation data.
"""
from spanav_tbi.processing.tfr import get_sid_level_tfr_df, get_group_level_tfr_df
from spanav_tbi.visualization.tfr_plots import all_sid_tfr_plots, all_group_tfr_plots


def plot_each_sid_tfr(test: bool, show: bool, save: bool) -> None:
    # Load df storing TFR at subject level
    sid_level_df = get_sid_level_tfr_df(test=test, load=True, save=False)

    # Only plot epoch types that were baseline-corrected (based on stasis) and extracted from wide windows
    mask = (sid_level_df['epo_type'].str.endswith('wide')) & (sid_level_df['epo_type'].str.startswith('bl'))
    plot_df = sid_level_df[mask]
    all_sid_tfr_plots(plot_df, 'heatmap', show, save)
    all_sid_tfr_plots(plot_df, 'topomap', show, save)
    all_sid_tfr_plots(plot_df, 'spectrum', show, save)


def plot_each_group_tfr(test: bool, show: bool, save: bool) -> None:
    # Load df storing TFR at group level
    group_level_df = get_group_level_tfr_df(load=True, test=test, save=False)

    # Only plot epoch types that were baseline-corrected (based on stasis) and extracted from wide windows
    mask = (group_level_df['epo_type'].str.endswith('wide')) & (group_level_df['epo_type'].str.startswith('bl'))
    plot_df = group_level_df[mask]
    all_group_tfr_plots(plot_df, 'heatmap', show, save)
    all_group_tfr_plots(plot_df, 'spectrum', show, save)


def run_tfr_plots(**kwargs):
    plot_each_sid_tfr(**kwargs)
    plot_each_group_tfr(**kwargs)


if __name__ == '__main__':
    run_tfr_plots(
        test=False,
        show=False,
        save=True,
    )
