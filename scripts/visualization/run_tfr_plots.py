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
    """Plot heatmaps and spectra (channel-aggregated) and topomaps for each subject."""
    sid_df = get_sid_level_tfr_df(test=test, load=True, save=False)
    all_sid_tfr_plots(sid_df, 'heatmap', show, save)
    all_sid_tfr_plots(sid_df, 'spectrum', show, save)
    all_sid_tfr_plots(sid_df, 'topomap', show, save)


def plot_each_group_tfr(test: bool, show: bool, save: bool) -> None:
    """Plot heatmaps and spectra (channel-aggregated) and topomaps for each group."""
    group_df = get_group_level_tfr_df(load=True, test=test, save=False)
    all_group_tfr_plots(group_df, 'heatmap', show, save)
    all_group_tfr_plots(group_df, 'spectrum', show, save)
    all_group_tfr_plots(group_df, 'topomap', show, save)


def run_tfr_plots(**kwargs):
    plot_each_sid_tfr(**kwargs)
    plot_each_group_tfr(**kwargs)


if __name__ == '__main__':
    run_tfr_plots(
        test=False,
        show=True,
        save=False,
    )
