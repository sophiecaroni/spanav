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
    # Load subject TFRs df with channels aggregated and plot heatmaps and spectra
    sid_df_ch_avg = get_sid_level_tfr_df(test=test, load=True, save=False, average_channels=True)
    all_sid_tfr_plots(sid_df_ch_avg, 'heatmap', show, save)
    all_sid_tfr_plots(sid_df_ch_avg, 'spectrum', show, save)

    # Load subject TFRs df with unaggregated channels and plot topomaps
    sid_df_ch_all = get_sid_level_tfr_df(test=test, load=True, save=False, average_channels=False)
    all_sid_tfr_plots(sid_df_ch_all, 'topomap', show, save)


def plot_each_group_tfr(test: bool, show: bool, save: bool) -> None:
    # Load group TFRs df with channels aggregated and plot heatmaps and spectra
    group_df_ch_avg = get_group_level_tfr_df(load=True, test=test, save=False, average_channels=True)
    all_group_tfr_plots(group_df_ch_avg, 'heatmap', show, save)
    all_group_tfr_plots(group_df_ch_avg, 'spectrum', show, save)

    # Load group TFRs df with unaggregated channels and plot topomaps
    group_df_ch_all = get_group_level_tfr_df(load=True, test=test, save=False, average_channels=False)
    all_group_tfr_plots(group_df_ch_all, 'topomap', show, save)


def run_tfr_plots(**kwargs):
    plot_each_sid_tfr(**kwargs)
    plot_each_group_tfr(**kwargs)


if __name__ == '__main__':
    run_tfr_plots(
        test=False,
        show=True,
        save=False,
    )
