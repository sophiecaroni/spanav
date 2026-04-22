"""
    Title: Power spectral density (PSD) plots

    Author: Sophie Caroni
    Date of creation: 21.04.2026

    Description:
    This script generates various PSD plots types for different levels of data aggregation.
"""
from spanav_tbi.processing.psd import get_sid_level_psd_df, get_group_level_psd_df
from spanav_tbi.visualization.psd_plots import all_sid_psd_plots, all_group_psd_plots


def plot_each_sid_psd(test: bool, show: bool, save: bool) -> None:
    # Load subject PSDs df with channels aggregated and plot heatmaps and spectra
    sid_df_ch_avg = get_sid_level_psd_df(test=test, load=True, save=False, average_channels=True)
    all_sid_psd_plots(sid_df_ch_avg, 'spectrum', show, save)

    # Load subject PSDS df with unaggregated channels and plot topomaps
    sid_df_ch_all = get_sid_level_psd_df(test=test, load=True, save=False, average_channels=False)
    all_sid_psd_plots(sid_df_ch_all, 'topomap', show, save)


def plot_each_group_psd(test: bool, show: bool, save: bool) -> None:
    # Load group PSDs df with channels aggregated and plot heatmaps and spectra
    group_df_ch_avg = get_group_level_psd_df(load=True, test=test, save=False, average_channels=True)
    all_group_psd_plots(group_df_ch_avg, 'spectrum', show, save)

    # Load group PSDs df with unaggregated channels and plot topomaps
    group_df_ch_all = get_group_level_psd_df(load=True, test=test, save=False, average_channels=False)
    all_group_psd_plots(group_df_ch_all, 'topomap', show, save)


def run_psd_plots(**kwargs):
    plot_each_sid_psd(**kwargs)
    plot_each_group_psd(**kwargs)


if __name__ == '__main__':
    run_psd_plots(
        test=False,
        show=True,
        save=False,
    )
