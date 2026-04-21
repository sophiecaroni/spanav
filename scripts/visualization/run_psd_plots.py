"""
    Title: Power spectral density (PSD) plots

    Author: Sophie Caroni
    Date of creation: 21.04.2026

    Description:
    This script generates various PSD plots types for different levels of data aggregation.
"""
import warnings
from spanav_tbi.processing.psd import get_sid_level_psd_df, get_group_level_psd_df
from spanav_tbi.visualization.psd_plots import iter_plot_sid_psd, iter_plot_group_psd


def plot_each_sid_psd(test: bool, show: bool, save: bool) -> None:
    sid_level_df = get_sid_level_psd_df(test=test, load=True, save=False)
    bl_df = sid_level_df[sid_level_df['epo_type'].str.startswith('bl')]
    if not bl_df.empty:
        iter_plot_sid_psd(bl_df, 'spectrum', show, save)
        iter_plot_sid_psd(bl_df, 'topomap', show, save)
    else:
        warnings.warn('No bl-corrected epoch types in subject-level df!')


def plot_each_group_psd(test: bool, show: bool, save: bool) -> None:
    group_level_df = get_group_level_psd_df(load=True, test=test, save=False)

    # Only plot bl-corrected
    bl_df = group_level_df[group_level_df['epo_type'].str.startswith('bl')]
    if not bl_df.empty:
        iter_plot_group_psd(bl_df, show, save)
    else:
        warnings.warn('No bl-corrected epoch types in group-level df!')


def run_psd_plots(**kwargs):
    plot_each_sid_psd(**kwargs)
    plot_each_group_psd(**kwargs)


if __name__ == '__main__':
    run_psd_plots(
        test=False,
        show=True,
        save=False,
    )
