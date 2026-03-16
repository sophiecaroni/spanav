"""
********************************************************************************
    Title: Time-frequency Representation (TFR) plots

    Author: Sophie Caroni
    Date of creation: 16.03.2026

    Description:
    This script generate figures for different aggregation levels of TFR objects.
********************************************************************************
"""
from spanav_tbi.processing.tfr import get_sid_level_tfr_df, get_group_level_tfr_df
from spanav_tbi.visualization.tfr_plots import iter_plot_sid_tfr, iter_plot_group_tfr


def plot_each_sid_tfr(test: bool, show: bool, save: bool) -> None:
    # Load df storing TFR at subject level
    sid_level_df = get_sid_level_tfr_df(test=test, save=False)

    # Plot blMovOn epochs in a single figure
    sid_level_df_bl = sid_level_df[sid_level_df['epo_type'] == 'blMovOn']
    iter_plot_sid_tfr(sid_level_df_bl, show, save)

    # Plot all the other epoch-types
    sid_level_df_nobl = sid_level_df[sid_level_df['epo_type'] != 'blMovOn']
    iter_plot_sid_tfr(sid_level_df_nobl, show, save)


def plot_each_group_tfr(test: bool, show: bool, save: bool) -> None:
    # Load df storing TFR at group level
    group_level_df = get_group_level_tfr_df(test=test, save=False)

    # Plot blMovOn epochs in a single figure
    group_level_df_bl = group_level_df[group_level_df['epo_type'] == 'blMovOn']
    iter_plot_group_tfr(group_level_df_bl, show, save)

    # Plot all the other epoch-types
    group_level_df_nobl = group_level_df[group_level_df['epo_type'] != 'blMovOn']
    iter_plot_group_tfr(group_level_df_nobl, show, save)


def run_tfr_plots(**kwargs):
    plot_each_sid_tfr(**kwargs)
    plot_each_group_tfr(**kwargs)


if __name__ == '__main__':
    run_tfr_plots(
        test=False,
        show=False,
        save=True,
    )
