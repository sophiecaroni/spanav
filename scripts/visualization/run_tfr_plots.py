"""
********************************************************************************
    Title: Spectrogram plots of Time-frequency Representations (TFR)

    Author: Sophie Caroni
    Date of creation: 16.03.2026

    Description:
    This script generate spectrograms for different aggregation levels of TFR objects.
********************************************************************************
"""
from spanav_tbi.processing.tfr import get_sid_level_tfr_df, get_group_level_tfr_df
from spanav_tbi.visualization.tfr_plots import iter_plot_sid_tfr, iter_plot_group_tfr


def plot_each_sid_tfr(test: bool, show: bool, save: bool) -> None:
    # Load df storing TFR at subject level
    sid_level_df = get_sid_level_tfr_df(test=test, load=True, save=False)

    # Only plot stasis-bl corrected epoch-types
    bl_mask = sid_level_df['epo_type'].str.startswith('bl')
    sid_level_df_bl = sid_level_df[bl_mask]
    iter_plot_sid_tfr(sid_level_df_bl, 'tfr', show, save)
    iter_plot_sid_tfr(sid_level_df_bl, 'topomap', show, save)


def plot_each_group_tfr(test: bool, show: bool, save: bool) -> None:
    # Load df storing TFR at group level
    group_level_df = get_group_level_tfr_df(load=True, test=test, save=False)

    # Only plot stasis-bl corrected epoch-types
    bl_mask = group_level_df['epo_type'].str.startswith('bl')
    group_level_df_bl = group_level_df[bl_mask]
    iter_plot_group_tfr(group_level_df_bl, 'tfr', show, save)
    iter_plot_group_tfr(group_level_df_bl, 'topomap', show, save)


def run_tfr_plots(**kwargs):
    plot_each_sid_tfr(**kwargs)
    plot_each_group_tfr(**kwargs)


if __name__ == '__main__':
    run_tfr_plots(
        test=False,
        show=False,
        save=True,
    )
