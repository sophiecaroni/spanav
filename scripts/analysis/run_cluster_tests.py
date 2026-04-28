"""
    Title: TFR and PSD cluster permutation tests

    Author: Sophie Caroni
    Date of creation: 22.04.2026

    Description:
    This script performs mass-univariate cluster-based permutation tests on channel-averaged TFR and PSD
    data, separately per subjects group.
"""
from spanav_tbi.processing.tfr import get_sid_level_tfr_df
from spanav_tbi.processing.psd import get_sid_level_psd_df
from spanav_tbi.analysis.cluster_tests import (
    run_cluster_test_tfr,
    run_cluster_test_psd,
    format_cluster_test_results,
    save_cluster_test_results,
)


def run_cluster_tests(
        dev: bool = False,
        verbose: bool = False,
        save: bool = False,
        **kwargs,
) -> None:
    """
    Run spectral cluster permutation tests on channel-averaged TFR and PSD data, separately per group.
    :param dev: bool, if True loads test data and reduces n_permutations to 10.
    :param verbose: bool, whether to print a summary of results per group.
    :param save: bool, whether to save results to disk.
    :param kwargs: forwarded to run_cluster_test_tfr and run_cluster_test_psd (e.g. effects).
    """
    kwargs.setdefault('n_permutations', 10 if dev else 1000)

    tfr_df = get_sid_level_tfr_df(test=dev, save=False, load=True, average_channels=True)
    psd_df = get_sid_level_psd_df(test=dev, save=False, load=True, average_channels=True)
    tfr_df = tfr_df[tfr_df['epo_type'].isin(TFR_EPO_TYPES)]
    psd_df = psd_df[psd_df['epo_type'].isin(PSD_EPO_TYPES)]

    spct_objects = [
        ('tfr', tfr_df, run_cluster_test_tfr),
        ('psd', psd_df, run_cluster_test_psd),
    ]

    for group in tfr_df['group'].unique():
        for spct_obj, df, run_fn in spct_objects:
            group_df = df[df['group'] == group]
            results, included_sids = run_fn(group_df, factor_cols=FACTORS, effects=EFFECTS, **kwargs)

            if verbose:
                print(format_cluster_test_results(spct_obj, results, group, included_sids))

            if save:
                save_cluster_test_results(group, spct_obj, results, included_sids)


if __name__ == '__main__':
    EFFECTS = ['A']  # main effect of the condition
    TFR_EPO_TYPES = ['blObjPres_wide', 'blMovOn_wide', 'blContMov_wide']
    PSD_EPO_TYPES = ['blObjPres', 'blMovOn', 'blContMov']
    FACTORS = ['cond', 'epo_type']

    run_cluster_tests(
        dev=True,
        verbose=True,
        save=True,
    )
