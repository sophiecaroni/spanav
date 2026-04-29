"""
    Title: TFR and PSD cluster permutation tests

    Author: Sophie Caroni
    Date of creation: 22.04.2026

    Description:
    This script performs mass-univariate cluster-based permutation tests on channel-averaged TFR and PSD
    data, separately per subjects group.
"""
from spanav_tbi.analysis.cluster_tests import (
    run_cluster_test_tfr,
    run_cluster_test_psd,
    run_psd_ch_cluster_test
)
from spanav_tbi.analysis.stat_utils import format_cluster_test_results, save_cluster_test_results
from spanav_eeg_utils.io_utils import get_groups_letters


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
    kwargs.setdefault('n_permutations', 10 if dev else 1024)

    spct_objects = [
        ('tfr', run_cluster_test_tfr),
        ('psd', run_cluster_test_psd),
        ('ch_psd', run_psd_ch_cluster_test),
    ]

    for group in get_groups_letters():
        for spct_obj, run_fn in spct_objects:
            results, included_sids = run_fn(group, factor_cols=FACTORS, effects=EFFECTS, **kwargs)

            if verbose:
                print(format_cluster_test_results(spct_obj, results, group, included_sids))

            if save:
                save_cluster_test_results(group, spct_obj, results, included_sids)


if __name__ == '__main__':
    EFFECTS = ['A']  # main effect of the condition
    FACTORS = ['cond', 'epo_type']

    run_cluster_tests(
        dev=False,
        verbose=True,
        save=False,
    )
