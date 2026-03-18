"""
    Title: Spectral processing pipeline

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script performs spectral processing (power spectra, oscillations, TFR) of EEG.
"""
from spanav_tbi.processing.psd import get_epo_level_psd_df, get_sid_level_psd_df, get_group_level_psd_df
from spanav_tbi.processing.osc import get_epo_level_osc_df, get_sid_level_osc_df, get_group_level_osc_df
from spanav_tbi.processing.tfr import get_epo_level_tfr_df, get_sid_level_tfr_df, get_group_level_tfr_df


def run_psd_processing(test: bool, load: bool, verbose: bool, save: bool) -> None:
    print(
        f"\n\t#### Compute PSD across different aggregation-levels and store results in tables #### "
    )
    log_scale = (True, False)

    print(
        f"\n\t#### 1) Get epoch-level PSD table ####"
    )
    for ls in log_scale:
        print(
            f"\n\t\t#### ({'Log scale' if ls else 'Linear scale'}) ####"
        )
        epo_level_psd_df = get_epo_level_psd_df(load=load, test=test, save=save, log=ls)
        if verbose:
            print(f"\t\t{epo_level_psd_df = }")

    print(
        f"\n\t#### 2) Get subject-level PSD table ####"
    )
    for ls in log_scale:
        print(
            f"\n\t\t#### ({'Log scale' if ls else 'Linear scale'}) ####"
        )

        sid_level_df = get_sid_level_psd_df(load=load, test=test, save=save, log=ls)
        if verbose:
            print(f"\t\t{sid_level_df = }")

    print(
        f"\n\t#### 3) Get group-level PSD table"
    )

    for ls in log_scale:
        print(
            f"\n\t\t#### ({'Log scale' if ls else 'Linear scale'}) ####"
        )
        group_level_df = get_group_level_psd_df(load=load, test=test, save=save, log=ls)
        if verbose:
            print(f"\t\t{group_level_df = }")


def run_osc_processing(test: bool, load: bool, verbose: bool, save: bool) -> None:
    print(
        f"\n\t#### Compute oscillatory features across different aggregation-levels and store results in tables #### "
    )
    print(
        f"\n\t\t#### 1) Get epoch-level oscillatory features table ####"
    )

    epo_level_osc_df = get_epo_level_osc_df(load=load, test=test, save=save)
    if verbose:
        print(f"\t\t{epo_level_osc_df = }")

    print(
        f"\n\t\t#### 2) Get subject-level oscillatory features table ####"
    )

    sid_level_df = get_sid_level_osc_df(load=load, test=test, save=save)
    if verbose:
        print(f"\t\t{sid_level_df = }")

    print(
        f"\n\t\t#### 3) Get group-level oscillatory features table"
    )
    group_level_df = get_group_level_osc_df(load=load, test=test, save=save)
    if verbose:
        print(f"\t\t{group_level_df = }")


def run_tfr_processing(test: bool, load: bool, verbose: bool, save: bool) -> None:
    print(
        f"\n\t#### Compute TFR across different aggregation-levels and store results in tables #### "
    )
    print(
        f"\n\t\t#### 1) Get epoch-level TFR table ####"
    )

    epo_level_tfr_df = get_epo_level_tfr_df(test=test, load=load, save=save)
    if verbose:
        print(f"{epo_level_tfr_df = }")

    print(
        f"\n\t\t#### 2) Get subject-level TFR table ####"
    )

    sid_level_tfr_df = get_sid_level_tfr_df(test=test, load=load, save=save)
    if verbose:
        print(f"\t\t{sid_level_tfr_df = }")

    print(
        f"\n\t\t#### 3) Get group-level TFR table"
    )
    group_level_tfr_df = get_group_level_tfr_df(test=test, load=load, save=save)
    if verbose:
        print(f"\t\t{group_level_tfr_df = }")


def run_spectral_processing(**kwargs) -> None:

    run_psd_processing(**kwargs)
    run_osc_processing(**kwargs)
    run_tfr_processing(**kwargs)


if __name__ == '__main__':
    run_spectral_processing(
        test=False,
        load=True,
        verbose=True,
        save=False,
    )

