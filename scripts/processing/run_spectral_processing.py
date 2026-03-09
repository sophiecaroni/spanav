from spanav_tbi.processing.psd import get_epo_level_psd_df, get_sid_level_psd_df, get_group_level_psd_df
from spanav_tbi.processing.osc import get_epo_level_osc_df, get_sid_level_osc_df, get_group_level_osc_df
# from spanav_tbi.processing.tfr import get_tfr_df, get_agg_tfr_df


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
            print(epo_level_psd_df)

    print(
        f"\n\t#### 2) Get subject-level PSD table ####"
    )
    for ls in log_scale:
        print(
            f"\n\t\t#### ({'Log scale' if ls else 'Linear scale'}) ####"
        )

        sid_level_df = get_sid_level_psd_df(load=load, test=test, save=save, log=ls)
        if verbose:
            print(sid_level_df)

    print(
        f"\n\t#### 3) Get group-level PSD table"
    )

    for ls in log_scale:
        print(
            f"\n\t\t#### ({'Log scale' if ls else 'Linear scale'}) ####"
        )
        group_level_df = get_group_level_psd_df(load=load, test=test, save=save, log=ls)
        if verbose:
            print(group_level_df)


def run_osc_processing(test: bool, load: bool, verbose: bool, save: bool) -> None:
    print(
        f"\n\t#### Compute oscillatory features across different aggregation-levels and store results in tables #### "
    )
    print(
        f"\n\t\t#### 1) Get epoch-level oscillatory features table ####"
    )

    epo_level_osc_df = get_epo_level_osc_df(load=load, test=test, save=save)
    if verbose:
        print(epo_level_osc_df)

    print(
        f"\n\t\t#### 2) Get subject-level oscillatory features table ####"
    )

    sid_level_df = get_sid_level_osc_df(load=load, test=test, save=save)
    if verbose:
        print(sid_level_df)

    print(
        f"\n\t\t#### 3) Get group-level oscillatory features table"
    )
    group_level_df = get_group_level_osc_df(load=load, test=test, save=save)
    if verbose:
        print(group_level_df)


def run_spectral_processing(**kwargs) -> None:

    run_psd_processing(**kwargs)
    run_osc_processing(**kwargs)


if __name__ == '__main__':
    run_spectral_processing(
        test=False,
        load=False,
        verbose=True,
        save=True,
    )

