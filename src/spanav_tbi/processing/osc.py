"""
    Title: Oscillations

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains functions for computing and storing oscillatory features of EEG.
"""
import warnings
import pandas as pd
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.spectral_utils as spct
import spanav_eeg_utils.comp_utils as cmp
from spanav_tbi.processing.psd import get_epo_level_psd_df

# Suppress MNE filename convention warning
warnings.filterwarnings(
    "ignore",
    message=r".*does not conform to MNE naming conventions.*",
    category=RuntimeWarning,
)


def get_epo_level_osc_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
) -> pd.DataFrame:
    if load:
        file_path = io.get_tables_path() / f'osc_df_epo_level.csv'
        return pd.read_csv(file_path, index_col=0, dtype={'sid': str})  # make sure subject ID's are strings

    # Get spectra, always in linear space (needed for osc-computations)
    psd_df = get_epo_level_psd_df(test=test, space='lin')
    df_rows = []

    # Define grouping variables for computing the oscillatory features
    group_by = ['sid', 'cond', 'epo_type']
    for (sid, cond, epo_type), grouped_df in psd_df.groupby(group_by):
        if len(grouped_df) > 1:
            raise ValueError(
                f'Should have one PSD per subject, condition and epoch-type, got {len(grouped_df)} for {sid = }, {cond = }, {epo_type = }'
                f'\n\t{grouped_df = }')

        # Select PSD of frontal channels
        epos_psd = grouped_df.copy().reset_index().loc[0, 'psd']
        frontal_chs = [ch for ch in epos_psd.ch_names if ch.startswith('F')]
        frontal_psd = epos_psd.copy().pick(frontal_chs)
        freqs = epos_psd.freqs

        # Iterate over epochs to compute one observation of oscillatory feature each
        for epoch_idx in range(len(frontal_psd)):
            epo_psd = frontal_psd._data[epoch_idx].mean(axis=0)  # average across channels
            if epo_psd.shape != freqs.shape:
                raise ValueError(
                    f'PSD and freqs should have the same shape, got {epo_psd.sape} PSD and {freqs.shape} freqs'
                    f'\n\t{grouped_df = }')

            for band in ['theta', 'alpha']:

                # Extract absolute and relative band power
                abs_pw_log = spct.get_band_power(epo_psd, freqs, band, rel=False, space='log')   # log space to improve normality of feature distribution
                rel_pw_log = spct.get_band_power(epo_psd, freqs, band, rel=True, space='log')   # log space to improve normality of feature distribution

                # Model PSD
                psd_model = spct.model_psd(epo_psd, freqs, max_n_peaks=3)  # limit max_n_peaks to our relevant canonical bands

                # Extract FOOOF oscillatory SNR
                osc_snr = spct.compute_osc_snr(psd_model, band)

                # Extract power of modeled peaks in the band (if any - otherwise will be nan)
                pk_pw_log = spct.get_modeled_peak_power(psd_model, band, space='log')  # log space to improve normality of feature distribution

                row = dict(
                    sid=sid,
                    group=prs.get_group_letter(sid),
                    cond=cond,
                    epo_type=epo_type,
                    epo_n=epoch_idx,
                    band=band,
                    abs_pw_log=abs_pw_log,
                    rel_pw_log=rel_pw_log,
                    osc_snr=osc_snr,
                    pk_pw_log=pk_pw_log,
                )

                df_rows.append(row)

    # Create df
    epo_level_df = pd.DataFrame(df_rows)
    epo_level_df['sid'] = epo_level_df['sid'].astype('str')
    epo_level_df.sort_values(by=['sid', 'cond'])  # sort conveniently

    if save:
        file_path = io.get_tables_path() / f'osc_df_epo_level.csv'
        epo_level_df.to_csv(file_path)

    return epo_level_df


def get_sid_level_osc_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
) -> pd.DataFrame:
    if load:
        file_path = io.get_tables_path() / 'osc_df_sid_level.csv'
        return pd.read_csv(file_path, index_col=0, dtype={'sid': str})  # make sure subject ID's are strings

    # Load epoch-level PSD dataframe
    epo_level_df = get_epo_level_osc_df(load=True, test=test, save=False)

    # Average across epochs of the same group_cols values
    group_cols = ['sid', 'cond', 'epo_type', 'band']
    grouped_df = epo_level_df.groupby(group_cols, as_index=False)
    sid_level_df = grouped_df.agg(
        group=('group', 'first'),
        abs_pw_log_avg=('abs_pw_log', 'mean'),
        abs_pw_log_std=('abs_pw_log', 'std'),
        rel_pw_log_avg=('rel_pw_log', 'mean'),
        rel_pw_log_std=('rel_pw_log', 'std'),
        osc_snr_avg=('osc_snr', 'mean'),
        osc_snr_std=('osc_snr', 'std'),
        n_epochs=('sid', 'size'),
        pk_pw_log_avg=('pk_pw_log', 'mean'),  # mean peak power among epochs with a peak - nans (no peaks) are skipped
        pk_pw_log_std=('pk_pw_log', 'std'),
        n_peak_epochs=('pk_pw_log', 'count'),  # track count of not nan epochs, to quantify presence of peaks
    )

    # Replace with zeros the NaNs introduced as std if there was only one row to average across
    cmp.fix_std_singleton(
        sid_level_df, ["abs_pw_log_std",  "rel_pw_log_std", "osc_snr_std"],
        n_col="n_epochs"
    )
    cmp.fix_std_singleton(
        sid_level_df, ["pk_pw_log_std"],
        n_col="n_peak_epochs"  # for peak power this correction has to be considering only the epochs having a peak
    )

    if save:
        fname = 'osc_df_sid_level.csv'
        file_path = io.get_tables_path() / fname
        sid_level_df.to_csv(file_path)

    return sid_level_df


def get_group_level_osc_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
) -> pd.DataFrame:
    if load:
        file_path = io.get_tables_path() / 'osc_df_group_level.csv'
        return pd.read_csv(file_path, index_col=0, dtype={'sid': str})  # make sure subject ID's are strings

    # Load subject-level PSD dataframe
    sid_level_df = get_sid_level_osc_df(load=True, test=test, save=False)

    # Average across subject spectra of the same group_cols values
    group_cols = ['group', 'cond', 'epo_type', 'band']
    group_level_df = sid_level_df.groupby(group_cols, as_index=False).agg(

        abs_pw_log_avg=('abs_pw_log_avg', 'mean'),
        abs_pw_log_std=('abs_pw_log_avg', 'std'),
        rel_pw_log_avg=('rel_pw_log_avg', 'mean'),
        rel_pw_log_std=('rel_pw_log_avg', 'std'),
        osc_snr_avg=('osc_snr_avg', 'mean'),
        osc_snr_std=('osc_snr_avg', 'std'),
        n_sids=('group', 'size'),
        pk_pw_log_avg=('pk_pw_log_avg', 'mean'),
        pk_pw_log_std=('pk_pw_log_avg', 'std'),
        n_sids_with_peak=('pk_pw_log_avg', 'count'),  # sids contributing a peak value
        n_peak_epochs=('n_peak_epochs', 'sum'),  # total of not nan epochs across sids, to quantify overall presence of peaks
    )

    # Replace with zeros the NaNs appearing as std (if there was only one row to average across)
    cmp.fix_std_singleton(
        group_level_df, ["abs_pw_log_std", "rel_pw_log_std", "osc_snr_std"],
        n_col="n_sids"
    )
    cmp.fix_std_singleton(
        sid_level_df, ["pk_pw_log_std"],
        n_col="n_peak_epochs"  # for peak power this correction has to be considering only the epochs having a peak
    )

    if save:
        fname = 'osc_df_group_level.csv'
        file_path = io.get_tables_path() / fname
        group_level_df.to_csv(file_path)

    return group_level_df
