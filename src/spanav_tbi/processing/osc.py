"""
    Title: Oscillations

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains functions for computing and storing oscillatory features of EEG.
"""
import warnings
import numpy as np
import pandas as pd
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.spectral_utils as spct
import spanav_eeg_utils.comp_utils as cmp

from spanav_tbi.processing.psd import get_epo_level_psd_df
from fooof.analysis import get_band_peak_fm

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
        file_path = io.get_tables_path() / 'osc_df_epo_level.csv'
        return pd.read_csv(file_path, index_col=0, dtype={'sid': str})  # make sure subject ID's are strings

    # Compute spectra in the linear space
    psd_df = get_epo_level_psd_df(load=load, save=save, test=test, space='lin')
    bands = ['theta']
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

            # Model PSD
            psd_model = spct.model_psd(epo_psd, freqs, max_n_peaks=3)  # limit max_n_peaks to our relevant canonical bands

            for band in bands:
                # Detect peaks
                band_freqs = spct.get_band_freqs(band)
                pk_pw = get_band_peak_fm(psd_model, band_freqs, select_highest=True)[1]  # select power of the highest peak
                pk = False if np.isnan(pk_pw) else True

                # Define a row of the df
                row = dict(
                    sid=sid,
                    group=prs.get_group_letter(sid),
                    cond=cond,
                    epo_type=epo_type,
                    epo_n=epoch_idx,
                    band=band,
                    abs_pw=spct.get_band_power(epo_psd, freqs, band, rel=False),  # Compute abs power
                    rel_pw=spct.get_band_power(epo_psd, freqs, band, rel=True),  # Compute rel power
                    osc_snr=spct.compute_osc_snr(psd_model, band),  # Compute oscillatory SNR
                    pk=pk,
                    pk_pw=pk_pw,
                )

                df_rows.append(row)

    # Create df
    epo_level_df = pd.DataFrame(df_rows)
    epo_level_df['sid'] = epo_level_df['sid'].astype('str')
    epo_level_df.sort_values(by=['sid', 'cond'])  # sort conveniently

    if save:
        file_path = io.get_tables_path() / 'osc_df_epo_level.csv'
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
    group_cols = ['sid', 'group', 'cond', 'epo_type', 'band']
    grouped_df = epo_level_df.groupby(group_cols, as_index=False)
    sid_level_df = grouped_df.agg(
        abs_pw_avg=('abs_pw', 'mean'),
        abs_pw_std=('abs_pw', 'std'),
        rel_pw_avg=('rel_pw', 'mean'),
        rel_pw_std=('rel_pw', 'std'),
        osc_snr_avg=('osc_snr', 'mean'),
        osc_snr_std=('osc_snr', 'std'),
        n_epochs=('sid', 'size'),
    )
    cmp.fix_std_singleton(sid_level_df, ["abs_pw_std", "rel_pw_std", "osc_snr_std"], n_col="n_epochs")  # replace with zeros the NaNs appearing as std (if there was only one row to average across)

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
        abs_pw_avg=('abs_pw_avg', 'mean'),
        abs_pw_std=('abs_pw_avg', 'std'),
        rel_pw_avg=('rel_pw_avg', 'mean'),
        rel_pw_std=('rel_pw_avg', 'std'),
        osc_snr_avg=('osc_snr_avg', 'mean'),
        osc_snr_std=('osc_snr_avg', 'std'),
        n_sids=('group', 'size'),
    )
    cmp.fix_std_singleton(group_level_df, ["abs_pw_std", "rel_pw_std", "osc_snr_std"], n_col="n_sids")  # replace with zeros the NaNs appearing as std (if there was only one row to average across)

    if save:
        fname = 'osc_df_group_level.csv'
        file_path = io.get_tables_path() / fname
        group_level_df.to_csv(file_path)

    return group_level_df
