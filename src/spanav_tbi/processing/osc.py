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

    psd_df = get_epo_level_psd_df(load=load, save=save, test=test, space='lin')  # load power spectra in linear scale
    bands = ['theta']
    df_rows = []

    # Define variables within which the PSD will be computed (for each of these there will be one)
    group_by = ['sid', 'block', 'cond', 'epo_type', 'n_epo']
    for (sid, block, cond, epo_type, n_epo), grouped_df in psd_df.groupby(group_by):

        # Average across blocks of the grouping_vars-combination condition (of the same epo_type, cond, and sid (and n_epo when not avg_across_epochs)
        mean_psd_df = (
            grouped_df
            .groupby('freq', as_index=False)['pw_avg']
            .mean()
        )

        psd = mean_psd_df['pw_avg'].to_numpy()
        freqs = mean_psd_df['freq'].to_numpy()
        psd_model = spct.model_psd(psd, freqs,
                              max_n_peaks=3)  # limit max_n_peaks bc we only care about alpha/theta (and perhaps gamma)

        for band in bands:
            # Detect peaks
            band_freqs = spct.get_band_freqs(band)
            pk_pw = get_band_peak_fm(psd_model, band_freqs, select_highest=True)[1]
            pk = False if np.isnan(pk_pw) else True

            # Define a row of the df
            row = dict(
                sid=sid,
                group=prs.get_group_letter(sid),
                cond=cond,
                block=block,
                epo_type=epo_type,
                n_epo=n_epo,
                band=band,
                abs_pw=spct.get_band_power(psd, freqs, band, rel=False),  # Compute abs power
                rel_pw=spct.get_band_power(psd, freqs, band, rel=True),  # Compute rel power
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

    # For each subject, average metrics of the same condition and epoch-type across different blocks
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

    # For each group, average metrics of the same condition and epoch-type across different subjects
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
