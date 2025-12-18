"""
********************************************************************************
    Title: Processing EEG

    Author: Sophie Caroni
    Date of creation: 02.10.2025

    Description:
    This script contains functions for processing EEG data.
********************************************************************************
"""
import mne
import numpy as np
import pandas as pd
import os

from utils.spectral_utils import compute_psd, get_band_power, model_psd, compute_osc_snr, get_band_freqs
from utils.gen_utils import get_sids, set_for_save, get_wd, parse_epo_filename, parse_prepro_filename, \
    get_exp_phase
from fooof.analysis import get_band_peak_fm


def compute_avg_epo_psd(
        rec: mne.Epochs,
        fmin: float = 0.0,
        fmax: float = np.inf,
        test: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param rec:
    :param test:
    :param fmin:
    :param fmax: default takes all frequencies (up to Nyquist frequency)
    :param plot:
    :return: PSD
    """
    ch_psd, freqs = compute_psd(rec, log_space=True, fmin=fmin, fmax=fmax, test=test)
    rec_psd_avg = np.mean(np.mean(ch_psd, axis=1), axis=0)
    rec_psd_std = np.mean(np.std(ch_psd, axis=1) / np.sqrt(ch_psd.shape[1]), axis=0)
    return rec_psd_avg, rec_psd_std, freqs


def compute_psd_by_key(
        epos_dict: dict[str, mne.Epochs],
) -> dict[str, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]:
    """
    Compute PSD for each recording of a dict.
    :param epos_dict:
    :return:
    """
    psds = {}
    for epo_key, epo_rec in epos_dict.items():
        if epo_rec is None or len(epo_rec) == 0:
            psds[epo_key] = (None, None, None)
        else:
            fmax = epo_rec.info['lowpass']
            psds[epo_key] = compute_avg_epo_psd(epo_rec, fmax=fmax)
    return psds


def get_psd_df(
        log: bool = False,
        test: bool = False,
        load: bool = True,
        save: bool = False,
        segmented_epochs: bool = False,
) -> pd.DataFrame:
    if load:
        file_pref = 'SEG_' if segmented_epochs else ''
        file_name = f'{file_pref}psd_df_log.csv' if log else f'{file_pref}psd_df_lin.csv'
        # file_name = f'psd_df_log_WITH04.csv' if log else f'psd_df_lin_WITH04.csv'
        psd_df = pd.read_csv(f'{get_wd()}/data/{get_exp_phase()}/{file_name}', index_col=0, dtype={'sid': str})  # make sure subject ID's are strings
    else:
        sids = get_sids(test=test, include_04=True)
        print(
            f"{sids = }"
        )
        df_rows = []
        for sid in sids:

            sid_epo_files = os.listdir(f'{get_wd()}/data/{get_exp_phase()}/{sid}/eeg/Epo')
            sid_raw_files = os.listdir(f'{get_wd()}/data/{get_exp_phase()}/{sid}/eeg/RawPreprocessed')  # Also compute metrics on full raw data
            sid_files = sid_raw_files + sid_epo_files

            for file in sid_files:
                if file.endswith('.fif'):

                    if file.startswith('RS'):
                        continue

                    if file.endswith('-epo.fif'):
                        if (file.startswith('SEG') and segmented_epochs) or (not file.startswith('SEG') and not segmented_epochs):
                            rec = mne.read_epochs(f'{get_wd()}/data/{get_exp_phase()}/{sid}/eeg/Epo/{file}', preload=True, verbose=False)
                            cond, block_n, epo_type = parse_epo_filename(file)
                        else:
                            rec = None
                    else:
                        assert file.endswith('-raw.fif')
                        rec = mne.io.read_raw_fif(f'{get_wd()}/data/{get_exp_phase()}/{sid}/eeg/RawPreprocessed/{file}', preload=True, verbose=False)
                        cid = parse_prepro_filename(file)
                        cond = cid.split('_')[-2]
                        block_n = cid.split('_')[-1]
                        epo_type = 'Raw'

                    if rec is None or len(rec) == 0:
                        continue

                    # Compute PSD in each subject, epoch and channel, and then average across them
                    fmin, fmax = 1, 45
                    # psd_kwargs = {} if file.endswith('-epo.fif') else {'n_fft': 250}
                    sfreq = int(rec.info['sfreq'])
                    psd_kwargs = dict(
                        method="welch",
                        n_fft=sfreq,
                        n_per_seg=sfreq,  # windows_length = n_per_seg / sfreq, so n_per_seg=sfreq makes 1s windows
                        n_overlap=int(sfreq/2),  # 50% overlap, common
                        window="hamming"  # common
                    )
                    psd = compute_psd(rec, fmin=fmin, fmax=fmax, verbose=False, **psd_kwargs)
                    full_psd, freqs = psd.get_data(return_freqs=True)
                    if log:
                        full_psd = np.log10(full_psd)

                    # Compute average PSD
                    if file.endswith('-epo.fif'):

                        # 1) compute average of full_psd (n_epochs, n_channels, n_freqs) across channels
                        psd_ch_mean = full_psd.mean(axis=1)  # (n_epochs, n_freqs)

                        # 2) compute mean and std of psd_ch_mean (n_epochs, n_freqs) across epochs
                        psd_avg = psd_ch_mean.mean(axis=0)  # (n_freqs,)
                        psd_std = psd_ch_mean.std(axis=0)  # (n_freqs,)
                    else:
                        assert file.endswith('-raw.fif')
                        # compute average and std of full_psd (n_channels, n_freqs)  across channels
                        psd_avg = full_psd.mean(axis=0)  # (n_freqs,)
                        psd_std = full_psd.std(axis=0)  # (n_freqs,)

                    # Define rows of the df (one row for each frequency-point of the psd
                    for freq, pw, std in zip(freqs, psd_avg, psd_std):
                        df_rows.append({
                            'sid': sid,
                            'cond': cond,
                            'block': block_n,
                            'epo_type': epo_type,
                            # 'epo_len': epo_len,
                            'freq': freq,
                            'pw_avg': pw,
                            'pw_std': std,
                        })

        psd_df = pd.DataFrame(df_rows)
        psd_df['sid'] = psd_df['sid'].astype('str')

        assert (psd_df['freq'].unique() == list(range(fmin, fmax+1))).all()

        if save:
            files_path = f'{get_wd()}/data/{get_exp_phase()}'
            file_pref = 'SEG_' if segmented_epochs else ''
            file_name = f'{file_pref}psd_df_log.csv' if log else f'{file_pref}psd_df_lin.csv'
            psd_df.to_csv(f'{set_for_save(files_path)}/{file_name}')
    return psd_df


def get_psd_avg_df(
        log: bool = False,
        load: bool = True,
        save: bool = False,
        segmented_epochs: bool = False,
) -> pd.DataFrame:
    if load:
        file_pref = 'SEG_' if segmented_epochs else ''
        file_name = f'{file_pref}psd_avg_df_log.csv' if log else f'{file_pref}psd_avg_df_lin.csv'
        # file_name = f'psd_avg_df_log_WITH04.csv' if log else f'psd_avg_df_lin_WITH04.csv'
        psd_avg_df = pd.read_csv(f'{get_wd()}/data/{get_exp_phase()}/{file_name}', index_col=0, dtype={'sid': str})  # make sure subject ID's are strings
    else:
        psd_df = get_psd_df(load=True, log=False, segmented_epochs=segmented_epochs)  # always start by linear df (to apply log afterwards)

        psd_df = psd_df[~(psd_df['sid'] == '04')]

        # For each patient, average PSD of the same condition and epoch-type across different blocks
        print(f"sids = {psd_df['sid'].unique()}")
        df_subj = psd_df.groupby(
            ['sid', 'cond', 'epo_type', 'freq'], as_index=False).agg(
            **{
                "sid": ("sid", 'first'),
                "freq": ("freq", 'first'),
                "pw_avg": ('pw_avg', 'mean'),
            }
        )

        if log:
            # Convert in log space
            df_subj['pw_avg'] = np.log10(df_subj['pw_avg'])

        # Then average PSD of the same epoch-type and condition across different patients
        psd_avg_df = df_subj.groupby(
            ['cond', 'epo_type', 'freq'], as_index=False).agg(
            **{
                "freq": ("freq", 'first'),
                "pw_avg": ('pw_avg', 'mean'),
                "pw_std": ('pw_avg', 'std'),
                "N": ('pw_avg', 'count'),
            }
        )

        raw = psd_avg_df[psd_avg_df["epo_type"] == "Raw"]
        grp_to_check = raw.groupby("cond")["N"].agg(["min", "max"])
        assert (grp_to_check['min'] == grp_to_check['max']).all()

        if save:
            files_path = f'{get_wd()}/data/{get_exp_phase()}'
            file_pref = 'SEG_' if segmented_epochs else ''
            file_name = f'{file_pref}psd_avg_df_log.csv' if log else f'{file_pref}psd_avg_df_lin.csv'
            psd_avg_df.to_csv(f'{set_for_save(files_path)}/{file_name}')
    return psd_avg_df


def get_band_metrics_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
        segmented_epochs: bool = False,
) -> pd.DataFrame:
    if load:
        file_pref = 'SEG_' if segmented_epochs else ''
        osc_df = pd.read_csv(f'{get_wd()}/data/{get_exp_phase()}/{file_pref}band_metrics_df.csv', index_col=0, dtype={'sid': str})  # make sure subject ID's are strings
        # osc_df = pd.read_csv(f'{get_wd()}/data/{get_exp_phase()}/osc_df_WITH04.csv', index_col=0, dtype={'sid': str})  # make sure subject ID's are strings
    else:
        psd_df = get_psd_df(test=test, log=False, segmented_epochs=segmented_epochs)  # load power spectra in linear scale
        bands = ['theta'] if test else ['theta', 'alpha', '38-42']
        df_rows = []

        print(f"sids = {psd_df['sid'].unique()}")
        for epo_type, single_epo_type_df in psd_df.groupby('epo_type'):
            for cond, single_cond_df in single_epo_type_df.groupby('cond'):
                for sid, single_sid_df in single_cond_df.groupby('sid'):

                    # Average first across blocks of the same condition (within each participant)
                    mean_psd_df = (
                        single_sid_df
                        .groupby('freq', as_index=False)['pw_avg']
                        .mean()
                    )

                    psd = mean_psd_df['pw_avg'].to_numpy()
                    freqs = mean_psd_df['freq'].to_numpy()
                    psd_model = model_psd(psd, freqs, max_n_peaks=3)  # limit max_n_peaks bc we only care about alpha/theta (and perhaps gamma)

                    for band in bands:
                        # Detect peak
                        band_freqs = get_band_freqs(band)
                        pk_pw = get_band_peak_fm(psd_model, band_freqs, select_highest=True)[1]
                        pk = False if np.isnan(pk_pw) else True

                        # Define rows of the df
                        df_rows.append({
                            'sid': sid,
                            'cond': cond,
                            'epo_type': epo_type,
                            'band': band,
                            'abs_pw': get_band_power(psd, freqs, band, rel=False),  # Compute abs power
                            'rel_pw': get_band_power(psd, freqs, band, rel=True),  # Compute rel power
                            'osc_snr': compute_osc_snr(psd_model, band),  # Compute oscillatory SNR
                            'pk': pk,
                            'pk_pw': pk_pw,
                        })

        osc_df = pd.DataFrame(df_rows)
        osc_df['sid'] = osc_df['sid'].astype('str')

        if save:
            files_path = f'{get_wd()}/data/{get_exp_phase()}'
            file_pref = 'SEG_' if segmented_epochs else ''
            osc_df.to_csv(f'{set_for_save(files_path)}/{file_pref}band_metrics_df.csv')
    return osc_df
