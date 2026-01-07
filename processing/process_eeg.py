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
from utils.gen_utils import get_pids, set_for_save, parse_epo_filename, parse_prepro_filename, get_tables_path, \
    get_eeg_path, get_clean_eeg_path
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
        load: bool = True,
        test: bool = False,
        save: bool = False,
        log: bool = False,
        avg_across_epochs: bool = True,
) -> pd.DataFrame:
    if load:
        file_name = ('psd_df_log.csv' if log else 'psd_df_lin.csv') if avg_across_epochs else (
            'psd_df_epo_log.csv' if log else 'psd_df_epo_lin.csv')
        file_path = set_for_save(get_tables_path()) / file_name
        psd_df = pd.read_csv(file_path, index_col=0, dtype={'pid': str})  # make sure participant ID's are strings
    else:
        pids = get_pids(test=test)
        all_parts = []
        sfreq = 250
        psd_kwargs = dict(
            fmin=1,
            fmax=45,
            method="welch",
            n_fft=sfreq,
            n_per_seg=sfreq,
            # windows_length will be n_per_seg / sfreq, so setting n_per_seg=sfreq will make windows_length 1s
            n_overlap=int(sfreq / 2),  # 50% overlap, common
            window="hamming"  # common
        )

        for pid in pids:

            epo_path = get_eeg_path() / 'Epo' / pid
            pid_epo_files = [file for file in os.listdir(epo_path) if file.endswith('.fif')]
            raw_path = get_clean_eeg_path() / pid
            pid_raw_files = [file for file in os.listdir(raw_path) if file.endswith('final_raw.fif')]  # Also compute metrics on full raw data
            pid_files = pid_epo_files # + pid_raw_files

            for i, file in enumerate(pid_files):
                if test and i > 0:
                    break
                if file.startswith('RS'):  # ignore for the moment
                    continue

                is_epo = file.endswith('-epo.fif')

                if is_epo:
                    rec = mne.read_epochs(epo_path / file, preload=True, verbose=False)
                    cond, block_n, epo_type = parse_epo_filename(file, pid=pid)
                else:
                    rec = mne.io.read_raw_fif(raw_path / file, preload=True, verbose=False)
                    cond, block_n, epo_type = parse_prepro_filename(file)

                if rec is None or len(rec) == 0:
                    continue

                # Compute PSD in each recording (first within and epoch and channel, then average across them to get a PSD for the entire recording)
                # psd_kwargs = {} if file.endswith('-epo.fif') else {'n_fft': 250}
                psd = compute_psd(rec, verbose=False, **psd_kwargs)
                full_psd, freqs = psd.get_data(return_freqs=True)
                if log:
                    full_psd = np.log10(full_psd)

                # Prepare columns with base information to include to each df/row
                base_cols = dict(pid=pid, cond=cond, block=block_n, epo_type=epo_type)

                # Compute average PSD
                if avg_across_epochs:
                    if is_epo:
                        # 1) compute average of full_psd (n_epochs, n_channels, n_freqs) across channels
                        psd_ch_mean = full_psd.mean(axis=1)  # (n_epochs, n_freqs)

                        # 2) compute mean and std of psd_ch_mean (n_epochs, n_freqs) across epochs
                        psd_avg = psd_ch_mean.mean(axis=0)  # (n_freqs,)
                        psd_std = psd_ch_mean.std(axis=0)  # (n_freqs,)
                    else:  # continuous data
                        # compute average and std of full_psd (n_channels, n_freqs)  across channels
                        psd_avg = full_psd.mean(axis=0)  # (n_freqs,)
                        psd_std = full_psd.std(axis=0)  # (n_freqs,)

                    # Define sub_df for this rec (one row for each frequency-point of the PSD)
                    file_freqs = freqs
                    file_pws = psd_avg
                    file_stds = psd_std

                else:
                    if not is_epo:  # not averaging across epochs only has to be implemented for epoched recordings
                        continue

                    # 1) compute average of full_psd (n_epochs, n_channels, n_freqs) across channels
                    psd_ch_mean = full_psd.mean(axis=1)  # (n_epochs, n_freqs)

                    # 2) compute std of full_psd (n_epochs, n_channels, n_freqs) across channels
                    psd_ch_std = full_psd.std(axis=1)  # (n_epochs, n_freqs)

                    # Define sub_df for this rec (one row for each eppch and frequency-point of the PSD)
                    n_epochs = psd_ch_mean.shape[0]
                    base_cols['n_epo'] = np.repeat(np.arange(1, n_epochs+1),
                                                   len(freqs))  # repeats n_epochs range (from 0 to tot nr of epochs) for all epochs
                    file_freqs = np.tile(freqs, n_epochs)  # repeats freqs range for all epochs
                    file_pws = psd_ch_mean.reshape(-1)  # flatten, so that all epochs follow in the col
                    file_stds = psd_ch_std.reshape(-1)  # flatten, so that all epochs follow in the col

                # Define sub_df for this rec (one row for each frequency-point of the PSD)
                all_parts.append(pd.DataFrame({
                    **base_cols,
                    'freq': file_freqs,
                    'pw_avg': file_pws,
                    'pw_std': file_stds,
                }))

        psd_df = pd.concat(all_parts, ignore_index=True)

        assert (psd_df['freq'].unique() == list(range(psd_kwargs['fmin'], psd_kwargs['fmax']+1))).all()

        if save:
            file_name = ('psd_df_log.csv' if log else 'psd_df_lin.csv') if avg_across_epochs else (
                'psd_df_epo_log.csv' if log else 'psd_df_epo_lin.csv')
            file_path = set_for_save(get_tables_path()) / file_name
            psd_df.to_csv(file_path)

    return psd_df


def get_psd_avg_df(
        load: bool = True,
        save: bool = False,
        log: bool = False,
) -> pd.DataFrame:
    if load:
        file_name = 'psd_avg_df_log.csv' if log else 'psd_avg_df_lin.csv'
        file_path = set_for_save(get_tables_path()) / file_name
        psd_avg_df = pd.read_csv(file_path, index_col=0, dtype={'pid': str})  # make sure participant ID's are strings
    else:
        psd_df = get_psd_df(load=True, log=False)  # always start by linear df (to apply log afterwards)

        # For each patient, average PSD of the same condition and epoch-type across different blocks
        df_subj = psd_df.groupby(
            ['pid', 'cond', 'epo_type', 'freq'], as_index=False).agg(
            pw_avg=('pw_avg', 'mean'),
        )

        if log:
            # Convert in log space
            df_subj['pw_avg'] = np.log10(df_subj['pw_avg'])

        # Then average PSD of the same epoch-type and condition across different participants
        psd_avg_df = df_subj.groupby(
            ['cond', 'epo_type', 'freq'], as_index=False).agg(
            pw_avg=('pw_avg', 'mean'),
            pw_std=('pw_avg', 'std'),
            N=('pw_avg', 'count'),
        )
        psd_avg_df.loc[psd_avg_df["N"] == 1, "pw_std"] = 0.0  # replace with zeros the NaNs appearing as std if there's only one participant

        raw = psd_avg_df[psd_avg_df["epo_type"] == "Raw"]
        grp_to_check = raw.groupby("cond")["N"].agg(["min", "max"])
        assert (grp_to_check['min'] == grp_to_check['max']).all()

        if save:
            file_name = 'psd_avg_df_log.csv' if log else 'psd_avg_df_lin.csv'
            file_path = set_for_save(get_tables_path()) / file_name
            psd_avg_df.to_csv(file_path)
    return psd_avg_df


def get_band_metrics_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
        avg_across_epochs: bool = True,
) -> pd.DataFrame:
    if load:
        file_name = 'band_metrics_df.csv' if avg_across_epochs else 'band_metrics_df_epo.csv'
        file_path = set_for_save(get_tables_path()) / file_name
        osc_df = pd.read_csv(file_path, index_col=0, dtype={'pid': str})  # make sure participant ID's are strings
    else:
        psd_df = get_psd_df(test=test, log=False, avg_across_epochs=avg_across_epochs)  # load power spectra in linear scale
        bands = ['theta'] if test else ['theta', 'alpha', '38-42']
        df_rows = []

        # Define variables within which the PSD will be computed (for each of these there will be one)
        group_by = ['pid', 'cond', 'epo_type'] if avg_across_epochs else ['pid', 'cond', 'epo_type', 'n_epo']
        for by_vals, grouped_df in psd_df.groupby(group_by):

            # Average across blocks of the grouping_vars-combination condition (of the same epo_type, cond, and pid (and n_epo when not avg_across_epochs)
            mean_psd_df = (
                grouped_df
                .groupby('freq', as_index=False)['pw_avg']
                .mean()
            )

            psd = mean_psd_df['pw_avg'].to_numpy()
            freqs = mean_psd_df['freq'].to_numpy()
            psd_model = model_psd(psd, freqs, max_n_peaks=3)  # limit max_n_peaks bc we only care about alpha/theta (and perhaps gamma)

            for band in bands:

                # Detect peaks
                band_freqs = get_band_freqs(band)
                pk_pw = get_band_peak_fm(psd_model, band_freqs, select_highest=True)[1]
                pk = False if np.isnan(pk_pw) else True

                if avg_across_epochs:
                    pid, cond, epo_type = by_vals
                else:
                    pid, cond, epo_type, n_epo = by_vals

                # Define a row of the df
                row = dict(
                    pid=pid,
                    cond=cond,
                    epo_type=epo_type,
                    band=band,
                    abs_pw=get_band_power(psd, freqs, band, rel=False),  # Compute abs power
                    rel_pw=get_band_power(psd, freqs, band, rel=True),  # Compute rel power
                    osc_snr=compute_osc_snr(psd_model, band),  # Compute oscillatory SNR
                    pk=pk,
                    pk_pw=pk_pw,
                )
                if not avg_across_epochs:
                    # Also add n_epo col to the row dict
                    items = list(row.items())  # convert dict items to list of tuples
                    items.insert(2, ('n_epo', n_epo))  # this allows to insert the new item at the preferred index
                    row = dict(items)  # re convert to dict

                df_rows.append(row)

        # Create df
        osc_df = pd.DataFrame(df_rows)
        osc_df['pid'] = osc_df['pid'].astype('str')
        osc_df.sort_values(by=['pid', 'cond'])  # sort conveniently

        if save:
            file_name = 'band_metrics_df.csv' if avg_across_epochs else 'band_metrics_df_epo.csv'
            file_path = set_for_save(get_tables_path()) / file_name
            osc_df.to_csv(file_path)
    return osc_df
