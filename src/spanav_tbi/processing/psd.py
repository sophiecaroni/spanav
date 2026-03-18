"""
    Title: Power Spectral Density (PSD) utilities for EEG.

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains functions to compute and store in tables power spectra of EEG.
"""
import mne
import numpy as np
import pandas as pd
import spanav_eeg_utils.spectral_utils as spct
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.comp_utils as cmp
from spanav_eeg_utils.spanav_utils import get_epo_types
from mne.epochs import BaseEpochs, EpochsArray, Epochs


def compute_avg_epo_psd(
        rec: Epochs,
        fmin: float = 0.0,
        fmax: float = np.inf,
        test: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param rec:
    :param test:
    :param fmin:
    :param fmax: default takes all frequencies (up to Nyquist frequency)
    :return: PSD
    """
    ch_psd, freqs = spct.compute_psd(rec, log_space=True, fmin=fmin, fmax=fmax, test=test)
    rec_psd_avg = np.mean(np.mean(ch_psd, axis=1), axis=0)
    rec_psd_std = np.mean(np.std(ch_psd, axis=1) / np.sqrt(ch_psd.shape[1]), axis=0)
    return rec_psd_avg, rec_psd_std, freqs


def compute_psd_by_key(
        epos_dict: dict[str, Epochs],
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


def get_epo_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        log: bool = False,
) -> pd.DataFrame:
    if load:
        fname = 'psd_df_epo_level_log.csv' if log else 'psd_df_epo_level_lin.csv'
        file_path = io.get_tables_path() / fname
        return pd.read_csv(file_path, index_col=0, dtype={'sid': str})  # make sure subject ID's are strings

    psd_kwargs = spct.get_psd_kwargs()
    sids = io.get_sids(test=test)
    epo_types = get_epo_types()
    all_epo_entries = []
    for sid in sids:
        for epo_type in epo_types:
            blocks = io.get_sid_blocks(sid, test=test)
            for block in blocks:

                epo_fpath = io.get_epo_data_path(epo_type=epo_type, sid=sid, acq=block)
                rec = mne.read_epochs(epo_fpath, preload=False, verbose=False, proj=False)

                if rec is None or len(rec) == 0:
                    continue

                rec.load_data()

                # Compute PSD in each recording (first within and epoch and channel, then average across them to get a PSD for the entire recording)
                psd = spct.compute_psd(rec, verbose=False, **psd_kwargs)
                full_psd, freqs = psd.get_data(return_freqs=True)
                if log:
                    full_psd = np.log10(full_psd)

                # Prepare columns with base information to include to each df/row
                cond = prs.get_stim(sid, block)
                base_cols = dict(
                    sid=sid,
                    group=prs.get_group_letter(sid),
                    cond=cond,
                    block=block[-1],
                    epo_type=epo_type
                )

                # 1) compute average of full_psd (n_epochs, n_channels, n_freqs) across channels
                psd_ch_mean = full_psd.mean(axis=1)  # (n_epochs, n_freqs)

                # 2) compute std of full_psd (n_epochs, n_channels, n_freqs) across channels
                psd_ch_std = full_psd.std(axis=1)  # (n_epochs, n_freqs)

                # Define sub_df for this rec (one row for each eppch and frequency-point of the PSD)
                n_epochs = psd_ch_mean.shape[0]
                base_cols['n_epo'] = np.repeat(np.arange(1, n_epochs + 1),
                                               len(freqs))  # repeats n_epochs range (from 0 to tot nr of epochs) for all epochs
                file_freqs = np.tile(freqs, n_epochs)  # repeats freqs range for all epochs
                file_pws = psd_ch_mean.reshape(-1)  # flatten, so that all epochs follow in the col
                file_stds = psd_ch_std.reshape(-1)  # flatten, so that all epochs follow in the col

                # Define sub_df for this rec (one row for each frequency-point of the PSD)
                all_epo_entries.append(pd.DataFrame({
                    **base_cols,
                    'freq': file_freqs,
                    'pw_avg': file_pws,
                    'pw_std': file_stds,
                }))

    epo_level_psd_df = pd.concat(all_epo_entries, ignore_index=True)

    assert (epo_level_psd_df['freq'].unique() == list(range(psd_kwargs['fmin'], psd_kwargs['fmax']+1))).all()

    if save:
        fname = 'psd_df_epo_level_log.csv' if log else 'psd_df_epo_level_lin.csv'
        file_path = io.get_tables_path() / fname
        epo_level_psd_df.to_csv(file_path)

    return epo_level_psd_df


def get_sid_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        log: bool = False,
) -> pd.DataFrame:
    if load:
        fname = 'psd_df_sid_level_log.csv' if log else 'psd_df_sid_level_lin.csv'
        file_path = io.get_tables_path() / fname
        return pd.read_csv(file_path, index_col=0, dtype={'sid': str})  # make sure subject ID's are strings

    # Load epoch-level PSD dataframe
    epo_level_df = get_epo_level_psd_df(load=True, log=False, test=test, save=False)  # always start by linear df (to apply log afterwards)

    # For each subject, average PSD of the same condition and epoch-type across different blocks
    group_cols = ['sid', 'group', 'cond', 'epo_type', 'freq']
    grouped_df = epo_level_df.groupby(group_cols, as_index=False)
    sid_level_df = grouped_df.agg(
        pw_avg=('pw_avg', 'mean'),
        pw_std=('pw_avg', 'std'),
        n_epochs=('sid', 'size'),
    )
    cmp.fix_std_singleton(sid_level_df, ["pw_std"], n_col="n_epochs")  # replace with zeros the NaNs appearing as std (if there was only one row to average across)

    if log:
        # Convert in log space
        sid_level_df['pw_avg'] = np.log10(sid_level_df['pw_avg'])

    if save:
        fname = 'psd_df_sid_level_log.csv' if log else 'psd_df_sid_level_lin.csv'
        file_path = io.get_tables_path() / fname
        sid_level_df.to_csv(file_path)

    return sid_level_df


def get_group_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        log: bool = False,
) -> pd.DataFrame:
    if load:
        fname = 'psd_df_group_level_log.csv' if log else 'psd_df_group_level_lin.csv'
        file_path = io.get_tables_path() / fname
        return pd.read_csv(file_path, index_col=0, dtype={'sid': str})  # make sure subject ID's are strings

    # Load subject-level PSD dataframe
    sid_level_df = get_sid_level_psd_df(load=True, log=False, test=test, save=False)  # always start by linear df (to apply log afterwards)

    # Fors each group, average PSD across different subjects
    group_cols = ['group', 'cond', 'epo_type', 'freq']
    grouped_df = sid_level_df.groupby(group_cols, as_index=False)
    group_level_df = grouped_df.agg(
        pw_avg=('pw_avg', 'mean'),
        pw_std=('pw_avg', 'std'),
        n_sids=('group', 'size'),
    )
    cmp.fix_std_singleton(sid_level_df, ["pw_std"], n_col="n_sids")  # replace with zeros the NaNs appearing as std (if there was only one row to average across)

    if log:
        # Convert in log space
        group_level_df['pw_avg'] = np.log10(group_level_df['pw_avg'])

    if save:
        fname = 'psd_df_group_level_log.csv' if log else 'psd_df_group_level_lin.csv'
        file_path = io.get_tables_path() / fname
        group_level_df.to_csv(file_path)

    return group_level_df


def average_channels_across_epochs(epo_in: BaseEpochs) -> EpochsArray:
    info = mne.create_info(
        ch_names=["avg_channels"],
        sfreq=epo_in.info["sfreq"],
        ch_types="eeg"
    )
    data = epo_in.get_data()
    ch_avg = np.average(data, axis=1)

    epo_out = mne.EpochsArray(
        ch_avg[:, np.newaxis, :],  # keep 3D
        info=info,
        events=epo_in.events,
        event_id=epo_in.event_id
    )
    return epo_out
