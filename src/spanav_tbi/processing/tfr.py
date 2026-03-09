"""
    Title: Time-Frequency (TFR)

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains functions to compute and store TFR of EEG.
"""
import numpy as np
import pandas as pd

import spanav_eeg_utils.comp_utils
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.spanav_utils as sn

from mne.time_frequency import EpochsTFR, AverageTFR, read_tfrs, combine_tfr
from mne.epochs import BaseEpochs
from spanav_tbi.processing.psd import average_channels_across_epochs

TFR = EpochsTFR | AverageTFR


def combine_tfr_series(tfr_series):
    """
    Combine a pandas Series of MNE TFR objects.
    """
    return combine_tfr(list(tfr_series))


def compute_tfr(
        epo_rec: BaseEpochs,
        log: bool = True,
        norm: bool = True,
) -> EpochsTFR:
    """
    Compute TFR as in Convertino et al., 2023
    :param epo_rec:
    :param log:
    :param norm:
    :return:
    """

    freqs = np.logspace(np.log10(2), np.log10(60), 40)
    # freqs = np.linspace(2, 70, 40)
    n_cycles = freqs / 2  # Convertino uses 5 but we can't because we have shorter epochs, so use a specific cycle for each freq (n_cycles_f = f/2)
    # zero_mean = False  # don't correct morlet wavelet to be of mean zero; To have a true wavelet zero_mean should be True but here for illustration purposes it helps to spot the evoked response.

    tfr = epo_rec.compute_tfr(
        "morlet",
        freqs,
        n_cycles=n_cycles,
        average=False,  # don't average across epochs at this stage
        zero_mean=False,    # don't correct morlet wavelet to be of mean zero; To have a true wavelet zero_mean should be True but here for illustration purposes it helps to spot the evoked response.
        return_itc=False,
    )

    if log:
        # Ensure strictly positive before log
        data = tfr.data
        # eps = np.finfo(data.dtype).tiny if np.issubdtype(data.dtype, np.floating) else 1e-20
        # data = np.maximum(data, eps)
        log_data = np.log(data)
        tfr.data = log_data

    if norm:
        tfr = custom_tfr_norm(tfr)
    return tfr


def custom_tfr_norm(tfr_in: TFR) -> TFR:
    """
    Normalize TFR as in Convertino et al., 2023
    :param tfr_in:
    :return:
    """
    tfr_out = tfr_in.copy()
    data_in = tfr_in.data

    # Normalize by sum over frequencies at each time point (per epoch, channel, time)
    denom = data_in.sum(axis=2, keepdims=True)  # axis=2 because (epochs, ch, freq, time)
    data = data_in / denom

    tfr_out.data = data
    return tfr_out


def compute_cond_epo_tfr(sid, cids: list[str], epo_type) -> EpochsTFR:
    epo_rec_full = spanav_eeg_utils.comp_utils.get_concat_epo_recs(sid, cids, epo_type)

    # Average across channels
    epo_rec_ch_avg = average_channels_across_epochs(epo_rec_full)  # average across channels

    # Compute TFR on all epochs and return
    return compute_tfr(epo_rec_ch_avg, log=True, norm=True)  # log-transform and normalize


def get_epo_level_tfr_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
) -> pd.DataFrame:

    # Compute TFRs for each epoch type
    epo_types = sn.get_task_epo_types(test)

    tfr_records = []
    epo_types = sn.get_task_epo_types(test) if epo_types is None else epo_types
    sids = io.get_sids(test=test)
    for epo_type in epo_types:
        for sid in sids:
            sid_cids = io.get_sid_blocks(sid, test=test)
            cids_by_cond = sn.group_cids_by_cond(sid, test, cids=sid_cids)

            # Get one concatenated recording of epochs of the same condition
            for cond, cids in cids_by_cond.items():

                try:
                    if load:
                        # Read exported files
                        fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-epo_tfr.h5'
                        fpath = io.set_for_save(io.get_outputs_path(sid) / 'TFR' / sid) / fname
                        cond_epo_tfr = read_tfrs(fpath, verbose=False)

                    else:
                        cond_epo_tfr = compute_cond_epo_tfr(sid, cids, epo_type)

                        if save:
                            # Export
                            fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-epo_tfr.h5'
                            fpath = io.set_for_save(io.get_outputs_path(sid) / 'TFR' / sid) / fname
                            cond_epo_tfr.save(fpath, overwrite=True)

                except (FileNotFoundError, OSError):
                    print(f"\nFile not found for {sid, cond, epo_type = }. Continuing...")
                    continue

                # Store as df entry
                tfr_entry = dict(
                    sid=sid,
                    group=prs.get_group_letter(sid),
                    cond=cond,
                    epo_type=epo_type,
                    tfr=cond_epo_tfr,
                )

                tfr_records.append(tfr_entry)

    return pd.DataFrame.from_records(tfr_records)


def get_sid_level_tfr_df(
        test: bool = False,
        save: bool = False,
) -> pd.DataFrame:
    # Load epoch-level TFR dataframe
    epo_level_df = get_epo_level_tfr_df(test, load=True, save=False)

    # For each subject, aggregate TFR of the same condition and epoch-type across different blocks
    # group_cols = ['sid', 'group', 'cond', 'epo_type']
    # grouped_df = epo_level_df.groupby(group_cols, as_index=False)
    # sid_level_df = grouped_df['tfr'].apply(lambda tfr: tfr.average()).reset_index(drop=True)
    sid_level_df = epo_level_df.copy()
    sid_level_df['tfr'] = epo_level_df['tfr'].apply(
            lambda tfr: tfr.average(method='mean', dim='epochs')
    )

    if save:
        # Export each subject-level TFR object
        for i, row in sid_level_df.iterrows():
            sid = row['sid']
            cond = row['cond']
            epo_type = row['epo_type']
            sid_tfr = row['tfr']
            fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_tfr.h5'
            fpath = io.set_for_save(io.get_outputs_path(sid) / 'TFR' / sid) / fname
            sid_tfr.save(fpath, overwrite=True)

    return sid_level_df


def get_group_level_tfr_df(
        test: bool = False,
        save: bool = False,
) -> pd.DataFrame:
    # Load subject-level TFR dataframe
    sid_level_df = get_sid_level_tfr_df(test, save=False)

    # For each group, average TFR of the same condition and epoch-type across different subjects
    group_cols = ['group', 'cond', 'epo_type']
    grouped_df = sid_level_df.groupby(group_cols, as_index=False)
    group_level_df = grouped_df['tfr'].apply(combine_tfr_series).reset_index(drop=True)

    if save:
        # Export each group-level TFR object
        for i, row in group_level_df.iterrows():
            group = row['group']
            cond = row['cond']
            epo_type = row['epo_type']
            group_tfr = row['tfr']
            fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_tfr.h5'
            fpath = io.set_for_save(io.get_outputs_path(group_letter=group) / 'TFR') / fname
            group_tfr.save(fpath, overwrite=True)

    return group_level_df

