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
import spanav_eeg_utils.spanav_utils as sn
from mne.epochs import Epochs
from mne.time_frequency import read_spectrum, EpochsSpectrum, combine_spectrum, Spectrum


def average_psd_series(psd_series) -> Spectrum:
    """
    Combine a pandas Series of MNE Spectrum objects by weigthed average.
    """
    return combine_spectrum(list(psd_series))


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
    ch_psd = spct.compute_psd(rec, log_space=True, fmin=fmin, fmax=fmax, test=test)
    freqs = ch_psd.freqs
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


def compute_cond_psd(sid, cids: list[str], epo_type: str, log: bool) -> EpochsSpectrum:
    # Get default parameters for of PSD computation
    psd_kwargs = spct.get_psd_kwargs()

    # Concatenate epoched recordings across block of the same condition (and subject)
    epo_rec_full = cmp.get_concat_epo_recs(sid, cids, epo_type)

    # Compute PSD on all epochs and channels and return
    return spct.compute_psd(epo_rec_full, log_space=log, **psd_kwargs)


def get_epo_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        log: bool = False,
) -> pd.DataFrame:
    # Get PSD within each epoch
    sids = io.get_sids(test=test)
    epo_types = sn.get_task_epo_types(test=test)
    psd_records = []
    for sid in sids:
        for epo_type in epo_types:
            sid_cids = io.get_sid_blocks(sid, test=test)
            cids_by_cond = sn.group_cids_by_cond(sid, test, cids=sid_cids)

            # Get one concatenated recording of epochs of the same condition
            for cond, cids in cids_by_cond.items():
                try:
                    if load:
                        # Read exported files
                        scale = 'log' if log else 'lin'
                        fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-epo_scale-{scale}_psd.h5'
                        fpath = io.get_outputs_path(sid) / 'PSD' / f'sub-{sid}' / fname
                        psd = read_spectrum(fpath)

                    else:
                        # Compute PSD of the recording
                        psd = compute_cond_psd(sid, cids, epo_type, log=log)  # get a PSD in each channel and epoch
                        if save:
                            # Export PSD object
                            scale = 'log' if log else 'lin'
                            fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-epo_scale-{scale}_psd.h5'
                            fpath = io.set_for_save(io.get_outputs_path(sid) / 'PSD' / f'sub-{sid}') / fname
                            psd.save(fpath, overwrite=True)

                except (FileNotFoundError, OSError):
                    print(f"\nFile not found for {sid, cond, epo_type = }. Continuing...")
                    continue

                # Store as df entry
                psd_entry = dict(
                    sid=sid,
                    group=prs.get_group_letter(sid),
                    cond=cond,
                    epo_type=epo_type,
                    psd=psd,
                )
                psd_records.append(psd_entry)

    return pd.DataFrame.from_records(psd_records)


def get_sid_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        log: bool = False,
) -> pd.DataFrame:
    if load:
        sids = io.get_sids(test=test)
        epo_types = sn.get_task_epo_types(test=test)

        # Additionally try to load bl-corrected version of epoch-types
        epo_types = epo_types + [f'bl{epo_type}' for epo_type in epo_types if epo_type != 'Stasis']

        psd_records = []
        for sid in sids:
            conds = prs.get_conds(sid=sid)
            for cond in conds:
                for epo_type in epo_types:
                    try:
                        # Read exported files (always in linear scale)
                        fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_scale-lin_psd.h5'
                        fpath = io.get_outputs_path(sid) / 'PSD' / f'sub-{sid}' / fname
                        psd = read_spectrum(fpath)

                        # Store as df entry(s)
                        psd_entry = dict(
                            sid=sid,
                            group=prs.get_group_letter(sid),
                            cond=cond,
                            epo_type=epo_type,
                            psd=psd,
                        )
                        psd_records.append(psd_entry)

                    except (FileNotFoundError, OSError):
                        print(f"\nFile not found for {sid, cond, epo_type = } (scale-lin). Continuing...")

        return pd.DataFrame.from_records(psd_records)

    # Load epoch-level PSD dataframe
    epo_level_df = get_epo_level_psd_df(test=test, load=True, save=False)

    # For each subject, PSD of the same condition and epoch-type were concatenated across different blocks; so simply
    # average across epochs of the concatenated PSD object to get one PSD for subject, condition and epoch-type
    sid_level_df = epo_level_df.copy()
    sid_level_df['psd'] = epo_level_df['psd'].apply(
            lambda row_psd: row_psd.average(method='mean')  # averages across epochs by implementation
    )

    # Baseline correct movement-onset epochs with stasis epochs, as in Convertino et al., 2023
    sid_level_df = _stasis_bl_corr(sid_level_df)

    if save:
        # Export each subject-level PSD object
        for i, row in sid_level_df.iterrows():
            sid, cond, epo_type, psd = row['sid'], row['cond'], row['epo_type'], row['psd']
            scale = 'log' if log else 'lin'
            fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_scale-{scale}_psd.h5'
            psd._inst_type = mne.Evoked  # use this (with any non-Epochs class) to prevent bug with read_spectrum
            fpath = io.set_for_save(io.get_outputs_path(sid) / 'PSD' / sid) / fname
            psd.save(fpath, overwrite=True)

    return sid_level_df


def get_group_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        log: bool = False,
) -> pd.DataFrame:
    if load:
        groups = io.get_groups_letters()
        epo_types = sn.get_task_epo_types(test=test)

        # Additionally try to load bl-corrected version of epoch-types
        epo_types = epo_types + [f'bl{epo_type}' for epo_type in epo_types if epo_type != 'Stasis']

        psd_records = []
        for group in groups:
            conds = prs.get_conds(group=group)
            for cond in conds:
                for epo_type in epo_types:
                    try:
                        # Read exported files
                        scale = 'log' if log else 'lin'
                        fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_scale-{scale}_psd.h5'
                        fpath = io.get_outputs_path(group_letter=group) / 'PSD' / fname
                        cond_psd = read_spectrum(fpath)

                        # Store as df entry(s)
                        psd_entry = dict(
                            group=group,
                            cond=cond,
                            epo_type=epo_type,
                            psd=cond_psd,
                        )
                        psd_records.append(psd_entry)

                    except (FileNotFoundError, OSError):
                        print(f"\nFile not found for {group, cond, epo_type = }. Continuing...")

        return pd.DataFrame.from_records(psd_records)

    # Load subject-level PSD
    sid_level_df = get_sid_level_psd_df(test=test, load=True, save=False)

    # For each group, average PSD of the same condition and epoch-type across different subjects
    group_cols = ['group', 'cond', 'epo_type']
    grouped_df = sid_level_df.groupby(group_cols, as_index=False)
    group_level_df = grouped_df['psd'].apply(average_psd_series).reset_index(drop=True)

    if save:
        # Export each group-level PSD object
        for i, row in group_level_df.iterrows():
            group, cond, epo_type, psd = row['group'], row['cond'], row['epo_type'], row['psd']
            scale = 'log' if log else 'lin'
            fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_scale-{scale}_psd.h5'
            fpath = io.set_for_save(io.get_outputs_path(group_letter=group) / 'PSD') / fname
            psd.save(fpath, overwrite=True)

    return group_level_df


def _stasis_bl_corr(input_df: pd.DataFrame):
    group_cols = ['sid', 'group', 'cond']
    grouped_df = input_df.groupby(group_cols, as_index=False)
    new_rows = []
    for (sid, group, cond), subdf in grouped_df:

        # Apply baseline correction using Stasis PSD on PSD of all other epoch types
        bl_corr_records = spct.spectral_bl_corr_from_df(subdf, 'epo_type', 'psd', 'Stasis')

        # Add grouping columns information to bl_corr_records (which will be broadcasted to match len in bl_corr_records)
        bl_corr_records.update(dict(
            sid=sid,
            group=group,
            cond=cond,
        ))

        # Turn into a one-line df for the new baseline-corrected PSD and append, by treating the BL-corrected as new epoch-types named with suffix "bl"
        new_rows.append(pd.DataFrame(bl_corr_records))

    # Add new rows relative to baseline-corrected PSD (of the new 'blMovOn' epoch_type) into output df
    output_df = pd.concat([input_df] + new_rows, ignore_index=True)
    return output_df
