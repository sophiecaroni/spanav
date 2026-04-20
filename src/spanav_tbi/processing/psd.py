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
import warnings
from mne.epochs import Epochs
from mne.time_frequency import read_spectrum, EpochsSpectrum, combine_spectrum, Spectrum


def compute_group_psd(psd_series: pd.Series) -> Spectrum:
    """
    Combine a pandas Series of MNE Spectrum objects.
    Every Spectrum weights 1 on the average - this is suited for example when averaging across subjects of the same group
    (where every subject should weight the same).
    """
    return combine_spectrum(list(psd_series), weights='equal')


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


def normalize_psd(psd_in: EpochsSpectrum) -> EpochsSpectrum:
    """
    Normalize PSD.
    :param psd_in:
    :return:
    """
    psd_out = psd_in.copy()
    data_in = psd_in._data

    # Normalize by sum across frequencies (per epoch and channel)
    denom = data_in.sum(axis=-1, keepdims=True)  # axis=-1 because (epochs, ch, freq)
    data_norm = data_in / denom
    psd_out._data = data_norm
    return psd_out


def compute_cond_psd(sid: str, cids: list[str], epo_type: str, space: str = 'log') -> EpochsSpectrum | None:
    """
    Compute PSD of data recorded in one stimulation condition.
    Concatenates epoch-objects from different blocks of the same stimulation condition. Then follows a procedure
    similar to Convertino et al. (2023).
    :param sid:
    :param cids:
    :param epo_type:
    :param space: str, defaults to 'log' to log-transform the spectrum (as in the reference paper) - but this function
                  is also used in the oscillatory features pipeline where the spectrum needs to be modeled in linear space.
    :return:
    """
    # Get default parameters for of PSD computation
    psd_kwargs = spct.get_psd_kwargs()

    # Concatenate epoched recordings across block of the same condition (and subject)
    epo_rec_full = cmp.get_concat_epo_recs(sid, cids, epo_type)

    if epo_rec_full is not None:
        # Compute PSD on all epochs and channels
        psd = spct.compute_psd(epo_rec_full, log_space=False, **psd_kwargs)  # don't log yet

        # Normalize PSD
        norm_psd = normalize_psd(psd)

        if space == 'log':
            norm_psd._data = np.log10(norm_psd._data)

        return norm_psd
    return None


def get_epo_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        average_epochs: bool = False,
        space: str = 'log',
) -> pd.DataFrame:
    assert space in ['lin', 'log'], f'Please pass a valid space, accepted are "lin" and "log" (got {space})'
    # Get PSD within each epoch
    sids = io.get_sids(test=test)
    epo_types = sn.get_task_epo_types(test=test)
    psd_records = []
    for sid in sids:
        group = prs.get_group_letter(sid)
        sid_cids = io.get_sid_blocks(sid, test=test)
        if not sid_cids:
            continue
        cids_by_cond = sn.group_cids_by_cond(sid, test, cids=sid_cids)
        for epo_type in epo_types:
            # Get one concatenated recording of epochs of the same condition
            for cond, cids in cids_by_cond.items():
                fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-epo_psd_{space}.h5'
                fpath = io.get_outputs_path(sid) / 'PSD' / f'sub-{sid}' / fname

                if load:
                    if not fpath.exists():
                        warnings.warn(f"\nFile {fname} not found at {fpath.parent}. Skipping epo-level PSD...")
                        continue

                    # Load spectrum from exported file
                    psd = read_spectrum(fpath)

                else:
                    psd = compute_cond_psd(sid, cids, epo_type, space=space)
                    if psd is None:
                        warnings.warn(f"\nPSD is None for {fname} (epo rec file likely not found). Skipping epo-level PSD...")
                        continue
                    if save:
                        fpath = io.set_for_save(io.get_outputs_path(sid) / 'PSD' / f'sub-{sid}') / fname
                        psd.save(fpath, overwrite=True)

                if average_epochs:
                    psd = psd.average(method='mean')

                # Append as df entry
                psd_records.append(dict(
                    sid=sid,
                    group=group,
                    cond=cond,
                    epo_type=epo_type,
                    psd=psd,
                ))

    return pd.DataFrame.from_records(psd_records)


def _average_psd_channels(psd) -> object:
    ch_psds = []
    for ch in psd.ch_names:
        ch_psd = psd.copy().pick(ch)
        mne.rename_channels(ch_psd.info, {ch: 'ch_mean'})
        ch_psds.append(ch_psd)
    return combine_spectrum(ch_psds)


def get_sid_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
        average_channels: bool = False,
) -> pd.DataFrame:
    if load:
        sids = io.get_sids(test=test)
        epo_types = sn.get_task_epo_types(test=test)

        # Additionally try to load bl-corrected version of epoch-types
        epo_types = epo_types + [f'bl{epo_type}' for epo_type in epo_types if epo_type != 'Stasis']

        psd_records = []
        for sid in sids:
            group = prs.get_group_letter(sid)
            conds = prs.get_conds(sid=sid)
            for cond in conds:
                for epo_type in epo_types:
                    fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_psd.h5'
                    fpath = io.get_outputs_path(sid) / 'PSD' / f'sub-{sid}' / fname
                    if not fpath.exists():
                        warnings.warn(f"\nFile {fname} not found at {fpath.parent}. Continuing...")
                        continue

                    # Load spectrum from exported file
                    psd = read_spectrum(fpath)

                    if average_channels:
                        psd = _average_psd_channels(psd)

                    # Append as df entry
                    psd_records.append(dict(
                        sid=sid,
                        group=group,
                        cond=cond,
                        epo_type=epo_type,
                        psd=psd,
                    ))

        return pd.DataFrame.from_records(psd_records)

    # Load epoch-level PSD dataframe with average_epochs=True - average at the moment of loading is more efficient
    sid_level_df = get_epo_level_psd_df(test=test, load=True, save=False, average_epochs=True)

    # Baseline correct movement-onset epochs with stasis epochs, as in Convertino et al., 2023
    sid_level_df = _stasis_bl_corr(sid_level_df)

    if save:
        # Export each subject-level PSD object
        for i, row in sid_level_df.iterrows():
            sid, cond, epo_type, psd = row['sid'], row['cond'], row['epo_type'], row['psd']
            fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_psd.h5'
            psd._inst_type = mne.Evoked  # use this (with any non-Epochs class) to prevent bug with read_spectrum
            fpath = io.set_for_save(io.get_outputs_path(sid) / 'PSD' / sid) / fname
            psd.save(fpath, overwrite=True)

    return sid_level_df


def get_group_level_psd_df(
        load: bool = True,
        test: bool = False,
        save: bool = False,
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
                    fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_psd.h5'
                    fpath = io.get_outputs_path(group_letter=group) / 'PSD' / fname
                    if not fpath.exists():
                        print(f"\nFile {fname} not found at {fpath.parent}. Continuing...")
                        continue
                    # Load spectrum from exported file
                    cond_psd = read_spectrum(fpath)
                    
                    # Append as df entry
                    psd_records.append(dict(
                        group=group,
                        cond=cond,
                        epo_type=epo_type,
                        psd=cond_psd,
                    ))

        return pd.DataFrame.from_records(psd_records)

    # Load subject-level PSD, with average_channels=True - we need one channel per sid to compute group spectra (since every sid has a different montage)
    sid_level_df = get_sid_level_psd_df(test=test, load=True, save=False, average_channels=True)

    # For each group, average PSD of the same condition and epoch-type across different subjects
    group_cols = ['group', 'cond', 'epo_type']
    grouped_df = sid_level_df.groupby(group_cols, as_index=False)
    group_level_df = grouped_df['psd'].apply(compute_group_psd).reset_index(drop=True)

    if save:
        # Export each group-level PSD object
        for i, row in group_level_df.iterrows():
            group, cond, epo_type, psd = row['group'], row['cond'], row['epo_type'], row['psd']
            fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_psd.h5'
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
