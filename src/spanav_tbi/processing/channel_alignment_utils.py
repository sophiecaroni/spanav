"""
    Title: Channel alignment utilities

    Author: Sophie Caroni
    Date of creation: 22.04.2026

    Description:
    Shared utilities for aligning channel sets (unique to each subject) across subjects.
"""
import mne
import numpy as np
import pandas as pd
from mne import Epochs
from mne.time_frequency import AverageTFR, Spectrum
from spanav_eeg_utils.io_utils import get_all_preproc_raw_fpaths, get_outputs_path
from spanav_eeg_utils.parsing_utils import get_sid_from_fname


def _pad_spectral_channels(obj: AverageTFR | Spectrum, all_ch_names: list[str]) -> AverageTFR | Spectrum:
    """
    Add channels missing from all_ch_names as NaNs in spectral objects, and reorder to the canonical all_ch_names order.
    :param obj: AverageTFR or Spectrum, the spectral object to pad.
    :param all_ch_names: list[str], the target channel set (union across all subjects), sorted.
    :return: AverageTFR or Spectrum, padded to contain channels in all_ch_names, reordered.
    """
    missing_chs = [ch for ch in all_ch_names if ch not in obj.ch_names]
    if not missing_chs:
        return obj.reorder_channels(all_ch_names)

    # Get a channel object as template (to ensure it inherits all info fields from the object) and replace is data with Nans
    template = obj.copy().pick([obj.ch_names[0]])
    if isinstance(obj, AverageTFR):
        template.data[:] = np.nan
    else:
        template._data[:] = np.nan

    # Retrieve channel positions - new channels will need this info field to be set and not inherited from the template
    ch_pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']

    # Create new channel objects and add them to the main spectral object, finally reorder all channel names
    new_chs = []
    for ch in missing_chs:
        new_ch = template.copy()
        mne.rename_channels(new_ch.info, {new_ch.ch_names[0]: ch})

        # Replace template channel's position to the standard head-frame position
        if ch in ch_pos:
            new_ch.info['chs'][0]['loc'][:3] = ch_pos[ch]

        new_chs.append(new_ch)

    return obj.copy().add_channels(new_chs).reorder_channels(all_ch_names)


def _pad_rec_channels(rec: Epochs, all_ch_names: list[str]) -> Epochs:
    missing_chs = [ch for ch in all_ch_names if ch not in rec.ch_names]
    if not missing_chs:
        return rec.reorder_channels(all_ch_names)

    # Get a channel object as template (to ensure it inherits all info fields from the object) and replace is data with Nans
    template = rec.copy().pick([rec.ch_names[0]])
    template._data[:] = np.nan

    # Retrieve channel positions - new channels will need this info field to be set and not inherited from the template
    ch_pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']

    # Create new channel objects and add them to the main spectral object, finally reorder all channel names
    new_chs = []
    for ch in missing_chs:
        new_ch = template.copy()
        mne.rename_channels(new_ch.info, {new_ch.ch_names[0]: ch})

        # Replace template channel's position to the standard head-frame position
        if ch in ch_pos:
            new_ch.info['chs'][0]['loc'][:3] = ch_pos[ch]

        new_chs.append(new_ch)

    return rec.copy().add_channels(new_chs).reorder_channels(all_ch_names)


def align_spectral_channels(df: pd.DataFrame, obj_col: str) -> pd.DataFrame:
    """
    Aligns all objects in df[obj_col] to the union channel set, by padding missing channels with NaNs.
    :param df: pd.DataFrame, must have a column obj_col containing AverageTFR or Spectrum objects.
    :param obj_col: str, name of the column holding spectral objects (e.g. 'tfr' or 'psd').
    :return: pd.DataFrame, copy of df with all spectral objects sharing the same channel set.
    """
    df = df.copy()
    all_ch_names = set()
    for obj in df[obj_col]:
        all_ch_names.update(obj.ch_names)
    all_ch_names = sorted(all_ch_names)
    df[obj_col] = df[obj_col].apply(lambda obj: _pad_spectral_channels(obj, all_ch_names))
    return df


def reconstruct_missing_channels(rec: Epochs, sid: str, epo_label: str, verbose: bool = False) -> Epochs:
    assert len(rec.info['bads']) == 0, 'Bad channels already present!'
    present_chs = rec.ch_names
    required_chs = get_cohort_channels()
    missing_chs = list(set(required_chs) - set(present_chs))
    if verbose:
        print(
            f"Interpolating {missing_chs} for subject {sid}, {epo_label} rec"
        )

    # Add missing channels by 0-padding
    padded_epo_rec = _pad_rec_channels(rec, required_chs)

    # Set 0-padded channels as bads and interpolate them
    padded_epo_rec.info['bads'] = missing_chs
    padded_epo_rec.interpolate_bads(verbose=verbose)
    return padded_epo_rec


def extract_cohort_channels(verbose: bool = False, save: bool = False):
    """Extract union of channels across all cohort subjects."""
    all_raw_fnames = get_all_preproc_raw_fpaths()
    all_chs = set()
    sids = []
    for fname in all_raw_fnames:
        raw = mne.io.read_raw_fif(fname, preload=False, verbose=False)

        # Only extract channel names from one file per subject
        sid = get_sid_from_fname(fname.stem)
        if sid in sids:
            continue

        chs = raw.ch_names
        if verbose:
            print(
                f"Extracting channels from {fname.stem} with {len(chs)} channels."
            )
        all_chs.update(set(raw.ch_names))
        sids.append(sid)
    all_chs_ordered = sorted(list(all_chs))
    if verbose:
        print(
            f"==> Final cohort channel set: {len(all_chs_ordered)} channels"
        )
    if save:
        fpath = get_outputs_path() / 'cohort_channel_set.csv'
        np.savetxt(fpath, np.array(all_chs_ordered), fmt='%s', header='ch_names')


def get_cohort_channels():
    fpath = get_outputs_path() / 'cohort_channel_set.csv'
    return np.loadtxt(fpath, dtype=str).tolist()
