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
from mne.time_frequency import AverageTFR, Spectrum


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
