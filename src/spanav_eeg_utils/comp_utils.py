"""
    Title: Computational utilities

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains helper functions for computations.
"""
import mne
import pandas as pd
import spanav_eeg_utils.parsing_utils as prs
from mne import EpochsArray
from spanav_eeg_utils.io_utils import get_epo_data_path


def fix_std_singleton(df: pd.DataFrame, std_cols: list[str], n_col: str) -> None:
    for std_col in std_cols:
        df.loc[df[n_col] == 1, std_col] = 0.0  # replace with zeros the NaNs appearing as std (if there was only one row to average across)


def get_concat_epo_recs(
        sid: str,
        cids_to_concat: list[str],
        epo_type: str,
        # epo_recs: list[EpochsFIF],
) -> EpochsArray | None:
    """
    Load epoched recordings and concatenate them.
    :param sid:
    :param cids_to_concat:
    :param epo_type:
    :return:
    """
    recs_list = []
    for cid in cids_to_concat:
        real_cid = prs.get_stim(sid, acq=cid)
        task = 'RS' if real_cid.lower().startswith('rs') else 'SpaNav'
        epo_path = get_epo_data_path(epo_type, sid, acq=real_cid, task=task)
        if epo_path.exists():
            epo_rec = mne.read_epochs(epo_path, preload=False, proj=False, verbose=False)
            recs_list.append(epo_rec)
    if recs_list:
        return mne.concatenate_epochs(recs_list)
    return None
