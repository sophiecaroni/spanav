"""
********************************************************************************
    Title: I/O utilities

    Author: Sophie Caroni
    Date of creation: 18.02.2026

    Description:
    This script contains input/output helper functions.
********************************************************************************
"""
import os
import pandas as pd
import re

from pathlib import Path
from spanav_eeg_utils.config_utils import get_server
from spanav_eeg_utils.spanav_utils import get_group_letter
from spanav_eeg_utils.parsing_utils import parse_prepro_fname


def set_for_save(
        save_path: Path,
) -> Path:
    """

    :param save_path:
    :return:
    """
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_main_path(
        server: bool | None = None,
) -> Path:
    """

    :param server:
    :return:
    """
    SERVER = get_server() if server is None else server
    if SERVER:
        return Path('/Volumes/Hummel-Data/TI/SpatialNavigation/WP7.3_EEG')
    else:
        return Path('/Volumes/My Passport/SpaNav/Sophie_backup')


def get_raw_eeg_path(
        sid: str,
) -> Path:
    root = get_main_path()
    group = get_group_letter(sid)
    return root / 'raw' / f'BIDS_Data_WP73{group}' / f'sub-{sid}' / 'ses-1' / 'eeg'


def get_beh_path(
        sid: str,
) -> Path:
    root = get_main_path()
    group = get_group_letter(sid)
    return root / 'raw' / f'BIDS_Data_WP73{group}' / f'sub-{sid}' / 'ses-1' / 'beh'


def get_epo_path(
        sid: str,
) -> Path:
    root = get_main_path()
    group = get_group_letter(sid)
    return root / 'epo' / f'WP73{group}' / f'sub-{sid}'


def get_derivatives_path(
        sid: str,
) -> Path:
    root = get_main_path()
    group = get_group_letter(sid)
    return root / 'intermediate' / f'WP73{group}' / f'sub-{sid}'


def get_data_path(
        proc_stage: str,
        sid: str,
        acq: str | None = None,
        task: str | None = 'SpaNav',
) -> Path:
    include_fname = acq is not None and task is not None
    fname = f'sub-{sid}_ses-1_task-{task}_acq-{acq}' if include_fname else ''  # BIDS name
    fpath, fext = None, None
    proc_stage = proc_stage.lower()  # easier for following comparison with strings

    if 'raw' in proc_stage:
        fext = 'vhdr'
        fpath = get_raw_eeg_path(sid)

    elif 'beh' in proc_stage:
        fext = 'txt'
        fpath = get_beh_path(sid)
        fname += '_beh'

    else:
        fext = 'fif'

        if 'epo' in proc_stage:
            fext = 'fif'
            fpath = get_epo_path(sid)
            fname += '_desc-epo_eeg'

        else:
            deriv_path = get_derivatives_path(sid)

            if 'annot' in proc_stage and 'reannot' not in proc_stage:
                fpath = deriv_path / '01_annot'
                fname += '_desc-annot_eeg'

            elif 'filt' in proc_stage.lower():
                fpath = deriv_path / '02_filt_ds'
                fname += '_desc-FiltDs_eeg'

            elif 'reannot' in proc_stage:
                fpath = deriv_path / '03_reannot'
                fname += '_desc-reannot_eeg'

            elif 'ica' in proc_stage:
                fpath = deriv_path / '04_ica'
                fname += '_desc-ica_eeg'

            elif 'reconst' in proc_stage:
                fpath = deriv_path / '05_reconstructed'
                fname += '_desc-reconst_eeg'

            elif 'preproc' in proc_stage:
                fpath = deriv_path / '06_preproc'
                fname += '_desc-preproc_eeg'

    if fpath is None or fext is None:
        raise ValueError('Please pass a valid proc_stage!')

    set_for_save(fpath)

    if include_fname:
        return fpath / f"{fname}.{fext}"
    else:
        return fpath


def get_clean_eeg_path(
        sid: str,
        acq: str | None = None,
        task: str | None = 'SpaNav'
) -> Path:
    return get_data_path('preproc', sid, acq, task)


def get_outputs_path(
        sid: str | None = None,
) -> Path:
    root = get_main_path()
    if sid is None:
        return set_for_save(root / 'outputs')
    else:
        # If a subject ID is passed, then the group-specific path is returned
        group = get_group_letter(sid)
        return set_for_save(root / 'outputs' / f'WP73{group}')


def get_tables_path(
) -> Path:
    outputs_path = get_outputs_path()
    return outputs_path / 'Tables'


def get_sids(
    test: bool = False,
) -> list[str]:
    """

    :param test:
    :return:
    """
    raw_path = get_main_path() / 'raw'
    sids = []
    for element in os.listdir(raw_path):
        group_path = os.path.join(raw_path, element)
        if os.path.isdir(group_path):
            group_sids = [
                sid.replace('sub-73', '') for sid in os.listdir(group_path)
                if os.path.isdir(os.path.join(group_path, sid))
            ]
            sids += group_sids
    sids = sorted(sids)
    return sids if not test else [sids[0]]


def get_sid_cids(
        sid: str,
        test: bool = False,
) -> list[str]:
    """

    :param sid:
    :param task:
    :param test:
    :return:
    """
    cids = []
    raw_path = get_clean_eeg_path(sid, task='SpaNav')
    sid_files = os.listdir(raw_path)
    for file in sid_files:
        if file.endswith('final_raw.fif') or file.endswith('iclean-raw.fif') :
            cid, _, _ = parse_prepro_fname(file)
            cids.append(cid)
            if len(cids) > 0 and test:  # stop after finding first cid when in testing mode
                break
            else:
                continue
    return cids


def get_sid_cid_from_block(
        sid: str,
        block_n: int | str,
) -> str:
    """

    :param sid:
    :param block_n:
    :return:
    """
    stim_conds = "|".join(('HF', 'iTBS', 'cTBS'))
    stim_file_path = get_main_path() / 'Raw' / sid / 'stimulations.xlsx'
    conv_table = pd.read_excel(stim_file_path)
    block_str_variants = [f'block{block_n}', f'block_{block_n}']  # possible variants of how block was reported in excel file
    block_str_variants_lower = {v.lower() for v in block_str_variants}  # normalize to lower once

    block_str = None
    for col in conv_table.columns:
        col_lower = col.lower()
        if col_lower in block_str_variants_lower:
            block_str = col
            break
    else:
        # This runs only if the loop completes with no break
        raise ValueError(
            f'File stimulations.xlsx of subject {sid} does not contain a valid column for block {block_n}'
        )

    conv_cell = str(conv_table.loc[0, block_str])
    match = re.search(stim_conds, conv_cell)

    if match:
        return f'task_{match.group(0)}_{block_n}'  # group(0) returns the entire matched string
    else:
        raise ValueError(
            f'Conversion cell for block {block_n} in stimulations.xlsx of subject {sid} does not contain a valid stimulation '
            f'condition ID (contains "{conv_cell}")')


def get_block_stim(
        sid: str,
        block_n: int | str,
) -> str:
    """

    :param sid:
    :param block_n:
    :return:
    """
    stim_conds = "|".join(('HF', 'iTBS', 'cTBS'))
    cid = get_sid_cid_from_block(sid, block_n)
    return f'task_{re.search(stim_conds, cid).group(0)}'  # group(0) returns the entire matched string



def get_ti_positions(
        sid: str,
) -> list:
    """

    :param sid:
    :return:
    """
    exp_grp = get_group_letter(sid)
    stim_file_path = get_main_path() / f'Data_WP73{exp_grp}' / 'TI_and_EEG' / 'Montage' / f'73{sid}'
    spanav_files = list(stim_file_path.glob(f'log_73{sid}*.csv'))
    if len(spanav_files) != 1:
        raise RuntimeError(f'Expected exactly one SpaNav CSV for {sid}, found {len(spanav_files)}')
    conv_table = pd.read_csv(spanav_files[0], sep=';')
    eeg_ti_positions = conv_table.loc[:, 'Old channel name'].to_list()
    return eeg_ti_positions

