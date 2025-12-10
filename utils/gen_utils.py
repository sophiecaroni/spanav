"""
********************************************************************************
    Title: Helper functions

    Author: Sophie Caroni
    Date of creation: 06.10.2025

    Description:
    This script contains helper functions to facilitate recurrent tasks.
********************************************************************************
"""
import matplotlib.pyplot as plt
import os
from contextlib import contextmanager

import mne
import numpy as np
import pandas as pd
import re

SEED = 81025


@contextmanager
def plot_context(
):
    increase = 2
    params = {
        'figure.dpi': 300,
        'axes.grid': False,
        'font.size': 7 + increase,              # General fontsize
        'axes.titlesize': 7 + increase,         # Subplot titles
        'figure.titlesize': 9 + increase,       # Overall figure title
        'axes.labelsize': 7 + increase,         # Axis labels (x and y)
        'xtick.labelsize': 6 + increase,        # X-axis tick labels
        'ytick.labelsize': 6 + increase,        # Y-axis tick labels
        'legend.fontsize': 6 + increase,        # Legend text
        'legend.title_fontsize': 6 + increase,       # Legend title

        # Line widths
        'axes.linewidth': 1,         # Border (spines) width
        'xtick.major.width': 1,      # X tick line width
        'ytick.major.width': 1,      # Y tick line width

        # Tick appearance
        'xtick.major.size': 2,       # Tick length
        'ytick.major.size': 2
    }
    with plt.rc_context(rc=params):
        yield


def set_for_save(
        save_path: str,
) -> str:
    """

    :param save_path:
    :return:
    """
    os.makedirs(save_path, exist_ok=True)
    return save_path


def layout_subplots_grid(
        n: int,
        max_cols: int = 6,
) -> tuple[int, int]:
    """

    :param n:
    :param max_cols:
    :return:
    """
    ncols = int(np.ceil(np.sqrt(n)))
    ncols = min(max(ncols, 1), max_cols)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def get_trigger_str(
       trigger: str,
) -> str:
    """

    :param trigger:
    :return:
    """
    trigger_map = {
        'rs_start': 'Stimulus/S  8',  # RS begins
        'enc_instr_gone': 'Stimulus/S  2',  # encoding instruction disappears
        'retr_obj_gone': 'Stimulus/S 10',  # object presentated in retrieval disappears
        'task_rs_start': 'Stimulus/S 14',  # task-RS begins (at the beginning of encoding and retrieval phases, 5s or 15s)
        'trial_start': 'Stimulus/S  4',  # object is shown
        'trial_end': 'Response/R  2',  # S32  # object is collected
    }
    return trigger_map[trigger]


def get_nrows_ncols(
        subplots_elements: list
) -> tuple[int, int]:
    n = int(len(subplots_elements))
    nrows, ncols = (2, int(np.ceil(n/2))) if n > 2 else (1, n)
    return nrows, ncols


def get_sids(
    test: bool = False,
    include_04: bool = False,
) -> list[str]:
    """

    :param test:
    :param include_04:
    :return:
    """
    rec_folders = os.listdir(f'{get_wd()}/data')
    sids = sorted([f for i, f in enumerate(rec_folders) if not (f.startswith('.') or f.startswith('test') or f.startswith('to_start') or f.endswith('csv'))])
    if not include_04:
        sids.remove('04')
    return sids if not test else ['04']


def get_conds(
        sid: str,
        task: bool,
        test: bool = False,
) -> list[str]:
    """

    :param sid:
    :param task:
    :param test:
    :return:
    """
    if sid == '02':
        if task:
            return ['task_HF'] if test else [
                'task_HF',
                'task_iTBS',
                'task_cTBS'
            ]
        else:
            return ['RS_EO'] if test else [
                'RS_EO',
                'RS_EC'
            ]
    elif sid == '03':
        if task:
            return ['task_HF'] if test else [
                'task_HF',
                'task_iTBS'
            ]
        else:
            return ['RS_pre_EO']
    else:
        if task:
            return ['task_HF'] if test else [
                'task_HF',
                'task_iTBS',
                'task_cTBS'
            ]
        else:
            return ['RS_pre_EO'] if test else [
                'RS_pre_EO',
                'RS_pre_EC',
                'RS_post_EO',
                'RS_post_EC',
            ]


def get_sid_cids(
        sid: str,
        task: bool,
        test: bool = False,
) -> list[str]:
    """

    :param sid:
    :param task:
    :param test:
    :return:
    """
    cids = []
    sid_files = os.listdir(f'{get_wd()}/data/{sid}/eeg/RawPreprocessed')
    for file in sid_files:
        if file.endswith('.fif'):
            if not task and file.startswith('RS'):
                cid = parse_prepro_filename(file)
                cids.append(cid)
            elif task and file.startswith('task'):
                cid = parse_prepro_filename(file)
                cids.append(cid)
            else:
                continue
    return cids if not test else [cids[0]]


def get_band_freqs(
        band: str,
) -> tuple[float, float]:
    """

    :param band:
    :return:
    """
    if band == 'theta':
        return 4.0, 8.0
    elif band == 'alpha':
        return 8.0, 12.0
    elif band == 'beta':
        return 12.0, 30.0


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
    conv_table = pd.read_excel(f'{get_wd()}/data/{sid}/stimulations.xlsx')
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


def get_cid_with_block(
        sid: str,
        cid: str,
) -> str:
    """

    :param sid:
    :param cid:
    :return:
    """
    if sid == '02':
        return {
            'task_HF': 'task_HF_1',
            'task_iTBS': 'task_iTBS_2',
            'task_cTBS': 'task_cTBS_3'
        }[cid]
    elif sid == '03':
        return {
            'task_HF': 'task_HF_1',
            'task_iTBS': 'task_iTBS_2',
        }[cid]


def reveal_cid(
        sid: str,
        cid: str | None = None,
        block_n: int | str | None = None
):
    """

    :param sid:
    :param cid:
    :param block_n:
    :return:
    """
    if cid is not None:
        if cid[-1] in ['1', '2', '3', '4', '5', '6']:
            print(f'Condition ID of subject {sid} is already the full one: {cid}')
            return cid
        else:
            if sid == '02' or sid == '03':
                return get_cid_with_block(sid, cid)
            else:
                raise ValueError(f'Something is wrong here. {sid = }, {cid = }')
    else:
        return get_sid_cid_from_block(sid, block_n)


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
    sid_files = os.listdir(f'{get_wd()}/data/{sid}')
    conv_file = [file_name for file_name in sid_files if file_name.startswith('SpaNav') and file_name.endswith('csv')][0]
    conv_table = pd.read_csv(f'{get_wd()}/data/{sid}/{conv_file}', sep=';')
    eeg_ti_positions = conv_table.loc[:, 'Old channel name'].to_list()
    return eeg_ti_positions


def get_task_epo_types(
        test: bool = False,
) -> list:
    """

    :param test:
    :return:
    """
    return [
        'ObjPres',
        'MovOn',
        'ContMov',
        'Static',
    ] if not test else ['ContMov']


def get_epo_types(
) -> list:
    """

    :return:
    """
    return [
        # 'RS',
        'ObjPres',
        'MovOn',
        'ContMov',
        'Static',
    ]


def get_wd(
        ext=True,
):
    """

    :param ext:
    :return:
    """
    if ext:
        return '/Volumes/My Passport/SpaNav/Sophie_backup'
    else:
        return '/Users/sophiecaroni/epfl_hes/spanav-tbi'


def parse_epo_filename(
        filename: str
) -> tuple[str, str | None, str]:
    """

    :param filename:
    :return:
    """
    if filename.startswith('RS'):
        m = re.match(r"RS_(.+?)_(.+?)-epo\.fif$", filename)
        if m:
            rs_cond, epo_type = m.groups()
            cond = f'RS_{rs_cond}'
        else:
            other_m = re.match(r"RS_(.+?)-epo\.fif$", filename)
            (epo_type, ) = other_m.groups()
            cond = 'RS'
        block_n = 0
    else:
        m = re.match(r"task_(.+?)_(.+?)_(.+?)-epo\.fif$", filename)
        if m:
            cond, block_n, epo_type = m.groups()
        else:
            other_m = re.match(r"task_(.+?)_(.+?)-epo\.fif$", filename)
            cond, epo_type = other_m.groups()
            block_n = 0

    return cond, block_n, epo_type


def parse_prepro_filename(
        filename: str
) -> str | None:
    """

    :param filename:
    :return:
    """
    cid = None
    if filename.startswith('RS'):
        m = re.match(r"RS_(.+?)-raw\.fif$", filename)
        if m:
            (rs_cond, ) = m.groups()
            cid = f'RS_{rs_cond}'
    else:
        m = re.match(r"task_(.+?)-raw\.fif$", filename)
        if m:
            (task_cid, ) = m.groups()
            cid = f'task_{task_cid}'

    return cid


def get_ch_by_region(
        info: mne.Info,
) -> dict:
    """

    :param info:
    :return:
    """
    ch_by_region = dict(Left=[], Midline=[], Right=[])
    for ch in info.ch_names:
        last_char = ch[-1].lower()  # in 10/20, last char codes hemisphere
        if last_char == 'z':
            ch_by_region['Midline'].append(ch)
        elif last_char.isdigit():
            if int(last_char) % 2 == 0:
                ch_by_region['Right'].append(ch)
            else:
                ch_by_region['Left'].append(ch)
        else:
            print(f'Skipping channel {ch}')
    return ch_by_region


if __name__ == '__main__':
    pass