"""
********************************************************************************
    Title: Helper functions

    Author: Sophie Caroni
    Date of creation: 06.10.2025

    Description:
    This script contains helper functions to facilitate recurrent tasks.
********************************************************************************
"""
import matplotlib.figure
import matplotlib.pyplot as plt
import os
import mne
import numpy as np
import pandas as pd
import re

from contextlib import contextmanager
from pathlib import Path

SERVER = False
PILOT = False
SEED = 81025


@contextmanager
def plot_context(
):
    increase = 0
    params = {
        'figure.dpi': 300,
        'axes.grid': False,
        'font.size': 7 + increase,              # General fontsize
        'axes.titlesize': 7 + increase,         # Subplot titles
        'figure.titlesize': 9 + increase,       # Overall figure title
        'axes.labelsize': 7 + increase,         # Axis labels (x and y)
        'xtick.labelsize': 6 + increase,        # X-axis tick labels
        'ytick.labelsize': 6 + increase,        # Y-axis tick labels
        'legend.fontsize': 5 + increase,        # Legend text
        'legend.title_fontsize': 5 + increase,       # Legend title

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
        save_path: str | Path,
) -> str | Path:
    """

    :param save_path:
    :return:
    """
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_figure(
        save_path: str | Path,
        file_name: str,
        fig: matplotlib.figure.Figure | None = None,
        dpi: int = 900,
        prevent_overwrite: bool = False,
        **kwargs,
) -> None:
    save_path = set_for_save(save_path)
    # final_path = save_path / file_name
    final_path = f'{save_path}/{file_name}'

    if prevent_overwrite and os.path.exists(final_path):
        prefix = 'NO_PSD_KWARGS'
        final_path = f'{save_path}/{prefix}_{file_name}'

    if fig is None:
        plt.savefig(final_path, dpi=dpi, **kwargs)
    else:
        fig.savefig(final_path, dpi=dpi, **kwargs)


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
) -> list[str]:
    """

    :param test:
    :return:
    """
    raw_dir = get_eeg_path() / '00_raw'
    rec_folders = os.listdir(raw_dir)
    sids = sorted([f for i, f in enumerate(rec_folders)
                   if not (f.startswith('.') or f.startswith('test') or f.startswith('to_start') or f.endswith('csv')
                           or f.startswith('WITH'))
                   ])
    return sids if not test else sids[0]


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
    raw_dir = get_eeg_path() / '03_ica' / sid
    sid_files = os.listdir(raw_dir)
    for file in sid_files:
        if file.endswith('final_raw.fif'):
            if not task and file.startswith('RS'):
                cid = parse_prepro_filename(file)
                cids.append(cid)
            elif task and file.startswith('task'):
                cid = parse_prepro_filename(file)
                cids.append(cid)
            else:
                continue
    return cids if not test else [cids[0]]


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
    stim_file_path = get_eeg_path() / '00_raw' / sid / 'stimulations.xlsx'
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
        block_n: int | str | None = None,
        pilot: bool = PILOT,
):
    """

    :param sid:
    :param cid:
    :param block_n:
    :param pilot:
    :return:
    """
    if pilot:
        if cid is not None:
            if cid[-1] in ['1', '2', '3', '4', '5', '6'] or cid.startswith('RS'):
                print(f'Condition ID of subject {sid} is already the full one: {cid}')
                return cid
            else:
                if sid == '02' or sid == '03':
                    return get_cid_with_block(sid, cid)
                else:
                    raise ValueError(f'Something is wrong here. {sid = }, {cid = }')
        else:
            return get_sid_cid_from_block(sid, block_n)
    else:
        # Keep blinding atm
        return f'block{block_n}' if isinstance(block_n, int) or isinstance(block_n, np.int_) else (block_n if block_n.startswith('block') else f'block{block_n}')


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

    stim_file_path = get_eeg_path() / '00_raw' / sid
    spanav_files = list(stim_file_path.glob('SpaNav_*.csv'))
    if len(spanav_files) != 1:
        raise RuntimeError(f'Expected exactly one SpaNav CSV for {sid}, found {len(spanav_files)}')
    conv_table = pd.read_csv(spanav_files[0], sep=';')
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


def get_main_path(
        server: bool = SERVER,
) -> Path:
    """

    :param server:
    :return:
    """
    if server:
        return Path('/Volumes/Hummel-Data/mnt/Hummel-Data/TI/TI_EEG')
    else:
        return Path('/Volumes/My Passport/SpaNav/Sophie_backup')


def get_eeg_path(
        server: bool = SERVER,
) -> Path:
    main_root = get_main_path()
    if server:
        return main_root / 'EEG'
    else:
        exp_phase = get_exp_phase()
        return main_root / 'data' / exp_phase / 'EEG'


def get_behav_path(
        server: bool = SERVER,
) -> Path:
    main_root = get_main_path()
    if server:
        return main_root / 'behav'
    else:
        exp_phase = get_exp_phase()
        return main_root / 'data' / exp_phase / 'behav'


def get_outputs_path(
        server: bool = SERVER,
) -> Path:
    main_root = get_main_path()
    if server:
        return main_root / 'outputs'
    else:
        exp_phase = get_exp_phase()
        return main_root / 'outputs' / exp_phase


def get_tables_path(
) -> Path:
    output_root = get_outputs_path()
    return output_root / 'tables'


def get_exp_phase(
        pilot: bool = False,
) -> str:
    if pilot:
        return 'pilot'
    else:
        return 'main'


def parse_epo_filename(
        filename: str,
        pilot: bool = PILOT,
) -> tuple[str, str | None, str]:
    """

    :param filename:
    :param pilot:
    :return:
    """
    if filename.startswith('RS'):
        block_n = None
        m = re.match(r"RS_(.+?)_(.+?)-epo\.fif$", filename)
        if m:
            rs_cond, epo_type = m.groups()
            cond = f'RS_{rs_cond}'
        else:
            other_m = re.match(r"RS_(.+?)-epo\.fif$", filename)
            (epo_type, ) = other_m.groups()
            cond = 'RS'
    else:
        if pilot:
            m = re.match(r".*task_(.+?)_(.+?)_(.+?)-epo\.fif$", filename)  # .* allows anything before
            if m:
                cond, block_n, epo_type = m.groups()
            else:
                other_m = re.match(r".*task_(.+?)_(.+?)-epo\.fif$", filename)  # .* allows anything before
                cond, epo_type = other_m.groups()
                block_n = None
        else:
            m = re.match(r".*block(.+?)_(.+?)-epo\.fif$", filename)  # .* allows anything before
            block_n, epo_type = m.groups()
            cond = f'block{block_n}'

    return cond, block_n, epo_type


def parse_prepro_filename(
        filename: str,
        pilot: bool = PILOT,
) -> tuple[str | None, str | None, str]:
    """

    :param filename:
    :param pilot:
    :return:
    """
    cid = None
    epo_type = 'Raw'
    block_n = None
    if filename.startswith('RS'):
        block_n = None
        m = re.match(r"RS_(.+?)-raw\.fif$", filename)
        if m:
            (rs_cond, ) = m.groups()
            cid = f'RS_{rs_cond}'
    else:
        if pilot:
            m = re.match(r".*task_(.+?)-raw\.fif$", filename)  # .* allows anything before
            if m:
                (task_cid, ) = m.groups()
                cid = f'task_{task_cid}'
                block_n = cid[-1]
        else:
            m = re.match(r".*block(.+?)_(.+?)_raw\.fif$", filename)  # .* allows anything before
            block_n, _ = m.groups()
            cid = f'block{block_n}'

    return cid, block_n, epo_type


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


def get_epo_palette(
) -> dict:
    return {
        'ContMov': 'tab:green',
        'Static': 'tab:blue',
        'MovOn': 'OrangeRed',
        'ObjPres': 'orange',
        'Raw': 'purple',
    }


if __name__ == '__main__':
    pass