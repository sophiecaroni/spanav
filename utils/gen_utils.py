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
import configparser

from contextlib import contextmanager
from pathlib import Path

config = configparser.ConfigParser()
config.read('../config.ini')

PILOT = config.getboolean('General', 'pilot')
SERVER = config.getboolean('General', 'server')
BLINDING = config.getboolean('General', 'blinding')


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

    # Other specific customizations are stored in the despine.mplstyle file
    style_path = (Path(__file__).resolve().parent / ".." / "visualization" / "despine.mplstyle").resolve()

    # Create figure applying the custom context
    with plt.rc_context(rc=params), plt.style.context(str(style_path)):
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


def get_pids(
    test: bool = False,
) -> list[str]:
    """

    :param test:
    :return:
    """
    raw_dir = get_eeg_path() / '00_raw'
    rec_folders = os.listdir(raw_dir)
    pids = sorted([f for i, f in enumerate(rec_folders) if not (f.startswith('.')) and not (f.startswith('test'))])
    return pids if not test else [pids[0]]


def get_conds(
        pid: str,
        task: bool,
        test: bool = False,
) -> list[str]:
    """

    :param pid:
    :param task:
    :param test:
    :return:
    """
    if pid == '02':
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
    elif pid == '03':
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


def get_pid_cids(
        pid: str,
        test: bool = False,
) -> list[str]:
    """

    :param pid:
    :param task:
    :param test:
    :return:
    """
    cids = []
    raw_dir = get_clean_eeg_path() / pid
    pid_files = os.listdir(raw_dir)
    for file in pid_files:
        if file.endswith('final_raw.fif') or file.endswith('iclean-raw.fif') :
            cid, _, _ = parse_prepro_filename(file)
            cids.append(cid)
            if len(cids) > 0 and test:  # stop after finding first cid when in testing mode
                break
            else:
                continue
    return cids


def get_pid_cid_from_block(
        pid: str,
        block_n: int | str,
) -> str:
    """

    :param pid:
    :param block_n:
    :return:
    """
    stim_conds = "|".join(('HF', 'iTBS', 'cTBS'))
    stim_file_path = get_eeg_path() / '00_raw' / pid / 'stimulations.xlsx'
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
            f'File stimulations.xlsx of participant {pid} does not contain a valid column for block {block_n}'
        )

    conv_cell = str(conv_table.loc[0, block_str])
    match = re.search(stim_conds, conv_cell)

    if match:
        return f'task_{match.group(0)}_{block_n}'  # group(0) returns the entire matched string
    else:
        raise ValueError(
            f'Conversion cell for block {block_n} in stimulations.xlsx of participant {pid} does not contain a valid stimulation '
            f'condition ID (contains "{conv_cell}")')


def get_cid_with_block(
        pid: str,
        cid: str,
) -> str:
    """

    :param pid:
    :param cid:
    :return:
    """
    if pid == '02':
        return {
            'task_HF': 'task_HF_1',
            'task_iTBS': 'task_iTBS_2',
            'task_cTBS': 'task_cTBS_3'
        }[cid]
    elif pid == '03':
        return {
            'task_HF': 'task_HF_1',
            'task_iTBS': 'task_iTBS_2',
        }[cid]


def reveal_cid(
        pid: str,
        cid: str | None = None,
        block_n: int | str | None = None,
        pilot: bool = PILOT,
):
    """

    :param pid:
    :param cid:
    :param block_n:
    :param pilot:
    :return:
    """
    if pilot:
        if cid is not None:
            if cid[-1] in ['1', '2', '3', '4', '5', '6'] or cid.startswith('RS'):
                print(f'Condition ID of participant {pid} is already the full one: {cid}')
                return cid
            else:
                if pid == '02' or pid == '03':
                    return get_cid_with_block(pid, cid)
                else:
                    raise ValueError(f'Something is wrong here. {pid = }, {cid = }')
        else:
            return get_pid_cid_from_block(pid, block_n)
    else:
        # Keep blinding atm
        return f'block{block_n}' if isinstance(block_n, int) or isinstance(block_n, np.int_) else (block_n if block_n.startswith('block') else f'block{block_n}')


def get_block_stim(
        pid: str,
        block_n: int | str,
) -> str:
    """

    :param pid:
    :param block_n:
    :return:
    """
    stim_conds = "|".join(('HF', 'iTBS', 'cTBS'))
    cid = get_pid_cid_from_block(pid, block_n)
    return f'task_{re.search(stim_conds, cid).group(0)}'  # group(0) returns the entire matched string


def get_ti_positions(
        pid: str,
) -> list:
    """

    :param pid:
    :return:
    """

    stim_file_path = get_eeg_path() / '00_raw' / pid
    spanav_files = list(stim_file_path.glob('SpaNav_*.csv'))
    if len(spanav_files) != 1:
        raise RuntimeError(f'Expected exactly one SpaNav CSV for {pid}, found {len(spanav_files)}')
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


def get_clean_eeg_path(
        server: bool = SERVER,
) -> Path:
    main_root = get_main_path()
    if server:
        return main_root / 'EEG'
    else:
        exp_phase = get_exp_phase()
        clean_folder = '03_ica' if exp_phase == 'main' else 'RawClean'
        return main_root / 'data' / exp_phase / 'EEG' / clean_folder

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
        print(f"\n\n\n ### {exp_phase.upper()} ### \n\n\n ")
        return main_root / 'outputs' / exp_phase


def get_tables_path(
) -> Path:
    output_root = get_outputs_path()
    return output_root / 'tables'


def get_exp_phase(
        pilot: bool = PILOT,
) -> str:
    if pilot:
        return 'pilot'
    else:
        return 'main'


def parse_epo_filename(
        filename: str,
        pilot: bool = PILOT,
        pid: str | None = None,
) -> tuple[str, str | None, str]:
    """

    :param filename:
    :param pilot:
    :param pid:
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

            if BLINDING:
                runned_blocks = len(get_pid_cids(pid, test=False))
                if runned_blocks == 4:  # this is a patient (4 blocks runned)
                    cond = 'A' if int(block_n) in (1, 4) else 'B'  # blocks 1-4 of conditions ABBA
                else:  # this is a healthy control (6 blocks runned)
                    cond = 'A' if int(block_n) in (1, 6) else ('B' if int(block_n) in (2, 5) else 'C')  # blocks 1-6 of conditions ABCCBA
            else:
                cond = reveal_cid(pid, block_n=block_n)

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


def get_cond_palette(
) -> dict:
    if BLINDING or not PILOT:
        return {
            'A': plt.get_cmap('tab20')(18),
            'B': plt.get_cmap('tab20')(19),
        }
    else:
        return {
            'HF': plt.get_cmap('tab20')(0),
            'cTBS': plt.get_cmap('tab20')(2),
            'iTBS': plt.get_cmap('tab20')(3),
        }


def map_epo_type_labels(
) -> dict:
    return {
        'ObjPres': 'Object Presentation',
        'MovOn': 'Movement onset',
        'ContMov': 'Continuous movement',
        'Static': 'Static',
    }


def map_metric_labels(

) -> dict:
    return {
        'abs_pw': 'Absolute band-power',
        'rel_pw': 'Relative band-power',
        'osc_snr': 'FOOOF-based SNR',
    }


def map_metric_label(
        metric_str: str,
) -> str:
    return map_metric_labels()[metric_str]


def map_band_labels(

) -> dict:
    return {
        'theta': 'Theta',
        'alpha': 'Alpha',
        '38-42': 'Narrow Gamma',
    }


def get_band_label(
        metric_str: str,
) -> str:
    return map_metric_labels()[metric_str]


if __name__ == '__main__':
    pass