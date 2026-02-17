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
        save_dir: str,
        fname: str,
        fig: matplotlib.figure.Figure | None = None,
        sid: str | None = None,
        dpi: int = 900,
        prevent_overwrite: bool = False,
        **kwargs,
) -> None:
    save_path = set_for_save(get_outputs_path(sid) / save_dir)
    full_save_path = save_path / fname

    if prevent_overwrite and os.path.exists(full_save_path):
        prefix = 'NEW_'
        full_save_path = save_path / f'{prefix}_{fname}'

    if fig is None:
        plt.savefig(full_save_path, dpi=dpi, **kwargs)
    else:
        fig.savefig(full_save_path, dpi=dpi, **kwargs)


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
):
    """

    :param sid:
    :param cid:
    :param block_n:
    :return:
    """
    # Keep blinding atm
    return f'block{block_n}' if isinstance(block_n, int) or isinstance(block_n, np.int_) else (block_n if block_n.startswith('block') else f'block{block_n}')


def get_group_letter(
        sid: str,
) -> str:
    return 'T' if 't' in sid.lower() else 'A'


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


def parse_epo_fname(
        fname: str,
        sid: str | None = None,
) -> tuple[str, str | None, str]:
    """

    :param fname:
    :param sid:
    :return:
    """
    if fname.startswith('RS'):
        block_n = None
        m = re.match(r"RS_(.+?)_(.+?)-epo\.fif$", fname)
        if m:
            rs_cond, epo_type = m.groups()
            cond = f'RS_{rs_cond}'
        else:
            other_m = re.match(r"RS_(.+?)-epo\.fif$", fname)
            (epo_type, ) = other_m.groups()
            cond = 'RS'
    else:
        m = re.match(r".*block(.+?)_(.+?)-epo\.fif$", fname)  # .* allows anything before
        block_n, epo_type = m.groups()

        BLINDING = get_blinding()
        if BLINDING:
            runned_blocks = len(get_sid_cids(sid, test=False))
            if runned_blocks == 4:  # this is a patient (4 blocks runned)
                cond = 'A' if int(block_n) in (1, 4) else 'B'  # blocks 1-4 of conditions ABBA
            else:  # this is a healthy control (6 blocks runned)
                cond = 'A' if int(block_n) in (1, 6) else ('B' if int(block_n) in (2, 5) else 'C')  # blocks 1-6 of conditions ABCCBA
        else:
            cond = reveal_cid(sid, block_n=block_n)

    return cond, block_n, epo_type


def parse_prepro_fname(
        fname: str,
) -> tuple[str | None, str | None, str]:
    """

    :param fname:
    :return:
    """
    cid = None
    epo_type = 'Raw'
    block_n = None
    if fname.startswith('RS'):
        block_n = None
        m = re.match(r"RS_(.+?)-raw\.fif$", fname)
        if m:
            (rs_cond, ) = m.groups()
            cid = f'RS_{rs_cond}'
    else:
        m = re.match(r".*block(.+?)_(.+?)_raw\.fif$", fname)  # .* allows anything before
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
    BLINDING = get_blinding()
    print(
        f"{BLINDING = }"
    )
    if BLINDING:
        return {
            'A': plt.get_cmap('tab20')(18),
            'B': plt.get_cmap('tab20')(19),
        }
    else:
        return {
            # 'HF': '#4293f5',  # blue
            'HF': '#a558ed',
            # 'cTBS': '#f77c99',  # pink
            'cTBS': '#f5a6d4',
            # 'iTBS':  '#5ad676',  # green
            'iTBS': '#edcd58',
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
        band_str: str,
) -> str:
    return map_band_labels()[band_str]


if __name__ == '__main__':
    print(get_sids())