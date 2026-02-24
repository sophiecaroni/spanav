"""
********************************************************************************
    Title: SpaNav-study specific utilities

    Author: Sophie Caroni
    Date of creation: 06.10.2025

    Description:
    This script contains helper functions specific to the SpaNav study.
********************************************************************************
"""
import matplotlib.pyplot as plt
import mne
import numpy as np

from spanav_eeg_utils.config_utils import get_blinding


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
        'Stasis',
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
        'Stasis',
    ]


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
        'Stasis': 'tab:blue',
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
        'Stasis': 'Stasis',
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

