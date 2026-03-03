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
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.config_utils as cfg


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
    BLINDING = cfg.get_blinding()
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


def get_full_pid(
        in_pid: str,
) -> str:
    in_pid = in_pid.lower()  # work in lowercase

    # Case 1: in_pid is already the correct participant ID
    if in_pid.startswith('73') and len(in_pid) == 5:
        out_pid = in_pid

    # Case 2: in_pid is missing the initial 73
    elif (in_pid.startswith('t') or in_pid.startswith('a')) and 2 <= len(in_pid) <= 3:  # len is 2 if participant number is not in 02d format
        in_pid = f'{in_pid[0]}{int(in_pid[1:]):02d}'  # make sure participant number is in 02d format
        out_pid = f'73{in_pid}'

    # Case 3: in_pid isn't any of the above cases -> invalid, re-ask input and recall function
    else:
        in_pid = input(f'Given participant ID ({in_pid}) is unrecognized. Please input a valid one: ')
        out_pid = get_full_pid(in_pid)

    return out_pid.upper()  # letters in participant IDs are always capitalized


def group_cids_by_cond(
        sid: str,
        test: bool,
        cids: list[str] | None = None,
) -> dict[str, list[str]]:
    if cids is None:
        cids = io.get_sid_cids(sid, test)
    cids_by_cond = {}
    for cid in cids:
        cond = prs.get_stim(sid, cid)
        cids_by_cond.setdefault(cond, []).append(cid)
    return cids_by_cond


