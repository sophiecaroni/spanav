"""
********************************************************************************
    Title: SpaNav-study specific utilities

    Author: Sophie Caroni
    Date of creation: 06.10.2025

    Description:
    This script contains helper functions specific to the SpaNav study.
********************************************************************************
"""
import mne
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.io_utils as io


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


def get_full_sid(
        in_sid: str,
) -> str:
    in_sid = in_sid.lower()  # work in lowercase

    # Case 1: in_sid is already the correct participant ID
    if in_sid.startswith('73') and len(in_sid) == 5:
        out_sid = in_sid

    # Case 2: in_sid is missing the initial 73
    elif (in_sid.startswith('t') or in_sid.startswith('a')) and 2 <= len(in_sid) <= 3:  # len is 2 if participant number is not in 02d format
        out_sid = f'73{in_sid[0]}{int(in_sid[1:]):02d}'  # make sure participant number is in 02d format

    # Case 3: in_sid isn't any of the above cases -> invalid, re-ask input and recall function
    else:
        in_sid = input(f'Given participant ID ({in_sid}) is unrecognized. Please input a valid one: ')
        out_sid = get_full_sid(in_sid)

    return out_sid.upper()  # letters in participant IDs are always capitalized


def group_cids_by_cond(
        sid: str,
        test: bool,
        cids: list[str] | None = None,
) -> dict[str, list[str]]:
    if cids is None:
        cids = io.get_sid_blocks(sid, test)
    cids_by_cond = {}
    for cid in cids:
        cond = prs.get_stim(sid, cid)
        cids_by_cond.setdefault(cond, []).append(cid)
    return cids_by_cond


