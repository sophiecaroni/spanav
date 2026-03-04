"""
********************************************************************************
    Title: Parsing utilities

    Author: Sophie Caroni
    Date of creation: 18.02.2026

    Description:
    This script contains helper functions to parse strings / file names.
********************************************************************************
"""
import re

from pathlib import Path


def get_group_letter(
        sid: str,
) -> str:
    if sid[-2:].isdigit():
        if 't' in sid.lower():
            return 'T'
        elif 'a' in sid.lower():
            return 'A'
    raise ValueError(f'Subject ID {sid} is invalid')


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

        cond = get_stim(sid, block_n)

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


def check_path_sid(fpath: Path) -> Path:
    group_dir_re = re.compile(r"^(?:(?:BIDS_Data|Data)_)WP73[AT]$", re.IGNORECASE)
    subject_dir_re = re.compile(r"^(?:sub[- ]*)?(?:(?:WP)?73)?(?P<grp>[TA])(?P<num>0[1-9]|[1-9]\d)$", re.IGNORECASE,)

    parts = list(fpath.parts)

    for i, part in enumerate(parts):

        if group_dir_re.match(part):  # don't edit group directories in this function
            continue

        match = subject_dir_re.match(part)
        if not match:
            continue

        grp = match.group("grp").upper()
        sid_n = int(match.group("num"))

        # Replace elements in parts with correct subject-direcotry name
        parts[i] = f"sub-{grp}{sid_n:02d}"

    return Path(*parts)


def get_stim(sid: str, acq: str | int) -> str:
    acq = str(acq)  # convert in case it was input the n of block_n as acq

    # Return unchanged if rsEEG recording (no stimulation protocol)
    if acq.lower().startswith('rs') or acq.lower().startswith('pre') or acq.lower().startswith('post'):
        return acq

    # Otherwise, extract block condition from table that maps blocks to stimulation conditions
    block = acq if acq.startswith('block') else f'block{acq}'
    from spanav_eeg_utils.io_utils import load_stim_mapping_table
    map_table = load_stim_mapping_table(sid)
    cond = map_table.loc[f'73{sid}', block.title()]

    return cond
