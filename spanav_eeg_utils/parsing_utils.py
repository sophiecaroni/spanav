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

from spanav_eeg_utils.config_utils import get_blinding
from spanav_eeg_utils.spanav_utils import reveal_cid, get_group_letter


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
            runned_blocks = 4 if get_group_letter(sid) == 'T' else 6
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