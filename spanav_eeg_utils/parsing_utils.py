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


def get_cid_cond(
        sid: str,
        cid: str,
) -> str:
    runned_blocks = 4 if sid.startswith('T') else 6
    block_n = cid[-1]
    if runned_blocks == 4:  # this is a patient (4 blocks runned)
        cond = 'A' if int(block_n) in (1, 4) else 'B'  # blocks 1-4 of conditions ABBA
    else:  # this is a healthy control (6 blocks runned)
        cond = 'A' if int(block_n) in (1, 6) else (
            'B' if int(block_n) in (2, 5) else 'C')  # blocks 1-6 of conditions ABCCBA
    return cond


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
            cond = get_cid_cond(sid, block_n)
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


def get_group_letter_from_path(
        path_parts: tuple[str, ...]
) -> str:
    # Search for group-dir (e.g. BIDS_Data_WP73T, or WP73A, or Data_WP73A, ...)
    exact_pattern = re.compile(
        r"^(?:WP)?73?([AT])(0[1-9]|[1-9][0-9])$"
    )

    for element in path_parts:
        el = element.upper()  # make sure we deal with upper case strings only

        # ---- Exact match case ----
        match = exact_pattern.match(el)
        if match:
            return match.group(1)  # A/T letter part is retured

        # ---- Substring case ----
        if "73T" in el:
            return "T"
        if "73A" in el:
            return "A"

    raise ValueError(f"Cannot infer group letter from {path_parts}.")

