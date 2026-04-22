"""
********************************************************************************
    Title: I/O utilities

    Author: Sophie Caroni
    Date of creation: 18.02.2026

    Description:
    This script contains input/output helper functions.
********************************************************************************
"""
import os
import pandas as pd
import spanav_eeg_utils.config_utils as cfg
import spanav_eeg_utils.parsing_utils as prs
import warnings
from pathlib import Path


def set_for_save(
        save_path: Path,
        check_sid_strings: bool = True,
) -> Path:
    """

    :param check_sid_strings:
    :param save_path:
    :return:
    """
    if check_sid_strings:
        # Check correctness of the path
        save_path = prs.check_path_sid(save_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_main_path(
        server: bool | None = None,
) -> Path:
    """

    :param server:
    :return:
    """
    SERVER = server or cfg.get_server()
    if SERVER:
        return cfg.get_server_root()
    else:
        return cfg.get_local_root()


def get_raw_eeg_path(
        sid: str,
        server: bool | None = None,
) -> Path:
    root = get_main_path(server=server)
    group = prs.get_group_letter(sid)
    return root / 'raw' / f'BIDS_Data_WP73{group}' / f'sub-{sid}' / 'ses-1' / 'eeg'


def get_raw_beh_path(
        sid: str,
        acq: str | None = None,
        task: str | None = 'SpaNav',
) -> Path:
    # Build behavioral directory
    root = get_main_path()
    group = prs.get_group_letter(sid)
    beh_dir = root / 'raw' / f'BIDS_Data_WP73{group}' / f'sub-{sid}' / 'ses-1' / 'beh'
    set_for_save(beh_dir)  # Make sure the directory exist

    # Optionally include a file in the path
    include_fname = acq is not None and task is not None  # both acq and task are needed in behavioral data filenames
    fname = get_base_bids_filename(sid=sid, task=task, acq=acq) if include_fname else ''  # BIDS name

    # Return paths
    if include_fname:
        fname += '_beh.txt'
        return beh_dir / fname
    return beh_dir


def get_epo_beh_tables_path(
        sid: str,
        fname: str | None = None,
) -> Path:
    outputs_path = get_outputs_path(sid=sid)
    epo_beh_tables_path = set_for_save(outputs_path / 'Epo' / sid)
    if fname:
        return epo_beh_tables_path / fname
    return epo_beh_tables_path


def get_epo_path(
        sid: str,
) -> Path:
    root = get_main_path()
    group = prs.get_group_letter(sid)
    return root / 'epo' / f'WP73{group}' / f'sub-{sid}'


def get_derivatives_path(
        sid: str,
) -> Path:
    root = get_main_path()
    group = prs.get_group_letter(sid)
    return root / 'intermediate' / f'WP73{group}' / f'sub-{sid}'


def get_base_bids_filename(
        sid: str,
        task: str | None,
        acq: str | None,
) -> str:
    fname = f'sub-{sid}_ses-1'
    if task is not None:
        fname += f'_task-{task}'
    if acq is not None:
        fname += f'_acq-{acq}'
    return fname


def get_cont_path(
        proc_stage: str,
        sid: str,
        acq: str | None = None,
        task: str | None = 'SpaNav',
        server: bool | None = None,
) -> Path:
    include_fname = acq is not None and task is not None  # both acq and task are needed in behavioral data filenames
    fname = get_base_bids_filename(sid=sid, task=task, acq=acq) if include_fname else ''  # BIDS name
    fpath, fext = None, None
    proc_stage = proc_stage.lower()  # easier for following comparison with strings

    if 'raw' in proc_stage:
        fext = 'vhdr'
        fpath = get_raw_eeg_path(sid, server)
        fname += '_eeg'

    else:
        fext = 'fif'
        deriv_path = get_derivatives_path(sid)

        if 'annot' in proc_stage and 'reannot' not in proc_stage:
            fpath = deriv_path / '01_annot'
            fname += '_desc-annot_eeg'

        elif 'filt' in proc_stage:
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


def get_beh_data_path(
        sid: str,
        acq: str | None = None,
        task: str | None = 'SpaNav',
) -> Path:
    include_fname = acq is not None and task is not None  # both acq and task are needed in behavioral data filenames
    fname = get_base_bids_filename(sid=sid, task=task, acq=acq) if include_fname else ''  # BIDS name
    fname += '_beh.txt'
    fpath = get_raw_beh_path(sid)
    set_for_save(fpath)

    if include_fname:
        return fpath / fname
    else:
        return fpath


def get_epo_data_path(
        epo_type: str,
        sid: str,
        acq: str | None = None,
        task: str | None = 'SpaNav'
) -> Path:
    include_fname = acq is not None  # acq is needed in epoched data filenames (task not)
    fname = get_base_bids_filename(sid=sid, task=task, acq=acq) if include_fname else ''  # BIDS name
    fname += f'_desc-{epo_type}_eeg.fif'
    fpath = get_epo_path(sid)
    set_for_save(fpath)

    if include_fname:
        return fpath / fname
    else:
        return fpath


def get_clean_eeg_path(
        sid: str,
        acq: str | None = None,
        task: str | None = 'SpaNav'
) -> Path:
    return get_cont_path('preproc', sid, acq, task)


def get_outputs_path(
        sid: str | None = None,
        group_parent_dir: str | None = None,
        group_letter: str | None = None,
) -> Path:
    root = get_main_path()
    outputs_path = root / 'outputs'

    if group_parent_dir:
        outputs_path /= group_parent_dir

    if sid is None and group_letter is None:
        return outputs_path

    # If a subject ID or a group letter is passed, then the group-specific path is returned
    group = group_letter or prs.get_group_letter(sid)

    outputs_path /= f"WP73{group}"

    return set_for_save(outputs_path)


def get_tables_path(
) -> Path:
    outputs_path = get_outputs_path()
    return set_for_save(outputs_path / 'tables')


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
                sid.replace('sub-', '') for sid in os.listdir(group_path)
                if os.path.isdir(os.path.join(group_path, sid))
            ]
            sids += group_sids
    sids = sorted(sids)
    return sids if not test else [sids[0]]


def get_group_sids(
        group_letter: str,
        test: bool = False,
) -> list[str]:
    """

    :param group_letter:
    :param test:
    :return:
    """
    group_path = get_main_path() / 'raw' / f'BIDS_Data_WP73{group_letter}'
    group_sids = []
    for element in os.listdir(group_path):
        if os.path.isdir(os.path.join(group_path, element)):
            group_sids.append(
                element.replace('sub-', '')
            )
    group_sids = sorted(group_sids)
    return group_sids if not test else [group_sids[0]]


def get_sid_blocks(
        sid: str,
        test: bool = False,
) -> list[str]:
    """

    :param sid:
    :param test:
    :return:
    """
    cids = []
    group = prs.get_group_letter(sid)
    potential_blocks = [f'block{i}' for i in range(1, 5)] if group == 'A' else [f'block{i}' for i in range(1, 7)]
    for block in potential_blocks:
        raw_path = get_clean_eeg_path(sid, acq=block, task='SpaNav')
        if raw_path.exists():
            cids.append(block)
            if test:  # stop after finding first cid when in testing mode
                break
    if not cids:
        warnings.warn(f'No cids found for {sid = } ! Returning cids = []', UserWarning)
    return cids


def load_stim_mapping_table(sid: str) -> pd.DataFrame:
    BLINDING = cfg.get_blinding()
    group = prs.get_group_letter(sid)
    table_fname = f'WP73{group}_RandomizationTable_BLIND.xlsx' if BLINDING else f'WP73{group}_RandomizationTable.xlsx'
    table_fpath = get_main_path() / f'Data_WP73{group}' / 'TI_and_EEG' / table_fname
    return pd.read_excel(table_fpath, index_col=0)


def get_ti_positions(
        sid: str,
) -> list:
    """

    :param sid:
    :return:
    """
    exp_grp = prs.get_group_letter(sid)
    ti_pos_fpath = get_main_path() / f'Data_WP73{exp_grp}' / 'TI_and_EEG' / 'Montage' / sid
    spanav_files = list(ti_pos_fpath.glob(f'log_{sid}*.csv'))
    if len(spanav_files) != 1:
        raise RuntimeError(f'Expected exactly one SpaNav CSV for {sid}, found {len(spanav_files)}')
    conv_table = pd.read_csv(spanav_files[0], sep=';')
    eeg_ti_positions = conv_table.loc[:, 'Old channel name'].to_list()
    return eeg_ti_positions


def get_groups_letters(
) -> list:
    root = get_main_path() / 'raw'
    groups = [element for element in os.listdir(root) if os.path.isdir(root / element)]
    groups_letters = [group[-1] for group in groups]
    return groups_letters


