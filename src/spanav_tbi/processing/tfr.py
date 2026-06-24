"""
    Title: Time-Frequency Representations (TFR)

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains functions to compute and store TFR of EEG.
"""
import re
import mne
import numpy as np
import pandas as pd
import warnings
import spanav_eeg_utils.comp_utils as cmp
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.spanav_utils as sn
import spanav_eeg_utils.spectral_utils as spct
from mne.time_frequency import EpochsTFR, AverageTFR, read_tfrs, combine_tfr
from mne.epochs import BaseEpochs

TFR = EpochsTFR | AverageTFR


def average_tfr_series(tfr_series) -> AverageTFR:
    """Combine a pandas Series of MNE TFR objects by weigthed average."""
    return combine_tfr(list(tfr_series))


def compute_tfr(
        epo_rec: BaseEpochs,
        epo_type: str,
        log: bool = True,
        norm: bool = True,
) -> EpochsTFR:
    """
    Compute TFR similarly to Convertino et al., 2023
    :param epo_rec:
    :param epo_type:
    :param log:
    :param norm:
    :return:
    """
    freqs = np.logspace(np.log10(2), np.log10(60), 40)  # Convertino uses 70 but we can only up to 60 (bc of LPF for TI)
    n_cycles = 5 if epo_type.endswith('wide') else freqs / 2  # Convertino uses 5 but in case of shorter (non-wide) epochs use a specific cycle for each frequency (half of the frequency)

    tfr = epo_rec.compute_tfr(
        "morlet",
        freqs,
        n_cycles=n_cycles,
        average=False,  # don't average across epochs at this stage
        return_itc=False,
    )

    if norm:
        tfr = custom_tfr_norm(tfr)

    if log:
        data = tfr.data
        log_data = np.log(data)
        tfr.data = log_data

    return tfr


def custom_tfr_norm(tfr_in: TFR) -> TFR:
    """
    Normalize TFR as in Convertino et al., 2023
    :param tfr_in:
    :return:
    """
    tfr_out = tfr_in.copy()
    data_in = tfr_in.data

    # Normalize by sum over frequencies at each time point (per epoch, channel, time)
    denom = data_in.sum(axis=2, keepdims=True)  # axis=2 because (epochs, ch, freq, time)
    data = data_in / denom

    tfr_out.data = data
    return tfr_out


def compute_cond_tfr(sid: str, cids: list[str], epo_type) -> EpochsTFR | None:
    # Concatenate epoched recordings across block of the same condition (and subject)
    epo_rec_full = cmp.get_concat_epo_recs(sid, cids, epo_type)

    # Compute TFR on all epochs and channels and return
    if epo_rec_full is not None:
        return compute_tfr(epo_rec_full, epo_type, log=True, norm=True)  # log-transform and normalize
    return None


def get_epo_level_tfr_df(
        test: bool = False,
        average_epochs: bool = False,
) -> pd.DataFrame:

    # Get TFRs within each epoch
    epo_types = sn.get_task_epo_types(test)
    epo_types += [f'{epo_type}_wide' for epo_type in epo_types]
    sids = io.get_sids(test=test)
    rows = []
    for sid in sids:
        group = prs.get_group_letter(sid)
        sid_cids = io.get_sid_blocks(sid, test=test)
        if not sid_cids:
            continue
        cids_by_cond = sn.group_cids_by_cond(sid, test, cids=sid_cids)
        for epo_type in epo_types:
            # Get one concatenated recording of epochs of the same condition
            for cond, cids in cids_by_cond.items():
                cond_epo_tfr = compute_cond_tfr(sid, cids, epo_type)
                if cond_epo_tfr is None:
                    warnings.warn(f"\nTFR is None for {sid, cond, epo_type} (epo rec file likely not found). Skipping epo-level TFR...")
                    continue

                if average_epochs:
                    # Average TFR across epochs
                    cond_epo_tfr = cond_epo_tfr.average(method='mean', dim='epochs')

                # Store as df entry
                rows.append(dict(
                    sid=sid,
                    group=group,
                    cond=cond,
                    epo_type=epo_type,
                    tfr=cond_epo_tfr,
                ))

    return pd.DataFrame.from_records(rows)


def _average_tfr_channels(tfr) -> object:
    ch_tfrs = []
    for ch in tfr.ch_names:
        ch_tfr = tfr.copy().pick(ch)
        mne.rename_channels(ch_tfr.info, {ch: 'ch_mean'})
        ch_tfrs.append(ch_tfr)
    return combine_tfr(ch_tfrs)


def get_sid_level_tfr_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
        verbose: bool = False,
        ch_avg: bool = False,
) -> pd.DataFrame:
    if load:
        ch_avg_label = 'ch-avg' if ch_avg else 'ch-all'
        rows = []

        # Load subject-level existing TFR files 
        outputs_root = io.get_outputs_path()
        fname_pattern = re.compile(
            rf'sub-(?P<sid>.+)_acq-(?P<cond>.+)_desc-(?P<epo_type>.+)_level-sid_{ch_avg_label}_tfr\.h5'
        )
        tfr_fpaths = outputs_root.glob(f'WP73*/TFR/sub-*/sub-*_level-sid_{ch_avg_label}_tfr.h5')
        for fpath in sorted(tfr_fpaths):
            match = fname_pattern.fullmatch(fpath.name)
            if match is None:
                if verbose:
                    warnings.warn(f"\nFile {fpath.name} does not match the expected naming. Skipping...")
                continue
            cond_epo_tfr = read_tfrs(fpath, verbose=False)
            sid = match['sid']
            rows.append(dict(
                sid=sid,
                group=prs.get_group_letter(sid),
                cond=match['cond'],
                epo_type=match['epo_type'],
                tfr=cond_epo_tfr,
            ))

            if test:
                break  # stops after the first iteration

        return pd.DataFrame.from_records(rows)

    # Load epoch-level TFR dataframe with average_epochs=True to average across epochs at loading time (more efficient)
    sid_level_df = get_epo_level_tfr_df(test, average_epochs=True)
    if sid_level_df.empty:
        raise ValueError(f"Sid-level TFR dataframe is empty: \n\t{sid_level_df}")

    # Crop wide epochs to their central 1s window (as in Convertino et al., 2023)
    is_wide = sid_level_df['epo_type'].str.endswith('_wide')
    sid_level_df.loc[is_wide, 'tfr'] = sid_level_df.loc[is_wide, 'tfr'].apply(_crop_wide_to_central)

    # Baseline correct movement-onset epochs with stasis epochs (as in Convertino et al., 2023) using Stasis or Stasis_wide as baseline, respectively
    normal_df = sid_level_df[~sid_level_df['epo_type'].str.endswith('_wide')]
    if not normal_df.empty:
        normal_df = _stasis_bl_corr(normal_df, bl_name='Stasis')
    wide_df = sid_level_df[sid_level_df['epo_type'].str.endswith('_wide')]
    if not wide_df.empty:
        wide_df = _stasis_bl_corr(wide_df, bl_name='Stasis_wide')
    sid_level_df = pd.concat([normal_df, wide_df], ignore_index=True)

    if ch_avg:
        sid_level_df['tfr'] = sid_level_df['tfr'].apply(lambda t: _average_tfr_channels(t))

    if save:
        # Export each subject-level TFR object
        for i, row in sid_level_df.iterrows():
            sid, cond, epo_type, tfr = row['sid'], row['cond'], row['epo_type'], row['tfr']
            ch_avg_label = 'ch-avg' if ch_avg else 'ch-all'
            fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_{ch_avg_label}_tfr.h5'
            fpath = io.set_for_save(io.get_outputs_path(sid) / 'TFR' / f'sub-{sid}') / fname
            tfr.save(fpath, overwrite=True)
    return sid_level_df


def get_group_level_tfr_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
        ch_avg: bool = False,
) -> pd.DataFrame:
    if load:
        ch_avg_label = 'ch-avg' if ch_avg else 'ch-all'
        rows = []

        # Load group-level existing TFR files 
        outputs_root = io.get_outputs_path()
        fname_pattern = re.compile(
            rf'group-(?P<group>.+)_acq-(?P<cond>.+)_desc-(?P<epo_type>.+)_level-group_{ch_avg_label}_tfr\.h5'
        )
        tfr_fpaths = outputs_root.glob(f'WP73*/TFR/group-*_level-group_{ch_avg_label}_tfr.h5')
        for fpath in sorted(tfr_fpaths):
            match = fname_pattern.fullmatch(fpath.name)
            if match is None:
                warnings.warn(f"\nFile {fpath.name} does not match the expected naming. Skipping...")
                continue
            cond_epo_tfr = read_tfrs(fpath, verbose=False)
            rows.append(dict(
                group=match['group'],
                cond=match['cond'],
                epo_type=match['epo_type'],
                tfr=cond_epo_tfr,
            ))

            if test:
                break  # stops after the first iteration

        # Create and return dataframe
        return pd.DataFrame.from_records(rows)

    # For each group, average TFR of the same condition and epoch-type across different subjects
    sid_level_df = get_sid_level_tfr_df(test, load=True, save=False, ch_avg=ch_avg)  # if ch_avg, take already subject-level channel averaged
    group_cols = ['group', 'cond', 'epo_type']
    grouped_df = sid_level_df.groupby(group_cols, as_index=False)
    group_level_df = grouped_df['tfr'].apply(average_tfr_series).reset_index(drop=True)

    if save:
        for i, row in group_level_df.iterrows():
            group, cond, epo_type, group_tfr = row['group'], row['cond'], row['epo_type'], row['tfr']
            ch_avg_label = 'ch-avg' if ch_avg else 'ch-all'
            fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_{ch_avg_label}_tfr.h5'
            fpath = io.set_for_save(io.get_outputs_path(group_letter=group) / 'TFR') / fname
            group_tfr.save(fpath, overwrite=True)
    return group_level_df


def _crop_wide_to_central(tfr: AverageTFR, reset_times: bool = False) -> AverageTFR:
    center = (tfr.times[0] + tfr.times[-1]) / 2
    cropped = tfr.copy().crop(tmin=center - 0.5, tmax=center + 0.5, include_tmax=False)
    if reset_times:
        cropped.shift_time(0, relative=False)  # sets times to start from 0
    return cropped


def _stasis_bl_corr(input_df: pd.DataFrame, bl_name: str = 'Stasis'):
    group_cols = ['sid', 'group', 'cond']
    grouped_df = input_df.groupby(group_cols, as_index=False)
    new_rows = []
    for (sid, group, cond), subdf in grouped_df:

        # Apply baseline correction using Stasis TFR on TFR of all other epoch types
        if bl_name not in subdf['epo_type'].values:
            warnings.warn(f"Baseline '{bl_name}' not found for {sid=}, {cond=}. Skipping baseline correction.")
            continue
        bl_corr_records = spct.spectral_bl_corr_from_df(subdf, 'epo_type', 'tfr', bl_name)

        # Add grouping columns information to bl_corr_records (which will be broadcasted to match len in bl_corr_records)
        bl_corr_records.update(dict(
            sid=sid,
            group=group,
            cond=cond,
        ))

        # Turn into a one-line df for the new baseline-corrected TFR and append, by treating the BL-corrected as new epoch-types named with suffix "bl"
        new_rows.append(pd.DataFrame(bl_corr_records))

    # Add new rows relative to baseline-corrected TFR (of the new 'blMovOn' epoch_type) into output df
    output_df = pd.concat([input_df] + new_rows, ignore_index=True)
    return output_df

