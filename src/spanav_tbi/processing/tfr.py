"""
    Title: Time-Frequency Representations (TFR)

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains functions to compute and store TFR of EEG.
"""
import numpy as np
import pandas as pd

import spanav_eeg_utils.comp_utils as cmp
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.spanav_utils as sn
import spanav_eeg_utils.spectral_utils as spct
from mne.time_frequency import EpochsTFR, AverageTFR, read_tfrs, combine_tfr
from mne.epochs import BaseEpochs

TFR = EpochsTFR | AverageTFR


def average_tfr_series(tfr_series) -> AverageTFR:
    """
    Combine a pandas Series of MNE TFR objects by weigthed average.
    """
    return combine_tfr(list(tfr_series))


def compute_tfr(
        epo_rec: BaseEpochs,
        epo_type: str,
        log: bool = True,
        norm: bool = True,
) -> EpochsTFR:
    """
    Compute TFR as in Convertino et al., 2023
    :param epo_rec:
    :param epo_type:
    :param log:
    :param norm:
    :return:
    """

    freqs = np.logspace(np.log10(2), np.log10(70), 40)
    # freqs = np.linspace(2, 70, 40)
    n_cycles = 5 if epo_type.endswith('wide') else freqs / 2  # Convertino uses 5 but in case of shorter (non-wide) epochs use a specific cycle for each frequency (half of the frequency)
    zero_mean = True  # wether to correct morlet wavelet to be of mean zero; set to True to have a true wavelet, but False better for illustration purposes

    tfr = epo_rec.compute_tfr(
        "morlet",
        freqs,
        n_cycles=n_cycles,
        average=False,  # don't average across epochs at this stage
        zero_mean=zero_mean,
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


def compute_cond_tfr(sid: str, cids: list[str], epo_type) -> EpochsTFR:
    # Concatenate epoched recordings across block of the same condition (and subject)
    epo_rec_full = cmp.get_concat_epo_recs(sid, cids, epo_type)

    # Compute TFR on all epochs and channels and return
    return compute_tfr(epo_rec_full, epo_type, log=True, norm=True)  # log-transform and normalize


def get_epo_level_tfr_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
        average: bool = False,
) -> pd.DataFrame:

    # Get TFRs within each epoch
    epo_types = sn.get_task_epo_types(test)
    epo_types += [f'{epo_type}_wide' for epo_type in epo_types]
    sids = io.get_sids(test=test)
    tfr_entries = []
    for epo_type in epo_types:
        for sid in sids:
            group = prs.get_group_letter(sid)
            sid_cids = io.get_sid_blocks(sid, test=test)
            if not sid_cids:
                continue
            cids_by_cond = sn.group_cids_by_cond(sid, test, cids=sid_cids)

            # Get one concatenated recording of epochs of the same condition
            for cond, cids in cids_by_cond.items():
                fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-epo_tfr.h5'
                fpath = io.get_outputs_path(sid) / 'TFR' / f'sub-{sid}' / fname
                if load:
                    if fpath.exists():
                        # Read exported files
                        cond_epo_tfr = read_tfrs(fpath, verbose=False)
                    else:
                        print(f"\nFile {fname} not found at {fpath.parent}. Continuing...")
                        continue
                else:
                    cond_epo_tfr = compute_cond_tfr(sid, cids, epo_type)

                    if save:
                        # Export
                        fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-epo_tfr.h5'
                        fpath = io.set_for_save(io.get_outputs_path(sid) / 'TFR' / f'sub-{sid}') / fname
                        cond_epo_tfr.save(fpath, overwrite=True)

                if average:
                    cond_epo_tfr = cond_epo_tfr.average(method='mean', dim='epochs')

                # Store as df entry
                tfr_entry = dict(
                    sid=sid,
                    group=group,
                    cond=cond,
                    epo_type=epo_type,
                    tfr=cond_epo_tfr,
                )
                tfr_entries.append(tfr_entry)

    return pd.DataFrame.from_records(tfr_entries)


def get_sid_level_tfr_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
) -> pd.DataFrame:
    if load:
        sids = io.get_sids(test=test)
        epo_types = sn.get_task_epo_types(test=test)

        # Additionally try to load bl-corrected version of epoch-types
        epo_types += [f'bl{epo_type}' for epo_type in epo_types if epo_type != 'Stasis']
        epo_types += [f'{epo_type}_wide' for epo_type in epo_types]

        tfr_records = []
        for sid in sids:
            sid_tfr_dir = io.get_outputs_path(sid) / 'TFR' / f'sub-{sid}'
            group = prs.get_group_letter(sid)
            conds = prs.get_conds(sid=sid)
            for cond in conds:
                for epo_type in epo_types:
                    fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_tfr.h5'
                    fpath = io.get_outputs_path(sid) / 'TFR' / f'sub-{sid}' / fname
                    if fpath.exists():
                        # Read exported files
                        cond_epo_tfr = read_tfrs(sid_tfr_dir/fname, verbose=False)
                    else:
                        print(f"\nFile {fname} not found at {fpath.parent}. Continuing...")
                        continue

                    # Store as df entry
                    tfr_entry = dict(
                        sid=sid,
                        group=group,
                        cond=cond,
                        epo_type=epo_type,
                        tfr=cond_epo_tfr,
                    )
                    tfr_records.append(tfr_entry)

        return pd.DataFrame.from_records(tfr_records)

    # Load epoch-level TFR dataframe with average=True to average across epochs (to avoid keeping all TFRs in memorry)
    sid_level_df = get_epo_level_tfr_df(test, load=True, save=False, average=True)

    # Crop wide epochs to their central 1s window (as in Convertino et al., 2023)
    is_wide = sid_level_df['epo_type'].str.endswith('_wide')
    sid_level_df.loc[is_wide, 'tfr'] = sid_level_df.loc[is_wide, 'tfr'].apply(_crop_wide_to_central)

    # Baseline correct movement-onset epochs with stasis epochs (as in Convertino et al., 2023) using Stasis or Stasis_wide as baseline, respectively
    normal_df = sid_level_df[~sid_level_df['epo_type'].str.endswith('_wide')]
    wide_df = sid_level_df[sid_level_df['epo_type'].str.endswith('_wide')]
    normal_df = _stasis_bl_corr(normal_df, bl_name='Stasis')
    wide_df = _stasis_bl_corr(wide_df, bl_name='Stasis_wide')
    sid_level_df = pd.concat([normal_df, wide_df], ignore_index=True)

    if save:
        # Export each subject-level TFR object
        for i, row in sid_level_df.iterrows():
            sid, cond, epo_type, tfr = row['sid'], row['cond'], row['epo_type'], row['tfr']
            fname = f'sub-{sid}_acq-{cond}_desc-{epo_type}_level-sid_tfr.h5'
            fpath = io.set_for_save(io.get_outputs_path(sid) / 'TFR' / f'sub-{sid}') / fname
            tfr.save(fpath, overwrite=True)

    return sid_level_df


def get_group_level_tfr_df(
        test: bool = False,
        load: bool = True,
        save: bool = False,
) -> pd.DataFrame:
    if load:
        groups = io.get_groups_letters()
        epo_types = sn.get_task_epo_types(test=test)

        # Additionally try to load bl-corrected version of epoch-types
        epo_types += [f'bl{epo_type}' for epo_type in epo_types if epo_type != 'Stasis']
        epo_types += [f'{epo_type}_wide' for epo_type in epo_types]

        tfr_records = []
        for group in groups:
            conds = prs.get_conds(group=group)
            for cond in conds:
                for epo_type in epo_types:
                    fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_tfr.h5'
                    fpath = io.get_outputs_path(group_letter=group) / 'TFR' / fname
                    if fpath.exists():
                        # Read exported files
                        cond_epo_tfr = read_tfrs(fpath, verbose=False)
                    else:
                        print(f"\nFile {fname} not found at {fpath.parent}. Continuing...")
                        continue

                    # Store as df entry(s)
                    tfr_entry = dict(
                        group=group,
                        cond=cond,
                        epo_type=epo_type,
                        tfr=cond_epo_tfr,
                    )
                    tfr_records.append(tfr_entry)

        return pd.DataFrame.from_records(tfr_records)

    # Load subject-level TFR dataframe
    sid_level_df = get_sid_level_tfr_df(test, load=True, save=False)

    # For each group, average TFR of the same condition and epoch-type across different subjects
    group_cols = ['group', 'cond', 'epo_type']
    grouped_df = sid_level_df.groupby(group_cols, as_index=False)
    del sid_level_df
    group_level_df = grouped_df['tfr'].apply(average_tfr_series).reset_index(drop=True)

    if save:
        # Export each group-level TFR object
        for i, row in group_level_df.iterrows():
            group, cond, epo_type, group_tfr = row['group'], row['cond'], row['epo_type'], row['tfr']
            fname = f'group-{group}_acq-{cond}_desc-{epo_type}_level-group_tfr.h5'
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

