import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.spanav_utils as sn
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.config_utils as config
from mne import Epochs
from mne.epochs import EpochsFIF
from mne.baseline import rescale
from autoreject import AutoReject

EPO_TYPES = set(sn.get_epo_types())
EPO_LEN_COMPARISON_METRICS = {'psd', 'band_pw', 'evk', 'snr', 'osc_snr'}

SEED = config.get_seed()


def get_all_epo_objects(
        raw_rec: mne.io.BaseRaw | None = None,
        sid: str | None = None,
        cid: str | None = None,
        load: bool = False,
        save: bool = False,
        verbose: bool = True,
        test: bool = False,
        epo_types: list | None = None,
) -> dict:
    if save or load:
        assert sid is not None, "Subject ID (sid) can't be None with the current save/load parameters."
        assert cid is not None, "Condition ID (cid) can't be None with the current save/load parameters."
    epo_by_type = {}
    if cid.startswith('RS'):
        epo_types = ['RS']
    else:
        epo_types = sn.get_task_epo_types(test=test) if epo_types is None else epo_types
    for epo_type in epo_types:

        if verbose:
            print(
                f"\n\n{epo_type = }\n\n"
            )

        epo_rec = get_epo_rec(epo_type, sid, cid, raw_rec=raw_rec, load=load, save=save, verbose=verbose)
        epo_by_type[epo_type] = epo_rec

    return epo_by_type


def get_epo_rec(
        epo_type: str,
        sid: str,
        block: str,
        raw_rec: mne.io.BaseRaw | None = None,
        load: bool = True,
        save: bool = False,
        verbose: bool = False,
        epo_def_df: pd.DataFrame | None = None,
) -> EpochsFIF | None | Epochs:
    # Check epo type validity
    if epo_type not in EPO_TYPES:
        raise ValueError(f"Invalid epo_type: {epo_type!r}. Expected one of {EPO_TYPES}.")

    if load:
        real_cid = prs.get_stim(sid, acq=block)
        task = 'RS' if real_cid.lower().startswith('rs') else 'SpaNav'
        files_path = io.get_epo_data_path(epo_type, sid, acq=real_cid, task=task)
        try:
            epo_rec_clean = mne.read_epochs(files_path, preload=True, verbose=False, proj=False)
            return epo_rec_clean
        except FileNotFoundError:
            print(f'\nFile not found for subject {sid} - potentially not existing \n\t--> returning None')
            return None
    else:
        assert raw_rec is not None, "Raw recording (raw_rec_start can't be None with the current save/load parameters."

        if epo_type == 'ObjPres':
            epo_rec = get_obj_pres_epochs(raw_rec)

        elif epo_type in ['ContMov', 'Stasis', 'MovOn']:
            epo_def = get_epo_def(sid, block) if epo_def_df is None else epo_def_df
            epo_rec = get_epo_from_intervals(epo_def, epo_type, raw_rec)

        else:  # if epo_type == 'RS':
            epo_rec = get_rs_epochs(raw_rec)

        if epo_rec is not None:
            epo_rec.plot_drop_log(show=verbose)
            if not verbose:
                plt.close()

            # Clean epochs
            epo_rec_clean = clean_epos(epo_rec, epo_type, verbose=verbose)
            if len(epo_rec) == 0:
                epo_rec_clean = None
            else:
                if save:
                    real_cid = prs.get_stim(sid, acq=block)
                    task = 'RS' if real_cid.lower().startswith('rs') else 'SpaNav'
                    files_path = io.get_epo_data_path(epo_type, sid, acq=real_cid, task=task)
                    epo_rec_clean.save(files_path, overwrite=True)
            return epo_rec_clean

        else:
            return None


def get_obj_pres_epochs(
        raw_rec: mne.io.BaseRaw,
        wide: bool = False,
) -> mne.Epochs:
    kwargs = {
        'verbose': True,
        'baseline': None,  # baseline was already applied on raw_rec
    }
    all_events, all_event_ids = mne.events_from_annotations(raw_rec, verbose=kwargs['verbose'])
    obj_pres_gone = sn.get_trigger_str('retr_obj_gone')

    # When wide=True, the initial window must cover ±3s from the center of each 1s epoch:
    # centers are at -2.5, -1.5, -0.5 → widest range needed is [-5.5, 2.5]
    tmin_init = -5.5 if wide else -3.0
    tmax_init = 2.5 if wide else 0.0

    epo_3s = mne.Epochs(
        raw_rec,
        all_events, event_id={obj_pres_gone: all_event_ids[obj_pres_gone]},
        # use events in all_events that match event_id
        tmin=tmin_init,
        tmax=tmax_init,
        preload=True,
        reject_by_annotation=True,  # reject segments marked as bad
        **kwargs
    )
    epo_3s.plot_drop_log(show=True)

    # Reset these to prevent from keeping them in new short epochs based on the copy of epo_3s to define 'e'
    epo_3s.reject = None
    epo_3s.reject_tmin = None
    epo_3s.reject_tmax = None

    # Loop to define 1s epochs (or 6s wide epochs centered on each 1s epoch)
    epo_1s = []
    epo_len = 1 - 1 / raw_rec.info['sfreq']
    for s in (-3.0, -2.0, -1.0):
        if wide:
            center = s + 0.5  # center of the 1s epoch
            e = epo_3s.copy().crop(tmin=center - 3.0, tmax=center + 3.0)
        else:
            e = epo_3s.copy().crop(tmin=s, tmax=s + epo_len)  # [-3,-2], [-2,-1], [-1,0]
        e_1s = mne.EpochsArray(
            data=e.get_data(),  # baseline already applied to data
            info=e.info.copy(),
            events=e.events.copy(),
            event_id=e.event_id.copy(),
            tmin=0.0,
            raw_sfreq=e.info["sfreq"],
            **kwargs
        )
        epo_1s.append(e_1s)

    return mne.concatenate_epochs(epo_1s)


def get_rs_epochs(
        raw_rec: mne.io.BaseRaw,
) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw_rec, duration=1,
        verbose=False,
        preload=True,
        reject_by_annotation=True,
    )


def get_epo_def(
        sid: str,
        block: str | int,
) -> pd.DataFrame:
    file_path = io.get_epo_beh_tables_path(sid, 'eeg_epochs.csv')
    epo_table = pd.read_csv(file_path)

    # Select rows selative to the retrieval block od the condition ID
    block_n = int(block[-1])
    epo_table_block = epo_table[epo_table['RetrievalBlock'] == block_n]
    return epo_table_block


def check_alignment(
        raw_rec: mne.io.BaseRaw,
        events_table: pd.DataFrame,
):
    """
    This function checks that EEG recording and events (behavioral data) are correctly aligned, i.e. cover the same recording duration.
    :param raw_rec:
    :param events_table:
    :return:
    """
    block_start = events_table.loc[:, 'BlockStart'].to_numpy()[0]
    block_end = events_table.loc[:, 'BlockEnd'].to_numpy()[0]
    block_duration = round(block_end - block_start, 1)
    raw_rec_duration = round(raw_rec.duration, 1)

    if block_duration != raw_rec_duration:
        raise ValueError(
            f'Problem in aligning events (behavioral data) to EEG! \n\t{block_duration = }, {raw_rec_duration = }'
            f'\n\tdf of block:\n\t{events_table}')
    else:
        print('EEG recording and events (behavioral data) correctly aligned.')


def get_epo_from_intervals(
        df_epo_intervals: pd.DataFrame,
        epo_type: str,
        raw_rec: mne.io.BaseRaw,
        epo_len: float = 1.0,
) -> mne.Epochs | None:
    check_alignment(raw_rec, df_epo_intervals)

    # Subset the df to rows relative to epochs of argument epo_type
    epoch_type_df = df_epo_intervals[df_epo_intervals['EpochType'] == epo_type].copy()

    # Get start of each epoch (in samples)
    sfreq = raw_rec.info['sfreq']
    epo_start_samples = (epoch_type_df['EpochStart'] * sfreq).to_numpy(int)

    # As raw_rec was probably previously cropped (e.g. encoding-task cropped out), align epoch-timings from df to cropped raw
    first = raw_rec.first_samp
    starts = epo_start_samples + first

    # Define events in mne format: [start tp, 0, epo_id]
    epo_id = get_epo_type_id(epo_type)
    events = np.column_stack([
        starts,
        np.zeros(len(starts), int),
        np.full(len(starts), epo_id, int)
    ])

    if events.size > 0:
        # Create Epochs object
        tmax = epo_len - epo_len / sfreq
        epochs = mne.Epochs(
            raw_rec, events, event_id={epo_type: epo_id},
            tmin=0, tmax=tmax, baseline=None, preload=True,
            flat=None, picks=mne.pick_types(raw_rec.info, eeg=True, exclude='bads'),
            reject_by_annotation=True,  # reject segments marked as bad
            verbose=False,
        )
        return epochs
    else:
        return None


def get_epo_type_id(
        epo_type: str,
) -> int:
    epo_types_ids = {
        'Stasis': 111,
        'MovOn': 222,
        'ContMov': 333,
    }
    return epo_types_ids[epo_type]


def get_retrieval_raw_rec(
        sid: str,
        cid: str,
        verbose: bool = True,
) -> mne.io.BaseRaw:
    file_path = io.get_cont_path('preproc', sid, acq=cid)
    raw_rec = mne.io.read_raw_fif(file_path, preload=True, verbose=verbose)

    # Update onsets of annotations/triggers (bc if raw_rec was cropped, the onsets of triggers are not updated to the times of the new (cropped) rec)
    t0 = raw_rec.first_time
    onsets = raw_rec.annotations.onset - t0

    # Baseline correct (each trial with its initial 3s of RS)
    raw_corr = task_bl_corr(raw_rec, verbose=verbose)

    # Crop retrieval recording based on annotated triggers
    desc = raw_rec.annotations.description
    retr_start = onsets[desc == sn.get_trigger_str('trial_start')][0]  # start of the first trial
    retr_end = onsets[desc == sn.get_trigger_str('trial_end')][-1]  # end of the last trial
    raw_corr.crop(tmin=retr_start, tmax=retr_end)
    if verbose:
        print(
            f"\n-> Cropped raw recording to match retrieval-task: "
            f"final duration of {raw_corr.duration}s (from {raw_corr.first_time}s to {raw_rec._last_time}s of original recording)"
        )

    return raw_corr


def get_raw_to_epoch(
        sid: str,
        cid: str,
) -> mne.io.BaseRaw:
    # if cid.startswith('RS'):
    #     return p(sid, cid, load=True, save=False, verbose=False)
    # else:
    return get_retrieval_raw_rec(sid, cid, verbose=True)


def task_bl_corr(
        raw_rec: mne.io.BaseRaw,
        verbose: bool = False
) -> mne.io.BaseRaw:
    """
    Apply baseline-correction on task continuous data, using for each trial's segment its initial 3s of RS as baseline.
    :param raw_rec: initial continuous recording
    :param verbose:
    :return: baseline-corrected continuous recording
    """
    trigger_trial_start = sn.get_trigger_str('trial_start')
    trigger_trial_end = sn.get_trigger_str('trial_end')
    annots = raw_rec.annotations

    # Initialise variables
    corr_data = raw_rec.get_data().copy()
    in_trial = False
    trial_start_s = None

    # Iterate over trials, and baseline-correct each with rescale() using the 3s-RS at the beginning of the trial
    for tr, on in zip(annots.description, annots.onset):

        # Get trial start and end times
        if tr == trigger_trial_start:
            trial_start_s = on
            in_trial = True
        elif tr == trigger_trial_end and in_trial:
            trial_end_s = on
            trial_start, trial_end = raw_rec.time_as_index([trial_start_s, trial_end_s])  # convert start/end to samples

            # Define trial data and times
            trial_data = corr_data[:, trial_start:trial_end]
            trial_times = np.arange(trial_data.shape[1]) / raw_rec.info[
                'sfreq']  # times are relative to the segment being corrected, not the whole rec, so they should start from 0

            # Apply baseline-correction using 3s-RS
            trial_corr = rescale(
                trial_data,
                trial_times,
                baseline=(0.0, 3.0),
                verbose=False
            )
            corr_data[:, trial_start:trial_end] = trial_corr  # replace data in corr_data with trial's corrected data

    # Create new Raw object with baseline-corrected data and raw_rec info+annotations
    raw_corr = mne.io.RawArray(
        corr_data,
        raw_rec.info,
        verbose=False,
        first_samp=raw_rec.first_samp
    ).set_annotations(raw_rec.annotations.copy())
    if verbose:
        print('\n-> Baseline-correcting continuous recording (using first 3s of each trial as its baseline)')
    return raw_corr


def clean_epos(
        epo_rec: mne.Epochs,
        epo_label: str,
        verbose: bool = False,
) -> mne.Epochs | None:
    # Re-reference to common average
    epo_rec.set_eeg_reference('average')

    if len(epo_rec) > 0:
        n_epo = len(epo_rec)
        if n_epo >= 5:
            cv = 5
            ar = AutoReject(cv=cv, random_state=SEED, verbose=verbose)
            epo_rec_clean = ar.fit_transform(epo_rec)
        else:
            print(f'\n\nToo few epochs (n={n_epo}) in {epo_label} to apply autoreject! Applying manual epoch-cleaning.\n\n')

            # Compute PTP per channel per epoch
            data = epo_rec.get_data()  # (n_epochs, n_channels, n_times)
            ptp_ch = np.ptp(data, axis=-1)  # (n_epochs, n_channels)

            # Drop clearly bad epochs, where there are
            th_epoch = 100e-6  # 100 µV; ev. change to 120e-6 for more clemency
            bad_epochs = (ptp_ch > th_epoch).any(axis=1)  # shape (n_epochs,)
            good_epochs = ~bad_epochs
            epo_rec = epo_rec[good_epochs]
            ptp_ch = ptp_ch[good_epochs]
            n_epochs = len(epo_rec)

            if n_epochs > 0:
                # If there are still epochs left: reinterpolate channels that show >= 150 µV in at least 50% of epochs
                th_ch = 150e-6  # 150 µV
                frac_bad_per_ch = (ptp_ch > th_ch).mean(axis=0)
                globally_bad_idx = np.where(frac_bad_per_ch >= 0.5)[0]  # bad in ≥50% of epochs
                globally_bad_ch = [epo_rec.ch_names[i] for i in globally_bad_idx]

                # Interpolate bad channels
                epo_rec_clean = epo_rec.copy()
                epo_rec_clean.info['bads'] = globally_bad_ch
                epo_rec_clean.interpolate_bads(reset_bads=True)
            else:
                epo_rec_clean = epo_rec.copy()
        return epo_rec_clean
    else:
        return None
