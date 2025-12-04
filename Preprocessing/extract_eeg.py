import mne
import numpy as np
import pandas as pd
from mne import Epochs
from mne.epochs import EpochsFIF

from utils.spectral_utils import compute_psd, get_band_power, compute_osc_snr, model_psd
from preprocessing.preprocess_eeg import basic_preproc_raw
from utils.gen_utils import get_epo_types, get_task_epo_types, set_for_save, get_trigger_str, get_sids, get_cids, get_real_cid, \
    get_wd
from mne.baseline import rescale
from autoreject import AutoReject

EPO_TYPES = set(get_epo_types())
EPO_LEN_COMPARISON_METRICS = {'psd', 'band_pw', 'evk', 'snr', 'osc_snr'}


def get_all_epo_objects(
        raw_rec: mne.io.BaseRaw | None = None,
        sid: str | None = None,
        cid: str | None = None,
        load: bool = False,
        save: bool = False,
        verbose: bool = True,
        test: bool = False,
) -> dict:
    if save or load:
        assert sid is not None, "Subject ID (sid) can't be None with the current save/load parameters."
        assert cid is not None, "Condition ID (cid) can't be None with the current save/load parameters."
    epo_by_type = {}
    if cid.startswith('RS'):
        epo_types = ['RS']
    else:
        epo_types = get_task_epo_types(test=test)
    for epo_type in epo_types:

        if verbose:
            print(
                f"\n\n{epo_type = }\n\n"
            )

        epo_rec = get_epo_rec(epo_type, sid, cid, raw_rec=raw_rec, load=load, save=save)
        epo_by_type[epo_type] = epo_rec

    return epo_by_type


def get_epo_rec(
        epo_type: str,
        sid: str,
        cid: str,
        raw_rec: mne.io.BaseRaw | None = None,
        load: bool = True,
        save: bool = False,
) -> EpochsFIF | None | Epochs:
    # Check epo type validity
    if epo_type not in EPO_TYPES:
        raise ValueError(f"Invalid epo_type: {epo_type!r}. Expected one of {EPO_TYPES}.")

    if load:
        real_cid = get_real_cid(sid, block_n=cid[-1]) if cid.startswith('block') else get_real_cid(sid, cid=cid)
        files_path = f'{get_wd()}/Data/{sid}/eeg/Epo'
        file_name = f'{real_cid}_{epo_type}-epo.fif' if not epo_type.startswith('RS') else f'{real_cid}-epo.fif'
        try:
            epo_rec_clean = mne.read_epochs(f'{files_path}/{file_name}', preload=True, verbose=False)
            return epo_rec_clean
        except FileNotFoundError:
            print(f'\nFile {file_name} not found for subject {sid} - potentially not existing \n\t--> returning None')
            return None
    else:
        assert raw_rec is not None, "Raw recording (raw_rec_start can't be None with the current save/load parameters."

        if epo_type == 'ObjPres':
            epo_rec = get_obj_pres_epochs(raw_rec)

        elif epo_type in ['ContMov', 'Static', 'MovOn']:
            epo_def = get_epo_def_table(sid, cid)
            epo_rec = get_epo_from_intervals(epo_def, epo_type, raw_rec)

        else:  # if epo_type == 'RS':
            epo_rec = get_rs_epochs(raw_rec)

        if epo_rec is not None:
            epo_rec.plot_drop_log(show=True)

            # Clean epochs
            if len(epo_rec) > 0:
                epo_rec_clean = clean_epos(epo_rec, epo_type)

                if save:
                    real_cid = get_real_cid(sid, block_n=cid[-1]) if cid.startswith('block') else get_real_cid(sid, cid=cid)
                    files_path = f'{get_wd()}/Data/{sid}/eeg/Epo'
                    file_name = f'{real_cid}_{epo_type}-epo.fif' if not epo_type.startswith('RS') else f'{real_cid}-epo.fif'
                    epo_rec_clean.save(f'{set_for_save(files_path)}/{file_name}', overwrite=True)
            else:
                epo_rec_clean = None
            return epo_rec_clean

        else:
            return None


def get_obj_pres_epochs(
        raw_rec: mne.io.BaseRaw,
) -> mne.Epochs:
    obj_pres_len = 1
    kwargs = {
        'verbose': True,
        'baseline': None,  # baseline was already applied on raw_rec
    }
    all_events, all_event_ids = mne.events_from_annotations(raw_rec, verbose=kwargs['verbose'])
    obj_pres_gone = get_trigger_str('retr_obj_gone')
    epo3 = mne.Epochs(
        raw_rec,
        all_events, event_id={obj_pres_gone: all_event_ids[obj_pres_gone]},
        # use events in all_events that match event_id
        tmin=-3.0,
        tmax=0.0,
        preload=True,
        reject_by_annotation=True,  # reject segments marked as bad
        **kwargs
    )
    epo3.plot_drop_log(show=True)

    # Reset these to prevent from keeping them in new shorter epochs then leading to warnings/errors because of different times
    epo3.reject = None
    epo3.reject_tmin = None
    epo3.reject_tmax = None
    parts = []

    # Loop to define 1s epochs
    epo_len = 1 - 1 / raw_rec.info['sfreq']
    for s in (-3.0, -2.0, -1.0):
        e = epo3.copy().crop(tmin=s, tmax=s + epo_len)  # [-3,-2], [-2,-1], [-1,0]
        epo1 = mne.EpochsArray(
            data=e.get_data(),  # baseline already applied to data
            info=e.info.copy(),
            events=e.events.copy(),
            event_id=e.event_id.copy(),
            tmin=0.0,
            raw_sfreq=e.info["sfreq"],
            **kwargs
        )
        parts.append(epo1)

    return epo3 if obj_pres_len == 3 else mne.concatenate_epochs(parts)


def get_rs_epochs(
        raw_rec: mne.io.BaseRaw,
) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw_rec, duration=1,
        verbose=False,
        preload=True,
        reject_by_annotation=True,
    )


def get_epo_def_table(
        sid: str,
        cid: str,
) -> pd.DataFrame:
    file_path = f'{get_wd()}/Data/{sid}/eeg/Epo'
    epo_table = pd.read_csv(f'{file_path}/eeg_epochs.csv')

    # Select rows selative to the retrieval block od the condition ID
    block_n = int(cid[-1]) if sid != '02' else int(get_real_cid(sid, cid)[-1])
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
        epo_len: int = 1,
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
        'Static': 111,
        'MovOn': 222,
        'ContMov': 333,
    }
    return epo_types_ids[epo_type]


def get_retrieval_raw_rec(
        sid: str,
        cid: str,
        verbose: bool = False,
) -> mne.io.BaseRaw:
    raw_rec = basic_preproc_raw(sid, cid, load=True, save=False, verbose=verbose)
    if raw_rec is not None:

        # Update onsets of annotations/triggers (bc if raw_rec was cropped, the onsets of triggers are not updated to the times of the new (cropped) rec)
        t0 = raw_rec.first_time
        onsets = raw_rec.annotations.onset - t0

        # Baseline correct (each trial with its initial 3s of RS)
        raw_corr = task_bl_corr(raw_rec, verbose=verbose)

        # Crop retrieval recording based on annotated triggers
        desc = raw_rec.annotations.description
        retr_start = onsets[desc == get_trigger_str('trial_start')][0]  # start of the first trial
        retr_end = onsets[desc == get_trigger_str('trial_end')][-1]  # end of the last trial
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
    if cid.startswith('RS'):
        return basic_preproc_raw(sid, cid, load=True, save=False, verbose=False)
    else:
        return get_retrieval_raw_rec(sid, cid, verbose=False)


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
    trigger_trial_start = get_trigger_str('trial_start')
    trigger_trial_end = get_trigger_str('trial_end')
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


def compare_epo_len_old(
        char_to_compare: str,
        lens_to_compare: list,
        test: bool = False,
):
    valid_comparisons = {'psd', 'band_pw', 'evk', 'snr', 'osc_snr'}
    if char_to_compare not in valid_comparisons:
        raise ValueError(f"Invalid char_to_compare: {char_to_compare!r}. Expected one of {valid_comparisons}.")

    sids = get_sids(test=test)
    cids = get_cids(task=True, test=test)
    bands = ['theta'] if test else ['theta', 'alpha']
    char_dict = {}
    for ln in lens_to_compare:
        char_dict[f'{ln}s'] = {}
        for cid in cids:
            char_dict[f'{ln}s'][cid] = {band: [] for band in bands} if char_to_compare in ['band_pw', 'osc_snr'] else []
            for sid in sids:

                raw_rec = get_raw_to_epoch(sid, cid)
                epo = get_epo_rec(raw_rec=raw_rec, epo_type='ObjPres', sid=sid, cid=cid, load=False, obj_pres_len=ln)

                # Compute char_to_compare by subject
                if char_to_compare in {'psd', 'band_pw', 'osc_snr'}:

                    # Compute PSD in each epoch and channel, and then average across them
                    fmin, fmax = 1, 40
                    psd_epoxch = compute_psd(epo, fmin=fmin, fmax=fmax, verbose=False)
                    psd_data, freqs = psd_epoxch.get_data(return_freqs=True)
                    epo_avg_psd = psd_data.mean(axis=(0, 1))  # Average across epochs and channels

                    # Interpolate PSD within a fixed set of freqs so that resolution of PSD of different epoch-lengths is fairly comparable
                    fix_freqs = np.arange(fmin, fmax + 1e-9, 0.5)  # Define freqs grid
                    psd_interp = np.interp(fix_freqs, freqs, epo_avg_psd)

                    if char_to_compare == 'psd':
                        char_dict[f'{ln}s'][cid].append(
                            psd_interp
                        )
                    else:  # if char_to_compare == 'band_pw' or char_to_compare == 'osc_snr':
                        for band in bands:
                            if char_to_compare == 'band_pw':
                                char_dict[f'{ln}s'][cid][band].append(
                                    get_band_power(psd_interp, fix_freqs, band, rel=False)
                                    # Compute theta/alpha abs power
                                )

                            else:  # char_to_compare == 'osc_snr'
                                psd_model = model_psd(psd_interp, fix_freqs,
                                                      max_n_peaks=4)  # limit max_n_peaks bc we only care about alpha/theta
                                char_dict[f'{ln}s'][cid][band].append(
                                    compute_osc_snr(psd_model, band)
                                )

                elif char_to_compare == 'evk':
                    char_dict[f'{ln}s'][cid].append(
                        epo.average()  # Average epochs to get to evoked response
                    )

            # Average across subjects
            if char_to_compare == 'psd':
                char_dict[f'{ln}s'][cid] = (
                # Store in a tuple average and std of PSD + frequencies used for computing it
                    np.mean(char_dict[f'{ln}s'][cid], axis=0),
                    np.std(char_dict[f'{ln}s'][cid], axis=0),
                    fix_freqs
                )

            elif char_to_compare in ['band_pw', 'osc_snr']:
                for band in bands:
                    char_dict[f'{ln}s'][cid][band] = (
                    # Store in a tuple average and std of metric (band_pw or osc_snr) across sids
                        np.mean(char_dict[f'{ln}s'][cid][band]),
                        np.std(char_dict[f'{ln}s'][cid][band])
                    )

            elif char_to_compare == 'evk':
                char_dict[f'{ln}s'][cid] = mne.grand_average(char_dict[f'{ln}s'][cid])

    if char_to_compare.endswith('snr'):
        return pd.DataFrame.from_dict(char_dict)
    else:
        return char_dict


def compare_objepo_len(
        metric: str,
        lens_to_compare: list[float] | list[int],
        sid: str | None = None,
        test: bool = False,
) -> pd.DataFrame:
    """

    :param metric:
    :param lens_to_compare:
    :param sid: data of the subject IDs to include; if None (default) uses all subjects.
    :param test:
    :return:
    """
    if metric not in EPO_LEN_COMPARISON_METRICS:
        raise ValueError(f"Invalid metric: {metric!r}. Expected one of {EPO_LEN_COMPARISON_METRICS}.")

    sids = get_sids(test=test) if sid is None else [sid]
    cids = get_cids(task=True, test=test)
    bands = ['theta'] if test else ['theta', 'alpha']
    res_df_rows = []
    for cid in cids:
        for sid in sids:
            raw_rec = get_raw_to_epoch(sid, cid)
            for ln in lens_to_compare:
                epo = get_epo_rec(raw_rec=raw_rec, epo_type='ObjPres', sid=sid, cid=cid, load=False, obj_pres_len=ln)

                # Compute char_to_compare by subject
                if metric in {'psd', 'band_pw', 'osc_snr'}:

                    # Compute PSD in each epoch and channel, and then average across them
                    fmin, fmax = 1, 40
                    psd_epoxch = compute_psd(epo, fmin=fmin, fmax=fmax, verbose=False)
                    psd_data, freqs = psd_epoxch.get_data(return_freqs=True)
                    epo_avg_psd = psd_data.mean(axis=(0, 1))  # Average across epochs and channels

                    # Interpolate PSD within a fixed set of freqs so that resolution of PSD of different epoch-lengths is fairly comparable
                    fix_freqs = np.arange(fmin, fmax + 1e-9, 0.5)  # Define freqs grid
                    psd_interp = np.interp(fix_freqs, freqs, epo_avg_psd)

                    if metric == 'psd':
                        # Define rows of the df (one row for each frequency-point of the psd
                        for freq, pw in zip(fix_freqs, psd_interp):
                            res_df_rows.append({
                                'epo_s': ln,
                                'sid': sid,
                                'cid': cid,
                                'freq': freq,
                                'pw': pw,
                            })

                    else:  # if metric == 'band_pw' or metric == 'osc_snr':

                        for band in bands:
                            if metric == 'band_pw':

                                res_df_rows.append({
                                    'epo_s': ln,
                                    'sid': sid,
                                    'cid': cid,
                                    'band': band,
                                    'pw': get_band_power(psd_interp, fix_freqs, band)  # Compute abs power
                                })

                            else:  # metric == 'osc_snr'
                                psd_model = model_psd(psd_interp, fix_freqs,
                                                      max_n_peaks=4)  # limit max_n_peaks bc we only care about alpha/theta
                                res_df_rows.append({
                                    'epo_s': ln,
                                    'sid': sid,
                                    'cid': cid,
                                    'band': band,
                                    'osc_snr': compute_osc_snr(psd_model, band)  # Compute oscillatory SNR
                                })

                else:  # metric == 'evk'
                    # Initiate a row of the df
                    df_row = {
                        'epo_s': ln,
                        'sid': sid,
                        'cid': cid,
                        'evk': epo.average()  # Average epochs to get to evoked response
                    }
                    res_df_rows.append(df_row)

    if metric == 'psd':
        df_psd_long = pd.DataFrame(res_df_rows)
        df_psd_wide = (
            df_psd_long
            .pivot_table(index=['sid', 'cid', 'epo_s'], columns='freq', values='pw')
            .sort_index()
        )
        return df_psd_wide
    else:
        return pd.DataFrame(res_df_rows)


def clean_epos(
        epo_rec: mne.Epochs,
        epo_label: str,
) -> mne.Epochs:
    n_epo = len(epo_rec)
    if n_epo >= 5:
        cv = 5
        ar = AutoReject(cv=cv)
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
