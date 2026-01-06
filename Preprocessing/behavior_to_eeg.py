import warnings
import pandas as pd
import numpy as np

from utils.gen_utils import set_for_save, get_eeg_path, get_behav_path

SEGMENT_EPOCHS = True


def get_times_retrieval_phases(
        pid: str,
) -> dict[int: tuple[float, float]]:
    """
    Retrieve start and end times of each retrieval phase, based on TaskLog.txt
    :param pid: participant ID
    :return: dict containing block IDs as keys, and tuples with start and end times (of block's retrieval phases) as values
    """
    retrieval_starts_to_check = []
    file_name = 'TaskLog.txt'
    file_path = get_behav_path() / pid / file_name
    with open(file_path, 'r') as f:
        times_by_retrieval_phase = {}
        block_n = 1
        for line in f:
            line = line.strip()  # remove trailing /n
            if 'Retrieval Start' in line:
                line_split = line.split(',')
                retr_start = float(line_split[0])
                next_line_split = next(f).split(',')
                retr_end = float(next_line_split[0])
                times_by_retrieval_phase[block_n] = (retr_start, retr_end)
                retrieval_starts_to_check.append(retr_start)
                block_n += 1

    if retrieval_starts_to_check != sorted(retrieval_starts_to_check):
        warnings.warn(
            f'\n\n### Warning! Unusual times acorss blocks - possible interruption detected {pid = } ### \n\n')

    return times_by_retrieval_phase


def get_trace_df(
        pid: str,
) -> pd.DataFrame:
    tracelog = []

    file_name = 'TraceLog.txt'
    file_path = get_behav_path() / pid / file_name
    with open(file_path, 'r') as f:
        for raw in f:
            line = raw.strip()

            # Skip some lines
            if line == '' or line.startswith('Time') or line.startswith('#') or line.startswith('---'):
                continue

            parts = line.split(',')  # list of the parts of the line
            if len(parts) == 1:  # lines "Number ----------------------"
                time = parts
                x = y = None
            elif len(parts) == 2:
                time, angle = parts
                x = y = None
            elif len(parts) == 4:
                time, angle, x, y = parts
            else:
                continue

            tracelog.append({
                'time': float(time),
                'x': float(x) if x else np.nan,
                'y': float(y) if y else np.nan,
            })

    # Transform into a dataframe
    df = pd.DataFrame(tracelog)

    # As "The row is recorded every 100 ms if there's any navigation (translation or rotation). And, if there's no
    # recording, it means that s/he was stationary during the period." (June), insert syntehtic rows when needed
    t = df['time'].to_numpy()
    reset = np.r_[False, np.diff(t) < 0]          # True where time decreases
    df["segment_id"] = np.cumsum(reset)           # 0,1,2,... in file order
    tolerance = 1e-6  # tolerance
    sample_s = 0.1  # 100 ms, to treat two times as equal if they differ by less than one microsecond
    synthethic_rows = []

    for seg_id, seg in df.groupby("segment_id", sort=False):
        seg = seg.copy()

        # Sort only inside the segment (safe)
        seg.sort_values("time", inplace=True, kind="mergesort")
        seg.reset_index(drop=True, inplace=True)

        ts = seg["time"].to_numpy()
        dt = np.diff(ts)

        synthetic_rows = []
        for i, d in enumerate(dt):
            if d > sample_s + tolerance:
                for tt in np.arange(ts[i] + sample_s, ts[i + 1] - tolerance, sample_s):
                    synthetic_rows.append({'time': float(tt), 'x': np.nan, 'y': np.nan, 'segment_id': seg_id})

        if synthetic_rows:
            seg = pd.concat([seg, pd.DataFrame(synthetic_rows)], ignore_index=True)
            seg.sort_values("time", inplace=True, kind="mergesort")
            seg.reset_index(drop=True, inplace=True)

        synthethic_rows.append(seg)

    df = pd.concat(synthethic_rows, ignore_index=True)

    # Final check
    if df['time'].tolist() != sorted(df['time']):
        warnings.warn(
            f'\n\n### Warning! Unusual times acorss blocks - possible interruption detected {pid = } ### \n\n')

    return df


def get_retrieval_df(
        pid: str,
) -> pd.DataFrame:

    file_name = 'RetrievalLog.txt'
    file_path = get_behav_path() / pid / file_name
    df = pd.read_csv(
        file_path,
        sep=',',
        comment='#',  # ignore rows that start with #
    )

    # Rename some cols
    df.drop('Block', axis=1, inplace=True)  # true block as we intend it is named "Round" here
    df = df.rename(columns={
        'Round': 'Block',
        'Index': 'Trial',
        'StartTime': 'starttime',
        'EndTime(Nav.)': 'endtime'
    })

    # Invalid strings (e.g. new starting behavioral data) --> nan
    df['starttime'] = pd.to_numeric(df['starttime'], errors='coerce')
    df['endtime'] = pd.to_numeric(df['endtime'], errors='coerce')
    df['Block'] = pd.to_numeric(df['Block'], errors='coerce').astype('Int64')
    df['Trial'] = pd.to_numeric(df['Trial'], errors='coerce').astype('Int64')

    # Drop NaN rows (trials missing a start time)
    df = df[df['starttime'].notna()].copy().reset_index(drop=True)

    return df


def extract_behav_events(
        pid: str,
        retrieval_times: dict[int: tuple[float, float]],
        retrieval_df: pd.DataFrame,
        trace_df: pd.DataFrame,
        save: bool = False,
        test: bool = False,
) -> pd.DataFrame:
    events = []

    # Iterate over blocks
    for block_n, (retr_start, retr_end) in retrieval_times.items():
        if block_n > 1 and test:
            break
        block_trials = retrieval_df.copy()[retrieval_df['Block'] == block_n]  # only consider trials in the block
        block_trials.reset_index(drop=True, inplace=True)
        assert ((block_trials['starttime'] >= retr_start) & (block_trials[
                                                                 'endtime'] <= retr_end)).all(), f'Something is wrong with block times! {block_trials["starttime"]}\n{block_trials["endtime"]}'

        # Extract stimulation condition of the block
        condition = block_n  #get_block_stim(pid, block_n)

        # Iterate over trials (rows) of the block
        for row_idx, trial_row in block_trials.iterrows():
            start, end = trial_row['starttime'], trial_row['endtime']
            trial_n = trial_row['Trial']

            trial_trace_df = select_trial_df(pid, block_n, trace_df, start, end)

            # Create new column state; set to Moving when there are values in x and y, otherwise to Static
            trial_trace_df['state'] = trial_trace_df.apply(
                lambda r: 'Moving' if pd.notna(r['x']) and pd.notna(r['y']) else 'Static', axis=1)

            # Change each "Moving" value that is preceded by "Static" to "MovOn" (movement onset)
            trial_trace_df['state'] = trial_trace_df.apply(
                lambda r: 'MovOn' if (r.name > 0 and trial_trace_df.loc[r.name - 1, 'state'] == 'Static' and r[
                    'state'] == 'Moving') else r['state'], axis=1
            )

            # Iterate over states (Static, MovOn, Moving) of the block
            current_state = trial_trace_df.loc[0, 'state']
            state_start = trial_trace_df.loc[0, 'time']
            for i in range(1, len(trial_trace_df)):  # in every line of dataframe trial_df
                if trial_trace_df.loc[i, 'state'] != current_state:  # when there is a change in state
                    state_end = trial_trace_df.loc[i - 1, 'time']  # i take the last time of the previous state
                    state_len = state_end - state_start
                    events.append({
                        'RetrievalBlock': block_n,
                        'Condition': condition,
                        'Trial': trial_n,
                        'TrialStart': start,
                        'TrialEnd': end,
                        'State': current_state,
                        'StateStart': state_start,
                        'StateEnd': state_end,
                        'Duration': state_len,
                    })

                    # Get next state and its time for the next iteration
                    current_state = trial_trace_df.loc[i, 'state']
                    state_start = trial_trace_df.loc[i, 'time']

            # Add the last state of the trial
            state_end = trial_trace_df.loc[len(trial_trace_df) - 1, 'time']
            state_len = state_end - state_start
            events.append({
                'RetrievalBlock': block_n,
                'Condition': condition,
                'Trial': trial_n,
                'TrialStart': start,
                'TrialEnd': end,
                'State': current_state,
                'StateStart': state_start,
                'StateEnd': state_end,
                'Duration': state_len,

            })

    # Create df
    events_df = pd.DataFrame(events)

    # Check events durations
    assert all(events_df['Duration']) >= 0, (f'Something is wrong, some events have negative duration')

    if save:
        file_name = 'behav_events.csv'
        file_path = set_for_save(get_eeg_path() / 'Epo' / pid) / file_name
        events_df.to_csv(file_path, index=False)

    return events_df


def select_trial_df(
        pid: str,
        block_n: int,
        trace_df: pd.DataFrame,
        trial_start: float,
        trial_end: float
):
    block_trace_df = trace_df.copy()

    if pid == '05':
        dt_full = trace_df['time'].diff()
        reset_indices = dt_full[dt_full < 0].index
        if len(reset_indices) > 0:
            reset_idx = reset_indices[0]
            if block_n in (1, 2):
                # Before the interruption: use the first part of the TraceLog
                block_trace_df = trace_df.iloc[:reset_idx].copy()
            else:
                # After the interruption: use the second part
                block_trace_df = trace_df.iloc[reset_idx:].copy()

    # Select rows in trace_df for which the timing is within the start/end of the trial
    trial_df = block_trace_df[
        (block_trace_df['time'] >= trial_start) &
        (block_trace_df['time'] <= trial_end)
        ].copy().sort_values("time").reset_index(
        drop=True)  # only select timing of the trial (with 50 ms of padding each side - like June did - to remove uncertain points at the trial edges)

    # Add some padding (with 50 ms - like June did) to remove uncertain points at the trial edges
    pad = 0.05

    # Do not trim start if the first state is static (as per June's example:
    # "People often stay stationary in the beginning of a trial,
    # For instance,
    # 781.307, ----------------------
    # 782.660,167.187
    # Then, we can say that this one was static during 781.307 ~ (782.660 - 50ms)."
    is_static = trial_df['x'].isna() & trial_df['y'].isna()

    end_bound = trial_end - pad  # Always trim end

    trial_df = trial_df[
        (trial_df["time"] <= end_bound) &
        ((is_static & (trial_df["time"] >= trial_start)) |
         (~is_static & (trial_df["time"] >= trial_start + pad))
         )
        ].copy().reset_index(drop=True)

    return trial_df


def segment_epoch(
        events_list,
        epoch_info: dict,
        epoch_type: str,
        epoch_start: float,
        epoch_end: float,
        segment_len_s: float = 1.0,
):
    total_len = epoch_end - epoch_start
    if total_len <= 0:
        return

    # If segment_len_s invalid or longer than epoch, keep one epoch
    if (segment_len_s is None) or (segment_len_s <= 0) or (segment_len_s >= total_len):
        events_list.append({
            **epoch_info,
            'EpochType': epoch_type,
            'EpochStart': epoch_start,
            'EpochEnd': epoch_end,
            'EpochDuration': total_len,
            'SubEpoch': 0,
        })
        return

    n_segments = int(total_len // segment_len_s)
    if n_segments == 0:
        events_list.append({
            **epoch_info,
            'EpochType': epoch_type,
            'EpochStart': epoch_start,
            'EpochEnd': epoch_end,
            'EpochDuration': total_len,
            'SubEpoch': 0,
        })
        return

    for seg_idx in range(n_segments):
        seg_start = epoch_start + seg_idx * segment_len_s
        seg_end = seg_start + segment_len_s
        if seg_end > epoch_end + 1e-6:
            break
        events_list.append({
            **epoch_info,
            'EpochType': epoch_type,
            'EpochStart': seg_start,
            'EpochEnd': seg_end,
            'EpochDuration': segment_len_s,
            'SubEpoch': seg_idx,
        })


def define_eeg_epochs(
        events_df,
        pid: str,
        save: bool = False,
        verbose: bool = False,
        segment_epochs: bool = SEGMENT_EPOCHS,
) -> pd.DataFrame:
    # Epoch parameters
    movonset_epo_window = (-0.5, 0.5)
    static_epo_window = (0, 1.0)
    mov_epo_window_start = movonset_epo_window[1]
    mov_min_s = 1.0 + mov_epo_window_start  # 1s in Convertino, but we shift to avoid overlap with movement onset epochs
    mov_max_s = 3.0 + mov_epo_window_start  # to ignore part of movements after 3s of moving # 3s in Convertino, but since we shifted also the minimum of 0.5 to avoid overlap with movement onset epochs, it would not make sense to set to 3s then only have actually 2.5 to extract in epochs of 1s
    static_min_s = 2.0
    static_min_s_before_mov = 1.0

    events = []
    # Iterate over blocks
    for block in events_df['RetrievalBlock'].unique():
        block_df = events_df[events_df['RetrievalBlock'] == block]
        block_condition = block_df['Condition'].unique()[0]

        # Iterate over trials
        for trial in block_df['Trial'].unique():
            trial_df = block_df[block_df['Trial'] == trial].reset_index(drop=True)

            # Iterate over states of the trial
            for i, row in trial_df.iterrows():

                # Reset times of the block to match its EEG recording (bc block times in events_df are continuous
                # across blocks/stim. condition, while EEG aren't so their times always start from 0)
                abs_start, abs_end, duration = row['StateStart'], row['StateEnd'], row['Duration']
                block_n = row['RetrievalBlock']
                retrieval_start, retrieval_end = get_times_retrieval_phases(pid=pid)[block_n]
                state_start = abs_start - retrieval_start
                if verbose:
                    print(
                        f"\nblock: {row['Condition']} ({block_n})"
                        f"\n\t: {retrieval_start = }"
                        f"\n\t: {abs_start = }"
                        f"\n\t: {state_start = }"
                    )

                # Define epochs
                base_event_info = {
                    'RetrievalBlock': block,
                    'BlockStart': retrieval_start,
                    'BlockEnd': retrieval_end,
                    'Condition': block_condition,
                    'TrialNumber': trial,
                    'TrialStart': row['TrialStart'],
                    'TrialEnd': row['TrialEnd'],
                }

                # 1. Static epochs
                if row['State'] == 'Static' and duration >= static_min_s:
                    if segment_epochs:
                        segment_epoch(
                            events_list=events,
                            epoch_info=base_event_info,
                            epoch_type='Static',
                            epoch_start=state_start,
                            epoch_end=state_start + duration,
                        )

                    else:
                        stat_epoch_start = state_start + static_epo_window[0]
                        stat_epoch_end = state_start + static_epo_window[1]
                        events.append({
                            **base_event_info,
                            'EpochType': 'Static',
                            'EpochStart': stat_epoch_start,
                            'EpochEnd': stat_epoch_end,
                            'EpochDuration': stat_epoch_end - stat_epoch_start,
                        })

                # 2. Movement onset epochs
                if row['State'] == 'MovOn':

                    # Movement onset is constrained by preceding immobility and following movement, so can't be the first state and has to have one state after it
                    if 0 < i < len(trial_df) - 1:
                        prev = trial_df.loc[i - 1]
                        following = trial_df.loc[i + 1]

                        # Check if preceding and following states qualify
                        if (
                                prev['State'] == 'Static' and prev['Duration'] >= static_min_s_before_mov
                        ) and (
                                following['State'] == 'Moving' and following['Duration'] >= mov_min_s
                        ):

                            # Save aside movement onset time (relative to block)
                            movon_start = state_start

                            # Define MovOn epoch
                            movon_epoch_start = movon_start + movonset_epo_window[0]
                            movon_epoch_end = movon_start + movonset_epo_window[1]
                            if segment_epochs:
                                segment_epoch(
                                    events_list=events,
                                    epoch_info=base_event_info,
                                    epoch_type='MovOn',
                                    epoch_start=movon_epoch_start,
                                    epoch_end=movon_epoch_end,
                                )
                            else:
                                events.append({
                                    **base_event_info,
                                    'EpochType': 'MovOn',
                                    'EpochStart': movon_epoch_start,
                                    'EpochEnd': movon_epoch_end,
                                    'EpochDuration': movon_epoch_end - movon_epoch_start,
                                })

                            # 3. Continuous movement epochs - only exist after MovOn epochs
                            # Define epoch start - right after MovOn window to avoid overlapping portions
                            contmov_epoch_start = movon_start + movonset_epo_window[
                                1]  # onset of movement + end ov MovOn epoch

                            # Define epoch end: end at earlier end between movement end and movement portions exceeding mov_max_s after onset
                            max_end = movon_start + mov_max_s
                            mov_state_end = following[
                                                'StateEnd'] - retrieval_start  # as done above with abs_start (but here we are in following state so need this again)
                            contmov_epoch_end = min(mov_state_end, max_end)

                            # Append to events
                            if segment_epochs:
                                segment_epoch(
                                    events_list=events,
                                    epoch_info=base_event_info,
                                    epoch_type='ContMov',
                                    epoch_start=contmov_epoch_start,
                                    epoch_end=contmov_epoch_end,
                                )
                            else:
                                events.append({
                                    **base_event_info,
                                    'EpochType': 'ContMov',
                                    'EpochStart': contmov_epoch_start,
                                    'EpochEnd': contmov_epoch_end,
                                    'EpochDuration': contmov_epoch_end - contmov_epoch_start,
                                })

    events_df = pd.DataFrame(events)

    if save:
        file_name = 'eeg_epochs.csv'
        file_path = set_for_save(get_eeg_path() / 'Epo' / pid) / file_name
        events_df.to_csv(file_path, index=False)

    return events_df


if __name__ == '__main__':
    pass
