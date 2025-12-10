import warnings
import pandas as pd
from utils.gen_utils import set_for_save, get_block_stim, get_wd


def get_blocks_times(
        sid: str,
) -> dict[int: tuple[float, float]]:
    """
    Retrieve start and end times of each task-block, based on TaskLog.txt
    :param sid: subject ID
    :return: dict containing block IDs as keys, and tuples with start and end times as values
    """
    block_starts_to_check = []
    with open(f'{get_wd()}/Data/{sid}/behavior/TaskLog.txt', 'r') as f:
        times_by_block = {}
        block_n = 1
        for line in f:
            line = line.strip()  # remove trailing /n
            if 'Retrieval Start' in line:
                line_split = line.split(',')
                block_start = float(line_split[0])
                next_line_split = next(f).split(',')
                block_end = float(next_line_split[0])
                times_by_block[block_n] = (block_start, block_end)
                block_starts_to_check.append(block_start)
                block_n += 1

    if block_starts_to_check != sorted(block_starts_to_check):
        warnings.warn(f'\n\n### Warning! Unusual times acorss blocks - possible interruption detected {sid = } ### \n\n')

    return times_by_block


def get_trace_df(
        sid: str,
) -> pd.DataFrame:

    tracelog = []
    with open(f'{get_wd()}/Data/{sid}/behavior/TraceLog.txt', 'r') as f:
        trial_counter = 0
        new_trial_flag = False
        for line in f:
            line = line.strip()
            if '----------------------' in line:
                if not new_trial_flag:  #if the line has  ----------------------, it means we are in a new trial
                    trial_counter += 1
                    new_trial_flag = True  # new_trial_flag is set to True until we find a normal line without ----, to avoid double increase of trial when there are 2 Num,-----
                continue
            new_trial_flag = False

            if line == '' or line.startswith('Time') or line.startswith('#'): #to ignore the first line of TraceLog
                continue

            parts = line.split(',') #list of the parts of the line
            if len(parts) == 2:
                time, angle = parts
                x = y = None
            elif len(parts) == 4:
                time, angle, x, y = parts
            else:
                continue

            try:
                tracelog.append({
                    'time': float(time),
                    'x': float(x) if x else None,
                    'y': float(y) if y else None,
                })
            except ValueError:
                continue

    # Transform into a dataframe
    df = pd.DataFrame(tracelog)

    if df['time'].tolist() != sorted(df['time']):
        warnings.warn(f'\n\n### Warning! Unusual times acorss blocks - possible interruption detected {sid = } ### \n\n')

    return df


def get_retrieval_df(
        sid: str,
) -> pd.DataFrame:
    df = pd.read_csv(
        f'{get_wd()}/Data/{sid}/behavior/RetrievalLog.txt',
        sep=',',
        comment='#',  # ignore rows that start with #
    )

    # Rename some cols
    df.drop('Block', axis=1, inplace=True)  # true block as we intend it is named "Round" here
    df = df.rename(columns={
        'Round': 'Block',
        'StartTime': 'starttime',
        'EndTime(Nav.)': 'endtime'
    })

    # Invalid strings (e.g. new starting behavioral data) --> nan
    df['starttime'] = pd.to_numeric(df['starttime'], errors='coerce')
    df['endtime'] = pd.to_numeric(df['endtime'], errors='coerce')
    df['Block'] = pd.to_numeric(df['Block'], errors='coerce')

    # Drop NaN rows (trials missing a start time)
    df = df[df['starttime'].notna()].copy().reset_index(drop=True)

    return df


def extract_behavioral_events(
        sid: str,
        block_times: dict[int: tuple[float, float]],
        retrieval_df: pd.DataFrame,
        df_trace: pd.DataFrame,
        save: bool = False,
) -> pd.DataFrame:

    events = []

    # Iterate over blocks
    for block_n, (start_blk, end_blk) in block_times.items():
        block_trials = retrieval_df.copy()[retrieval_df['Block'] == block_n]  # only consider trials in the block
        block_trials.reset_index(drop=True, inplace=True)
        assert ((block_trials['starttime'] >= start_blk) & (block_trials['endtime'] <= end_blk)).all(), f'Something is wrong with block times! {block_trials["starttime"]}\n{block_trials["endtime"]}'

        # Extract stimulation condition of the block
        condition = get_block_stim(sid, block_n)

        # Iterate over trials (rows) of the block
        for phase_idx, row in block_trials.iterrows():
            start, end = row['starttime'], row['endtime']
            trial_trace_df = select_trial_df(sid, block_n, df_trace, start, end)

            # Create new column state; set to Moving when there are values in x and y, otherwise to Static
            trial_trace_df['state'] = trial_trace_df.apply(lambda r: 'Moving' if pd.notna(r['x']) and pd.notna(r['y']) else 'Static', axis=1)

            # Change each "Moving" value that is preceded by "Static" to "MovOn" (movement onset)
            trial_trace_df['state'] = trial_trace_df.apply(
                lambda r: 'MovOn' if (r.name > 0 and trial_trace_df.loc[r.name-1, 'state'] == 'Static' and r['state'] == 'Moving') else r['state'], axis=1
            )

            # Iterate over states (Static, MovOn, Moving) of the block
            current_state = trial_trace_df.loc[0, 'state']
            state_start = trial_trace_df.loc[0, 'time']
            for i in range(1, len(trial_trace_df)): #in every line of dataframe trial_df
                if trial_trace_df.loc[i, 'state'] != current_state:  # when there is a change in state
                    state_end = trial_trace_df.loc[i-1, 'time'] #i take the last time of the previous state
                    state_len = state_end - state_start
                    movon_window = None
                    if current_state == 'MovOn':
                        movon_window = f"{max(0, state_start-3):.3f} – {state_start+3:.3f}" #windowing of 6 sec for the pre movement
                    events.append({
                        'RetrievalBlock': block_n,
                        'Condition': condition,
                        'Trial': phase_idx+1,
                        'State': current_state,
                        'StartTime': state_start,
                        'EndTime': state_end,
                        'Duration': state_len,
                        'MovOnWindow': movon_window
                    })

                    # Get next state and its time for the next iteration
                    current_state = trial_trace_df.loc[i, 'state']
                    state_start = trial_trace_df.loc[i, 'time']

            # Add the last state of the trial
            state_end = trial_trace_df.loc[len(trial_trace_df)-1, 'time']
            state_len = state_end - state_start
            movon_window = None
            if current_state == 'MovOn':
                movon_window = f"{max(0, state_start-3):.3f} – {state_start+3:.3f}"
            events.append({
                'RetrievalBlock': block_n,
                'Condition': condition,
                'Trial': phase_idx+1,
                'State': current_state,
                'StartTime': state_start,
                'EndTime': state_end,
                'Duration': state_len,
                'MovOnWindow': movon_window
            })

    # Create df
    events_df = pd.DataFrame(events)

    # Check events durations
    assert all(events_df['Duration']) >= 0, (f'Something is wrong, some events have negative duration')

    if save:
        file_path = f'{get_wd()}/Data/{sid}/eeg/Epo'
        events_df.to_csv(f'{set_for_save(file_path)}/behavioral_events.csv', index=False)

    return events_df


def select_trial_df(
        sid: str,
        block_n: int,
        df_trace: pd.DataFrame,
        trial_start: float,
        trial_end: float
):
    df_trace_block = df_trace.copy()

    if sid == '05':
        dt_full = df_trace['time'].diff()
        reset_indices = dt_full[dt_full < 0].index
        if len(reset_indices) > 0:
            reset_idx = reset_indices[0]
            if block_n in (1, 2):
                # Before the interruption: use the first part of the TraceLog
                df_trace_block = df_trace.iloc[:reset_idx].copy()
            else:
                # After the interruption: use the second part
                df_trace_block = df_trace.iloc[reset_idx:].copy()

    trial_df = df_trace_block[(df_trace_block['time'] >= trial_start + 0.05) & (df_trace_block['time'] <= trial_end - 0.05)].copy()  # only select timing of the trial (with 50 ms of padding each side - like June did - to remove uncertain points at the trial edges)
    trial_df.reset_index(drop=True, inplace=True)  # reset of indexes
    return trial_df


def define_eeg_epochs(
        event_table,
        sid: str,
        save: bool = False,
        verbose: bool = False,
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
    for block in event_table['RetrievalBlock'].unique():
        block_df = event_table[event_table['RetrievalBlock'] == block]
        block_condition = block_df['Condition'].unique()[0]

        # Iterate over trials
        for trial in block_df['Trial'].unique():
            trial_df = block_df[block_df['Trial'] == trial].reset_index(drop=True)

            # Iterate over states of the trial
            for i, row in trial_df.iterrows():

                # Reset times of the block to match its EEG recording (bc block times in event_table are continuous
                # across blocks/stim. condition, while EEG aren't so their times always start from 0)
                abs_start, abs_end, duration = row['StartTime'], row['EndTime'], row['Duration']
                block_n = row['RetrievalBlock']
                block_start, block_end = get_blocks_times(sid=sid)[block_n]
                state_start = abs_start - block_start
                if verbose:
                    print(
                        f"\nblock: {row['Condition']} ({block_n})"
                        f"\n\t: {block_start = }"
                        f"\n\t: {abs_start = }"
                        f"\n\t: {state_start = }"
                    )

                # Define epochs
                # 1. Static epochs
                if row['State'] == 'Static' and duration >= static_min_s:
                    stat_epoch_start = state_start + static_epo_window[0]
                    stat_epoch_end = state_start + static_epo_window[1]
                    events.append({
                        'RetrievalBlock': block,
                        'BlockStart': block_start,
                        'BlockEnd': block_end,
                        'Condition': block_condition,
                        'TrialNumber': trial,
                        'EpochType': 'Static',
                        'EpochStart': stat_epoch_start,
                        'EpochEnd': stat_epoch_end,
                        'EpochDuration': stat_epoch_end-stat_epoch_start,
                    })

                # 2. Movement onset epochs
                if row['State'] == 'MovOn':

                    # Movement onset is constrained by preceding immobility and following movement, so can't be the first state and has to have one state after it
                    if 0 < i < len(trial_df)-1:
                        prev = trial_df.loc[i-1]
                        following = trial_df.loc[i+1]

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
                            events.append({
                                'RetrievalBlock': block,
                                'BlockStart': block_start,
                                'BlockEnd': block_end,
                                'Condition': block_condition,
                                'TrialNumber': trial,
                                'EpochType': 'MovOn',
                                'EpochStart': movon_epoch_start,
                                'EpochEnd': movon_epoch_end,
                                'EpochDuration': movon_epoch_end-movon_epoch_start,
                            })

                            # 3. Continuous movement epochs - only exist after MovOn epochs
                            # Define epoch start - right after MovOn window to avoid overlapping portions
                            contmov_epoch_start = movon_start + movonset_epo_window[1]  # onset of movement + end ov MovOn epoch

                            # Define epoch end: end at earlier end between movement end and movement portions exceeding mov_max_s after onset
                            max_end = movon_start + mov_max_s
                            mov_state_end = following['EndTime'] - block_start  # as done above with abs_start (but here we are in following state so need this again)
                            contmov_epoch_end = min(mov_state_end, max_end)

                            # Append to events
                            events.append({
                                'RetrievalBlock': block,
                                'BlockStart': block_start,
                                'BlockEnd': block_end,
                                'Condition': block_condition,
                                'TrialNumber': trial,
                                'EpochType': 'ContMov',
                                'EpochStart': contmov_epoch_start,
                                'EpochEnd': contmov_epoch_end,
                                'EpochDuration': contmov_epoch_end-contmov_epoch_start,
                            })

    events_df = pd.DataFrame(events)

    if save:
        file_path = f'{get_wd()}/Data/{sid}/eeg/Epo'
        events_df.to_csv(f'{set_for_save(file_path)}/eeg_epochs.csv', index=False)

    return events_df


if __name__ == '__main__':
    pass


