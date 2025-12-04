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
        warnings.warn('\n\n### Warning! Unusual times acorss blocks - possible interruption detected ### \n\n')

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
                if not new_trial_flag: #if the line has  ----------------------, it means we are in a new trial
                    trial_counter += 1
                    new_trial_flag = True # new_trial_flag is set to True until we find a normal line without ----, to avoid double increase of trial when there are 2 Num,-----
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
                    'time': float(time), #conversion into numbers from strings
                    'x': float(x) if x else None,
                    'y': float(y) if y else None,
                })
            except ValueError:
                continue

    df_trace = pd.DataFrame(tracelog) #conversion of tracelog into a data frame

    if df_trace['time'].tolist() != sorted(df_trace['time']):
        warnings.warn('\n\n### Warning! Unusual times acorss blocks - possible interruption detected ### \n\n')

    return df_trace


def get_retrieval_df(
        sid: str,
) -> pd.DataFrame:
    df = pd.read_csv(f'{get_wd()}/Data/{sid}/behavior/RetrievalLog.txt', sep=',', comment='#') #ignore rwaws that start with #
    df.drop('Block', axis=1, inplace=True)  # true block as we intended it is here "Round"

    # Rename some cols
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

    all_events = []
    for block_n, (start_blk, end_blk) in block_times.items():
        block_trials = retrieval_df.copy()[  # only consider trials in the block
            (retrieval_df['Block'] == block_n) &
            (retrieval_df['starttime'] >= start_blk) &
            (retrieval_df['endtime'] <= end_blk)]
        block_trials.reset_index(drop=True, inplace=True)

        # Extract stimulation condition of the block
        condition = get_block_stim(sid, block_n)

        for phase_idx, row in block_trials.iterrows(): #for every raw in block_trials
            start, end = row['starttime'], row['endtime']

            trial_df = select_trial_df(sid, block_n, df_trace, start, end)

            # set state
            trial_df['state'] = trial_df.apply(lambda r: 'Moving' if pd.notna(r['x']) and pd.notna(r['y']) else 'Static', axis=1)

            #  pre-moving state
            trial_df['state'] = trial_df.apply(
                lambda r: 'MovOn' if (r.name > 0 and trial_df.loc[r.name-1,'state']=='Static' and r['state']=='Moving') else r['state'], axis=1
            )

            # group togheter the same group
            current_state = trial_df.loc[0, 'state']
            start_time = trial_df.loc[0, 'time']

            for i in range(1, len(trial_df)): #in every line of dataframe trial_df
                if trial_df.loc[i, 'state'] != current_state:#i see the state of the prevous line with the following one
                    end_time = trial_df.loc[i-1, 'time'] #i take the last time of the previous state
                    duration = end_time - start_time #and calculate the duration
                    pre_movement_window = None
                    if current_state == 'MovOn':
                        pre_movement_window = f"{max(0,start_time-3):.3f} – {start_time+3:.3f}" #windowing of 6 sec for the pre movement
                    all_events.append({
                        'RetrievalBlock': block_n,
                        'Condition': condition,
                        'Trial': phase_idx+1,
                        'State': current_state,
                        'StartTime': start_time,
                        'EndTime': end_time,
                        'Duration': duration,
                        'MovOnWindow': pre_movement_window
                    })
                    # start of a new state
                    current_state = trial_df.loc[i,'state']
                    start_time = trial_df.loc[i,'time']

            # add the last state of the trial
            end_time = trial_df.loc[len(trial_df)-1, 'time']
            duration = end_time - start_time
            pre_movement_window = None
            if current_state == 'MovOn':
                pre_movement_window = f"{max(0,start_time-3):.3f} – {start_time+3:.3f}"
            all_events.append({
                'RetrievalBlock': block_n,
                'Condition': condition,
                'Trial': phase_idx+1,
                'State': current_state,
                'StartTime': start_time,
                'EndTime': end_time,
                'Duration': duration,
                'MovOnWindow': pre_movement_window
            })

    # build the table and visualization
    event_table = pd.DataFrame(all_events)

    assert all(event_table['Duration']) >= 0, (
        f'Something is wrong, some events have negative duration'
    )

    if save:
        file_path = f'{get_wd()}/Data/{sid}/eeg/Epo'
        event_table.to_csv(f'{set_for_save(file_path)}/behavioral_events.csv', index=False)

    return event_table


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


def extract_eeg_epochs(
        event_table,
        sid: str,
        save: bool = False,
        verbose: bool = False,
) -> pd.DataFrame:
    # Parameters
    movement_onset_window = (-0.5, 0.5)
    stationary_window = (0, 1)
    wide_window = 6.0

    epochs = []

    for block in event_table['RetrievalBlock'].unique():
        block_df = event_table[event_table['RetrievalBlock'] == block]
        block_condition = block_df['Condition'].unique()[0]

        for trial in block_df['Trial'].unique():
            trial_df = block_df[block_df['Trial'] == trial].reset_index(drop=True)

            for i in range(len(trial_df)):
                row = trial_df.loc[i]

                # Reset times of the block to match its EEG recording (bc block times in event_table are continuous
                # across blocks/stim. condition, while EEG aren't so their times always start from 0)
                abs_start, abs_end, duration = row['StartTime'], row['EndTime'], row['Duration']
                block_n = row['RetrievalBlock']
                block_start, block_end = get_blocks_times(sid=sid)[block_n]
                start = abs_start - block_start
                end = abs_end - block_start
                if verbose:
                    print(
                        f"\nblock: {row['Condition']} ({block_n})"
                        f"\n\t: {block_start = }"
                        f"\n\t: {abs_start = }"
                        f"\n\t: {start = }"
                    )

                # Retrieve participant's state
                state = row['State']

                # static
                if state == 'Static' and duration >= 2.0:
                    epoch_start = start + stationary_window[0]
                    epoch_end = start + stationary_window[1]
                    wide_start = max(0, start - wide_window/2)
                    wide_end = epoch_end + wide_window/2
                    epochs.append({
                        'RetrievalBlock': block,
                        'BlockStart': block_start,
                        'BlockEnd': block_end,
                        'Condition': block_condition,
                        'TrialNumber': trial,
                        'EpochType': 'Static',
                        'EpochStart': epoch_start,
                        'EpochEnd': epoch_end,
                        'WideStart': wide_start,
                        'WideEnd': wide_end
                    })

                # movement onset
                if state == 'MovOn':
                    if i > 0:
                        prev = trial_df.loc[i-1]
                        if prev['State'] == 'Static' and prev['Duration'] >= 1.0:
                            epoch_start = start + movement_onset_window[0]
                            epoch_end = start + movement_onset_window[1]
                            wide_start = max(0, start - wide_window/2)
                            wide_end = epoch_end + wide_window/2
                            epochs.append({
                                'RetrievalBlock': block,
                                'BlockStart': block_start,
                                'BlockEnd': block_end,
                                'Condition': block_condition,
                                'TrialNumber': trial,
                                'EpochType': 'MovOn',
                                'EpochStart': epoch_start,
                                'EpochEnd': epoch_end,
                                'WideStart': wide_start,
                                'WideEnd': wide_end
                            })

                # continuous movement
                if state == 'Moving' and duration >= 3.0:
                    # wide epoch fixed of 6 s in the onset
                    wide_start = max(0, start - wide_window/2)
                    wide_end = wide_start + wide_window

                    # movement inside the wide window
                    continuous_start = max(start, wide_start)
                    continuous_end = min(end, wide_end)

                    epochs.append({
                        'RetrievalBlock': block,
                        'BlockStart': block_start,
                        'BlockEnd': block_end,
                        'Condition': block_condition,
                        'TrialNumber': trial,
                        'EpochType': 'ContMov',
                        'EpochStart': continuous_start,
                        'EpochEnd': continuous_end,
                        'WideStart': wide_start,
                        'WideEnd': wide_end
                    })

    epochs_table = pd.DataFrame(epochs)

    # display and save the table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    if save:
        file_path = f'{get_wd()}/Data/{sid}/eeg/Epo'
        epochs_table.to_csv(f'{set_for_save(file_path)}/eeg_epochs.csv', index=False)

    return epochs_table


if __name__ == '__main__':
    pass


