"""
    Title: Resting-state (rs) EEG preprocessing pipeline

    Author: Sophie Caroni
    Created: 03.03.2026

    Description:
    This script performs an interactive, semi-automated preprocessing of rsEEG data.
"""
import matplotlib
import spanav_eeg_utils.spanav_utils as sn
from spanav_tbi.preprocessing.rs_eeg_preprocessing import manual_inspection, filter_and_ds, ica_pipeline, inspect_final_data

matplotlib.use("tkagg")


def run_interactive_rs_preprocessing():
    # Set run configuration
    dev_mode_input = input("Do you want to run in dev mode? (yes 'y'/ no 'n'): ").strip().lower()
    dev_mode = dev_mode_input.startswith('y')

    # Subject(s) selection
    subject_input = 't01' if dev_mode else input("Enter comma-separated subject IDs (e.g. t01, a02): ").strip()
    subjects = [f'{sn.get_full_sid(i.strip())}' for i in subject_input.split(',')]

    # Recording(s) selection
    all_recs = 'n' if dev_mode else input("Do you want to process all recordings available ? (y/n): ").strip().lower()

    # Process each subject at once
    for subject in subjects:

        # Define name of rsEEG recordings
        if all_recs.startswith('y'):
            recs = ['eo_pre', 'eo_post', 'ec_pre', 'ec_post']
        else:
            recs_input = 'eo_pre' if dev_mode else input('Enter rsEEG recs to process, COMMA SEPARATED AND IN THE FORMAT "eo_pre, eo_post, ec_pre, ec_post": ').strip().lower()
            recs = [i.strip() for i in recs_input.split(',')]

        for rec in recs:
            task, acq = rec.split('_')
            task = task.upper()
            task = 'Rest' + task

            print(f"\n=== Processing: recording {task, acq = } ===")

            insp_raw = manual_inspection(subject, acq, task, dev_mode)
            print(f"\nManual inspection completed for {subject=}, {acq, task = }")

            filt_ds_raw = filter_and_ds(subject, acq, task, dev_mode, raw=insp_raw)
            print(f"\nFiltering completed for {subject=}, {acq, task=}")

            final_raw = ica_pipeline(subject, acq, task, dev_mode, raw=filt_ds_raw)
            print(f"\nICA completed for {subject=}, {acq, task = }")

            print('\n\nPlease have a check at the final result.')
            inspect_final_data(final_raw)
            _ = input(f'\n\nCleaning for {subject=}, {acq, task = } done! \nPress enter to go to the next rec.')


if __name__ == '__main__':
    run_interactive_rs_preprocessing()
