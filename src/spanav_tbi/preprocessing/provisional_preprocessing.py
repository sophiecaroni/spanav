import mne
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
from pathlib import Path


def get_server_raw_path(subject: str, block: str, task: str) -> Path:
    group = prs.get_group_letter(subject)
    fname = f'sub-{subject}_ses-1_task-{task}_acq-{block}_eeg.vhdr'
    return io.get_raw_eeg_path(subject, server=True) / 'raw' / f'BIDS_Data_WP73{group}' / f'sub-{subject}' / 'ses-1' / 'eeg' / fname


def preproc_pipeline(subject: str, block: str, task: str = 'SpaNav', fast: bool = True) -> None:
    """
    Load a raw BrainVision recording from the server, apply automated preprocessing,
    and save as .fif to the local preproc directory.
    """
    raw_path = get_server_raw_path(subject, block, task)
    print(f"\tLoading : {raw_path}")
    raw = mne.io.read_raw_brainvision(raw_path, preload=False, verbose=False)
    full_duration = raw.times[-1]

    # Crop first 145 s of recording in any case because they are encoding phase and are not needed at this stage
    crop_start = 145.0
    if fast:
        CROP_END_S = 200.0  # crop end for fast mode (seconds)
        raw = raw.crop(crop_start, CROP_END_S)
    else:
        raw = raw.crop(crop_start, full_duration)

    raw.load_data()
    raw.filter(l_freq=1, h_freq=60, n_jobs=-1, verbose=False)
    raw.resample(sfreq=250, verbose=False)
    raw.notch_filter(freqs=50.0, verbose=False)
    raw.set_eeg_reference(projection=False, verbose=False)

    out_path = io.get_cont_path('preproc', subject, block, task)
    print(f"\tSaving : {out_path}")
    raw.save(out_path, overwrite=True, verbose=False)