"""
    Title: Resting-state (rs) EEG preprocessing

    Based on code by: Stavriani Skarvelaki (EPFL) and Paul de Fontaney (EPFL)
    Author: Sophie Caroni
    Created: 03.03.2026

    Description:
    This script contains steps to preprocess rsEEG data.
"""
import mne
import spanav_eeg_utils.io_utils as io
import spanav_eeg_utils.parsing_utils as prs
import spanav_tbi.preprocessing.preprocessing_utils as pp
from spanav_eeg_utils.plot_utils import get_cont_rec_plot_kwargs
from mne.io import BaseRaw


def manual_inspection(sid: str, acq: str, task: str, dev_mode: bool) -> BaseRaw:
    """
    Perform manual inspection of an EEG recording.
    :param sid: str, subject ID
    :param acq: str, recording acquisition label
    :param task: str, recording task label
    :param dev_mode: bool, whether to run in developement mode (crop data, no saving) or not
    :return: BaseRaw, recording after inspection
    """
    # Load the recording to inspect
    raw_path = io.get_cont_path('raw', sid, acq, task, server=True)
    print(f"\nLoading EEG data from: {raw_path}")
    raw = mne.io.read_raw_brainvision(raw_path, preload=False)
    if dev_mode:
        raw = raw.crop(0, 50)
    raw.load_data()

    # Add manual annotations of bad segments and channels
    print("\n\n# ===== ANNOTATION INSTRUCTIONS ===== #")
    _ = input(pp.get_annot_instructions())
    raw.plot(block=True, **get_cont_rec_plot_kwargs(raw))

    # Check channels PSD for additional detection of bad channels
    rec_dir = prs.get_rec_acq_dir(acq, task)
    fig_output_path = io.set_for_save(io.get_outputs_path(sid) / "Cleaning" / sid / rec_dir)
    pp.bad_channel_inspection(raw, fig_output_path)

    if not dev_mode:
        output_path = io.get_cont_path('annot', sid, acq, task)
        raw.save(output_path, overwrite=True)
    return raw


def filter_and_ds(sid: str, acq: str, task: str, dev_mode: bool, raw: BaseRaw | None = None) -> BaseRaw:
    """
    Apply phase delay correction, bandpass filter, and downsampling to an EEG recording.
    :param sid: str, subject ID
    :param acq: str, recording acquisition label
    :param task: str, recording task label
    :param dev_mode: bool, whether to run in development mode (no saving) or not
    :param raw: BaseRaw or None, if None the annotated recording is loaded from disk
    :return: BaseRaw, filtered and downsampled recording
    """
    if raw is None:
        # Load recording
        rec_path = io.get_cont_path('annot', sid, acq, task)
        print(f"Loading EEG data from: {rec_path}")
        raw = mne.io.read_raw_fif(rec_path, preload=False)
    raw.load_data()

    # Correct for phase delay of the low-pass filter
    pp.correct_phase_delay(raw)

    # Bandpass filter and downsample
    raw_filtered = pp.filter_and_ds(raw, l_freq=1, h_freq=60, sfreq=250)

    if not dev_mode:

        # Export final recording
        output_path = io.get_cont_path('filt', sid, acq, task)
        raw_filtered.save(output_path, overwrite=True)
    return raw_filtered


def ica_pipeline(sid: str, acq: str, task: str, dev_mode: bool, raw: BaseRaw | None = None) -> BaseRaw:
    """
    Run ICA artifact removal, interpolate bad channels, and re-reference to common average.
    :param sid: str, subject ID
    :param acq: str, recording acquisition label
    :param task: str, recording task label
    :param dev_mode: bool, whether to run in development mode (no saving) or not
    :param raw: BaseRaw or None, if None the filtered recording is loaded from disk
    :return: BaseRaw, preprocessed recording
    """
    if raw is None:
        raw_path = io.get_cont_path('filt', sid, acq, task)
        print(f"Loading EEG data from: {raw_path}")
        raw = mne.io.read_raw_fif(raw_path, preload=False)
    raw.load_data()

    # Re-annotate segments
    print("\n\n# ===== ANNOTATION INSTRUCTIONS ===== #")
    _ = input(pp.get_annot_instructions())
    raw.plot(block=True, **get_cont_rec_plot_kwargs(raw))

    if not dev_mode:
        reannot_path = io.get_cont_path('reannot', sid, acq, task)
        raw.save(reannot_path, overwrite=True)

    # Run ICA
    print("\n\n# ===== ICA 'BAD' COMPONENTS INSTRUCTIONS ===== ")
    _ = input(pp.get_ica_instructions())
    reconst_raw, ica = pp.run_ica(raw, sid, acq, task, save=(not dev_mode))

    # Interpolate bad channels
    print(f"\n\t Interpolating bad channels ...")
    reconst_raw.interpolate_bads(reset_bads=True)

    # Re-referencing
    print(f"\n\t Setting reference to common average...")
    reconst_raw.set_eeg_reference(projection=False)

    if not dev_mode:
        output_path = io.get_cont_path('preproc', sid, acq, task)
        reconst_raw.save(output_path, overwrite=True)
    return reconst_raw


def inspect_final_data(final_raw: BaseRaw) -> None:
    """
    Plot the fully preprocessed recording for a final visual check.
    :param final_raw: BaseRaw, preprocessed EEG recording to inspect
    """
    final_raw.plot(**get_cont_rec_plot_kwargs(final_raw))
