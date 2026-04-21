"""
    Title: EEG preprocessing utilities

    Based on code by: Stavriani Skarvelaki (EPFL) and Paul de Fontaney (EPFL)
    Adapted and integrated by: Sophie Caroni
    Integrated: 03.03.2026

    Description:
    This script contains helper functions for the preprocessing of EEG data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import spanav_eeg_utils.parsing_utils as prs
import spanav_eeg_utils.io_utils as io
from mne.io import BaseRaw
from mne.preprocessing import ICA, create_eog_epochs
from spanav_eeg_utils.config_utils import get_seed
from spanav_eeg_utils.plot_utils import save_figure, get_cont_rec_plot_kwargs

SEED = get_seed()


def get_annot_instructions() -> str:
    return (
        f"When the recording window pops up: \n"
        f"a) Press 'a' to open the annotation tab \n"
        f"b) Create a new description and name it BAD_segments \n"
        f"c) Mark bad segments by left-clicking and dragging on the noisy part of the recording \n"
        f"d) Mark bad channels (by clicking on their name on the left) \n"
        f"Press enter to continue. \n"
    )


def get_ica_instructions() -> str:
    return (
        f"When the next plots pop up: \n"
        f"a) Scroll through sources \n"
        f"b) Check topographies \n"
        f"c) Check properties (by clicking on topographies) \n"
        f"d) Then mark bad components (by clicking on their index above the topographies) \n"
        f"Press enter to continue. \n"
    )


def bad_channel_inspection(raw_rec: BaseRaw, output_directory: Path | str) -> None:
    """
    Plots the PSD of EEG channels to help identify bad channels visually, then prompts for additional bad channels.
    :param raw_rec: BaseRaw, EEG recording to inspect (bad channels updated in-place)
    :param output_directory: Path or str, directory to save the PSD figure
    """
    print("\nComputing PSD, might take some time...")
    psd = raw_rec.compute_psd(method='welch', fmin=0, fmax=200, n_fft=len(raw_rec.times),
                               n_overlap=0, window='hann', exclude="bads", reject_by_annotation=False)

    fig = psd.plot()
    plt.show(block=True)
    fig.savefig(f'{output_directory}/raw_psd.png')

    while True:
        additionnal_bads = input(
            "Specify any additionnal bad channels from PSD inspection"
            "\n\t⚠️ use ';' between channel names (e.g. Cz;FC1;Pz)"
            "\n\t⚠️ please also beware of capital letters (e.g. FC1 = fc1) ").strip().lower().split(';')
        if additionnal_bads == ['']:
            break
        invalid = [ch for ch in additionnal_bads if ch not in raw_rec.info['ch_names']]
        if invalid:
            print(f"Incorrect channel names detected: {invalid}. Please input them again.")
            continue
        raw_rec.info["bads"] += additionnal_bads
        print('Channels', additionnal_bads, 'added to bad channels')
        break


def correct_phase_delay(raw_rec: BaseRaw) -> None:
    """
    Corrects event onset times to account for the 8 ms phase delay caused by the low-pass filter.
    :param raw_rec: BaseRaw, EEG recording whose annotation onsets are updated in-place
    """
    for i, event in enumerate(raw_rec.annotations.description):
        raw_rec.annotations.onset[i] += 0.008


def filter_and_ds(raw_rec: BaseRaw, l_freq: float, h_freq: float, sfreq: float) -> BaseRaw:
    """
    Applies bandpass filter, downsampling, and notch filter to an EEG recording.
    :param raw_rec: BaseRaw, EEG recording to filter
    :param l_freq: float, lower cutoff frequency for the bandpass filter (Hz)
    :param h_freq: float, upper cutoff frequency for the bandpass filter (Hz)
    :param sfreq: float, target sampling frequency for downsampling (Hz)
    :return: BaseRaw, filtered and downsampled recording
    """
    b_filtered = raw_rec.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=-1)  # bandpass filter
    b_filtered = b_filtered.resample(sfreq=sfreq)  # resample
    b_filtered = b_filtered.notch_filter(freqs=[50])  # notch filter power-line noise
    return b_filtered


def run_ica(rec: BaseRaw, sid: str, acq: str, task: str, save: bool) -> tuple[BaseRaw, ICA]:
    """
    Runs ICA on an EEG recording and allows manual exclusion of artifact components.
    :param rec: BaseRaw, filtered EEG recording
    :param sid: str, subject ID
    :param acq: str, recording acquisition label
    :param task: str, recording task label
    :param save: bool, whether to save ICA and reconstructed recording to disk
    :return: tuple of (BaseRaw, ICA), reconstructed recording and fitted ICA instance
    """
    start_rec = rec.copy()

    n_comp = len(start_rec.info.ch_names) - len(start_rec.info["bads"]) - 1
    print(f"Computing ICA with {n_comp} components...")
    ica = ICA(n_components=n_comp, max_iter="auto", random_state=SEED)
    ica.fit(start_rec)
    if save:
        ica_path = io.get_cont_path('ica', sid, acq, task)
        ica.save(ica_path, overwrite=True)

    # Find bad components using automatic labelling
    muscle_ics, eog_ics = auto_find_bad_components(ica, start_rec)

    # Label manually bad components based on plots/traces
    inspect_components(ica, start_rec, sid, save)

    # Check whether manually detected components were also labelled as bad by automatic mne detection
    manual_ics = set(ica.exclude) - set(muscle_ics) - set(eog_ics)
    if manual_ics:
        decision = input(
            f"Some components marked as bad weren't classified as bad by automatic detection: "
            f"{sorted(manual_ics)}. Keep them excluded? [y/n]: "
        ).strip().lower()
        if decision.startswith('n'):
            ica.exclude = [ic for ic in ica.exclude if ic not in manual_ics]

    print(f"Applying ICA and excluding {len(ica.exclude)} components: {ica.exclude}...")
    reconst_data = start_rec.copy()
    ica.apply(reconst_data)
    if save:
        # Save excluded components to a txt file
        ch_path = io.set_for_save(io.get_outputs_path() / "ICA")
        with open(f"{ch_path}/excluded_components.txt", "w") as output:
            output.write(str(ica.exclude))

    # Visualize result
    print("Plot before/after")
    start_s, stop_s = float(0), float(20)  # need to be floats!
    result_fig = ica.plot_overlay(start_rec, start=start_s, stop=stop_s, verbose=False,
                                  title="Before (red) / after (black) cleaning muscle-ICs", show=True)
    if save:
        rec_block_dir = prs.get_rec_acq_dir(acq, task)
        save_figure(f'ICA/{sid}/{rec_block_dir}', f"result.png", result_fig, sid=sid)

    print(f"Scroll through final result...")
    reconst_data.plot(**get_cont_rec_plot_kwargs(reconst_data), block=True)

    if save:
        reconst_path = io.get_cont_path('reconst', sid, acq, task)
        reconst_data.save(reconst_path, overwrite=True)

    return reconst_data, ica


def inspect_components(
        fit_ica: ICA,
        fit_rec: mne.io.BaseRaw,
        sid: str,
        save: bool = True,
) -> None:
    """
    Plots ICA component properties and sources for manual marking of bad ICs in-place.
    :param fit_ica: ICA, fitted ICA object (bad components marked in-place)
    :param fit_rec: BaseRaw, EEG recording used to fit ICA
    :param sid: str, subject ID
    :param save: bool, whether to export component plots
    """
    # Label muscle and eyes components with automatic detection
    muscle_ics, _ = fit_ica.find_bads_muscle(fit_rec, verbose=False)
    eog_ch = 'Fp1' if 'Fp1' in fit_rec.ch_names else 'Fp2'
    eog_epochs = create_eog_epochs(fit_rec, ch_name=eog_ch, verbose=False)
    eyes_ics, _ = fit_ica.find_bads_eog(eog_epochs, ch_name=eog_ch, verbose=False)

    # Plot properties of each component
    prop_figs = fit_ica.plot_properties(fit_rec, show=False, verbose=False, picks=range(0, fit_ica.n_components))
    for c, prop_fig in enumerate(prop_figs):
        explained_var_ratio = fit_ica.get_explained_variance_ratio(fit_rec, components=[c])
        ratio_percent = round(100 * explained_var_ratio["eeg"], 2)
        prop_fig.suptitle(f"Explained variance: {ratio_percent}%", x=0.2)
        prop_fig.subplots_adjust()
        if save:
            save_figure(f'ICA/{sid}', f'properties_{c:02d}.png', prop_fig, sid=sid)
        plt.close(prop_fig)

    # Plot topographies all together
    topo_figs = fit_ica.plot_components(inst=fit_rec, show=False, verbose=False)
    topo_figs = topo_figs if isinstance(topo_figs, list) else [topo_figs]  # if plot_components is high, plot_components returns list of figures
    for n_fig, topo_fig in enumerate(topo_figs):
        if save:
            save_figure(f'ICA/{sid}', f'topos_{n_fig:02d}.png', topo_fig, sid=sid)
        plt.close(topo_fig)

    # Plot interactive window of components sources, to manually mark bad components
    fit_ica.plot_sources(fit_rec, block=True)


def auto_find_bad_components(
        fit_ica: ICA,
        fit_data: mne.io.BaseRaw,
) -> tuple[list, list]:
    """
    Finds bad ICA components using MNE's automatic muscle and EOG detection.
    :param fit_ica: ICA, fitted ICA object
    :param fit_data: BaseRaw, data used to fit ICA
    :return: tuple of (muscle_ics, eyes_ics), lists of bad component indices
    """
    # Find muscle-related components
    muscle_ics, _ = fit_ica.find_bads_muscle(fit_data)

    # Find eye-related components
    eog_ch = 'Fp1' if 'Fp1' in fit_data.ch_names else 'Fp2'
    eog_epochs = create_eog_epochs(fit_data, ch_name=eog_ch)
    eyes_ics, _ = fit_ica.find_bads_eog(eog_epochs, ch_name=eog_ch)

    return muscle_ics, eyes_ics
