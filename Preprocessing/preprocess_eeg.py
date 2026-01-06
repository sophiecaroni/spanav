"""
********************************************************************************
    Title: Cleaning EEG

    Author: Sophie Caroni
    Date of creation: 01.10.2025

    Description:
    This script contains functions for cleaning EEG data.
********************************************************************************
"""
import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA, create_eog_epochs, annotate_amplitude, annotate_muscle_zscore
from visualization.vis_eeg import compare_psds, plot_muscle_art
from utils.spectral_utils import compute_psd
from utils.gen_utils import set_for_save, save_figure, plot_context, SEED, get_trigger_str, reveal_cid, get_main_path, get_exp_phase


def load_raw(
        pid: str,
        cid: str,
        annotated_bads: bool,
        test: bool = False,
        verbose: bool = True,
) -> mne.io.BaseRaw | None:
    """

    :param pid: participant ID
    :param cid: condition ID
    :param annotated_bads:
    :param test:
    :param verbose:
    :return:
    """
    if annotated_bads:
        real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
        file_path = f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/RawAnnotated'
        file_name = f'{real_cid}_annot-raw.fif'
        raw_rec = mne.io.read_raw_fif(f'{file_path}/{file_name}', preload=True)

        # Print contained 'bad' annotations
        print(
            f"Rec annotations: \n\t- bad channels: {raw_rec.info['bads']}"
            f"\n\t- bad segments: {len([d for d in raw_rec.annotations.description if d.startswith('BAD')])}"
            f"\n\t- other annotations: {len([d for d in raw_rec.annotations.description if not d.startswith('BAD')])}"
        )
    else:
        # Load data
        raw_rec = mne.io.read_raw_brainvision(vhdr_fname=f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/RawRecorded/{cid}.vhdr', preload=True)

        # Make sure this recording does not contain 'bad' annotations
        assert not raw_rec.info['bads'], '\n\n#### Some channels are already set as bad!! ####\n\n'
        assert not raw_rec.info['bads'], '\n\n#### Some channels are already set as bad!! ####\n\n'

        # Crop from trigger defining start of RS/block recording
        annots = raw_rec.annotations
        if annots and not test:

            if cid.startswith('RS'):
                start_trigger = get_trigger_str('rs_start')
                end_trigger = None
                assert start_trigger in annots.description, f'starting trigger "{start_trigger}" not found in annotations'
                tmin = [o for o, d in zip(annots.onset, annots.description) if d == start_trigger][0]  # there should be one, but take first in any case
                tmax = tmin+4*60  # crop after 4 minutes for RS at the beginning/end of the experiment
            else:
                start_trigger = get_trigger_str('enc_instr_gone')
                end_trigger = get_trigger_str('trial_end')
                assert start_trigger in annots.description, f'starting trigger "{start_trigger}" not found in annotations'
                assert end_trigger in annots.description, f'ending trigger "{end_trigger}" not found in annotations'
                tmin = [o for o, d in zip(annots.onset, annots.description) if d == start_trigger][0]  # there should be one, but take first in any case
                tmax = [o for o, d in zip(annots.onset, annots.description) if d == end_trigger][-1]  # crop after last response

            raw_rec.crop(tmin=tmin, tmax=tmax)
            if verbose:
                print(f'\nCropping from first {start_trigger}...')
                if end_trigger is not None:
                    print(f'.. Cropping until {end_trigger}')

            # Update onsets of annotations/triggers to "new" times of the cropped recording
            t0 = raw_rec.first_time
            raw_rec.annotations.onset = raw_rec.annotations.onset - t0

        # Use only some seconds of data if in testing mode
        if test:
            raw_rec.crop(tmax=50.0)
            if verbose:
                print(f'\nCropping for testing...')

    if verbose:
        print(f'\n\t-> {raw_rec.times[-1]} sec of data, {len(raw_rec.times)} timepoints')

    return raw_rec


def annot_raw_manual(
        raw_rec: mne.io.BaseRaw,
        bad_chs: list | None = None,
        bad_seg_starts: list | None = None,
        bad_seg_lens: list | None = None,
        pid: str | None = None,
        cid: str | None = None,
        save: bool = False,
) -> mne.io.BaseRaw | None:
    """

    :param raw_rec:
    :param bad_chs:
    :param bad_seg_starts:
    :param bad_seg_lens:
    :param pid: participant ID
    :param cid: condition ID
    :param load:
    :param save:
    :return:
    """
    annot_raw_rec = raw_rec.copy()

    # Manually annotate bad channels
    if bad_chs:
        annot_raw_rec.info['bads'] = bad_chs

    # Manually annotate bad segments
    if bad_seg_starts:
        all_annots = annot_raw_rec.annotations  # keep existing annotations
        all_annots.append(mne.Annotations(bad_seg_starts, bad_seg_lens, ['BAD']*len(bad_seg_starts)))
        annot_raw_rec.set_annotations(all_annots)

    print(
        f"Final annotations: \n\t- bad channels: {annot_raw_rec.info['bads']}"
        f"\n\t- bad segments: {len([d for d in raw_rec.annotations.description if d.startswith('BAD')])}"
        f"\n\t- other annotations: {len([d for d in raw_rec.annotations.description if not d.startswith('BAD')])}"
    )

    if save:
        real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
        save_path = set_for_save(f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/RawAnnotated')
        annot_raw_rec.save(f'{save_path}/{real_cid}_annot-raw.fif', overwrite=True)

    return annot_raw_rec


def muscle_annot_raw_auto(
        raw_rec: mne.io.BaseRaw,
        muscle_threshold: float | None = None,
        show: bool = True,
) -> mne.io.BaseRaw:
    """
    Automatically detect muscle artifacts.
    :param raw_rec:
    :param muscle_threshold:
    :param show:
    :return:
    """
    rec_annot = raw_rec.annotations.copy()
    
    # Copy recording to detect bad segments - needs some processing
    raw_det = raw_rec.copy()
    raw_det.resample(285, verbose=False)  # to reduce computational time, but assuring the Nyquist freq is higher than 140 (highpass of the muscle detection filter)
    raw_det.notch_filter(freqs=50, verbose=False)  # you should notch filter before detecting muscle artifacts

    # Detecet muslce segments
    muscle_threshold = 5.0 if muscle_threshold is None else muscle_threshold
    annot_muscle, scores_muscle = annotate_muscle_zscore(
        raw_det,
        threshold=muscle_threshold,
        min_length_good=0.1,
    )
    plot_muscle_art(raw_det, scores_muscle, muscle_threshold, show=show)

    # Apply annotations to original unaltered recording
    raw_annot = raw_rec.copy()
    raw_annot.set_annotations(rec_annot + annot_muscle)

    return raw_annot


def prepro_for_ica(
        raw_rec: mne.io.BaseRaw,
) -> mne.io.BaseRaw:
    """

    :param raw_rec:
    :param muscle_threshold: z-score above which a segment is considered a muscle artifact; set it optimally by
    looking at the plot.
    :param pid:
    :param cid:
    :param show:
    :param save:
    :return:
    """
    raw_rec = raw_rec.copy()

    # Apply basic filters
    raw_rec.filter(l_freq=1, h_freq=60)
    raw_rec.notch_filter(freqs=50)

    # Downsample
    sfreq = 250
    raw_rec.resample(sfreq)

    # Automatically detect amplitude artifacts
    ann, _ = annotate_amplitude(
        raw_rec,
        peak={'eeg': 1e-6},
        flat=None,
        min_duration=0.01,  # 10 ms consecutive above-threshold
        picks='data'
    )
    raw_rec.set_annotations(raw_rec.annotations + ann)
    print(
        f"-> Automatically detected {len(ann)!r} (noisy) peaks\n\n"
    )

    return raw_rec


def get_ica(
        pid: str,
        cid: str,
        raw_rec: mne.io.BaseRaw,
        n_components_based_on_var: bool = False,
        n_components: int | None = None,
        load: bool = False,
        save: bool = False,
) -> tuple[mne.preprocessing.ica.ICA, mne.io.BaseRaw]:

    # Apply preprocessing steps needed to get optimal ICA
    ica_prepro_raw = prepro_for_ica(raw_rec)

    if load:
        assert n_components is not None, "Number of components (n_components) can't be None with the current save/load parameters."
        real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
        fitted_ica = mne.preprocessing.read_ica(f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/ICA/{real_cid}/{n_components}_comp/{real_cid}_ica.fif')
    else:

        # Define components number
        if n_components_based_on_var:
            n_components = 0.99
        else:
            n = ica_prepro_raw.info['nchan'] - len(ica_prepro_raw.info['bads'])
            n_components = n - 1

        # Fit ICA on the preprocessed data
        fitted_ica = ICA(n_components=n_components, random_state=SEED, verbose=False).fit(
            ica_prepro_raw,
            reject_by_annotation=True  # ignores 'bad' segmetns (as muscle artifacts previously set)
        )

        tot_exp_var = fitted_ica.get_explained_variance_ratio(ica_prepro_raw, ch_type='eeg')['eeg']
        print(
            f'--> Tot explained variance: {round(tot_exp_var * 100, 2)} %'
            f'\n\n\n'
        )

        if save:
            n_components = fitted_ica.n_components_
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            save_path = set_for_save(f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/ICA/{real_cid}/{n_components}_comp')
            fitted_ica.save(f'{save_path}/{real_cid}_ica.fif', overwrite=True)

    return fitted_ica, ica_prepro_raw


def get_ica_sources(
        fitted_ica: mne.preprocessing.ica.ICA,
        raw_rec: mne.io.BaseRaw,
        pid: str | None = None,
        cid: str | None = None,
        load: bool = False,
        save: bool = False,
) -> mne.io.BaseRaw:
    n_components = fitted_ica.n_components_
    if load:
        real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
        ica_sources = mne.io.Raw(f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/ICA/{real_cid}/{n_components}_comp/{real_cid}_icsources-raw.fif', preload=True)
    else:
        ica_sources = fitted_ica.get_sources(raw_rec)
        if save:
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            save_path = set_for_save(f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/ICA/{real_cid}/{n_components}_comp')
            ica_sources.save(f'{save_path}/{real_cid}_icsources-raw.fif', overwrite=True)

    return ica_sources


def plot_ics(
        ica: mne.preprocessing.ica.ICA,
        raw_rec: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
        pid: str | None = None,
        cid: str | None = None,
        test: bool = False,
        show: bool = False,
        save: bool = False,
):
    # Plot components topography
    with plot_context():
        nrows, ncols = ('auto', 'auto')  # if n_comp < 20 else (5, 5)
        picks = range(4) if test else None  # picks=None plots all ICs
        topo_figs = ica.plot_components(inst=raw_rec, nrows=nrows, ncols=ncols, picks=picks, show=False, verbose=False)
        topo_figs = topo_figs if isinstance(topo_figs, list) else [topo_figs]  # if plot_components ish high, plot_components returns list of figures
    for n_fig, topo_fig in enumerate(topo_figs):
        if save:
            n_components = ica.n_components_
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            save_path = f'../Outputs/ICA/{pid}/{real_cid}/{n_components}_comp'
            save_figure(save_path, f'topos_{n_fig:02}.png', fig=topo_fig)
        if show:
            plt.show()
        else:
            plt.close()

    # Plot properties
    with plot_context():
        picks = range(4) if test else range(ica.n_components_)  # picks=None plots the first 5 ICs
        prop_figs = ica.plot_properties(raw_rec, picks=picks, show=False, verbose=False)
    for c, fig in enumerate(prop_figs):
        explained_var_ratio = ica.get_explained_variance_ratio(raw_rec, components=[c])
        ratio_percent = round(100 * explained_var_ratio["eeg"], 2)
        fig.suptitle(f"Explained variance: {ratio_percent}%", x=0.2)
        fig.subplots_adjust()
        if save:
            save_figure(save_path, f'props_{c:02d}.png', fig=fig)

        if show:
            plt.show()
        else:
            plt.close()


def get_eyes_ics(
        ica: mne.preprocessing.ica.ICA,
        raw_rec: mne.io.BaseRaw,
        arbitrary_eyes_ics: list | None = None,
        pid: str | None = None,
        cid: str | None = None,
        show: bool = True,
        save: bool = False,
) -> list:
    with plot_context():
        ica = ica.copy()  # dont modify original ica instance

        # Create EOG epochs
        eog_ch = 'Fp1' if 'Fp1' in raw_rec.ch_names else 'Fp2'
        eog_epochs = create_eog_epochs(raw_rec, ch_name=eog_ch, show=False)

        if arbitrary_eyes_ics:
            eyes_ics = arbitrary_eyes_ics
            fig1 = ica.plot_components(eyes_ics, title='Selected eyes ICs:', show=False)
            fig2 = None
        else:

            # Find components correlated to EOG
            eyes_ics, scores = ica.find_bads_eog(eog_epochs, ch_name=eog_ch, show=False)
            eyes_ics = list(map(int, eyes_ics))

            # Plots 
            fig1 = ica.plot_components(eyes_ics, title='Detected eyes-ICs:', show=False)  # this only if no arbitrary_eyes_ics is passed
            fig2 = ica.plot_scores(scores, eyes_ics, title='ICs eyes scores', show=False)

        # Plot what the signal would look like by excluding eyes_ics
        ica.exclude = eyes_ics
        fig3 = ica.plot_overlay(eog_epochs.average(), verbose=False, show=False)
        start_s, stop_s = float(0), float(20)  # need to be floats!
        fig4 = ica.plot_overlay(raw_rec, start=start_s, stop=stop_s, verbose=False, title="Before (red) / after (black) cleaning eyes-ICs", show=False)

        if save:
            n_components = ica.n_components_
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            save_path = f'../Outputs/ICA/{pid}/{real_cid}/{n_components}_comp'
            if fig2 is not None:
                save_figure(save_path, 'eyes_ch_corrs.png', fig=fig2)
            file_name_suff = '_arbitrary' if arbitrary_eyes_ics else ''
            save_figure(save_path, f'eyes_comp_topos{file_name_suff}.png', fig=fig1)
            save_figure(save_path, f'eyes_evk_overlay{file_name_suff}.png', fig=fig3)
            save_figure(save_path, f'eyes_trace_overlay{file_name_suff}.png', fig=fig4)

        if show:
            plt.show()
        else:
            plt.close(fig1)
            if fig2 is not None:
                plt.close(fig2)
            plt.close(fig3)
            plt.close(fig4)

    print(f'--> Returned ICs (EOG): {eyes_ics}')
    return list(map(int, eyes_ics))


def get_muscle_ics(
        ica: mne.preprocessing.ica.ICA,
        raw_rec: mne.io.BaseRaw,
        arbitrary_muscle_ics: list | None = None,
        pid: str | None = None,
        cid: str | None = None,
        show: bool = True,
        save: bool = False,
) -> list:
    with plot_context():
        ica = ica.copy()  # dont modify original ica instance

        if arbitrary_muscle_ics:
            muscle_ics = arbitrary_muscle_ics
            fig1 = ica.plot_components(muscle_ics, title='Selected muscle ICs:', show=False)
            fig2 = None
        else:
            muscle_ics, scores = ica.find_bads_muscle(raw_rec, show=False)
            muscle_ics = list(map(int, muscle_ics))

            # Plots
            fig1 = ica.plot_components(muscle_ics, title='Detected muscle-ICs:', show=False)
            fig2 = ica.plot_scores(scores, muscle_ics, title='ICs muscle scores', show=False)  # this one only if no arbitrary_muscle_ics is passed

        # Plot what the signal would look like by excluding muscle_ics
        ica.exclude = muscle_ics
        start_s, stop_s = float(0), float(20)  # need to be floats!
        fig3 = ica.plot_overlay(raw_rec, start=start_s, stop=stop_s, verbose=False, title="Before (red) / after (black) cleaning muscle-ICs", show=False)

        if save:
            n_components = ica.n_components_
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            save_path = f'../Outputs/ICA/{pid}/{real_cid}/{n_components}_comp'
            if fig2 is not None:
                save_figure(save_path, 'muscle_ch_corrs.png', fig=fig2)
            file_name_pref = '_arbitrary' if arbitrary_muscle_ics else ''
            save_figure(save_path, f'muscle_comp_topos{file_name_pref}.png', fig=fig1)
            save_figure(save_path, f'muscle_trace_overlay{file_name_pref}.png', fig=fig3)
        if show:
            plt.show()
        else:
            plt.close(fig1)
            if fig2 is not None:
                plt.close(fig2)
            plt.close(fig3)

    print(f'--> Returned ICs (muscle): {muscle_ics}')
    return muscle_ics


def find_bad_ics(
        ica: mne.preprocessing.ica.ICA,
        raw_rec: mne.io.BaseRaw,
        arbitrary_eyes_ics: list | None = None,
        arbitrary_muscle_ics: list | None = None,
        pid: str | None = None,
        cid: str | None = None,
        show: bool = True,
        save: bool = False,
) -> np.ndarray:
    
    eyes_ics = get_eyes_ics(ica, raw_rec, arbitrary_eyes_ics, pid, cid, save=save, show=show)
    muscle_ics = get_muscle_ics(ica, raw_rec, arbitrary_muscle_ics, pid, cid, save=save, show=show)

    bad_ics = np.unique(eyes_ics + muscle_ics)
    tot_cis = ica.n_components
    perc_bad_ics = round((len(bad_ics) / tot_cis)*100, 1)
    print(f'\n==> BAD ICs: {bad_ics} \n\t{perc_bad_ics}% of ICs')

    return bad_ics


def apply_ica(
        ica: mne.preprocessing.ica.ICA,
        rec_start: mne.io.BaseRaw,
        ics_to_exclude: list,
        pid: str | None = None,
        cid: str | None = None,
        load: bool = False,
        save: bool = False,
) -> mne.io.BaseRaw:
    """

    :param ica:
    :param rec_start:
    :param pid:
    :param cid:
    :param ics_to_exclude:
    :param load:
    :param save:
    :return:
    """
    if save or load:  # Export cleaned data
        assert pid is not None, "Participant ID (pid) can't be None with the current save/load parameters."
        assert cid is not None, "Condition ID (cid) can't be None with the current save/load parameters."
    file_path = f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/RawClean'
    real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
    file_name = f'{real_cid}_iclean-raw.fif'
    if load:
        clean_rec = mne.io.Raw(f'{file_path}/{file_name}', preload=True)
    else:
        # Exclude bad ICs
        ica.exclude = ics_to_exclude

        # Applying ICA on the initial original data
        clean_rec = ica.apply(rec_start.copy())

        if save:
            # Export cleaned data
            clean_rec.save(f'{set_for_save(file_path)}/{file_name}', overwrite=True)

            # Export ICA with bad component(s) marked
            n_components = ica.n_components_
            save_path = set_for_save(f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/ICA/{real_cid}/{n_components}_comp')
            ica.save(f'{save_path}/{real_cid}_ica.fif', overwrite=True)

    return clean_rec


def plot_ica_result(
    ica: mne.preprocessing.ica.ICA,
    rec_start: mne.io.BaseRaw,
):
    """

    :param ica:
    :param rec_start:
    :return:
    """
    # Plot characteristic of EEG signal before and after
    # fig = ica.plot_overlay(
    ica.plot_overlay(
        rec_start,
        # Plot raw before ICA but noise free, as it's not helpful to visualize noise (that will not be included in the analyses anyway)
        exclude=ica.exclude, picks="eeg", title="Signals before (red) and after (black) ICA"
    )


def vis_and_set_art_chs(
        sure_art_chs: list,
        potential_art_chs: list,
        raw_rec: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
        test: bool = False,
):
    """

    :param sure_art_chs:
    :param potential_art_chs:
    :param raw_rec:
    :param test:
    :return:
    """
    # Initialize variables
    set_art_chs = []
    n = 0
    tot_dec = len([ch for ch in potential_art_chs if ch not in sure_art_chs])
    psd_sure_art_chs = compute_psd(raw_rec.copy().pick(sure_art_chs), test=False)

    # Iterate over channels potentially carrying the artifact
    for i, ch in enumerate(potential_art_chs):
        if i > 1 and test:
            break
        if ch not in sure_art_chs:

            # Create a dictionary containing psds to compare
            psd_by_ch_status = {'artifact channels': psd_sure_art_chs}
            psd_by_ch_status[ch] = compute_psd(raw_rec.copy().pick(ch), test=False)

            # Plot psds
            # psd_by_ch_status = {'pot_art': psd_by_ch_status['pot_art'], 'art': psd_by_ch_status['art_full'], }  # to invert
            compare_psds(psd_by_ch_status, show=True, lw=0.5, alpha=0.7)
            plt.show(block=True)  # makes the figure appear before the user input window
            decision = input(f'{n}/{tot_dec-1}: Does this channel carry the artifact? (y/n)')
            if decision.lower() == 'y':
                set_art_chs.append(ch)
            elif decision.lower() == 'break':
                break
            n += 1
    return set_art_chs


def load_bad_chs(
    pid: str,
    cid: str,
):
    """

    :param pid:
    :param cid:
    :return:
    """
    file_path = f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/RawClean'
    return np.loadtxt(f'{file_path}/{cid}_bad_channels', delimiter=',', dtype=str)


def export_bad_chs(
        data: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
        pid: str,
        cid: str,
):
    """

    :param data:
    :param pid:
    :param cid:
    :return:
    """
    save_path = f'/Volumes/My Passport/SpaNav/Sophie_backup/data/{get_exp_phase()}/{pid}/eeg/RawClean'
    bad_channels = data.info['bads']
    np.savetxt(f'{save_path}/{cid}_bad_channels', bad_channels, delimiter=',', fmt='%s')
    print(
        f"{cid}_bad_channels.txt saved in {save_path}"
    )


def basic_preproc_raw(
        pid: str | None = None,
        cid: str | None = None,
        raw_rec_start: mne.io.BaseRaw | None = None,
        load: bool = True,
        save: bool = False,
        verbose: bool = True,
) -> mne.io.BaseRaw | None:
    if save or load:
        assert pid is not None, "Participant ID (pid) can't be None with the current save/load parameters."
        assert cid is not None, "Condition ID (cid) can't be None with the current save/load parameters."
    real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
    file_path = f'{get_main_path()}/data/{get_exp_phase()}/{pid}/eeg/RawPreprocessed'
    file_name = f'{real_cid}-raw.fif'
    if load:
        try:
            raw_rec_end = mne.io.read_raw_fif(f'{file_path}/{file_name}', preload=True, verbose=verbose)
        except FileNotFoundError:
            print(f'File {file_name} not found for participant {pid} \n\t --> returning None')
            return None
    else:
        assert raw_rec_start is not None, "data can't be None with load=False (when data is to process)"
        raw_rec_end = raw_rec_start.copy()

        # Apply basic filters
        raw_rec_end.filter(l_freq=1, h_freq=60)
        raw_rec_end.notch_filter(freqs=50)

        # Downsample
        raw_rec_end.resample(250)

        # Interpolate bad channels
        raw_rec_end.interpolate_bads(
            reset_bads=True   # removes bads from info
        )

        if save:
            raw_rec_end.save(f'{set_for_save(file_path)}/{file_name}', overwrite=True)

    return raw_rec_end
