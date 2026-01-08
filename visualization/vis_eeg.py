"""
********************************************************************************
    Title: Visualizing EEG

    Author: Sophie Caroni
    Date of creation: 30.09.2025

    Description:
    This script contains functions to visualize EEG data.
********************************************************************************
"""
import matplotlib.pyplot as plt
import numpy as np
import warnings
import mne
import pandas as pd
import seaborn as sns
import os
import configparser
from utils.gen_utils import plot_context, save_figure, layout_subplots_grid, get_nrows_ncols, reveal_cid, \
                             get_ti_positions, get_ch_by_region, get_epo_palette, get_outputs_path, get_eeg_path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

config = configparser.ConfigParser()
config.read('../config.ini')

SEED = config.getint('General', 'seed')


cm = 1/2.54


def plot_ti_sensors(
        info,
        pid: str,
        show: bool = False,
        save: bool = False,
) -> None:
    ti_chs = get_ti_positions(pid)
    info_copy = mne.create_info(ch_names=info.ch_names+ti_chs, sfreq=info['sfreq'], ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info_copy.set_montage(montage)
    info_copy['bads'] = ti_chs

    sensor_colors = {ch_name: 'k' for ch_name in info.ch_names if ch_name in info.get_montage().ch_names}
    # Override colors for the highlighted channels
    highlighted_channels = []
    for ch_name in highlighted_channels:
        if ch_name in sensor_colors: # Ensure the channel exists in the montage
            sensor_colors[ch_name] = 'r'
        else:
            print(f"Warning: Channel '{ch_name}' not found in the montage, skipping highlight.")

    # Plot the sensors using 'topomap' kind to enable coloring
    fig = mne.viz.plot_sensors(
        info_copy,
        kind='topomap',
        show_names=True,
        sphere=None,
        axes=None,
        title='EEG and TI sensors (red = TI)',
        show=False
    )

    if save:
        save_path = get_eeg_path() / pid
        file_name = 'ti_sensors.png'
        save_figure(save_path, file_name, fig=fig)
    if show:
        fig.show()
    else:
        plt.close(fig)


def plot_single_ch_psd(
        ch_psd: mne.time_frequency.Spectrum | np.ndarray,
        title: str,
        freqs: np.ndarray | None = None,
        ax: Axes | None = None,
        show: bool = False,
        **kwargs,
) -> tuple[Axes, np.ndarray]:
    with plot_context():
        if isinstance(ch_psd, np.ndarray):
            assert freqs is not None, "Spectrum frequencies (freqs) can't be None with the current PSD (ch_psd) instance."
            psd_T = ch_psd.T
        else:
            freqs = ch_psd.freqs
            psd_T = ch_psd.get_data(exclude=[]).T
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.plot(freqs, psd_T, **kwargs)
        if ax is None:
            ax.set_ylabel(r'log(Power [$\mu$V])')
            ax.set_xlabel('Frequency [Hz]')

        if show:
            plt.show()
    return ax, psd_T


def ch_psd_subplots(
        raw_rec: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
        ch_psd: mne.time_frequency.Spectrum,
        cid: str | None = None,
        pid: str | None = None,
        test: bool = False,
        show: bool = False,
        save: bool = False,
        file_name_pref: str = None,
) -> None:
    with plot_context():
        ch_by_region = get_ch_by_region(raw_rec.info)  # not using mne.channels.make_1020_channel_selections because returns wrong grouping if there are bad channels

        # Plot channels of each region together (as subplots)
        for brain_region, chs in ch_by_region.items():

            nr_ch = len(chs)
            nrows, ncols = layout_subplots_grid(nr_ch)
            fig = psd_subplots(ch_psd, chs, nrows, ncols, test=test)

            if save:
                assert pid is not None, "Participant ID (pid) can't be None with save=True (when data is to save)"
                assert cid is not None, "Condition ID (cid) can't be None with save=True (when data is to save)"
                real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
                file_name = f'{file_name_pref}_psd_subplots' if file_name_pref is not None else 'psd_subplots'
                file_name += f"_{brain_region}"
                save_path = get_outputs_path() / 'PSD' / pid / real_cid
                save_figure(save_path, file_name, fig=fig, dpi=1200)
            if show:
                fig.show()
            else:
                plt.close()


def ch_psd_overlap(
        raw_rec: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
        ch_psd: mne.time_frequency.Spectrum,
        cid: str | None = None,
        pid: str | None = None,
        show: bool = False,
        save: bool = False,
        file_name_pref: str = None,
) -> None:
    with plot_context():
        fig = ch_psd.plot(picks=raw_rec.ch_names, average=False, show=False)

        if save:
            assert pid is not None, "Participant ID (pid) can't be None with save=True (when data is to save)"
            assert cid is not None, "Condition ID (cid) can't be None with save=True (when data is to save)"
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            file_name = f'{file_name_pref}_psd_overlap' if file_name_pref is not None else 'psd_overlap'
            save_path = get_outputs_path() / 'PSD' / pid / real_cid
            save_figure(save_path, file_name, fig=fig)
        if show:
            fig.show()
        else:
            plt.close(fig)


def ics_psd_subplots(
        ica_psd: mne.time_frequency.Spectrum,
        cid: str | None = None,
        pid: str | None = None,
        test: bool = False,
        show: bool = False,
        save: bool = False,
        file_name_suff: str = None,
) -> None:
    with plot_context():
        nr_ics = ica_psd.info['nchan']
        if nr_ics >= 30:
            grouped_ics = {f'chs_{i}': ica_psd.info['ch_names'][start:stop] for i, (start, stop) in enumerate(zip([0, nr_ics//2], [nr_ics//2, nr_ics]))}
        else:
            grouped_ics = {'all': ica_psd.info['ch_names']}

        # Plot channels of each region together (as subplots)
        for group, ics in grouped_ics.items():

            nr_ics = len(ics)
            nrows, ncols = layout_subplots_grid(nr_ics)

            fig = psd_subplots(ica_psd, ics, nrows, ncols, test=test)

            if save:
                assert pid is not None, "Participant ID (pid) can't be None with save=True (when data is to save)"
                assert cid is not None, "Condition ID (cid) can't be None with save=True (when data is to save)"
                n_components = ica_psd.info['nchan']
                real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
                file_name = 'subplots'
                file_name += f"_{file_name_suff}" if file_name_suff is not None else ''
                file_name += f"_{group}" if len(grouped_ics.keys()) > 1 else ''
                save_path = get_outputs_path() / 'PSD' / pid / real_cid / 'ICs' / f'{n_components}_com'
                save_figure(save_path, file_name, fig=fig)
            if show:
                fig.show()
            else:
                plt.close(fig)


def psd_subplots(
        psd: mne.time_frequency.Spectrum,
        psd_sources: list,
        nrows: int,
        ncols: int,
        test: bool,
) -> Figure:
    with plot_context():
        fig, ax = plt.subplots(nrows, ncols, figsize=(7*nrows * cm, 5*ncols * cm), sharey=True, sharex=True)
        ax = ax.flatten()

        for i, source in enumerate(psd_sources):
            if i > 10 and test:
                break

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                psd.copy().plot(dB=True, axes=ax[i], picks=[source], show=False)
                if i == 0:
                    ylabel = ax[i].get_ylabel()
                ax[i].set_ylabel('')
                ax[i].set_title(source)

        fig.supylabel(
            ylabel
        )
        fig.supxlabel(
            'Frequency [Hz]'
        )
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.4)

        return fig


def compare_ch_psds(
        psds: mne.time_frequency.Spectrum,
        test: bool = False,
        show: bool = False,
        save: bool = False,
) -> None:
    with plot_context():
        ax = None
        ch_names = list(psds.values())[0].ch_names
        for j, ch in enumerate(ch_names):
            if test and j > 2:
                break
            for i, cid in enumerate(psds.keys()):
                if i == 0:
                    _, ax = plt.subplots(1, 1, dpi=600, figsize=(9 * cm, 6 * cm), sharey=True, sharex=True)

                rec_psd = psds[cid]
                ch_psd = rec_psd.copy().pick([ch])
                ax, _ = plot_single_ch_psd(ch_psd, title=ch, ax=ax, label=cid, linewidth=0.7, alpha=0.6)
                ax.legend()

            if save:
                file_name = f'{ch}.png'
                comps = '-vs-'.join(list(psds.keys()))
                save_path = get_outputs_path() / 'PSD' / 'Comparisons' / comps
                save_figure(save_path, file_name)
            if show:
                plt.show()
            else:
                plt.close()


def compare_psds(
        psds: dict[str, mne.time_frequency.Spectrum],
        show: bool = False,
        save: bool = False,
        **kwargs,
) -> None:
    with plot_context():
        for i, key in enumerate(psds.keys()):
            if i == 0:
                fig, ax = plt.subplots(1, 1, figsize=(9 * cm, 6 * cm), sharey=True, sharex=True)

            rec_psd_by_ch = psds[key]

            # If multiple channels present, compute avergae across channels
            if len(rec_psd_by_ch.get_data(exclude=[])) > 1:
                rec_psd = np.average(rec_psd_by_ch.get_data(exclude=[]), axis=0).T
            else:
                rec_psd = rec_psd_by_ch.get_data(exclude=[]).T
            rec_psd_db = 10 * np.log10(rec_psd * 1e6)  # convert to dB
            freqs = rec_psd_by_ch.freqs

            # Set custom colors
            c = None
            if key == 'bad channels':
                c = 'r'
            elif key == 'good channels':
                c = 'k'
            ax.plot(freqs, rec_psd_db, label=key, c=c, **kwargs)
        ax.set_ylabel('Power [dB]')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend()
        fig.tight_layout()

        show_save_dpi = 600
        if save:
            file_name = 'psds.png'
            comps = '-vs-'.join(list(psds.keys()))
            save_path = get_outputs_path() / 'PSD' / 'Comparisons' / comps
            save_figure(save_path, file_name, file_namedpi=show_save_dpi)
        if show:
            fig.set_dpi(show_save_dpi)
            fig.show()
        else:
            plt.close(fig)


def plot_evk_from_df(
        evk_df: pd.DataFrame,
        facet_by: str,
        pid: str | None = None,
        cid: str | None = None,
        show: bool = False,
        save: bool = False,
        axes: Axes | None = None,
        **kwargs,
) -> Axes:
    with plot_context():

        # Define figure structure
        groups = evk_df[facet_by].unique()
        nrows, ncols = get_nrows_ncols(groups)
        if axes is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols * cm, 7 * nrows * cm), sharey=True)
        else:
            fig = axes.flatten()[0].figure
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.

        # Plot one evoked-rec per subplot
        for i, (ax, (facet_val, facet_df)) in enumerate(zip(axes, evk_df.groupby(facet_by))):
            evk_lst = facet_df['evk'].to_list()
            avg_evk = mne.grand_average(evk_lst) # compute average across patients
            avg_evk.plot(axes=ax, show=False, sphere=False, **kwargs)
            ax.set_title(facet_val)

            # Customize ax labels
            row, col = divmod(i, ncols)
            if row < nrows - 1:
                ax.set_xlabel('')
            if col > 0:
                ax.set_ylabel('')

        # General figure customization
        real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
        fig.suptitle(real_cid)

        if save:
            file_name = 'evk_traces.png'
            save_path = get_outputs_path() / 'Evk' / pid / real_cid
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
    return axes


def plot_evk_by_grp(
        recs_dict: dict,
        pid: str | None = None,
        cid: str | None = None,
        show: bool = False,
        save: bool = False,
        axes: Axes | None = None,
        **kwargs,
) -> Axes:
    with plot_context():

        # Define figure structure
        groups = list(recs_dict.keys())
        nrows, ncols = get_nrows_ncols(groups)
        if axes is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols * cm, 7 * nrows * cm), sharey=True)
        else:
            fig = axes.flatten()[0].figure
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.

        # # If the recs_dict contains repetitive strings in keys, set repetitive part as figure's title and keep unique parts as subplot titles
        keys = list(recs_dict)
        prefix = os.path.commonprefix(keys).rsplit("_", 1)[0]
        subtitles = [k.replace(prefix, "", 1) for k in keys]
        # fig.suptitle(prefix)

        real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
        fig.suptitle(real_cid)

        # Plot one evoked-rec per subplot
        for i, (ax, rec, title) in enumerate(zip(axes, recs_dict.values(), subtitles)):
            if rec is not None and len(rec) > 0:
                if isinstance(rec, mne.Evoked):
                    rec.plot(axes=ax, show=False, sphere=False, **kwargs)
                elif isinstance(rec, np.ndarray):
                    rec.mean().plot(axes=ax, show=False, sphere=False, **kwargs)
                else:
                    rec.average().plot(axes=ax, show=False, **kwargs)
                ax.set_title(title)
            else:
                ax.set_title(title)
                continue

            # Customize ax labels
            row, col = divmod(i, ncols)
            if row < nrows - 1:
                ax.set_xlabel('')
            if col > 0:
                ax.set_ylabel('')

        fig.tight_layout()

        if save:
            file_name = f'{real_cid}_evk_traces.png'
            save_path = get_outputs_path() / 'Evk' / pid
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            fig.show()
        else:
            plt.close()
    return axes
            

def plot_psd_avg(
        psd_avg: mne.time_frequency.Spectrum | np.ndarray,
        psd_std: np.ndarray | None = None,
        freqs: np.ndarray | None = None,
        show: bool = False,
        ax: Axes | None = None,
        **kwargs
):
    with plot_context():
        if isinstance(psd_avg, np.ndarray):
            assert freqs is not None, "Spectrum frequencies (freqs) can't be None with the current PSD (ch_psd) instance."
        else:
            freqs = psd_avg.freqs
            # if psd_std is not None:
            #     psd_std = psd_std.T  # this is done in plot_single_ch_psd for psd_avg
        ax, psd_plot_format = plot_single_ch_psd(psd_avg, freqs=freqs, title='', ax=ax, **kwargs)
        if psd_std is not None:
            kwargs.pop("label", None)
            ax.fill_between(freqs, psd_plot_format-psd_std, psd_plot_format+psd_std, alpha=0.2, **kwargs)
        if show:
            plt.show()
    return ax


def plot_psd_avg_by_grp(
        psd_avg_dict: dict[np.ndarray, np.ndarray, np.ndarray],
        pid: str | None = None,
        cid: str | None = None,
        show: bool = False,
        save: bool = False,
        axes : Axes | None = None,
        **kwargs
):
    """
    This function plots power spectra of different types/categories in different subplots each
    :param psd_avg_dict:
    :param pid:
    :param cid:
    :param show:
    :param save:
    :param axes:
    :param kwargs:
    :return:
    """
    created_fig = False
    base_plot_label = kwargs.pop("label", None)
    groups = list(psd_avg_dict.keys())

    with plot_context():

        # Define figure structure
        if axes is None:
            nrows, ncols = get_nrows_ncols(groups)
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols * cm, 7 * nrows * cm), sharey=True, sharex=True)
            created_fig = True
        else:
            fig = axes.figure if isinstance(axes, Axes) else axes.flatten()[0].figure

        # Plot one psd per subplot
        for i, (key, (psd_avg, psd_std, freqs)) in enumerate(psd_avg_dict.items()):
            ax = axes if isinstance(axes, Axes) else axes.flatten()[i]

            # General figure customization
            row, col = divmod(i, ncols)
            if row == nrows - 1:
                ax.set_xlabel('Frequency [Hz]')
            if col == 0:
                ax.set_ylabel(r'log(Power [$\mu$V])')
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            fig.suptitle(real_cid)

            # Plot
            if psd_avg is not None:
                label = base_plot_label if i == 0 else None  # label only once per facet_val
                ax = plot_psd_avg(psd_avg, psd_std, freqs, show=False, ax=ax, label=label, **kwargs)
                ax.set_title(key)
            else:
                ax.set_title(key)
                continue

        if save:
            file_name = f'{real_cid}_psds_by_epo_type.png'
            save_path = get_outputs_path() / 'PSD' / pid
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            if created_fig:
                plt.close(fig)
    return axes


def compare_epo_psd(
        df,
        super_col: str,
        plot_subj: str,
        show: bool = True,
        save: bool = False,
):
    """
    This function plots power spectra overlapping different lengths of epoch they were computed from and using suplots
    for different stimulation conditions.
    :param df:
    :param super_col: column containing the variable used to superimpose plots
    :param plot_subj:
    :param show:
    :param save:
    :return:
    """
    # Convert powers to logs
    df = df.copy()
    conds = df['cond'].unique()
    n_conds = len(conds)
    ncols = n_conds + 1 if super_col == 'epo_len' else n_conds  # extra col for average across conditions
    nrows = 1

    with plot_context():

        # Define figure structure
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols * cm, 10 * nrows * cm), sharey=True, sharex=True)
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.`

        # Create one subplot per condition, then superimpose plots of super_col levels
        for i_ax, (ax, (cid, cid_df)) in enumerate(zip(axes, df.groupby('cond'))):
            for lvl, lvl_df in cid_df.groupby(super_col):
                lvl_df = lvl_df.copy()
                lvl_df = lvl_df.sort_values("freq")

                psd = lvl_df['pw_avg'].to_numpy()
                sem = lvl_df["pw_std"].to_numpy() / np.sqrt(lvl_df["N"].to_numpy())
                freqs = lvl_df['freq'].to_numpy()

                # Plot
                palette = get_epo_palette()
                color = palette[lvl]

                labels = {
                    'ContMov': 'Continuous movement',
                    'Static': 'Static',
                    'MovOn': 'Movement onset',
                    'ObjPres': 'Object presentation',
                    'Raw': 'Continuous data',
                }

                label = labels[lvl] if i_ax == 0 else ''
                ax = plot_psd_avg(psd, sem, freqs, show=False, ax=ax, label=label, color=color)
                ax.set_title(cid)

        # # If we are comparing epochs-length: the last subplot put the average PSD across participants and conditions
        # if super_col == 'epo_len':
        #     for lvl, lvl_df in df.groupby(super_col):
        #         psd_avg = lvl_df.groupby(['freq'])['pw'].mean().values
        #         psd_std = lvl_df.groupby(['freq'])['pw'].std().values
        #         freqs = lvl_df['freq'].unique()
        #         axes[-1] = plot_psd_avg(psd_avg, psd_std, freqs, show=False, ax=axes[-1])
        #     axes[-1].set_title('Average across conditions')

        # Figure customizations
        fig.legend()  #loc='center right', bbox_to_anchor=(0.52, 0.7))
        suptitle = 'Object-presentation Epochs' if super_col == 'epo_s' else ''
        title_pids = 'Average across participants' if plot_subj.startswith('average') else f"Participant {plot_subj}"
        fig.suptitle(f"{suptitle}\n{title_pids}")
        fig.supylabel(r'log(Power [$\mu$V])')
        fig.supxlabel('Frequency [Hz]')
        fig.tight_layout()

        if save:
            save_path = get_outputs_path() / 'PSD' if plot_subj.startswith('average') else get_outputs_path() / 'PSD' / plot_subj
            file_name = f'psd_{super_col}.png'
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def compare_band_metric(
        df: pd.DataFrame,
        metric_name: str,
        super_col: str,
        show: bool = True,
        save: bool = False,
):
    """
    This function plots values of an oscillatory-band metric in an EEG band overlapping different participants and using
    subplots for different stimulation conditions.
    :param df:
    :param metric_name: name of the oscillatory-band metric
    :param super_col: column containing the variable used to superimpose plots
    :param show:
    :param save:
    :return:
    """
    # Convert powers to logs
    df = df.copy()
    df['metric'] = np.log10(df[metric_name]) if metric_name != 'osc_snr' else df[metric_name]

    n_cids = len(df['cond'].unique())
    nrows = 1
    ncols = n_cids + 1 if super_col == 'epo_len' else n_cids  # extra col for average across conditions
    with plot_context():
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols * cm, 10 * nrows * cm), sharey=True, sharex=True)
        _ = plot_band_metric_by_cat(df, cat_col='cond', metric_name='metric', show=False, axes=axes)

        # If we are comparing epochs-length, in the last subplot put the average PSD across participants and conditions
        if super_col == 'epo_len':
            axes[-1] = plot_band_metric(df, axes[-1], metric_name, show=False, legend=False)
            axes[-1].set_title('Average across Conditions')

        suptitle = 'Object-presentation Epochs' if super_col == 'epo_s' else ''
        pids = df['pid'].unique()
        title_pids = 'Absolute band-power' if metric_name == 'abs_pw' else ('Relative band-power' if metric_name == 'rel_pw' else 'SNR')
        fig.suptitle(f"{suptitle}\n{title_pids}")
        ylabel = r'log(Power [$\mu$V])' if metric_name != 'osc_snr' else 'SNR'
        fig.supylabel(ylabel)
        fig.supxlabel('Frequency bands')
        fig.tight_layout()

        if save:
            save_path = get_outputs_path() / 'PSD' if len(pids) > 1 else get_outputs_path() / 'PSD' / pids[0]
            file_name = f'{metric_name}_{super_col}.png'
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_band_metric_by_cat(
        band_metric_df: pd.DataFrame,
        cat_col: str,
        metric_name: str,
        show: bool = False,
        axes : Axes | None = None,
        **kwargs
):
    """
    This function plots a metric in an EEG band of different types/categories in different subplots each
    :param band_metric_df:
    :param cat_col: name of column to use to group data in different subplots
    :param metric_name: name of the oscillatory-band metric
    :param show:
    :param axes:
    :param kwargs:
    :return:
    """
    with plot_context():

        # Define figure structure
        cats = band_metric_df[cat_col].unique()
        if axes is None:
            nrows, ncols = get_nrows_ncols(cats)
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols * cm, 7 * nrows * cm), sharey=True, sharex=True)
        else:
            fig = axes.figure if isinstance(axes, Axes) else axes.flatten()[0].figure
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.

        # Plot each band's power of interest per subplot
        for i, (ax, (cat, cat_df)) in enumerate(zip(axes, band_metric_df.groupby(cat_col))):
            legend = i == 0

            ax = plot_band_metric(cat_df, ax, metric_name, show=False, legend=legend, **kwargs)
            ax.set_title(cat)

        # Customize figure
        if len(cats) > 2:
            fig.supylabel(r'log(Power [$\mu$V])')
            fig.supxlabel('Frequency bands')

        if show:
            plt.show()
    return axes


def plot_band_metric(
        metric_df: pd.DataFrame,
        ax: Axes,
        metric_name: str,
        show: bool = False,
        legend: bool = True,
        **kwargs
):
    with plot_context():
        palette = get_epo_palette()
        custom_labels = {
            'ContMov': 'Continuous movement',
            'Static': 'Static',
            'MovOn': 'Movement onset',
            'ObjPres': 'Object Presentation',
        }

        # Plot poin if there is only one observation, else violins/boxes
        if (metric_df.groupby(['band', 'epo_type'])[metric_name].count() == 1).any().any():

            # Add some jitter to points
            jitter = 0.05  # adjust as needed
            rng = np.random.default_rng(SEED)
            metric_df = metric_df.copy()
            band_cat = pd.Categorical(metric_df["band"])  # ensure band is categorical and get stable numeric codes
            metric_df["band_code"] = band_cat.codes.astype(float)  # convert to numeric codes
            metric_df["band_jitt"] = metric_df["band_code"] + rng.uniform(-jitter, jitter, size=len(metric_df))  # add jitter to numeric codes

            plot = sns.scatterplot(
                data=metric_df,
                x='band_jitt',
                y=metric_name,
                hue='epo_type',
                palette=palette,
                ax=ax,
                alpha=0.8,
                legend=legend,
                **kwargs
            )

        else:
            plot = sns.boxplot(
                data=metric_df,
                x='band',
                y=metric_name,
                hue='epo_type',
                palette=palette,
                ax=ax,
                saturation=0.7,
                # alpha=0.3,
                legend=legend,
                fill=True,
                # inner=None,
                # density_norm='count',
                **kwargs
            )
            # plot2 = sns.violinplot(
            #     data=metric_df,
            #     x='band',
            #     y=metric_name,
            #     hue='epo_type',
            #     palette=palette,
            #     ax=ax,
            #     saturation=1,
            #     legend=legend,
            #     fill=False,
            #     inner='point',
            #     density_norm='count',
            #     **kwargs
            # )
            # plot2.set(xlabel=None, ylabel=None)
        if legend:
            handles, current_labels = ax.get_legend_handles_labels()
            new_labels = [custom_labels.get(lbl, lbl) for lbl in current_labels]
            ax.legend(handles, new_labels, loc='upper right')  # , title='Epoch type'

        bands = metric_df['band'].unique()
        ax.set_xlim(-0.5, len(bands) - 0.5)  # no extra whitespace
        ax.margins(x=0.05)  # small outer margin
        plot.set(xlabel=None, ylabel=None)  # they are set elsewehre

        if show:
            plt.show()
    return ax


def plot_muscle_art(
        raw_rec: mne.io.BaseRaw,
        scores_muscle: list,
        muscle_threshold: float,
        show: bool = True,
):
    with plot_context():
        fig, ax = plt.subplots()
        ax.plot(raw_rec.times, scores_muscle)
        ax.axhline(y=muscle_threshold, color="r")
        ax.set(xlabel="Time [s]", ylabel="Z-score", title="Muscle activity")

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_preprocessing_result(
        raw_before: mne.io.BaseRaw,
        raw_after: mne.io.BaseRaw,
        pid: str | None = None,
        cid: str | None = None,
        show: bool = False,
        save: bool = False,
):
    # Include all channels in both before and after plots (also those marked as bad before preprocessing)
    picks = raw_after.ch_names

    # Resample initial recording
    raw_before_res = raw_before.copy().resample(250)

    plot_kwargs = dict(fmax=40, picks=picks, reject_by_annotation=False)

    # Create figure and plots
    with plot_context():
        fig, axs = plt.subplots(2, 2, figsize=(17*cm, 12*cm), sharey=True, sharex=True)

        # Before plots
        raw_before_res.plot_psd(average=True, ax=axs[0, 0], **plot_kwargs)
        axs[0, 0].set_title('Starting data - Averaged chs')
        raw_before_res.plot_psd(average=False, ax=axs[0, 1], **plot_kwargs)
        axs[0, 1].set_title('Starting data - Single chs')

        # After plots
        raw_after.plot_psd(average=True, ax=axs[1, 0], **plot_kwargs)
        axs[1, 0].set_title('Preprocessed data - Averaged chs')
        raw_after.plot_psd(average=False, ax=axs[1, 1], **plot_kwargs)
        axs[1, 1].set_title('Preprocessed data - Single chs')

        if save:
            real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
            save_path = get_outputs_path() / 'PSD' / pid / real_cid
            file_name = 'prepro_result.png'
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_schematic_epo_def(
        behav_events_df: pd.DataFrame,
        eeg_events_df: pd.DataFrame,
        pid: str | None = None,
        show: bool = False,
        save: bool = False,
) -> None:

    for block_n, block_df in behav_events_df.groupby('RetrievalBlock'):

        # Create one figure per block (containing all its trials as subplots)
        trials_in_block = len(block_df['Trial'].unique())
        nrows, ncols = layout_subplots_grid(n=trials_in_block, max_cols=2)
        with plot_context():
            fig, axs = plt.subplots(nrows, ncols, figsize=(17*ncols * cm, 4*nrows * cm))
            axs = axs.ravel()

            # Put each trial in a different subplot
            for i, (trial_n, trial_df) in enumerate(block_df.groupby('Trial')):
                ax = axs[i]

                # Subset dataframes to specific block and trial numers
                behav_data = trial_df.reset_index(drop=True)
                eeg_data = eeg_events_df[(eeg_events_df['RetrievalBlock'] == block_n) & (
                            eeg_events_df['TrialNumber'] == trial_n)].reset_index(drop=True)
                behav_data.drop(['RetrievalBlock', 'Trial', 'Condition'], axis=1,
                                inplace=True)  # drop now useless columns
                eeg_data.drop(['RetrievalBlock', 'TrialNumber', 'Condition'], axis=1,
                              inplace=True)  # drop now useless columns

                # Sort to make sure states/epochs are ordered on time
                behav_data = behav_data.sort_values("StateStart")
                eeg_data = eeg_data.sort_values("EpochStart")

                # Plot behavior, if in this trial there was any
                if len(behav_data) > 0:
                    plot_schematic_behavior(ax, behav_data)

                # Plot eeg epochs, if in this trial there were any
                if len(eeg_data) > 0:
                    plot_schematic_eeg_epochs(ax, eeg_data)

                # General plot customizations
                ax.legend(loc='upper left')
                ax.set_title(f'Trial {trial_n}')

            # hide unused axes
            for ax in axs[trials_in_block:]:
                ax.set_visible(False)

            fig.suptitle(f'Block {block_n}')
            fig.supxlabel('Time [s]')
            fig.tight_layout()

            if save:
                real_cid = reveal_cid(pid, block_n=int(block_n))
                save_path = get_outputs_path() / 'Epo' / pid / real_cid
                file_name = f'block{block_n}_trials_epoching.png'
                save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)


def plot_schematic_behavior(
        ax: Axes,
        behav_data: pd.DataFrame
) -> None:
    # Add needed x and y columns in behavioral plot
    y_movement = behav_data['State'].apply(lambda r: 1 if (r == 'Moving' or r == 'MovOn') else 0).to_numpy()  # Binarize movement (either present 1 or not 0)
    x_time = behav_data['StateStart'].to_numpy()

    # Duplicate last observation to allow plotting of full duration of last state
    x_time = np.r_[x_time, behav_data['StateEnd'].iloc[-1]]  # use np.r_ as row-wise concatenator
    y_movement = np.r_[y_movement, y_movement[-1]]  # use np.r_ as row-wise concatenator

    with plot_context():
        ax.step(
            x_time, y_movement,
            where="post",
            color="k",
            alpha=0.7,
            linestyle="dotted",
            linewidth=1,
            label='Behavior'
        )

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Stationary", "Moving"])

        # Retrieve start of first state and end of last one
        first_start = behav_data.loc[0, 'StateStart']
        last_end = behav_data['StateEnd'].iloc[-1]
        ax.set_xlim(first_start, last_end)


def plot_schematic_eeg_epochs(
        ax: Axes,
        eeg_epochs_df: pd.DataFrame,
) -> None:
    # Shift times based on block
    eeg_epochs_df['EpochStart'] = eeg_epochs_df['EpochStart'] + eeg_epochs_df['BlockStart']
    eeg_epochs_df['EpochEnd'] = eeg_epochs_df['EpochEnd'] + eeg_epochs_df['BlockStart']

    # One color per EpochType
    palette = get_epo_palette()

    # Vertical position
    patch_height = 0.1
    y_levels = {'Static': 0, 'MovOn': 0.5, 'ContMov': 1-patch_height}

    with plot_context():
        plotted_epos = []
        for etype, s, e in eeg_epochs_df[["EpochType", "EpochStart", "EpochEnd"]].itertuples(index=False):

            label = None if etype in plotted_epos else f'{etype} (x{len(eeg_epochs_df[eeg_epochs_df["EpochType"] == etype])})'
            color = palette[etype]
            rect = Rectangle(
                (s, y_levels[etype]),
                e - s,
                patch_height,
                facecolor=(color, 0.4),
                edgecolor=(color, 1),
                linewidth=1,
                label=label,
            )
            ax.add_patch(rect)
            plotted_epos.append(etype)


def compare_found_peaks(
        df: pd.DataFrame,
        metric_name: str,
        show: bool = True,
        save: bool = False,
):
    """

    :param metric_name: 
    :param df:
    :param show:
    :param save:
    :return:
    """
    facet_by = 'cond'
    nrows = 1
    ncols = len(df[facet_by].unique())
    with plot_context():

        # Define figure structure
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols * cm, 10 * nrows * cm), sharey=True, sharex=True)
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.

        # Plot each band's power per subplot
        for i, (ax, (facet_val, facet_df)) in enumerate(zip(axes, df.groupby(facet_by))):

            legend = i == 0

            # Create list to define plotting order
            bands = ["theta", "alpha", "38-42"]
            epo_types = ["Raw", "ObjPres", "Static", "MovOn", "ContMov"] if 'Raw' in facet_df['epo_type'].unique() else ["ObjPres", "Static", "MovOn", "ContMov"]

            # Sum numer of peaks for each band and epoch-type
            plot_df = (facet_df
                    .groupby(["band", "epo_type"])[metric_name]
                    .sum()
                    .unstack("epo_type", fill_value=0)
                    .reindex(index=bands, columns=epo_types))  # force defined order

            # Define bars
            n_bars = len(epo_types)
            bar_width = 0.8 / n_bars  # total band-group width ~0.8
            n_band_groups = len(bands)
            x0 = np.arange(n_band_groups)
            for i, etype in enumerate(epo_types):
                x = x0 + i * bar_width
                y = plot_df[etype].values
                color = get_epo_palette()[etype]
                ax.bar(x, y, width=bar_width, edgecolor="grey", label=etype, color=color)

            ax.set_xticks(x0 + (n_bars - 1) * bar_width / 2)
            ax.set_xticklabels(bands)
            ax.set_title(facet_val)

            if legend:
                ax.legend()

        # suptitle = 'Object-presentation Epochs' if super_col == 'epo_s' else ''
        fig.suptitle(f"Oscillatory peaks found by FOOOF")
        ylabel = '# Peaks (sum across subjects)'
        fig.supylabel(ylabel)
        fig.supxlabel('Frequency bands')
        fig.tight_layout()

        if save:
            pids = df['pid'].unique()
            save_path = get_outputs_path() / 'PSD' if len(pids) > 1 else get_outputs_path() / 'PSD' / pids[0]
            file_name = 'found_peaks.png'
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)













