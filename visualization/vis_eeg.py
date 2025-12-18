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
from utils.gen_utils import plot_context, save_figure, layout_subplots_grid, get_nrows_ncols, reveal_cid, \
                             get_ti_positions, get_ch_by_region, get_epo_palette, SEED, get_wd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

cm = 1/2.54


def plot_ti_sensors(
        info,
        sid: str,
        show: bool = False,
        save: bool = False,
) -> None:
    ti_chs = get_ti_positions(sid)
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
        save_figure(f'{get_wd()}/data/{sid}', 'ti_sensors.png', fig=fig)
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
        sid: str | None = None,
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
                assert sid is not None, "Subject ID (sid) can't be None with save=True (when data is to save)"
                assert cid is not None, "Condition ID (cid) can't be None with save=True (when data is to save)"
                real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
                file_name = f'{file_name_pref}_psd_subplots' if file_name_pref is not None else 'psd_subplots'
                file_name += f"_{brain_region}"
                save_figure(f'../outputs/PSD/{sid}/{real_cid}', file_name, fig=fig, dpi=1200)
            if show:
                fig.show()
            else:
                plt.close()


def ch_psd_overlap(
        raw_rec: mne.io.BaseRaw | mne.Epochs | mne.Evoked,
        ch_psd: mne.time_frequency.Spectrum,
        cid: str | None = None,
        sid: str | None = None,
        show: bool = False,
        save: bool = False,
        file_name_pref: str = None,
) -> None:
    with plot_context():
        fig = ch_psd.plot(picks=raw_rec.ch_names, average=False, show=False)

        if save:
            assert sid is not None, "Subject ID (sid) can't be None with save=True (when data is to save)"
            assert cid is not None, "Condition ID (cid) can't be None with save=True (when data is to save)"
            real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
            file_name = f'{file_name_pref}_psd_overlap' if file_name_pref is not None else 'psd_overlap'
            save_figure(f'../outputs/PSD/{sid}/{real_cid}', file_name, fig=fig)
        if show:
            fig.show()
        else:
            plt.close(fig)


def ics_psd_subplots(
        ica_psd: mne.time_frequency.Spectrum,
        cid: str | None = None,
        sid: str | None = None,
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
                assert sid is not None, "Subject ID (sid) can't be None with save=True (when data is to save)"
                assert cid is not None, "Condition ID (cid) can't be None with save=True (when data is to save)"
                n_components = ica_psd.info['nchan']
                real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
                file_name = 'subplots'
                file_name += f"_{file_name_suff}" if file_name_suff is not None else ''
                file_name += f"_{group}" if len(grouped_ics.keys()) > 1 else ''
                save_figure(f'../outputs/PSD/{sid}/{real_cid}/ICs/{n_components}_com', file_name, fig=fig)
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
                comps = '-vs-'.join(list(psds.keys()))
                save_figure(f'../outputs/PSD/Comparisons/{comps}', f'{ch}.png')
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
            comps = '-vs-'.join(list(psds.keys()))
            save_figure(f'../outputs/PSD/Comparisons/{comps}', 'psds.png', dpi=show_save_dpi)
        if show:
            fig.set_dpi(show_save_dpi)
            fig.show()
        else:
            plt.close(fig)


def plot_evk_from_df(
        evk_df: pd.DataFrame,
        cat_col: str,
        sid: str | None = None,
        cid: str | None = None,
        show: bool = False,
        save: bool = False,
        axes: Axes | None = None,
        **kwargs,
) -> Axes:
    with plot_context():

        # Define figure structure
        cats = evk_df[cat_col].unique()
        nrows, ncols = get_nrows_ncols(cats)
        if axes is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols * cm, 7 * nrows * cm), sharey=True)
        else:
            fig = axes.flatten()[0].figure
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.

        # Plot one evoked-rec per subplot
        for i, (ax, (cat, cat_df)) in enumerate(zip(axes, evk_df.groupby(cat_col))):
            evk_lst = cat_df['evk'].to_list()
            avg_evk = mne.grand_average(evk_lst) # compute average across patients
            avg_evk.plot(axes=ax, show=False, sphere=False, **kwargs)
            ax.set_title(cat)

            # Customize ax labels
            row, col = divmod(i, ncols)
            if row < nrows - 1:
                ax.set_xlabel('')
            if col > 0:
                ax.set_ylabel('')

        if save:
            real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
            save_figure(f'../outputs/Evk/{sid}/{real_cid}', 'evk_traces.png', dpi=900, bbox_inches='tight')
        if show:
            plt.show()
    return axes


def plot_evk_by_cat(
        recs_dict: dict,
        sid: str | None = None,
        cid: str | None = None,
        show: bool = False,
        save: bool = False,
        axes: Axes | None = None,
        segmented_epochs: bool = False,
        **kwargs,
) -> Axes:
    with plot_context():

        # Define figure structure
        cats = list(recs_dict.keys())
        nrows, ncols = get_nrows_ncols(cats)
        if axes is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols * cm, 7 * nrows * cm), sharey=True)
        else:
            fig = axes.flatten()[0].figure
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.

        # If the recs_dict contains repetitive strings in keys, set repetitive part as figure's title and keep unique parts as subplot titles
        keys = list(recs_dict)
        prefix = os.path.commonprefix(keys).rsplit("_", 1)[0]
        subtitles = [k.replace(prefix, "", 1) for k in keys]
        fig.suptitle(prefix)

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
            real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
            file_name = f'evk_traces.png' if not segmented_epochs else f'SEG_evk_traces.png'
            save_figure(f'../outputs/Evk/{sid}/{real_cid}', file_name, dpi=900, bbox_inches='tight')
        if show:
            fig.show()
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


def plot_psd_avg_by_cat(
        psd_avg_dict: dict[np.ndarray, np.ndarray, np.ndarray],
        sid: str | None = None,
        cid: str | None = None,
        show: bool = False,
        save: bool = False,
        axes : Axes | None = None,
        segmented_epochs: bool = False,
        **kwargs
):
    """
    This function plots power spectra of different types/categories in different subplots each
    :param psd_avg_dict:
    :param sid:
    :param cid:
    :param show:
    :param save:
    :param axes:
    :param segmented_epochs:
    :param kwargs:
    :return:
    """
    created_fig = False
    base_plot_label = kwargs.pop("label", None)
    cats = list(psd_avg_dict.keys())

    with plot_context():

        # Define figure structure
        if axes is None:
            nrows, ncols = get_nrows_ncols(cats)
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols * cm, 7 * nrows * cm), sharey=True, sharex=True)
            created_fig = True
        else:
            fig = axes.figure if isinstance(axes, Axes) else axes.flatten()[0].figure

        # Plot one psd per subplot
        for i, (key, (psd_avg, psd_std, freqs)) in enumerate(psd_avg_dict.items()):
            ax = axes if isinstance(axes, Axes) else axes.flatten()[i]
            if psd_avg is not None:
                label = base_plot_label if i == 0 else None  # label only once per cat
                ax = plot_psd_avg(psd_avg, psd_std, freqs, show=False, ax=ax, label=label, **kwargs)
                ax.set_title(key)
            else:
                ax.set_title(key)
                continue

        # Customize figure
        if len(cats) > 2:
            suplabels_fontsize = plt.rcParams['axes.labelsize']
            fig.supylabel(r'log(Power [$\mu$V])', fontsize=suplabels_fontsize)
            fig.supxlabel('Frequency [Hz]', fontsize=suplabels_fontsize)
        else:
            ax.set_ylabel(r'log(Power [$\mu$V])')
            ax.set_xlabel('Frequency [Hz]')

        if save:
            real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
            file_name = f'psds_by_epo_type.png' if not segmented_epochs else f'SEG_psds_by_epo_type.png'
            save_figure(f'../outputs/PSD/{sid}/{real_cid}', file_name, dpi=900, bbox_inches='tight')
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
        segmented_epochs: bool = False,
):
    """
    This function plots power spectra overlapping different lengths of epoch they were computed from and using suplots
    for different stimulation conditions.
    :param df:
    :param super_col: column containing the variable used to superimpose plots
    :param plot_subj:
    :param show:
    :param save:
    :param segmented_epochs:
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
                # sem_n = lvl_df.copy().reset_index().loc[0, 'N']
                # sem = lvl_df['pw_std'].to_numpy() / np.sqrt(sem_n)  # ok if N is constant
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

        # # If we are comparing epochs-length: the last subplot put the average PSD across subjects and conditions
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
        title_sids = 'Average across subjects' if plot_subj.startswith('average') else f"Subject {plot_subj}"
        fig.suptitle(f"{suptitle}\n{title_sids}")
        fig.supylabel(r'log(Power [$\mu$V])')
        fig.supxlabel('Frequency [Hz]')
        fig.tight_layout()

        if save:
            save_path = f'../outputs/PSD' if plot_subj.startswith('average') else f'../outputs/PSD/{plot_subj}'
            file_name = f'SEG_psd_{super_col}.png' if segmented_epochs else f'psd_{super_col}.png'
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
        segmented_epochs: bool = False,
):
    """
    This function plots values of an oscillatory-band metric in an EEG band overlapping different subjects and using
    subplots for different stimulation conditions.
    :param df:
    :param metric_name: name of the oscillatory-band metric
    :param super_col: column containing the variable used to superimpose plots
    :param show:
    :param save:
    :param segmented_epochs:
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

        # If we are comparing epochs-length, in the last subplot put the average PSD across subjects and conditions
        if super_col == 'epo_len':
            axes[-1] = plot_band_metric(df, axes[-1], metric_name, show=False, legend=False)
            axes[-1].set_title('Average across Conditions')

        suptitle = 'Object-presentation Epochs' if super_col == 'epo_s' else ''
        sids = df['sid'].unique()
        title_sids = 'Absolute band-power' if metric_name == 'abs_pw' else ('Relative band-power' if metric_name == 'rel_pw' else 'SNR')
        fig.suptitle(f"{suptitle}\n{title_sids}")
        ylabel = r'log(Power [$\mu$V])' if metric_name != 'osc_snr' else 'SNR'
        fig.supylabel(ylabel)
        fig.supxlabel('Frequency bands')
        fig.tight_layout()

        if save:
            save_path = f'../outputs/PSD' if len(sids) > 1 else f'../outputs/PSD/{sids[0]}'
            file_name = f'SEG_{metric_name}_{super_col}.png' if segmented_epochs else f'{metric_name}_{super_col}.png'
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
        sid: str | None = None,
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
            real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
            save_figure(f'../outputs/PSD/{sid}/{real_cid}', 'prepro_result.png', dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_schematic_epo_def(
        beh_events_df: pd.DataFrame,
        eeg_events_df: pd.DataFrame,
        block_n: int = 1,
        trial_n: int = 1,
        sid: str | None = None,
        show: bool = False,
        save: bool = False,
        segmented_epochs: bool = False,
) -> None:

    # Subset dataframes to specific block and trial numers
    beh_data = beh_events_df[(beh_events_df['RetrievalBlock'] == block_n) & (beh_events_df['Trial'] == trial_n)].reset_index(drop=True)
    eeg_data = eeg_events_df[(eeg_events_df['RetrievalBlock'] == block_n) & (eeg_events_df['TrialNumber'] == trial_n)].reset_index(drop=True)
    beh_data.drop(['RetrievalBlock', 'Trial', 'Condition'], axis=1, inplace=True)  # drop now useless columns
    eeg_data.drop(['RetrievalBlock', 'TrialNumber', 'Condition'], axis=1, inplace=True)  # drop now useless columns

    # Sort to make sure states/epochs are ordered on time
    beh_data = beh_data.sort_values("StateStart")
    eeg_data = eeg_data.sort_values("EpochStart")

    with plot_context():
        fig, ax = plt.subplots(1, 1, figsize=(17*cm, 2*cm))
        plot_schematic_behavior(ax, beh_data)
        plot_schematic_eeg_epochs(ax, eeg_data)
        ax.legend(loc='upper left')

        if save:
            real_cid = reveal_cid(sid, block_n=block_n)
            file_name = f'epoching_trial{trial_n}.png' if not segmented_epochs else f'SEG_epoching_trial{trial_n}.png'
            save_figure(f'../outputs/PSD/{sid}/{real_cid}', file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_schematic_behavior(
        ax: Axes,
        behavioral_data: pd.DataFrame
    ) -> None:
    # Add needed x and y columns in behavioral plot
    behavioral_data['y_Movement'] = behavioral_data['State'].apply(lambda r: 1 if (r == 'Moving' or r == 'MovOn') else 0)  # Binarize movement (either present 1 or not 0)
    behavioral_data['x_Time'] = behavioral_data['StateStart']

    with plot_context():
        ax.step(
            behavioral_data["x_Time"],
            behavioral_data["y_Movement"].astype('category'),
            where="post",
            color="k",
            alpha=0.7,
            linestyle="dotted",
            linewidth=1,
            label='Behavior'
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Stationary", "Moving"])


def plot_schematic_eeg_epochs(
        ax: Axes,
        eeg_epochs_df: pd.DataFrame,
) -> None:
    # Shift times based on block
    eeg_epochs_df['EpochStart'] = eeg_epochs_df['EpochStart'] + eeg_epochs_df['BlockStart']
    eeg_epochs_df['EpochEnd'] = eeg_epochs_df['EpochEnd'] + eeg_epochs_df['BlockStart']
    eeg_epochs_df['EpochStart'] = eeg_epochs_df['EpochStart']
    eeg_epochs_df['EpochEnd'] = eeg_epochs_df['EpochEnd']

    # one color per EpochType (no manual colors; uses matplotlib default cycle)
    epoch_types = eeg_epochs_df["EpochType"].unique()
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
        segmented_epochs: bool = False,
):
    """

    :param df:
    :param show:
    :param save:
    :param segmented_epochs:
    :return:
    """
    subplots_cat_col = 'cond'
    nrows = 1
    ncols = len(df[subplots_cat_col].unique())
    with plot_context():

        # Define figure structure
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols * cm, 10 * nrows * cm), sharey=True, sharex=True)
        axes = np.atleast_1d(axes).ravel()  # wraps into an array if is an Axes object; flattens.

        # Plot each band's power per subplot
        for i, (ax, (cat, cat_df)) in enumerate(zip(axes, df.groupby(subplots_cat_col))):

            legend = i == 0

            # Create list to define plotting order
            bands = ["theta", "alpha", "38-42"]
            epo_types = ["Raw", "ObjPres", "Static", "MovOn", "ContMov"] if 'Raw' in cat_df['epo_type'].unique() else ["ObjPres", "Static", "MovOn", "ContMov"]

            # Sum numer of peaks for each band and epoch-type
            plot_df = (cat_df
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
            ax.set_title(cat)

            if legend:
                ax.legend()

        # suptitle = 'Object-presentation Epochs' if super_col == 'epo_s' else ''
        fig.suptitle(f"Oscillatory peaks found by FOOOF")
        ylabel = '# Peaks (sum across subjects)'
        fig.supylabel(ylabel)
        fig.supxlabel('Frequency bands')
        fig.tight_layout()

        if save:
            sids = df['sid'].unique()
            save_path = f'../outputs/PSD' if len(sids) > 1 else f'../outputs/PSD/{sids[0]}'
            file_name = f'SEG_found_peaks.png' if segmented_epochs else f'found_peaks.png'
            save_figure(save_path, file_name, dpi=900, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)













