import warnings
import matplotlib.pyplot as plt

from preprocessing.behavior_to_eeg import get_times_retrieval_phases, get_trace_df, get_retrieval_df, extract_behav_events, \
    define_eeg_epochs
from extract_eeg import get_raw_to_epoch, get_epo_def, get_epo_rec
from utils.gen_utils import get_sid_cids, reveal_cid, save_figure, get_outputs_path
from visualization.vis_eeg import plot_evk_by_cat
from utils.gen_utils import plot_context

warnings.filterwarnings('ignore')  # Suppress all warnings


def gen_epo_tables(
        pids: list,
        segment_epoch_options: list,
        save: bool = False
) -> None:
    for pid in pids:
        block_times = get_times_retrieval_phases(pid)
        df_trace = get_trace_df(pid)
        retrieval_df = get_retrieval_df(pid)
        behav_events_df = extract_behav_events(pid, block_times, retrieval_df, df_trace)

        for segment_opt in segment_epoch_options:
            _ = define_eeg_epochs(behav_events_df, pid, segment_epochs=segment_opt, save=save)


def gen_contmov_epos(
        pids: list,
        segment_epoch_options: list[bool],
        test: bool = False,
) -> dict:
    epos_dict = {}
    for pid in pids:
        epos_dict[pid] = {}
        cids = get_sid_cids(pid, task=True, test=test)

        for cid in cids:
            epos_dict[pid][cid] = {}
            raw_rec = get_raw_to_epoch(pid, cid)

            for segment_bool in segment_epoch_options:
                epo_def_df = get_epo_def(pid, cid, segmented_epochs=segment_bool)
                epochs = get_epo_rec(
                    'ContMov',
                    pid,
                    cid,
                    raw_rec,
                    load=False,
                    epo_def_df=epo_def_df,
                    verbose=False
                )
                if epochs is not None:
                    if len(epochs) > 0:
                        key = 'ContMov_seg' if segment_bool else 'ContMov'
                        epos_dict[pid][cid][key] = epochs

    return epos_dict


def count_and_compare_epochs(
        epo_dict: dict,
) -> None:
    for pid, sid_dict in epo_dict.items():
        print(
            f"Participant {pid}"
        )
        for cid, cid_dict in sid_dict.items():
            print(
                f"\t{cid}"
            )
            for epoch_type, epochs in cid_dict.items():
                epoch_len = epoch_type.split('_')[-1]
                nr_of_epochs = len(epochs)
                print(
                    f"\t\t - {epoch_len}: {nr_of_epochs} epochs"
                )


def plot_contmov_epos(
        epos_dict: dict,
        show: bool = True,
        save: bool = False,
):
    with plot_context():
        for pid, sid_dict in epos_dict.items():
            for cid, epo_by_mov_def in sid_dict.items():
                plot_evk_by_cat(epo_by_mov_def, pid=pid, cid=cid, show=False, save=False)
                if save:
                    real_cid = reveal_cid(pid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(pid, cid=cid)
                    file_name = 'mov_durations.png'
                    save_path = get_outputs_path() / 'Evk' / pid / real_cid / file_name
                    save_figure(save_path, file_name, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


