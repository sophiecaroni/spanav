import warnings
import matplotlib.pyplot as plt

from preprocessing.behavior_to_eeg import get_times_retrieval_phases, get_trace_df, get_retrieval_df, extract_behavioral_events, \
    define_eeg_epochs
from extract_eeg import get_raw_to_epoch, get_epo_def, get_epo_rec
from utils.gen_utils import get_sid_cids, set_for_save, reveal_cid, save_figure
from visualization.vis_eeg import plot_evk_by_cat
from utils.gen_utils import plot_context

warnings.filterwarnings('ignore')  # Suppress all warnings


def gen_epo_tables(
        sids: list,
        segment_epoch_options: list,
        save: bool = False
) -> None:
    for sid in sids:
        block_times = get_times_retrieval_phases(sid)
        df_trace = get_trace_df(sid)
        retrieval_df = get_retrieval_df(sid)
        beh_events_df = extract_behavioral_events(sid, block_times, retrieval_df, df_trace)

        for segment_opt in segment_epoch_options:
            _ = define_eeg_epochs(beh_events_df, sid, segment_epochs=segment_opt, save=save)


def gen_contmov_epos(
        sids: list,
        segment_epoch_options: list[bool],
        test: bool = False,
) -> dict:
    epos_dict = {}
    for sid in sids:
        epos_dict[sid] = {}
        cids = get_sid_cids(sid, task=True, test=test)

        for cid in cids:
            epos_dict[sid][cid] = {}
            raw_rec = get_raw_to_epoch(sid, cid)

            for segment_bool in segment_epoch_options:
                epo_def_df = get_epo_def(sid, cid, segmented_epochs=segment_bool)
                epochs = get_epo_rec(
                    'ContMov',
                    sid,
                    cid,
                    raw_rec,
                    load=False,
                    epo_def_df=epo_def_df,
                    verbose=False
                )
                if epochs is not None:
                    if len(epochs) > 0:
                        key = 'ContMov_seg' if segment_bool else 'ContMov'
                        epos_dict[sid][cid][key] = epochs

    return epos_dict


def count_and_compare_epochs(
        epo_dict: dict,
) -> None:
    for sid, sid_dict in epo_dict.items():
        print(
            f"Subject {sid}"
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
        for sid, sid_dict in epos_dict.items():
            for cid, epo_by_mov_def in sid_dict.items():
                plot_evk_by_cat(epo_by_mov_def, sid=sid, cid=cid, show=False, save=False)
                if save:
                    real_cid = reveal_cid(sid, block_n=cid[-1]) if cid.startswith('block') else reveal_cid(sid, cid=cid)
                    save_figure(f'../outputs/Evk/{sid}/{real_cid}', 'mov_durations.png', bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


