from spanav_tbi.preprocessing.behavior_to_eeg import get_times_retrieval_phases, get_trace_df, get_retrieval_df, extract_beh_events, define_eeg_epochs
from spanav_tbi.visualization.vis_eeg import plot_schematic_epo_def
from spanav_eeg_utils.io_utils import get_sids


def run_beh_extraction(test: bool, show: bool, save: bool) -> None:
    sids = get_sids(test=test)
    for sid in sids:
        print(
            f"{sid = }"
        )

        block_times = get_times_retrieval_phases(sid)
        df_trace = get_trace_df(sid)
        retrieval_df = get_retrieval_df(sid)
        beh_events_df = extract_beh_events(sid, block_times, retrieval_df, df_trace, save=save)
        eeg_epochs_df = define_eeg_epochs(beh_events_df, sid, save=save)
        plot_schematic_epo_def(beh_events_df, eeg_epochs_df, show=show, save=save, sid=sid)


if __name__ == "__main__":
    run_beh_extraction(
        test=True,
        show=True,
        save=False,
    )
