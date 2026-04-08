from spanav_tbi.preprocessing.extract_eeg import get_raw_to_epoch, get_all_epo_objects
from spanav_tbi.processing.psd import compute_psd_by_key
from spanav_tbi.visualization.vis_eeg import plot_evk_by_grp, plot_psd_avg_by_grp
from spanav_eeg_utils.io_utils import get_sids, get_sid_blocks


def run_eeg_extraction(test: bool, show: bool, save: bool) -> None:
    sids = get_sids(test=test)
    for sid in sids:
        print(
            f"{sid = }"
        )

        cids = get_sid_blocks(sid, test=test)
        for cid in cids:
            print(
                f"\t{cid = }"
            )

            # Extract epochs
            raw_rec = get_raw_to_epoch(sid, cid)  # load clean continuous (raw) data
            epos_dict = get_all_epo_objects(raw_rec, sid=sid, cid=cid, save=save, load=False, verbose=False, test=test)  # Epoch raw data into different epoch-types

            # Visualize extracted epochs
            _ = plot_evk_by_grp(epos_dict, sid=sid, cid=cid, show=show, save=save)  # Plot evoked response

            # Compute and plot PSD for each epoch-type
            psd_by_rec = compute_psd_by_key(epos_dict)
            _ = plot_psd_avg_by_grp(psd_by_rec, sid, cid, show=show, save=save)


if __name__ == "__main__":
    run_eeg_extraction(
        test=False,
        show=True,
        save=True,
    )
