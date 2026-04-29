import pickle
from spanav_eeg_utils import io_utils as io


def format_cluster_test_results(obj_name: str, results: dict[str, dict], group: str, sids: list[str]) -> str:
    lines = [f"{obj_name.upper()} | group {group} ({len(sids)} sids)"]
    for effect, res in results.items():
        sig_clusters = [(i, pv) for i, pv in enumerate(res['cluster_pv']) if pv < 0.05]
        lines.append(
            f"\n\t{res['effect_label']}: "
            f"\n\t\t-> {len(sig_clusters)} significant cluster(s)"
            f"\n\t\t-> {res['significant'].sum()} significant points"
        )
        for i, pv in sig_clusters:
            lines.append(f"\t\t\tCluster {i}: p={pv:.3f}")
    text = '\n'.join(lines)
    return text


def save_cluster_test_results(group: str, obj_name: str, results: dict, sids: list[str]) -> None:
    out_dir = io.set_for_save(io.get_outputs_path() / 'stats')

    pkl_path = out_dir / f'group_{group}_{obj_name}_cluster_tests.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n\t ✅ Saved: {pkl_path}")

    txt_path = out_dir / f'group_{group}_{obj_name}_cluster_tests.txt'
    txt_path.write_text(format_cluster_test_results(obj_name.upper(), results, group, sids))
    print(f"\t ✅ Saved: {txt_path}")


def get_tfr_epo_types() -> list[str]:
    return ['blObjPres_wide', 'blMovOn_wide', 'blContMov_wide']


def get_psd_epo_types() -> list[str]:
    return ['blObjPres', 'blMovOn', 'blContMov']
