"""
    Title: Provisional preprocessing

    Author: Sophie Caroni
    Created: 20.04.2026

    Description:
    This script converts raw BrainVision recordings directly to provisional preprocessed .fif files, without data
    cleaning steps.

"""
from spanav_tbi.preprocessing.provisional_preprocessing import preproc_pipeline


def run_provisional_preprocessing(sids: list[str], acqs: list[str], fast: bool = True) -> None:
    for sid in sids:
        for block in acqs:
            print(f"\n=== {sid} | {block} ===")
            try:
                preproc_pipeline(sid, block, fast=fast)
                print(f"\tDone ✅")
            except FileNotFoundError as e:
                print(f"\t[SKIP] Raw file not found for {sid = }, {block = }")
            except Exception as e:
                print(f"\t[ERROR] {e} on {sid = }, {block = }")


if __name__ == '__main__':
    subjects = [
        '73T01', '73T02', '73A01', '73A02'
    ]
    blocks = [
        'block1', 'block2', 'block3', 'block4',
    ]

    fast = False
    run_provisional_preprocessing(subjects, blocks, fast=fast)

