"""
    Title: Data transfer runner

    Author: Sophie Caroni
    Date of creation: 08.07.2026

    Description:
    This script transfer specific set of files between local and server locations.
"""
from spanav_tbi.preprocessing.transfer_data import download_beh_raws, download_preproc_raws, download_epo_tables


def run_data_transfer() -> None:
    download_beh_raws()
    download_preproc_raws()
    download_epo_tables()


if __name__ == '__main__':
    run_data_transfer()
