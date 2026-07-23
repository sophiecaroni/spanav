"""
    Title: Data transfer runner

    Author: Sophie Caroni
    Date of creation: 08.07.2026

    Description:
    This script transfer specific set of files between local and server locations.
"""
import spanav_tbi.preprocessing.transfer_data as td

def run_data_transfer() -> None:
    td.download_beh_raws()
    td.download_preproc_raws()
    td.download_epo_tables()
    td.download_stim_mapping_tables()


if __name__ == '__main__':
    run_data_transfer()
