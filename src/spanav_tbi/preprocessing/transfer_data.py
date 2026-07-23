"""
    Title: Transfer data

    Author: Sophie Caroni
    Date of creation: 23.07.2026

    Description:
    This script contains functions for file transfer tasks (download/uplaod across local and server locations) that are
    recurrent throughout the SpaNav project.
"""
import spanav_eeg_utils.config_utils as cfg
from spanav_eeg_utils.transfer_utils import transfer_data


def download_beh_raws():
    src = cfg.get_server_root() / 'raw'
    dst = cfg.get_local_root() / 'raw'
    patterns = ['*_ses-1*_beh.txt']  # there are also behavioral log files of ses-0 (screening session)
    transfer_data(src=src, dst=dst, patterns=patterns)


def download_preproc_raws():
    src = cfg.get_server_root() / 'derivatives'
    dst = cfg.get_local_root() / 'derivatives'
    patterns = ['*_preproc*']  # only select preprocessed files from derivatives source folder
    transfer_data(src=src, dst=dst, patterns=patterns)


def download_epo_tables():
    src = cfg.get_server_root() / 'results'
    dst = cfg.get_local_root() / 'results'
    patterns = ['beh_events.csv', 'eeg_epochs.csv']  # only select epochs/events table files from results source folder
    transfer_data(src=src, dst=dst, patterns=patterns)


def download_stim_mapping_tables():
    for group in ['A', 'T']:
        src = cfg.get_server_root() / f'Data_WP73{group}' / 'TI_and_EEG'
        dst = cfg.get_local_root() / f'Data_WP73{group}' / 'TI_and_EEG'
        patterns = [f'*RandomizationTable*']  # only select randomization table files
        transfer_data(src=src, dst=dst, patterns=patterns)
