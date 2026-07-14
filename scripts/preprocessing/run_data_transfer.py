"""
    Title: Data transfer runner

    Author: Sophie Caroni
    Date of creation: 08.07.2026

    Description:
    This script copies a data tree between two locations, facilitating the recurrent need for transfer between local and
    server locations.
"""
from pathlib import Path
from spanav_eeg_utils.transfer_utils import transfer_data
import spanav_eeg_utils.config_utils as cfg


def run_data_transfer(src: Path, dst: Path) -> None:
    transfer_data(src=src, dst=dst)


if __name__ == '__main__':
    run_data_transfer(
        src=cfg.get_server_root() / 'derivatives',
        dst=cfg.get_local_root() / 'derivatives',
    )
