"""
********************************************************************************
    Title: Configuration utilities

    Author: Sophie Caroni
    Date of creation: 18.02.2026

    Description:
    This script contains helper functions to import running configuration prameters.
********************************************************************************
"""
import configparser
from pathlib import Path


def load_config(
        config_path: str | None = None
) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()

    if config_path is None:
        # default: config.ini next to the repo root (adjust if needed)
        config_path = Path(__file__).resolve().parents[1] / "config.ini"
    else:
        config_path = Path(config_path)

    cfg.read(config_path)
    return cfg


def get_server(config_path: str | None = None) -> bool:
    cfg = load_config(config_path)
    return cfg.getboolean("General", "server", fallback=True)


def get_blinding(config_path: str | None = None) -> bool:
    cfg = load_config(config_path)
    return cfg.getboolean("General", "blinding", fallback=True)


def get_seed(config_path: str | None = None) -> int:
    cfg = load_config(config_path)
    return cfg.getint("General", "seed", fallback=81025)
