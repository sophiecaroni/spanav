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
import platform
from pathlib import Path


def load_config(
        config_path: str | None = None
) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()

    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.ini"
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


def get_server_root(config_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    system = platform.system()

    if system == "Windows":
        root = cfg.get("Paths", "server_root_windows", fallback=r"\\sv-nas1.rcp.epfl.ch\Hummel-Data")
    elif system == "Darwin":
        root = cfg.get("Paths", "server_root_mac", fallback="/Volumes/Hummel-Data")
    else:
        # Linux/HPC: either set in config.ini or reuse windows UNC if mounted via smb
        root = cfg.get("Paths", "server_root_linux", fallback=r"\\sv-nas1.rcp.epfl.ch\Hummel-Data")

    return Path(root)


def get_local_root(config_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    root = cfg.get("Paths", "local_root", fallback=str(Path.home() / "local"))
    return Path(root)
