"""
********************************************************************************
    Title: Data transfer utilities

    Author: Sophie Caroni
    Date of creation: 08.07.2026

    Description:
    This script contains helper functions to copy data trees between storage locations via rsync.
********************************************************************************
"""
import platform
import shutil
import subprocess
from pathlib import Path


_IS_WINDOWS = platform.system() == 'Windows'

# Find rsync; for Windows, fall back _MSYS2_RSYNC location if 'rsync' isn't resolvable on PATH
_MSYS2_RSYNC = r'C:\msys64\usr\bin\rsync.exe'
_RSYNC_EXE = shutil.which('rsync') or (_MSYS2_RSYNC if _IS_WINDOWS else 'rsync')


def _to_rsync_path(path: Path) -> str:
    """
    Render a path in the form rsync expects. For Windows this means translating drive paths (e.g. 'C:\\data') into the
    needed '/c/data' format.
    :param path: Path, path to render
    :return: str, rsync-compatible path string
    """
    if not _IS_WINDOWS:
        return str(path)
    drive = path.drive
    if len(drive) == 2 and drive[1] == ':':
        return f"/{drive[0].lower()}{path.as_posix()[len(drive):]}"
    return path.as_posix()


def build_rsync_command(
        src: Path,
        dst: Path,
        patterns: list[str] | None = None,
        dry_run: bool = False,
) -> list[str]:
    """
    Assemble a rsync command that copies the contents of a source directory (src) into a destination directory (dst).
    :param src: Path, source directory whose contents are transferred
    :param dst: Path, destination directory the contents are written into
    :param patterns: list[str] | None, glob(s) to restrict the transfer to matching files (e.g. '*_preproc*'); None copies all
    :param dry_run: bool, whether to only simulate the transfer (``-n``), as used to build the confirmation preview
    :return: list[str], the rsync command and its arguments, ready to pass to ``subprocess.run``
    """
    command = [_RSYNC_EXE, '-avh']  # '-avh' for archive + verbose + human-readable
    if dry_run:
        command.append('-n')  # dry run flag

    # Keep only files matching patterns
    patterns = patterns or []  # assign empty list if None
    command += ['-m', '--include=*/', *[f'--include={p}' for p in patterns], '--exclude=*']

    # Add src to dst rsync command
    command += [f'{_to_rsync_path(src)}/', _to_rsync_path(dst)]  # trailing slash: copy contents of src, not src itself
    return command


def _run_rsync(command: list[str], verbose: bool = True) -> subprocess.CompletedProcess:
    """
    Run an assembled rsync command.
    :param command: list[str], the rsync command and its arguments
    :param verbose: bool, whether to print the output or not
    :return: subprocess.CompletedProcess, the completed rsync process
    """
    try:
        result = subprocess.run(
            args=command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("rsync failed")
        print("stdout:")
        print(e.stdout)
        print("stderr:")
        print(e.stderr)
        raise

    if verbose:
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    return result


def transfer_data(
        src: Path,
        dst: Path,
        verbose: bool = True,
) -> subprocess.CompletedProcess | None:
    """
    Copy a data tree from one directory to another via rsync, after user confirmation.
    :param src: Path, source directory whose contents are transferred
    :param dst: Path, destination directory the contents are written into
    :param verbose: bool, whether to print the rsync output of the live transfer
    :return: subprocess.CompletedProcess, the completed live transfer, or None if the user declined
    """
    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # When copying derivatives to the local working directory, only select the preprocessed files (_preproc)
    if 'derivatives' in src.parts and 'local' in dst.parts:
        patterns = ['*_preproc*']
    elif 'results' in src.parts and 'local' in dst.parts:
        patterns = ['beh_events.csv', 'eeg_epochs.csv']
    else:
        patterns = None

    # Give preview of what data is going to be transferred
    preview_command = build_rsync_command(src, dst, patterns=patterns, dry_run=True)
    print(f"\nPreview of {src} -> {dst}")
    print(f"Running: {' '.join(preview_command)}\n")
    _run_rsync(preview_command, verbose=True)

    # Require confirmation before performing the actual transfer
    if input("Proceed with the transfer? [y/N]: ").strip().lower() not in ('y', 'yes'):
        print("Transfer aborted.")
        return None

    # Perform actual transfer
    transfer_command = build_rsync_command(src, dst, patterns=patterns)
    print(f"\nTransferring {src} -> {dst}\n")
    return _run_rsync(transfer_command, verbose=verbose)
