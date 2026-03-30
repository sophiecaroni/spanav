#!/usr/bin/env python3

"""
********************************************************************************
    Title: BIDS format converter

    Original author: Andràs Puszta
    Refactored by: Sophie Caroni
    Last modified: 10.02.2026

    Description:
    This script allows to convert EEG files and organize them into a BIDS-
    compliant format. It was originally shared internally by Andràs Puszta and
    later refactored by Sophie Caroni.
********************************************************************************
"""

import os
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from mne_bids.copyfiles import copyfile_brainvision


def get_rs_file_suff(
        fname_parts: list,
) -> str:
    assert len(
        fname_parts) == 3, f'Unknown format of rsEEG file name. Need 3 elements, got {len(fname_parts)}: {fname_parts}'
    eyes_cond = fname_parts[2].upper()
    acq = fname_parts[1]
    return f"Rest{eyes_cond}_{acq}"


def get_impedances_file_suff(
        fname_parts: list,
) -> str:
    assert 0 < len(fname_parts) < 4, f'Unknown format of impedances file name. Need between 1 and 3 elements, got {len(fname_parts)}: {fname_parts}'

    # Conform to decided format: impedances_pre, impedances_mid1/2, impedances_post1/2
    if 'check' in fname_parts:  # special case 'SpaNav_check_imp1'
        nr = fname_parts[-1][-1]
        return f"impedances_mid{nr}"
    elif len(fname_parts) == 3:  # e.g. impedances_post_1
        nr = fname_parts[-1]
        return f"impedances_{fname_parts[1]}{nr}"
    elif len(fname_parts) == 2:  # e.g. impedances_posttask
        return '_'.join(fname_parts).replace('task', '')  # join the two words with _ ; remove 'task' if present
    else:  # it means fname_parts is just one word (the impedances file name)
        return f"{fname_parts[0]}_pre"


def get_block_file_suff(
        fname_parts: list,
) -> str:
    assert len(
        fname_parts) == 1, f'Unknown format of block file name. Need 1 element, got {len(fname_parts)}: {fname_parts}'
    return f"SpaNav_{''.join(fname_parts[0])}"


def auto_rename(
        file_name: str,
        file_path: str | Path,
) -> str:
    ses = 1  # in our experiment we only have one experimental session
    path_parts = file_path.split('/')
    sid = path_parts[-1]
    name_parts = file_name.replace('.vhdr', '').split('_')

    if file_name.lower().startswith('rs'):  # resting state
        new_file_name = f'{sid}_{ses}_{get_rs_file_suff(name_parts)}'

    elif 'im' in file_name.lower():  # for impedances; only use "im" because of existing files "imoedences" with typo
        new_file_name = f'{sid}_{ses}_{get_impedances_file_suff(name_parts)}'

    elif file_name.lower().startswith('block'):  # spanav task
        new_file_name = f'{sid}_{ses}_{get_block_file_suff(name_parts)}'

    else:
        new_file_name = f'{sid}_{ses}_{file_name}'

    return new_file_name


def parse_vhdr(file_path):
    """
    Parses a BrainVision .vhdr file to extract sampling rate and channel names.
    """
    sampling_frequency = 0
    channel_names = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith("SamplingInterval="):
                    try:
                        # Interval is in microseconds
                        interval = float(line.split('=')[1])
                        if interval > 0:
                            sampling_frequency = 1000000.0 / interval
                    except ValueError:
                        pass
                elif line.startswith("Ch") and "=" in line and "," in line:
                    # Format: Ch1=Fp1,,... or Ch1=Fp1,Reference,...
                    try:
                        parts = line.split('=', 1)[1].split(',')
                        name = parts[0]

                        # Don't append duplicates nor names that only are a digit
                        if name not in channel_names and not name.isdigit():
                            channel_names.append(name)
                    except IndexError:
                        pass
    except Exception as e:
        print(f"Warning: Could not fully parse header {file_path.name}: {e}")
        
    return sampling_frequency, channel_names


def parse_vmrk(file_path, sampling_freq):
    """
    Parses a BrainVision .vmrk file to extract events.
    """
    events = []
    if not file_path.exists() or sampling_freq <= 0:
        return events

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        start_processing = False
        for line in lines:
            line = line.strip()
            if line == "[Marker Infos]":
                start_processing = True
                continue
            
            if start_processing and line.startswith("Mk"):
                # Format: Mk<Number>=<Type>,<Description>,<Position in DataPoints>,...
                try:
                    _, content = line.split('=', 1)
                    parts = content.split(',')
                    
                    event_type = parts[0]
                    description = parts[1]
                    position = int(parts[2])
                    
                    # Calculate onset in seconds
                    onset = position / sampling_freq
                    
                    events.append({
                        "onset": round(onset, 5),
                        "duration": 0,  # Duration is usually not specified in simple markers
                        "trial_type": description
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Warning: Could not parse markers {file_path.name}: {e}")
        
    return events


def create_bids_root_files(bids_root, subjects):
    """
    Creates the required top-level BIDS files.
    """
    # 1. dataset_description.json
    desc = {
        "Name": "Automated EEG Dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "Authors": ["Automated Script"],
        "License": "CC0"
    }
    with open(bids_root / "dataset_description.json", 'w') as f:
        json.dump(desc, f, indent=4)

    # 2. README.md
    with open(bids_root / "README.md", 'w') as f:
        f.write("# Automated EEG Dataset\n\n")
        f.write(f"Converted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Structure created automatically based on filename parsing.\n")

    # 3. participants.tsv
    # Extract unique subjects from the list of processed items
    unique_subs = sorted(list(subjects))

    participants_file = bids_root / "participants.tsv"
    with open(participants_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["participant_id", "age", "sex"])
        for sub in unique_subs:
            writer.writerow([f"sub-{sub}", "n/a", "n/a"])

    # 4. participants.json (Sidecar)
    part_json = {
        "participant_id": {"Description": "Unique participant identifier"},
        "age": {"Description": "Age of the participant at time of testing", "Units": "years"},
        "sex": {"Description": "Biological sex of the participant"}
    }
    with open(bids_root / "participants.json", 'w') as f:
        json.dump(part_json, f, indent=4)


def process_directory(source_dir, output_dir):
    """
    Main processing loop.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track participants found
    participants_set = set()
    
    print(f"\nScanning {source_path}...\n")
    
    # Walk through all files
    for root, dirs, files in os.walk(source_path):
        for file in files:

            if file.startswith('WRONG'):
                continue

            # We focus on .vhdr files as the "anchor" for a run.
            # Associated .eeg and .vmrk files are handled when we find the .vhdr.
            if file.endswith(".vhdr"):
                
                # --- 1. Filename Parsing ---
                parts = file.split('_')

                if len(parts) < 4:
                    if AUTO_RENAME:
                        new_fname = auto_rename(file, root)
                        # print(f"{file = }\n{new_fname = }")
                        parts = new_fname.split('_')
                    else:
                        print(f"Skipping because filename does not match 'Subject_Session_Task_Acq' pattern: {file = }.")
                        continue

                # Expected format: SubjectID_SessionID_TaskName_AcqName.vhdr
                sid = parts[0]  # e.g. TIP34 or sub-TIP34
                raw_ses = parts[1]  # e.g. ses01, ses1, 1
                raw_task = parts[2]
                raw_acq = '_'.join(parts[3:])  # take all parts from third one as part of the acq

                if SUBJECTS and sid not in SUBJECTS:
                    print(f'\nSkipping {file = } because of subject not in subjects to process ({sid = })')
                    continue

                # Clean up Session ID: Extract digits strictly
                ses_match = re.search(r'\d+', raw_ses)
                if ses_match:
                    ses_id = ses_match.group()
                else:
                    # Fallback if no digits found (e.g. 'pre', 'post')
                    ses_id = raw_ses.replace("ses-", "").replace("ses", "")
                
                participants_set.add(sid)

                # Determine Task Name and sanitize (BIDS task names should be alphanumeric)
                task_name = re.sub(r'[^a-zA-Z0-9]', '', raw_task)

                # --- 2. Create Destination Directory ---
                # Structure: BIDS_EEG/sub-{id}/ses-{id}/eeg/
                dest_dir = output_path / f"sub-{sid}" / f"ses-{ses_id}" / "eeg"

                # --- 3. Define base BIDS Filename ---
                # Format: sub-{id}_ses-{id}_task-{task}__acq-{_acq}
                bids_basename = f"sub-{sid}_ses-{ses_id}_task-{task_name}_acq-{raw_acq}"

                print(f"\nProcessing {file = } \n\t ==> will generate files in format: {bids_basename}")

                # --- 4. Parse Metadata needed for next files---
                file_path_vhdr = Path(root) / file
                sfreq, channels = parse_vhdr(file_path_vhdr)

                if not TESTING_MODE:
                    # --- 5. Generate files in BIDS format ---

                    # --- 5a. _eeg.eeg, _eeg.vhdr, _eeg.vmrk, files ---
                    bids_eeg_fpath = dest_dir / (bids_basename + '_eeg.vhdr')
                    if not bids_eeg_fpath.exists():
                        # --- Rename _eeg.eeg, _eeg.vhdr, _eeg.vmrk, files ---
                        copyfile_brainvision(file_path_vhdr, bids_eeg_fpath, verbose=True)
                    else:
                        print(f"\n\tSkipping because already present in the directory: {bids_eeg_fpath}")

                    # --- 5b. _eeg.json file ---
                    bids_json_fpath = dest_dir / (bids_basename + "_eeg.json")
                    if not bids_json_fpath.exists():
                        # --- Generate _eeg.json file ---
                        eeg_json = {
                            "TaskName": task_name,
                            "AcqName": raw_acq,
                            "SamplingFrequency": sfreq,
                            "EEGReference": "n/a",
                            "PowerLineFrequency": 50,
                            "SoftwareFilters": "n/a",
                            "CapManufacturer": "n/a",
                            "EEGChannelCount": len(channels),
                            "Manufacturer": "BrainProducts"  # Assumed based on file type
                        }
                        with open(bids_json_fpath, 'w') as f:
                            json.dump(eeg_json, f, indent=4)
                    else:
                        print(f"\n\tSkipping because already present in the directory: {bids_json_fpath}")

                    # --- 5c. _channels.tsv file ---
                    bids_ch_fpath = dest_dir / (bids_basename + "_channels.tsv")
                    if not bids_ch_fpath.exists():
                        # --- Generate _channels.tsv ---
                        # Only write if it doesn't exist (shared by runs of same task usually, but here specific)
                        # Actually, BIDS allows per-run channels.tsv
                        with open(bids_ch_fpath, 'w', newline='') as f:
                            writer = csv.writer(f, delimiter='\t')
                            writer.writerow(["name", "type", "units"])
                            for ch in channels:
                                writer.writerow([ch, "EEG", "microV"])
                    else:
                        print(f"\n\tSkipping because already present in the directory: {bids_ch_fpath}")

                    # --- 5d. _events.tsv file ---
                    original_vmrk = file_path_vhdr.with_suffix(".vmrk")
                    events = parse_vmrk(original_vmrk, sfreq)
                    bids_events_fpath = dest_dir / (bids_basename + "_events.tsv")
                    if not bids_events_fpath.exists():
                        # --- Generate _events.tsv ---

                        with open(bids_events_fpath, 'w', newline='') as f:
                            writer = csv.writer(f, delimiter='\t')
                            writer.writerow(["onset", "duration", "trial_type"])
                            if events:
                                for ev in events:
                                    writer.writerow([ev["onset"], ev["duration"], ev["trial_type"]])
                            else:
                                writer.writerow(["0.0", "0.0", "start_placeholder"])

    # --- 9. Final Root Files ---
    if not TESTING_MODE:
        create_bids_root_files(output_path, participants_set)
        print("\nconversion Complete!")
        print(f"Data stored in: {output_path.absolute()}")


if __name__ == "__main__":

    # Configuration
    groups = ["T", "A"]
    for group in groups:
        src_dir_name = 'TBI' if group == 'T' else 'Healthy_Agematched'
        SOURCE_FOLDER = f"/Volumes/PlasMA_WP73/Raw/{src_dir_name}/TI_and_EEG/Raw"  # <--- EDIT THIS PATH if running directly
        OUTPUT_FOLDER = f"/Volumes/Hummel-Data/TI/SpatialNavigation/WP7.3_EEG/raw/BIDS_Data_WP73{group}"  # <--- EDIT THIS PATH if running directly
        AUTO_RENAME = True
        TESTING_MODE = False
        SUBJECTS = ['73A01', '73T01',  '73T02',  '73T03',  '73T04',  '73T05']  # set to None or empty to process all subjects possible

        # If you want to use command line arguments:
        import argparse

        parser = argparse.ArgumentParser(description="Automated BIDS Converter")
        parser.add_argument("--source", type=str, help="Path to source EEG directory")
        parser.add_argument("--output", type=str, default="BIDS_EEG", help="Output BIDS directory")

        args = parser.parse_args()

        if args.source:
            process_directory(args.source, args.output)
        else:
            # Use the hardcoded path from the top of the file
            print("No command line arguments provided. Using default configuration.")
            # Ask for input if default is placeholder
            if "Path\\To" in SOURCE_FOLDER:
                folder = input("Please enter the full path to your raw EEG folder: ").strip().replace('"', '')
                process_directory(folder, OUTPUT_FOLDER)
            else:
                process_directory(SOURCE_FOLDER, OUTPUT_FOLDER)
