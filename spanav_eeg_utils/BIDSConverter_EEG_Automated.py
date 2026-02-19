#!/usr/bin/env python3

"""
Automated BIDS Converter for EEG
--------------------------------
Scans a source directory for EEG files, parses their filenames according to
a specific convention (SubjectID_SessionID_...), and organizes them into
a BIDS-compliant directory structure.


"""

import os
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from mne_bids.copyfiles import copyfile_brainvision

# Configuration
group = "T"
SOURCE_FOLDER = f"/Volumes/Hummel-Data/TI/SpatialNavigation/WP7.3_EEG/Data_WP73{group}/TI_and_EEG/Raw"  # <--- EDIT THIS PATH if running directly
OUTPUT_FOLDER = f"/Volumes/Hummel-Data/TI/SpatialNavigation/WP7.3_EEG/raw/BIDS_Data_WP73{group}"  # <--- EDIT THIS PATH if running directly
AUTO_RENAME = True
TESTING_MODE = True


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
    assert len(fname_parts) == 1 or len(
        fname_parts) == 2, f'Unknown format of impedances file name. Need 1 or 2 elements, got {len(fname_parts)}: {fname_parts}'
    if len(fname_parts) == 2:
        return '_'.join(fname_parts).replace('task',
                                             '')  # join the two words with _ ; remove 'task' where it was added (e.g. posttask)
    else:  # it means fname_parts is just one word (the impedances file name)
        return f"{fname_parts[0]}_pre"  # join all words with - and append 'pre'


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
    elif file_name.lower().startswith('imp'):  # impedances
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
    
    print(f"Scanning {source_path}...")
    
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
                        print(f"Skipping {file}: Filename does not match 'Subject_Session_Task_Acq' pattern.")
                        continue

                # Expected format: SubjectID_SessionID_TaskName_AcqName.vhdr
                raw_sub = parts[0]  # e.g. TIP34 or sub-TIP34
                raw_ses = parts[1]  # e.g. ses01, ses1, 1
                raw_task = parts[2]
                raw_acq = '_'.join(parts[3:])  # take all parts from third one as part of the acq

                # if file in ['block3.vhdr', 'block1.vhdr'] and raw_sub.endswith('02'):
                #     print(f'\nSkipping {raw_sub} file {file}\n')
                #     continue
                if 'RS' not in file:
                    print(f'\nSkipping {file}\n')
                    continue

                # Clean up Subject ID
                sub_id = raw_sub.replace("sub-", "")
                
                # Clean up Session ID: Extract digits strictly
                ses_match = re.search(r'\d+', raw_ses)
                if ses_match:
                    ses_id = ses_match.group()
                else:
                    # Fallback if no digits found (e.g. 'pre', 'post')
                    ses_id = raw_ses.replace("ses-", "").replace("ses", "")
                
                participants_set.add(sub_id)

                # Determine Task Name and sanitize (BIDS task names should be alphanumeric)
                task_name = re.sub(r'[^a-zA-Z0-9]', '', raw_task)

                # --- 2. Create Destination Directory ---
                # Structure: BIDS_EEG/sub-{id}/ses-{id}/eeg/
                dest_dir = output_path / f"sub-{sub_id}" / f"ses-{ses_id}" / "eeg"
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # --- 3. Base BIDS Filename ---
                # Format: sub-{id}_ses-{id}_task-{task}__acq-{_acq}
                bids_basename = f"sub-{sub_id}_ses-{ses_id}_task-{task_name}_acq-{raw_acq}"

                print(f"Processing: {file} -> {bids_basename}")

                # --- 4. Parse Metadata ---
                file_path_vhdr = Path(root) / file
                sfreq, channels = parse_vhdr(file_path_vhdr)

                # Automatically rename and copy .vhdr, .vmrk, .eeg files
                dest_file_path = dest_dir / (bids_basename + '_eeg.vhdr')

                if not TESTING_MODE:
                    copyfile_brainvision(file_path_vhdr, dest_file_path, verbose=True)

                    # --- 6. Generate _eeg.json Sidecar ---
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

                    with open(dest_dir / (bids_basename + "_eeg.json"), 'w') as f:
                        json.dump(eeg_json, f, indent=4)

                    # --- 7. Generate _channels.tsv ---
                    # Only write if it doesn't exist (shared by runs of same task usually, but here specific)
                    # Actually, BIDS allows per-run channels.tsv
                    with open(dest_dir / (bids_basename + "_channels.tsv"), 'w', newline='') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow(["name", "type", "units"])
                        for ch in channels:
                            writer.writerow([ch, "EEG", "microV"])

                    # --- 8. Generate _events.tsv ---
                    original_vmrk = file_path_vhdr.with_suffix(".vmrk")
                    events = parse_vmrk(original_vmrk, sfreq)
                    events_tsv = dest_dir / (bids_basename + "_events.tsv")
                    with open(events_tsv, 'w', newline='') as f:
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