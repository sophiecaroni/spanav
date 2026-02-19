#!/usr/bin/env python3

##CRITICAL: Modify lines 179-180 if you have multiple underscores in your file (for example if you have Rev_2_ses2_tas.vhdr) to this
# raw_sub = parts[1] 
# raw_ses = parts[2]

"""
Automated BIDS Converter for EEG
--------------------------------
Scans a source directory for EEG files, parses their filenames according to
a specific convention (SubjectID_SessionID_...), and organizes them into
a BIDS-compliant directory structure.


"""

import os
import shutil
import json
import csv
import re
from pathlib import Path
from datetime import datetime

# Configuration
SOURCE_FOLDER = ""  # <--- EDIT THIS PATH if running directly
OUTPUT_FOLDER = ""  # <--- EDIT THIS PATH if running directly

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
                        "duration": 0, # Duration is usually not specified in simple markers
                        "trial_type": description
                    })
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Warning: Could not parse markers {file_path.name}: {e}")
        
    return events

def create_bids_root_files(bids_root, participants):
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
    unique_subs = sorted(list(participants))
    
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
            # We focus on .vhdr files as the "anchor" for a run. 
            # Associated .eeg and .vmrk files are handled when we find the .vhdr.
            if file.endswith(".vhdr"):
                
                # --- 1. Filename Parsing ---
                # Expected format: SubjectID_SessionID_RestOfName.vhdr
                parts = file.split('_')
                
                if len(parts) < 2:
                    print(f"Skipping {file}: Filename does not match 'Subject_Session_...' pattern.")
                    continue
                
                raw_sub = parts[0] # e.g. TIP34 or sub-TIP34
                raw_ses = parts[1] # e.g. ses01, ses1, 1
                
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
                
                # Determine Task Name
                # Everything after session ID is considered part of the task name
                # We strip the extension and sanitize
                if len(parts) > 2:
                    remainder = "_".join(parts[2:]).replace(".vhdr", "")
                    # Sanitize: BIDS task names should be alphanumeric
                    task_name = re.sub(r'[^a-zA-Z0-9]', '', remainder)
                else:
                    task_name = "rest" # Default if no other info provided
                
                if not task_name: task_name = "rest"

                # --- 2. Create Destination Directory ---
                # Structure: BIDS_EEG/sub-{id}/ses-{id}/eeg/
                dest_dir = output_path / f"sub-{sub_id}" / f"ses-{ses_id}" / "eeg"
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # --- 3. Base BIDS Filename ---
                # Format: sub-{id}_ses-{id}_task-{task}_eeg
                bids_basename = f"sub-{sub_id}_ses-{ses_id}_task-{task_name}_eeg"
                
                print(f"Processing: {file} -> {bids_basename}")

                # --- 4. Parse Metadata ---
                file_path_vhdr = Path(root) / file
                sfreq, channels = parse_vhdr(file_path_vhdr)
                
                # --- 5. Copy Files (.vhdr, .vmrk, .eeg) ---
                extensions = [".vhdr", ".vmrk", ".eeg"]
                for ext in extensions:
                    original_file = file_path_vhdr.with_suffix(ext)
                    if original_file.exists():
                        shutil.copy2(original_file, dest_dir / (bids_basename + ext))
                
                # --- 6. Generate _eeg.json Sidecar ---
                eeg_json = {
                    "TaskName": task_name,
                    "SamplingFrequency": sfreq,
                    "EEGReference": "n/a",
                    "PowerLineFrequency": 50,
                    "SoftwareFilters": "n/a",
                    "CapManufacturer": "n/a",
                    "EEGChannelCount": len(channels),
                    "Manufacturer": "BrainProducts" # Assumed based on file type
                }
                
                with open(dest_dir / (bids_basename + ".json"), 'w') as f:
                    json.dump(eeg_json, f, indent=4)
                    
                # --- 7. Generate _channels.tsv ---
                # Only write if it doesn't exist (shared by runs of same task usually, but here specific)
                # Actually, BIDS allows per-run channels.tsv
                with open(dest_dir / (bids_basename.replace("_eeg", "_channels.tsv")), 'w', newline='') as f:
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