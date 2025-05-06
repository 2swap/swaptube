#!/usr/bin/env python3
import os
import subprocess
import re

# Function to print the next 5 lines for lookahead
def print_lookahead(entries, start_index, lines=5):
    end_index = min(start_index + lines, len(entries))
    for i in range(start_index, end_index):
        print(entries[i][1])  # Print only the text part

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Record audio files based on record_list.tsv")
    parser.add_argument('project_name', help="Name of the project")
    args = parser.parse_args()

    PROJECT_NAME = args.project_name
    PROJECT_DIR = f"media/{PROJECT_NAME}"

    # Check if the directory exists, and if not, create it
    if not os.path.exists(PROJECT_DIR):
        os.makedirs(PROJECT_DIR)

    # Check if the record_list.tsv exists
    tsv_file_path = os.path.join(PROJECT_DIR, "record_list.tsv")
    if not os.path.exists(tsv_file_path):
        print(f"Error: {tsv_file_path} does not exist.")
        return

    # Read all entries from the tsv file
    entries = []
    with open(tsv_file_path, 'r') as tsv_file:
        for line in tsv_file:
            current_filename, current_text = line.rstrip('\n').split('\t')
            entries.append((current_filename, current_text))

    if not entries:
        print("Error: record_list.tsv is empty.")
        return

    """
    try:
        devices_output = subprocess.check_output(["arecord", "-l"], stderr=subprocess.STDOUT).decode()
        available_devices = []
        for line in devices_output.splitlines():
            if line.strip().startswith("card"):
                available_devices.append(line.strip())
        for idx, device in enumerate(available_devices):
            print(f"{idx}: {device}")
    except Exception as e:
        print("Error listing audio devices:", e)
        return

    selected_index = int(input("Enter the corresponding number: "))
    selected_line = available_devices[selected_index]
    m = re.search(r"card (\d+).*device (\d+):", selected_line)
    if m:
        card_num = m.group(1)
        device_num = m.group(2)
        selected_device = f"hw:{card_num},{device_num}"
    else:
        print("Could not parse device info.")
        return
    """

    input("Do 100 jumping jacks...")
    input("Smile...")
    input("Don't breathe in the mic...")
    input("Don't talk too fast, or too slow...")
    input("Press enter to start recording...")

    # Process each entry
    successes = 0
    for index in range(len(entries)):
        current_filename, current_text = entries[index]
        os.system('clear')
        # Check if the file already exists
        if os.path.exists(f"{PROJECT_DIR}/{current_filename}"):
            print(f"{current_filename} already exists in {PROJECT_DIR}. Skipping...")
            continue

        while True:
            print(f"-> {current_text}")
            # Print the next 5 entries for lookahead
            print_lookahead(entries, index + 1)

            # Start recording with ffmpeg in the background
            print("Recording... Press Enter to stop.")
            ffmpeg_log = os.path.join(PROJECT_DIR, "ffmpeg.log")
            ffmpeg_cmd = [
                'ffmpeg', '-f', 'alsa', '-ar', '44100',#, '-i', selected_device, 
                os.path.join(PROJECT_DIR, current_filename)
            ]

            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait for the user to stop recording
            input()

            # Stop the recording by terminating the ffmpeg process
            ffmpeg_process.terminate()
            ffmpeg_process.wait()

            print("Press enter to continue or 'u' to undo the last recording...")
            user_input = input()

            if user_input.lower() == 'u':
                print(f"Deleting {PROJECT_DIR}/{current_filename}...")
                os.remove(os.path.join(PROJECT_DIR, current_filename))
                # It will loop back to re-record this file
            else:
                successes += 1
                if successes % 10 == 1:
                    input("ARE YOU HYPED")
                if successes == 1:
                    input("Let's check the first audio output file...")
                    ffplay_cmd = [ 'ffplay', os.path.join(PROJECT_DIR, current_filename) ]
                    ffplay_process = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ffmpeg_process.wait()
                    user_check = input("Was it good? [y/n]")
                    if user_check != 'y':
                        print(f"Deleting {PROJECT_DIR}/{current_filename}...")
                        os.remove(os.path.join(PROJECT_DIR, current_filename))
                        continue
                break

if __name__ == "__main__":
    main()
