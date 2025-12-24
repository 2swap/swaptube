#!/usr/bin/env python3
import os
import subprocess
import re
import shutil
import argparse

def delete_temp_recordings(temp_recordings_dir):
    if os.path.exists(temp_recordings_dir):
        shutil.rmtree(temp_recordings_dir)

def setup_temp_recordings(temp_recordings_dir):
    delete_temp_recordings(temp_recordings_dir)
    os.makedirs(temp_recordings_dir)

# Function to print the next 5 lines for lookahead
def print_lookahead(entries, start_index, lines=5):
    end_index = min(start_index + lines, len(entries))
    for i in range(start_index, end_index):
        print(entries[i][1])  # Print only the text part

def prompt_user_to_archive_recordings(project_dir):
    wav_files = [f for f in os.listdir(project_dir) if f.endswith(".wav")]
    if not wav_files:
        return  # No files to archive, no prompt

    print("Do you want to archive the existing recordings? [y/N]")
    user_input = input().strip().lower()
    if user_input == 'y':
        # Find the lowest number not used by an existing archive directory
        uniq_id = 0
        while True:
            uniq_id += 1
            archive_dir = os.path.join(project_dir, "archive" + str(uniq_id))
            if not os.path.exists(archive_dir):
                break
        os.makedirs(archive_dir)
        for filename in wav_files:
            src_path = os.path.join(project_dir, filename)
            dest_path = os.path.join(archive_dir, filename)
            os.rename(src_path, dest_path)
        print(f"All wav recordings have been archived to {archive_dir}")
    else:
        print("No recordings were archived.")

def main():
    parser = argparse.ArgumentParser(description="Record audio files based on record_list.tsv")
    parser.add_argument('project_name', help="Name of the project")
    args = parser.parse_args()

    PROJECT_NAME = args.project_name
    PROJECT_DIR = f"media/{PROJECT_NAME}"

    temp_recordings_dir = os.path.join(PROJECT_DIR, "temp_recordings")
    setup_temp_recordings(temp_recordings_dir)

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

    prompt_user_to_archive_recordings(PROJECT_DIR)

    try:
        devices_output = subprocess.check_output(["arecord", "-l"], stderr=subprocess.STDOUT).decode()
        available_devices = []
        for line in devices_output.splitlines():
            if line.strip().startswith("card"):
                available_devices.append(line.strip())
    except Exception as e:
        print("Error listing audio devices:", e)
        return

    blue_devices = [device for device in available_devices if ('blue' in device.lower() or 'yeti' in device.lower())]
    if len(blue_devices) == 1:
        selected_line = blue_devices[0]
        print(f"Automatically selected microphone: {selected_line}")
    else:
        for idx, device in enumerate(available_devices):
            print(f"{idx}: {device}")
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

            temp_path = os.path.join(temp_recordings_dir, f"{current_filename}.wav")

            # Start recording with ffmpeg in the background
            print("Recording... Press Enter to stop.")
            ffmpeg_log = os.path.join(PROJECT_DIR, "ffmpeg.log")
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'alsa',
                '-ar', '48000',
                '-ac', '2',
                '-i', selected_device, 
                '-c:a', 'pcm_s32le',
                '-sample_fmt', 's32',
                temp_path
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
                print(f"Deleting {temp_path}...")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                # It will loop back to re-record this file
            else:
                # Move the temp file to the final destination
                final_path = os.path.join(PROJECT_DIR, current_filename)
                shutil.copyfile(temp_path, final_path)
                os.remove(temp_path)

                successes += 1
                if successes % 10 == 1:
                    input("ARE YOU HYPED")
                if successes == 1:
                    input("Let's check the first audio output file...")
                    ffplay_cmd = [ 'ffplay', final_path ]
                    ffplay_process = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ffmpeg_process.wait()
                    cont = False
                    while True:
                        user_check = input("Was it good? [y/n]")
                        if user_check == 'n':
                            print(f"Deleting {final_path}...")
                            os.remove(final_path)
                            cont = True
                            break
                        elif user_check == 'y':
                            print("Great! Let's continue.")
                            break
                        else:
                            print("Please answer with 'y' or 'n'.")
                            continue
                    if cont:
                        continue
                break

    delete_temp_recordings(temp_recordings_dir)

if __name__ == "__main__":
    main()
