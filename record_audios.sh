#!/bin/bash

# Check if project name argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <project_name>"
    exit 1
fi

PROJECT_NAME="$1"
PROJECT_DIR="media/$PROJECT_NAME"

# Check if the directory exists, and if not, create it
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi

# Check if the record_list.tsv exists
if [ ! -f "$PROJECT_DIR/record_list.tsv" ]; then
    echo "Error: $PROJECT_DIR/record_list.tsv does not exist."
    exit 2
fi

# Open the record_list.tsv on file descriptor 3
exec 3< "$PROJECT_DIR/record_list.tsv"

# Function to read the next 5 lines for lookahead
read_lookahead() {
    lookahead=()
    for i in {1..5}; do
        if IFS=$'\t' read -r next_filename next_text <&3; then
            lookahead+=("$next_text")
        else
            break
        fi
    done
}

# Read first line to initialize current variables
IFS=$'\t' read -r current_filename current_text <&3
read_lookahead

echo "Press enter to start recording..."
read

# Read from the file descriptor 3
while [ -n "$current_filename" ]; do
    clear

    while true; do
        # Check if the file already exists
        if [ -f "$PROJECT_DIR/$current_filename" ]; then
            echo "$current_filename already exists in $PROJECT_DIR. Skipping..."
            break
        fi

        echo "-> $current_text"
        # Show the lookahead (the next 5 entries)
        if [ "${#lookahead[@]}" -gt 0 ]; then
            for entry in "${lookahead[@]}"; do
                echo "$entry"
            done
        else
            echo "====="
        fi

        # Start recording in the background
        echo "Recording... Press Enter to stop."
        ffmpeg -f alsa -i default "$PROJECT_DIR/$current_filename" > "$PROJECT_DIR/ffmpeg.log" 2>&1 &
        
        # Capture the process ID
        FFMPEG_PID=$!

        # Verify the process ID
        if ! ps -p $FFMPEG_PID > /dev/null; then
            echo "Error: ffmpeg process $FFMPEG_PID does not exist. Check $PROJECT_DIR/ffmpeg.log for details. Exiting."
            exit 3
        fi

        # Wait for the Enter key
        read

        # Check if the process is still running before attempting to stop it
        if ps -p $FFMPEG_PID > /dev/null; then
            # Send 'q' to ffmpeg to stop recording
            kill -INT $FFMPEG_PID

            # Wait for ffmpeg to finish up and exit
            wait $FFMPEG_PID
        else
            echo "ffmpeg process $FFMPEG_PID has already exited. Check $PROJECT_DIR/ffmpeg.log for details."
        fi

        echo "Press enter to continue or 'u' to undo the last recording..."
        read -n1 input

        case $input in
            u|U) 
                echo
                echo "Deleting $PROJECT_DIR/$current_filename..."
                rm "$PROJECT_DIR/$current_filename"
                # It will loop back to re-record this file
                ;;
            "") 
                echo
                break
                ;;
            *) 
                echo
                echo "Invalid input. Try again."
                ;;
        esac
    done

    # Move to the next row
    current_filename="${lookahead[0]}"
    current_text="${lookahead[0]}"

    # Shift the lookahead array to remove the first item
    lookahead=("${lookahead[@]:1}")
    
    # Read the next lookahead entry to maintain the 5-line buffer
    if IFS=$'\t' read -r next_filename next_text <&3; then
        lookahead+=("$next_text")
    fi
done

# Close the file descriptor 3
exec 3<&-

