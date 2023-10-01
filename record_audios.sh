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

# Read from the file descriptor 3
while IFS=$'\t' read -r filename text <&3; do
    # Check if the file already exists
    if [ -f "$PROJECT_DIR/$filename" ]; then
        echo "$filename already exists in $PROJECT_DIR. Skipping..."
        continue
    fi

    echo "Next: $text"
    echo "Press enter to start recording..."
    read

    # Start recording in the background
    echo "Recording... Press Enter to stop."
    ffmpeg -f alsa -i default "$PROJECT_DIR/$filename" &
    # Get the process ID of ffmpeg
    FFMPEG_PID=$!

    # Wait for the Enter key
    read

    # Send SIGINT to ffmpeg to stop recording
    kill -INT $FFMPEG_PID

    # Wait for ffmpeg to finish up and exit
    wait $FFMPEG_PID

    echo "Finished recording $filename based on $text in $PROJECT_DIR"
    echo "Press enter to proceed to the next one..."
    read
done

# Close the file descriptor 3
exec 3<&-
