#!/bin/bash

check_media_player_exists() {
    # Try each player in order and pick the first available one
    if command -v "$MEDIA_PLAYER" >/dev/null 2>&1; then
        return 0
    fi

    for candidate in vlc mpv ffplay; do
        if command -v "$candidate" >/dev/null 2>&1; then
            MEDIA_PLAYER="$candidate"
            echo "play.sh: Using available media player: $MEDIA_PLAYER"
            return 0
        fi
    done

    echo "play.sh: Error - No supported media player found (tried: vlc, mpv, ffplay)."
    echo "Please install one of them or specify manually."
    exit 1
}

# Argument parsing
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "play.sh: Usage: $0 <ProjectName> [MediaPlayer]"
    exit 1
fi

PROJECT_NAME=$1
MEDIA_PLAYER=${2:-vlc}

check_media_player_exists
ultimate_subdir=$(ls -1d out/${PROJECT_NAME}/*/ 2>/dev/null | sort | tail -n 1)

if [ -n "$ultimate_subdir" ]; then
    file_path=""
    if [ -f "$ultimate_subdir${PROJECT_NAME}.mkv" ]; then
        file_path="$ultimate_subdir${PROJECT_NAME}.mkv"
    elif [ -f "$ultimate_subdir${PROJECT_NAME}.mp4" ]; then
        file_path="$ultimate_subdir${PROJECT_NAME}.mp4"
    fi

    if [ -n "$file_path" ] && [ -s "$file_path" ] && [ "$(stat -c%s "$file_path")" -ge 1024 ]; then
        echo "play.sh: Playing $file_path with $MEDIA_PLAYER ..."
        "$MEDIA_PLAYER" "$file_path" > /dev/null 2>&1
    else
        echo "play.sh: The output file is either smaller than 1KB, or does not exist."
    fi
else
    echo "play.sh: No output directory found for project ${PROJECT_NAME}."
fi

