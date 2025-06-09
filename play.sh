if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "play.sh: Usage: $0 <ProjectName> [MediaPlayer]"
    exit 1
fi

PROJECT_NAME=$1
MEDIA_PLAYER=${2:-vlc}
ultimate_subdir=$(ls -1d out/${PROJECT_NAME}/*/ 2>/dev/null | sort | tail -n 1)

# Check if the compile and run were successful
if [ -n "$ultimate_subdir" ]; then
    file_path=""
    if [ -f "$ultimate_subdir/${PROJECT_NAME}.mkv" ]; then
        file_path="$ultimate_subdir/${PROJECT_NAME}.mkv"
    elif [ -f "$ultimate_subdir/${PROJECT_NAME}.mp4" ]; then
        file_path="$ultimate_subdir/${PROJECT_NAME}.mp4"
    fi

    if [ -n "$file_path" ] && [ -s "$file_path" ] && [ $(stat -c%s "$file_path") -ge 1024 ]; then
        $MEDIA_PLAYER "$file_path" > /dev/null 2>&1
    else
        echo "play.sh: The output file is either smaller than 1KB, or does not exist."
    fi
else
    echo "play.sh: No output directory found for project ${PROJECT_NAME}."
fi
