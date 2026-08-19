check_command_available() {
    command -v "$1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "convert.sh: Error - Required command '$1' is not found. Please install it and try again."
        exit 1
    fi
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "convert.sh: Usage: $0 <ProjectName> [ffmpeg_path]"
    exit 1
fi

PROJECT_NAME=$1
FFMPEG=${2:-ffmpeg}

check_command_available "$FFMPEG"

PROJECT_OUTPUT="out/${PROJECT_NAME}"

if [ ! -d "$PROJECT_OUTPUT" ]; then
    echo "convert.sh: Error - No output directory exists for project '${PROJECT_NAME}'."
    exit 1
fi

LATEST=$(ls -1d "${PROJECT_OUTPUT}"/*/ 2>/dev/null | sort | tail -n 1)
if [ -z "$LATEST" ]; then
    echo "convert.sh: Error - No timestamped output exists for project '${PROJECT_NAME}'."
    exit 1
fi
LATEST=${LATEST%/}

SOURCE_VIDEO=""
if [ -f "${LATEST}/Video.mkv" ]; then
    SOURCE_VIDEO="${LATEST}/Video.mkv"
elif [ -f "${LATEST}/Video.mp4" ]; then
    SOURCE_VIDEO="${LATEST}/Video.mp4"
fi

if [ -z "$SOURCE_VIDEO" ] || [ "$(stat -c%s "$SOURCE_VIDEO")" -lt 1024 ]; then
    echo "convert.sh: Error - The newest output video is missing or smaller than 1 KiB: ${LATEST}"
    exit 1
fi

TARGET_VIDEO="${LATEST}/Video.mp4"

# If the source and target are the same file (already Video.mp4), convert to
# a temp file first so ffmpeg doesn't read and write the same file at once.
if [ -e "$TARGET_VIDEO" ] && [ "$SOURCE_VIDEO" -ef "$TARGET_VIDEO" ]; then
    TEMP_VIDEO="${LATEST}/Video.compat.mp4"
else
    TEMP_VIDEO="$TARGET_VIDEO"
fi

echo "convert.sh: Converting ${SOURCE_VIDEO} to ${TEMP_VIDEO} with ${FFMPEG}"

"$FFMPEG" -y -i "$SOURCE_VIDEO" \
    -map 0:v:0 -map 0:a? \
    -c:v libx264 -profile:v high -level 4.1 -pix_fmt yuv420p -crf 18 -preset medium \
    -c:a aac -b:a 192k -ar 48000 \
    -movflags +faststart \
    "$TEMP_VIDEO"

if [ $? -ne 0 ]; then
    echo "convert.sh: Error - ffmpeg failed."
    exit 1
fi

if [ "$TEMP_VIDEO" != "$TARGET_VIDEO" ]; then
    mv -f "$TEMP_VIDEO" "$TARGET_VIDEO"
fi

echo "convert.sh: Wrote ${TARGET_VIDEO} ($(du -h "$TARGET_VIDEO" | cut -f1))"
echo "convert.sh: Note - Discord only auto-embeds attachments under its per-upload size limit (10 MB on non-boosted servers). Lower -crf's value or add -b:v to cap bitrate if you need a smaller file."

if command -v nautilus > /dev/null 2>&1; then
    nautilus "${LATEST}" >/dev/null 2>&1 &
fi
