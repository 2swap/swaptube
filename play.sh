if [ $# -ne 1 ]; then
    echo "play.sh: Usage: $0 <ProjectName>"
    exit 1
fi

PROJECT_NAME=$1
ultimate_subdir=$(ls -1d out/${PROJECT_NAME}/*/ 2>/dev/null | sort | tail -n 1)

# Check if the compile and run were successful
if [ -n "$ultimate_subdir" ]; then
    file_path="$ultimate_subdir/${PROJECT_NAME}.mkv"

    if [ -s "$file_path" ] && [ $(stat -c%s "$file_path") -ge 1024 ]; then
        vlc "$file_path" > /dev/null 2>&1
    else
        echo "play.sh: The output file is either empty or smaller than 1KB."
    fi
else
    echo "play.sh: No output directory found for project ${PROJECT_NAME}."
fi
