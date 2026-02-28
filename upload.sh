# !/bin/bash
#
# Shell script which uploads completed swaptube output folders recursively
# to a gcloud bucket using `gcloud storage cp` command.
# The structure of the bucket is as follows:
# gs://swaptube-out/output_uploads/video_name/run_number/...
#
# The user will provide the video name.
# The completed output folder can then be found in `out/video_name/run_number/`.
# This script will identify the latest local run for the video,
# prompt the user to confirm the upload, and then proceed to upload.
#
# The script locates the bucket ID from the file `gcskey`
# in the user's home directory.

# Argument parsing
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <project_name>"
    echo "Example: $0 myproject"
    exit 1
fi

PROJECT_NAME=$1

OUTPUT_FOLDER=$(ls -1d out/${PROJECT_NAME}/*/ 2>/dev/null | sort | tail -n 1)

if [ -z "$OUTPUT_FOLDER" ]; then
    echo "No runs found for project '$PROJECT_NAME' in the 'out' directory."
    exit 1
fi

echo
echo "Latest run for project '$PROJECT_NAME': $OUTPUT_FOLDER"
echo "Contents of the run folder:"
ls -lah "$OUTPUT_FOLDER"
echo

read -p "Do you want to upload this run to the gcloud bucket? (y/n) "

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

echo "Proceeding with upload..."

# Upload the run folder to the gcloud bucket
DESTINATION="gs://swaptube-out/output_uploads/${PROJECT_NAME}/$(basename $OUTPUT_FOLDER)"
echo "Uploading to $DESTINATION..."
gcloud storage cp --recursive "$OUTPUT_FOLDER" "$DESTINATION"
