#!/bin/bash

# Check if the number of arguments is less than expected
if [ $# -ne 3 ]; then
    echo "go.sh: Suppose that in the Projects/ directory you have made a project called myproject.cpp."
    echo "go.sh: Usage: $0 <ProjectName> <VideoWidth> <VideoHeight>"
    echo "go.sh: Example: $0 myproject 640 360"
    exit 1
fi

PROJECT_NAME=$1
VIDEO_WIDTH=$2
VIDEO_HEIGHT=$3

# Assign the path to a variable
PROJECT_PATH="src/Projects/${PROJECT_NAME}.cpp"
TEMPFILE="src/Projects/.active_project.cpp"

# Check if the desired project exists
if [ ! -e $PROJECT_PATH ]; then
    echo "go.sh: Project ${PROJECT_NAME} does not exist."
    exit 1
fi
cp $PROJECT_PATH $TEMPFILE

(
    mkdir -p build
    cd build || exit

    clear

    echo "==============================================="
    echo "=================== COMPILE ==================="
    echo "==============================================="
    echo "go.sh: Running \`cmake ..\` from build directory"

    # Pass the variables to CMake as options
    cmake .. -DPROJECT_NAME_MACRO="${PROJECT_NAME}" -DVIDEO_WIDTH="${VIDEO_WIDTH}" -DVIDEO_HEIGHT="${VIDEO_HEIGHT}"

    echo "go.sh: Running \`make -j12\`"
    # build the project
    make -j12

    # Check if the build was successful
    if [ $? -ne 0 ]; then
      echo "go.sh: Build failed. Please check the build errors."
      exit 1
    fi

    echo "==============================================="
    echo "===================== RUN ====================="
    echo "==============================================="
    # Run the program
    ./swaptube

    if [ $? -ne 0 ]; then
      echo "go.sh: Execution failed in runtime."
      exit 1
    fi
)
SUCCESS=$?
ultimate_subdir=$(ls -1d out/${PROJECT_NAME}/*/ 2>/dev/null | sort | tail -n 1)
cp $TEMPFILE "$ultimate_subdir/Project.cpp"
rm $TEMPFILE

# Check if the compile and run were successful
if [ $SUCCESS -eq 0 ]; then
    if [ -n "$ultimate_subdir" ]; then
        file_path="$ultimate_subdir/${PROJECT_NAME}.mp4"

        if [ -s "$file_path" ] && [ $(stat -c%s "$file_path") -ge 1024 ]; then
            vlc "$file_path" > /dev/null 2>&1
        else
            echo "go.sh: The output file is either empty or smaller than 1KB."
        fi
    else
        echo "go.sh: No output directory found for project ${PROJECT_NAME}."
    fi
fi

