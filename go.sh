#!/bin/bash

# Function to check command availability
check_command() {
    command -v "$1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "go.sh: Error - Required command '$1' is not found. Please install it and try again."
        echo "go.sh: A list of all software dependencies can be found in README.md."
        exit 1
    fi
}

# Check for required commands
check_command "cmake"
check_command "make"
check_command "gnuplot"
if [ ! -s "../MicroTeX-master/build/LaTeX" ]; then
    echo "Error: ../MicroTeX-master/build/LaTeX does not exist. Please ensire that MicroTeX is configured."
    exit 1
fi

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

# Find the project file in any subdirectory under src/Projects
PROJECT_PATH=$(find src/Projects -type f -name "${PROJECT_NAME}.cpp" 2>/dev/null | head -n 1)
TEMPFILE="src/Projects/.active_project.cpp"

# Check if the desired project exists
if [ -z "$PROJECT_PATH" ]; then
    echo "go.sh: Project ${PROJECT_NAME} does not exist."
    exit 1
fi
cp "$PROJECT_PATH" "$TEMPFILE"

(
    mkdir -p build
    cd build

    if [ $? -ne 0 ]; then
        echo "go.sh: Unable to create and enter build directory."
        exit 1
    fi

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
if [ $SUCCESS -ne 0 ]; then
    rm "$TEMPFILE"
    exit $SUCCESS
fi

ultimate_subdir=$(ls -1d out/${PROJECT_NAME}/*/ 2>/dev/null | sort | tail -n 1)

# Check if the compile and run were successful
if [ -n "$ultimate_subdir" ]; then
    cp "$TEMPFILE" "$ultimate_subdir/Project.cpp"
    rm "$TEMPFILE"
    file_path="$ultimate_subdir/${PROJECT_NAME}.mkv"

    if [ -s "$file_path" ] && [ $(stat -c%s "$file_path") -ge 1024 ]; then
        vlc "$file_path" > /dev/null 2>&1
    else
        echo "go.sh: The output file is either empty or smaller than 1KB."
    fi
else
    echo "go.sh: No output directory found for project ${PROJECT_NAME}."
fi
