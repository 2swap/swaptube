#!/bin/bash

# Check if the number of arguments is less than expected
if [ $# -lt 1 ]; then
    echo "go.sh: Suppose that in the projects/ directory you have made a project called myproject.cpp."
    echo "go.sh: Example Usage: $0 myproject"
    exit 1
fi

# Assign the path to a variable
PROJECT_PATH="src/projects/$1.cpp"
TEMPFILE="src/projects/.active_project.cpp"

# Check if the desired project exists
if [ ! -e $PROJECT_PATH ]; then
    echo "go.sh: Project $1 does not exist."
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
    # if the build directory is empty, run CMake to generate build files
    if [ ! -e "CMakeCache.txt" ]; then
      cmake ..
    fi

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
    ./swaptube $1

    # Check if the execution was successful
    if [ $? -ne 0 ]; then
      echo "go.sh: Execution failed in runtime."
      exit 1
    fi
)
SUCCESS=$?
ultimate_subdir=$(ls -1d out/$1/*/ 2>/dev/null | sort | tail -n 1)
cp $TEMPFILE "$ultimate_subdir/Project.cpp"
rm $TEMPFILE

# Check if the compile and run were successful
if [ $SUCCESS -eq 0 ]; then
    if [ -n "$ultimate_subdir" ]; then
        vlc "$ultimate_subdir/$1.mp4" vlc://quit > /dev/null 2>&1
    else
        echo "go.sh: No output directory found for project $1."
    fi
fi
