#!/bin/bash


mkdir -p out
mkdir -p build

# Check if the number of arguments is less than expected
if [ $# -lt 1 ]; then
    echo "Example Usage: $0 test_mandelbrot"
    exit 1
fi

# Assign the path to a variable
PROJECT_PATH="src/projects/$1.cpp"

# Check if the desired project exists
if [ ! -e $PROJECT_PATH ]; then
    echo "Project $1 does not exist."
    exit 1
fi
cp $PROJECT_PATH src/projects/.active_project.cpp

cd build

clear

# if the build directory is empty, run CMake to generate build files
if [ ! -e "CMakeCache.txt" ]; then
  cmake ..
fi

# build the project
make -j12

# Check if the build was successful
if [ $? -ne 0 ]; then
  echo "Build failed. Please check the build errors."
  cd ..
  exit 1
fi

# Run the program
#valgrind ./swaptube $1
./swaptube $1

# Check if the execution was successful
if [ $? -ne 0 ]; then
  echo "Execution failed in runtime."
  cd ..
  exit 1
fi

cd ..

rm src/projects/.active_project.src

vlc out/$1.mp4

