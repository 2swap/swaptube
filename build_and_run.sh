#!/bin/bash

clear

mkdir -p out

# Check if the number of arguments is less than expected
if [ $# -lt 1 ]; then
    echo "Example Usage: $0 test_mandelbrot"
    exit 1
fi

# Check if the "build" directory exists
if [ ! -d "build" ]; then
  mkdir build
fi

cd build

# if the build directory is empty, run CMake to generate build files
if [ ! -e "CMakeCache.txt" ]; then
  cmake ..
fi

# build the project
make -j12

# Check if the build was successful
if [ $? -eq 0 ]; then
  # Run the program
  ./swaptube $1
else
  echo "Build failed. Please check the build errors."
fi

cd ..

