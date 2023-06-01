#!/bin/bash

clear

# Check if the number of arguments is less than expected
if [ $# -lt 1 ]; then
    echo "Example Usage: $0 Parity.json"
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
  ./swaptube ../config/$1
else
  echo "Build failed. Please check the build errors."
fi

cd ..

# add audio
# ffmpeg -i out/test.mp4 -i out/c4.mp3 -map 0:v -map 1:a -c:v copy out/output.mp4

# watch the result!
# vlc out/output.mp4