#!/bin/bash
(
    
cd src

if [ ! -d "HIP" ]; then
    echo "hipifyCUDA.sh: No HIP source directory found. Creating directory..."
    mkdir HIP
fi

CLEAN=0
# Handle any arguments
if [ $# -gt 0 ]; then
    if [ $1 == "-c" ]; then
        CLEAN=1
    else
        echo "hipifyCUDA.sh: Unknown argument '$1'."
        echo "Use flag \"-c\" to clear HIP source directory and re-convert all CUDA files."
        echo "By default only newly added CUDA files are converted."
        exit 1
    fi
fi

if [ $CLEAN == 1 ]; then
    if  [ ! -z "$(ls -A HIP)" ]; then
            echo "hipifyCUDA.sh: Clearing HIP source directory..."
            rm -r "HIP/"*
    else
        echo "hipifyCUDA.sh: HIP source directory already empty."
    fi
fi

command -v "hipify-perl" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "hipifyCUDA.sh: Error - command 'hipify-perl' not found."
    echo "The ROCm platform is required to convert to and run HIP files."
else
    echo "hipifyCUDA.sh: Converting CUDA files to HIP."
fi

cp -R "CUDA/"* "HIP/"

cd HIP

shopt -s globstar
for file in ./**/*.{cu,cuh}; do
    echo "Converting file "$file
    hipify-perl $file -inplace
done
shopt -u globstar

)