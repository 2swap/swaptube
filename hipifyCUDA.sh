#!/bin/bash

# Create HIP source directory if it does not exist
if [ ! -d "src/HIP" ]; then
    echo "hipifyCUDA.sh: No HIP source directory found. Creating directory..."
    mkdir src/HIP
fi

CLEAN=0
VERBOSE=0
FILE=""

# Handle and store any arguments
while getopts "cvf:h" flag; do
    case "$flag" in
        c) 
            CLEAN=1
            ;;
        v) 
            VERBOSE=1
            ;;
        f) 
            FILE="$OPTARG"
            ;;
        h) 
            echo "hipifyCUDA converts CUDA source files to HIP source files using the hipify-perl tool provided by the ROCm platform."
            echo "By default, only changed files will be converted."
            echo ""
            echo "Usage: $0 [-c] [-v] [-f <filename>] [-h]"
            echo "-c: Clear HIP source directory and re-convert all CUDA files."
            echo "-v: Verbose mode: print translation statistics for each file."
            echo "-f <filename>: Only convert the specified CUDA file. Do not include the 'CUDA/' or 'HIP/' prefix."
            echo "-h: Help: print this help message."
            exit 0
            ;; 
        *)
            echo "Use flag \"-h\" for usage information."
            exit 1
            ;;
    esac
done

# Check if hipify-perl is installed and functional
command -v "hipify-perl" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "hipifyCUDA.sh: Error - command 'hipify-perl' not found."
    echo "The ROCm platform is required to convert to and run HIP files."
    exit 1
fi

# Warn user if they try to use both the clean and file flag
if [ "$CLEAN" == 1 ] && [ "$FILE" != "" ]; then
    echo "hipifyCUDA.sh: Using both the clean and file flags will clear the HIP source directory and then convert only the specified file."
    echo "This will result in only one file being present in the HIP source directory after conversion."
    read -p "Are you sure you want to proceed? [y/N]: " choice
    case "$choice" in
        y|Y )
            echo "Proceeding with clean and file flags."
            ;;
        * )
            echo "Aborting."
            exit 1
            ;;
    esac
fi

( # Start conversion process in a subshell
cd src
echo "hipifyCUDA.sh: Starting HIP conversion process..."

# Attempts to empty HIP source directory if clean flag is used
if [ "$CLEAN" == 1 ]; then
    if  [ ! -z "$(ls -A HIP)" ]; then
            echo "hipifyCUDA.sh: Clearing HIP source directory..."
            rm -r "HIP/"*
    else
        echo "hipifyCUDA.sh: HIP source directory already empty."
    fi
fi

# Converts the selected file if the file flag is used
if [ "$FILE" != "" ]; then
    if [ -e "CUDA/${FILE}" ]; then
        if [ -e "HIP/${FILE}" ]; then
            rm "HIP/${FILE}"
        fi
        cp "CUDA/${FILE}" "HIP/${FILE}"
        cd HIP
        echo "Converting file ""$FILE"
        if [ "$VERBOSE" == 1 ]; then
            hipify-perl $FILE -inplace -print-stats
        else
            hipify-perl $FILE -inplace
        fi
        rm "${FILE}.prehip"
        echo "hipifyCUDA.sh: Converted file ${FILE}."
        exit 0
    else
        echo "hipifyCUDA.sh: Error - specified file 'CUDA/${FILE}' does not exist."
        exit 1
    fi
fi

FILES_CONVERTED=0

# Use hipify-perl to convert proper list of CUDA files
cd HIP
shopt -s globstar
if [ "$CLEAN" == 1 ]; then
    # Convert all files if clean flag is used
    cp -R "../CUDA/"* "."
    for file in ./**/*.{cu,cuh}; do
        echo "Converting file ${file}"
        if [ "$VERBOSE" == 1 ]; then
            hipify-perl $file -inplace -print-stats
            echo ""
        else
            hipify-perl $file -inplace
        fi
        rm "${file}.prehip"
        ((FILES_CONVERTED++))
    done
else
    # Convert only changed or new files otherwise
    for cudafile in ../CUDA/**/*.{cu,cuh}; do
        file="${cudafile#../CUDA/}"
        if [ ! -e "$file" ] || [ "../CUDA/${file}" -nt "$file" ]; then
            if [ -e "$file" ]; then
                rm "$file"
            fi
            mkdir -p "$(dirname "$file")"
            cp "../CUDA/${file}" "$file"
            echo "Converting file ${file}"
            if [ "$VERBOSE" == 1 ]; then
                hipify-perl $file -inplace -print-stats
                echo ""
            else
                hipify-perl $file -inplace
            fi
            rm "${file}.prehip"
            ((FILES_CONVERTED++))
        fi
    done
fi
shopt -u globstar

if [ "$FILES_CONVERTED" -eq 0 ]; then
    echo "hipifyCUDA.sh: No files converted. All HIP files are up to date."
else
    echo "hipifyCUDA.sh: Converted ${FILES_CONVERTED} file(s)."
fi
)