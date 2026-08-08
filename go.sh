#!/bin/bash

clear
check_command_available() {
    command -v "$1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "go.sh: Error - Required command '$1' is not found. Please install it and try again."
        echo "go.sh: A list of all software dependencies can be found in README.md."
        exit 1
    fi
}

# Check for required commands
check_command_available "cmake"
check_command_available "ninja"
check_command_available "gnuplot"
# Check if MicroTeX build exists
if [ ! -s "../MicroTeX-master/build/LaTeX" ]; then
    echo "Error: ../MicroTeX-master/build/LaTeX does not exist. MicroTeX is required for this project."
    echo "Install instructions are available at https://github.com/NanoMichael/MicroTeX"

    # Ask the user for confirmation
    read -p "Would you like to automatically re-install MicroTeX now? Installation process can be viewed in go.sh. [y/N]: " choice
    case "$choice" in
        y|Y )
            (
                set -e # Exit on error
                echo ">>> Cloning and building MicroTeX..."
                cd .. || exit 1
                rm MicroTeX-master -rf
                git clone --depth 1 https://github.com/NanoMichael/MicroTeX.git MicroTeX-master
                cd MicroTeX-master || exit 1
                mkdir -p build
                cd build || exit 1
                cmake ..
                make -j"$(nproc)"
            )
            ;;
        * )
    esac

    # Verify installation
    if [ ! -s "../MicroTeX-master/build/LaTeX" ]; then
        echo "Installation aborted or failed. Please follow the instructions manually: https://github.com/NanoMichael/MicroTeX"
        echo "HINT: If you are unable to install gtksourceviewmm-3.0 using your distro's package manager, try building it yourself using these instructions:"
        echo "https://github.com/end-4/dots-hyprland/issues/955#issuecomment-2486579754"
        exit 1
    fi

    echo "MicroTeX installation verified."
fi

# Check if the number of arguments is less or more than expected
if [ $# -lt 4 ]; then
    echo "go.sh: Suppose that in the Projects/ directory you have made a project called myproject.cpp."
    echo "go.sh: Usage: $0 <ProjectName> <VideoWidth> <VideoHeight> <Framerate> [optional extra flags]"
    echo "go.sh: Example: $0 myproject 640 360 30 -hx"
    exit 1
fi

PROJECT_NAME=$1
VIDEO_WIDTH=$2
VIDEO_HEIGHT=$3
FRAMERATE=$4
shift; shift; shift; shift;
# Check that the video dimensions are valid integers
if ! [[ "$VIDEO_WIDTH" =~ ^[0-9]+$ ]] || ! [[ "$VIDEO_HEIGHT" =~ ^[0-9]+$ ]] || ! [[ "$FRAMERATE" =~ ^[0-9]+$ ]]; then
    echo "go.sh: Error - Video width, height, and framerate must be valid integers."
    exit 1
fi
SAMPLERATE=48000

SKIP_RENDER=0
SKIP_SMOKETEST=0
AUDIO_HINTS=0
AUDIO_SFX=0
MIDI=0
MIDI_PITCH=0
MIDI_VELOCITY=0
MIDI_TRACKS=0
MIDI_BEND_RANGE=""
INVALID_FLAG=0
COMPUTE_LANG=""
# Parse flags
while getopts "snhxmpvtb:c:" flag; do
    case "$flag" in
        s) 
            SKIP_RENDER=1
            ;;
        n) 
            SKIP_SMOKETEST=1
            ;;
        h) 
            AUDIO_HINTS=1
            ;;
        x) 
            AUDIO_SFX=1
            ;;
        m)
            MIDI=1
            ;;
        p)
            MIDI_PITCH=1
            ;;
        v)
            MIDI_VELOCITY=1
            ;;
        t)
            MIDI_TRACKS=1
            ;;
        b)
            if ! [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
                echo "go.sh: Error - -b takes the pitch bend range in semitones, e.g. -b 24."
                exit 1
            fi
            MIDI_BEND_RANGE="$OPTARG"
            ;;
        c)  
            case "$OPTARG" in
                CUDA)
                    COMPUTE_LANG="CUDA"
                    ;;
                HIP)
                    COMPUTE_LANG="HIP"
                    ;;
                *)
                    echo "Invalid compute language specified: use CUDA or HIP"
                    exit 1
                    ;;
            esac
            ;;
        *)
            INVALID_FLAG=1
            ;;
    esac
done

# If the final flag is illegal, print an error message and exit
if [ $INVALID_FLAG -eq 1 ]; then
    echo "go.sh: Error - Invalid flag:"
    echo "-s means to only run the smoketest."
    echo "-n means to only run the render"
    echo "-h means to include audio hints."
    echo "-x means to include sound effects."
    echo "-m means to export a MIDI track of every sound effect, for a DAW."
    echo "-p means to carry each effect's pitch into the MIDI (implies -m)."
    echo "-v means to carry each effect's volume into MIDI velocity (implies -m)."
    echo "-t means to give each effect its own MIDI track (implies -m)."
    echo "   The midi flags cluster, so -mpvt turns all of them on at once."
    echo "-b sets the pitch bend range in semitones (default 24). Your instrument"
    echo "   must be set to the same number, or glides will play out of tune."
    echo "-c means to specify compute language (takes arguments \"CUDA\" or \"HIP\")"
    exit 1
fi

MIDI_OPTIONS="-"
if [ $MIDI -eq 1 ] || [ $MIDI_PITCH -eq 1 ] || [ $MIDI_VELOCITY -eq 1 ] || [ $MIDI_TRACKS -eq 1 ] || [ -n "$MIDI_BEND_RANGE" ]; then
    MIDI_OPTIONS="m"
    if [ $MIDI_PITCH    -eq 1 ]; then MIDI_OPTIONS="${MIDI_OPTIONS}p"; fi
    if [ $MIDI_VELOCITY -eq 1 ]; then MIDI_OPTIONS="${MIDI_OPTIONS}v"; fi
    if [ $MIDI_TRACKS   -eq 1 ]; then MIDI_OPTIONS="${MIDI_OPTIONS}t"; fi
    MIDI_OPTIONS="${MIDI_OPTIONS}${MIDI_BEND_RANGE}"
fi

# Find the project file in any subdirectory under src/Projects
PROJECT_PATH=$(find src/Projects -type f -name "${PROJECT_NAME}.cpp" 2>/dev/null | head -n 1)
TEMPFILE="src/Projects/.active_project.cpp"

# Check if the desired project exists
if [ -z "$PROJECT_PATH" ]; then
    echo "go.sh: Project ${PROJECT_NAME} does not exist."
    exit 1
fi
cp "$PROJECT_PATH" "$TEMPFILE"

# Generate a timestamp for this build
OUTPUT_FOLDER_NAME=$(date +"%Y-%m-%d_%H.%M.%S")
OUTPUT_DIR="out/${PROJECT_NAME}/${OUTPUT_FOLDER_NAME}"
mkdir -p "$OUTPUT_DIR"

INPUT_DIR="media/${PROJECT_NAME}"
mkdir -p "$INPUT_DIR/latex"

echo "go.sh: Building project ${PROJECT_NAME} with output folder name ${OUTPUT_FOLDER_NAME}"
(
    mkdir -p build
    cd build

    if [ $? -ne 0 ]; then
        echo "go.sh: Unable to create and enter build directory."
        exit 1
    fi

    # Print the command as run
    echo "$0 $*"
    echo ""

    echo "==============================================="
    echo "=================== COMPILE ==================="
    echo "==============================================="
    echo "go.sh: Running \`cmake ..\` from build directory"

    # Pass the variables to CMake as options
    cmake -G Ninja .. -DCOMPUTE_LANG="${COMPUTE_LANG}"

    echo "go.sh: Compiling..."
    # build the project
    ninja -j"$(nproc)"

    # Check if the build was successful
    if [ $? -ne 0 ]; then
        echo "go.sh: Build failed. Please check the build errors."
        exit 1
    fi

    echo "==============================================="
    echo "===================== RUN ====================="
    echo "==============================================="

    # Symlink "io_out" to the output directory for this project
    unlink io_out 2>/dev/null
    ln -s "../${OUTPUT_DIR}" io_out

    # Symlink "io_in" to the media assets directory
    unlink io_in 2>/dev/null
    ln -s "../${INPUT_DIR}" io_in

    # We redirect stderr to null since FFMPEG's encoder libraries tend to dump all sorts of junk there.
    # Swaptube errors are printed to stdout.

    # Smoketest
    if [ $SKIP_SMOKETEST -eq 0 ]; then
        ./swaptube 320 180 $FRAMERATE $SAMPLERATE smoketest $AUDIO_HINTS $AUDIO_SFX $MIDI_OPTIONS 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "go.sh: Execution failed in smoketest."
            exit 2
        fi
    fi

    # True render
    if [ $SKIP_RENDER -eq 0 ]; then
        # Clear all files from the smoketest
        rm io_out/* -rf
        ./swaptube $VIDEO_WIDTH $VIDEO_HEIGHT $FRAMERATE $SAMPLERATE render $AUDIO_HINTS $AUDIO_SFX $MIDI_OPTIONS 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "go.sh: Execution failed in render."
            exit 2
        fi
    fi

    exit 0
)
RESULT=$?

unlink "build/io_in"
unlink "build/io_out"
mv "$TEMPFILE" "$OUTPUT_DIR"

# Play video if compilation succeeded, and not in smoketest-only mode
if [ $RESULT -ne 1 ] && [ $SKIP_RENDER -eq 0 ]; then
    ./play.sh ${PROJECT_NAME}
fi

exit $RESULT
