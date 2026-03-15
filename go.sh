#!/bin/bash

check_command_available() {
    command -v "$1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "go.sh: Error - Required command '$1' is not found. Please install it and try again."
        echo "go.sh: A list of all software dependencies can be found in README.md."
        exit 1
    fi
}

find_windows_vcvars64() {
    local candidates=(
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
        "/c/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvars64.bat"
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvars64.bat"
    )
    local c
    for c in "${candidates[@]}"; do
        if [ -f "$c" ]; then
            cygpath -w "$c"
            return 0
        fi
    done

    if [ -f "/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" ]; then
        powershell.exe -NoProfile -Command "& '${env:ProgramFiles(x86)}\\Microsoft Visual Studio\\Installer\\vswhere.exe' -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\\Auxiliary\\Build\\vcvars64.bat'" | tr -d '\r' | head -n 1
        return 0
    fi
    return 1
}

find_windows_msys2_root() {
    local candidates=(
        "/c/msys64"
        "$HOME/scoop/apps/msys2/current"
    )
    local c
    for c in "${candidates[@]}"; do
        if [ -f "$c/mingw64/include/glib-2.0/glib.h" ]; then
            cygpath -m "$c"
            return 0
        fi
    done
    return 1
}

find_windows_ffmpeg_root() {
    local c
    for c in /c/ProgramData/chocolatey/lib/ffmpeg-shared/tools/* "$HOME/scoop/apps/ffmpeg-shared/current"; do
        [ -d "$c" ] || continue
        if [ -f "$c/include/libavcodec/avcodec.h" ] && [ -f "$c/lib/avcodec.lib" ]; then
            cygpath -m "$c"
            return 0
        fi
    done
    return 1
}

run_windows_dev_cmd() {
    local vcvars_bat="$1"
    local inner_cmd="$2"
    local runner_bat=".go_windows_dev_cmd.bat"
    {
        echo "@echo off"
        echo "call \"${vcvars_bat}\" >nul"
        echo "if errorlevel 1 exit /b 1"
        echo "${inner_cmd}"
        echo "exit /b %errorlevel%"
    } > "${runner_bat}"

    local runner_win
    runner_win="$(cygpath -w "${runner_bat}")"
    powershell.exe -NoProfile -Command "& '${runner_win}'; exit \$LASTEXITCODE"
    local rc=$?
    rm -f "${runner_bat}"
    return $rc
}

# Check for required commands
check_command_available "cmake"
check_command_available "ninja"
# gnuplot is used only for debug plot generation and is treated as Linux-only.
case "$(uname -s)" in
    Linux*)
        check_command_available "gnuplot"
        ;;
esac
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
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "go.sh: Suppose that in the Projects/ directory you have made a project called myproject.cpp."
    echo "go.sh: Usage: $0 <ProjectName> <VideoWidth> <VideoHeight> <Framerate> [-s|-h|-x|-hx]"
    echo "go.sh: Example: $0 myproject 640 360 30 -hx"
    exit 1
fi

PROJECT_NAME=$1
VIDEO_WIDTH=$2
VIDEO_HEIGHT=$3
FRAMERATE=$4
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
INVALID_FLAG=0
# If the 5th argument is provided, check if it is valid
if [ $# -eq 5 ]; then
    if [ "$5" == "-s" ]; then
        SKIP_RENDER=1
    elif [ "$5" == "-n" ]; then
        SKIP_SMOKETEST=1
    elif [ "$5" == "-h" ]; then
        AUDIO_HINTS=1
    elif [ "$5" == "-x" ]; then
        AUDIO_SFX=1
    elif [ "$5" == "-hx" ] || [ "$5" == "-xh" ]; then
        AUDIO_HINTS=1
        AUDIO_SFX=1
    else
        INVALID_FLAG=1
    fi
fi
# If the final flag is illegal, print an error message and exit
if [ $INVALID_FLAG -eq 1 ]; then
    echo "go.sh: Error - The 4th argument has 4 options:"
    echo "-s means to only run the smoketest."
    echo "-h means to include audio hints."
    echo "-x means to include sound effects."
    echo "-hx or -xh means to include both audio hints and sound effects."
    exit 1
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
mkdir -p "$OUTPUT_DIR/frames"

INPUT_DIR="media/${PROJECT_NAME}"
mkdir -p "$INPUT_DIR/latex"

is_windows_msys=0
case "${OSTYPE:-}" in
    msys*|cygwin*|win32*)
        is_windows_msys=1
        ;;
esac

SWAPTUBE_BIN="./swaptube"
if [ $is_windows_msys -eq 1 ]; then
    SWAPTUBE_BIN="./swaptube.exe"
fi

echo "go.sh: Building project ${PROJECT_NAME} with output folder name ${OUTPUT_FOLDER_NAME}"
(
    mkdir -p build
    cd build

    if [ $? -ne 0 ]; then
        echo "go.sh: Unable to create and enter build directory."
        exit 1
    fi

    clear

    # Print the command as run
    echo "$0 $*"
    echo ""

    echo "==============================================="
    echo "=================== COMPILE ==================="
    echo "==============================================="
    echo "go.sh: Running \`cmake ..\` from build directory"

    BUILD_JOBS="$(nproc 2>/dev/null || echo 8)"
    if [ $is_windows_msys -eq 1 ]; then
        VCVARS64_BAT="$(find_windows_vcvars64 | tr -d '\r')"
        if [ -z "$VCVARS64_BAT" ]; then
            echo "go.sh: Unable to locate vcvars64.bat. Install Visual Studio Build Tools with C++ workload."
            exit 1
        fi
        WINDOWS_RUNTIME_DIRS=""
        WINDOWS_CMAKE_ARGS="-DCMAKE_CXX_COMPILER=cl.exe"
        MSYS2_ROOT_HINT="$(find_windows_msys2_root || true)"
        if [ -n "$MSYS2_ROOT_HINT" ]; then
            WINDOWS_CMAKE_ARGS="${WINDOWS_CMAKE_ARGS} -DMSYS2_ROOT=\"${MSYS2_ROOT_HINT}\""
            WINDOWS_RUNTIME_DIRS="${WINDOWS_RUNTIME_DIRS}$(cygpath -w "${MSYS2_ROOT_HINT}/mingw64/bin" | tr -d '\r');"
            echo "go.sh: Using MSYS2_ROOT=${MSYS2_ROOT_HINT}"
        fi
        FFMPEG_ROOT_HINT="$(find_windows_ffmpeg_root || true)"
        if [ -n "$FFMPEG_ROOT_HINT" ]; then
            WINDOWS_CMAKE_ARGS="${WINDOWS_CMAKE_ARGS} -DFFMPEG_ROOT=\"${FFMPEG_ROOT_HINT}\""
            WINDOWS_RUNTIME_DIRS="${WINDOWS_RUNTIME_DIRS}$(cygpath -w "${FFMPEG_ROOT_HINT}/bin" | tr -d '\r');"
            echo "go.sh: Using FFMPEG_ROOT=${FFMPEG_ROOT_HINT}"
        fi
        echo "go.sh: Bootstrapping MSVC toolchain via $VCVARS64_BAT"
        run_windows_dev_cmd "$VCVARS64_BAT" "cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DPROJECT_NAME_MACRO=${PROJECT_NAME} -DAUDIO_HINTS=${AUDIO_HINTS} -DAUDIO_SFX=${AUDIO_SFX} ${WINDOWS_CMAKE_ARGS}"
    else
        # Pass the variables to CMake as options
        cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DPROJECT_NAME_MACRO="${PROJECT_NAME}" -DAUDIO_HINTS="${AUDIO_HINTS}" -DAUDIO_SFX="${AUDIO_SFX}"
    fi

    echo "go.sh: Compiling..."
    if [ $is_windows_msys -eq 1 ]; then
        run_windows_dev_cmd "$VCVARS64_BAT" "ninja -j${BUILD_JOBS}"
    else
        # build the project
        ninja -j"${BUILD_JOBS}"
    fi

    # Check if the build was successful
    if [ $? -ne 0 ]; then
        echo "go.sh: Build failed. Please check the build errors."
        exit 1
    fi

    echo "==============================================="
    echo "===================== RUN ====================="
    echo "==============================================="

    # Wire io paths for renderer input/output.
    if [ $is_windows_msys -eq 1 ]; then
        # Avoid Windows symlink/junction edge cases under Git Bash.
        rm -rf io_out io_in
        mkdir -p io_out/frames io_in
        cp -rf "../${INPUT_DIR}/." io_in/
    else
        rm -rf io_out
        ln -s "../${OUTPUT_DIR}" io_out
        rm -rf io_in
        ln -s "../${INPUT_DIR}" io_in
    fi

    # We redirect stderr to null since FFMPEG's encoder libraries tend to dump all sorts of junk there.
    # Swaptube errors are printed to stdout.

    # Smoketest
    if [ $SKIP_SMOKETEST -eq 0 ]; then
        if [ $is_windows_msys -eq 1 ]; then
            WINDOWS_RUNTIME_PREFIX=""
            if [ -n "$WINDOWS_RUNTIME_DIRS" ]; then
                WINDOWS_RUNTIME_PREFIX="set \"PATH=${WINDOWS_RUNTIME_DIRS}%PATH%\" && "
            fi
            run_windows_dev_cmd "$VCVARS64_BAT" "${WINDOWS_RUNTIME_PREFIX}swaptube.exe 160 90 ${FRAMERATE} ${SAMPLERATE} smoketest"
        else
            "$SWAPTUBE_BIN" 160 90 $FRAMERATE $SAMPLERATE smoketest
        fi
        SWAPTUBE_STATUS=$?
        echo "go.sh: Smoketest exit code ${SWAPTUBE_STATUS}"
        if [ $SWAPTUBE_STATUS -ne 0 ]; then
            echo "go.sh: Execution failed in smoketest."
            exit 2
        fi
    fi

    # True render
    if [ $SKIP_RENDER -eq 0 ]; then
        # Clear all files from the smoketest
        rm io_out/* -rf
        mkdir -p io_out/frames
        if [ $is_windows_msys -eq 1 ]; then
            WINDOWS_RUNTIME_PREFIX=""
            if [ -n "$WINDOWS_RUNTIME_DIRS" ]; then
                WINDOWS_RUNTIME_PREFIX="set \"PATH=${WINDOWS_RUNTIME_DIRS}%PATH%\" && "
            fi
            run_windows_dev_cmd "$VCVARS64_BAT" "${WINDOWS_RUNTIME_PREFIX}swaptube.exe ${VIDEO_WIDTH} ${VIDEO_HEIGHT} ${FRAMERATE} ${SAMPLERATE} render"
        else
            "$SWAPTUBE_BIN" $VIDEO_WIDTH $VIDEO_HEIGHT $FRAMERATE $SAMPLERATE render
        fi
        SWAPTUBE_STATUS=$?
        echo "go.sh: Render exit code ${SWAPTUBE_STATUS}"
        if [ $SWAPTUBE_STATUS -ne 0 ]; then
            echo "go.sh: Execution failed in render."
            exit 2
        fi
        if [ $is_windows_msys -eq 1 ]; then
            cp -rf io_out/. "../${OUTPUT_DIR}/"
        fi
    fi

    exit 0
)
RESULT=$?

if [ $is_windows_msys -eq 1 ]; then
    MSYS2_ARG_CONV_EXCL='*' cmd.exe //C "if exist build\\io_in rmdir /S /Q build\\io_in" >/dev/null 2>&1
    MSYS2_ARG_CONV_EXCL='*' cmd.exe //C "if exist build\\io_out rmdir /S /Q build\\io_out" >/dev/null 2>&1
else
    rm -rf "build/io_in"
    rm -rf "build/io_out"
fi
mv "$TEMPFILE" "$OUTPUT_DIR"

# Play video if compilation succeeded, and not in smoketest-only mode
if [ $RESULT -ne 1 ] && [ $SKIP_RENDER -eq 0 ]; then
    ./play.sh ${PROJECT_NAME}
fi

exit $RESULT
