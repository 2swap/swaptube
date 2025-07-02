cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
project(swaptube LANGUAGES CXX)

# Pass user-specified values as preprocessor macros
add_definitions(-DPROJECT_NAME_MACRO="${PROJECT_NAME_MACRO}")
add_definitions(-DVIDEO_WIDTH=${VIDEO_WIDTH})
add_definitions(-DVIDEO_HEIGHT=${VIDEO_HEIGHT})
add_definitions(-DFRAMERATE=${FRAMERATE})
add_definitions(-DSAMPLERATE=${SAMPLERATE})
add_definitions(-DSMOKETEST=${SMOKETEST})
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libswscale libavcodec libavformat libavdevice libavutil libavfilter)
message(STATUS "FFMPEG Libraries -> ${FFMPEG_LIBRARIES}")
pkg_check_modules(GLIB REQUIRED glib-2.0)
find_package(Cairo REQUIRED)
find_package(RSVG REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PNG REQUIRED)
find_package(glm REQUIRED)

# Initialize the sources list with the main project file. We will append to it soon.
set(SOURCES "src/misc/Main.cpp")

find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA found. Configuring for CUDA. Libraries -> ${CUDA_LIBRARIES}")
    set(GPU_LANGUAGE CUDA)
    enable_language(CUDA)
    set(GPU_COMPILER ${CMAKE_CUDA_COMPILER})
    file(GLOB_RECURSE CUDA_SOURCES "src/CUDA/cuda_main.cu")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES native) # Auto-detect available architectures
    endif()
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe> $<$<COMPILE_LANGUAGE:CUDA>:"--diag_suppress=20012">)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(WARNING "CUDA not found. A linker error may occur if you try to use scenes dependent on GPU acceleration.")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

include_directories(${PROJECT_NAME} ${FFMPEG_INCLUDE_DIRS} ${RSVG_INCLUDE_DIRS} ${GLIB_INCLUDE_DIRS})
include_directories("/usr/include/gdk-pixbuf-2.0")
include_directories("/usr/include/cairo")
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${GLIB_LIBRARIES} ${FFMPEG_LIBRARIES} ${RSVG_LIBRARIES} ${FDKAAC_LIBRARIES} cairo gobject-2.0 lzma ${PNG_LIBRARIES} ${GPU_LIBRARIES} ${X11_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE -Wreorder)

