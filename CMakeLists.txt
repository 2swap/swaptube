cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 17)
project(swaptube LANGUAGES CXX)

# Pass user-specified values as preprocessor macros
add_definitions(-DPROJECT_NAME_MACRO="${PROJECT_NAME_MACRO}")
add_definitions(-DVIDEO_WIDTH=${VIDEO_WIDTH})
add_definitions(-DVIDEO_HEIGHT=${VIDEO_HEIGHT})

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(FFmpeg REQUIRED)
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
    set(GPU_ARCHITECTURE ${CMAKE_CUDA_ARCHITECTURES})
    file(GLOB_RECURSE CUDA_SOURCES "src/CUDA/*.cu")
    include_directories(${CUDA_INCLUDE_DIRS})
elseif(HIP_FOUND)
    message(STATUS "HIP found. Configuring for HIP. Libraries -> ${HIP_LIBRARIES}")
    set(GPU_LANGUAGE HIP)
    enable_language(HIP)
    set(GPU_COMPILER ${HIP_COMPILER})
    file(GLOB_RECURSE HIP_SOURCES "src/ROCm/*.hip")
    include_directories(${HIP_INCLUDE_DIRS})
else()
    message(WARNING "Neither CUDA nor HIP was found. A linker error may occur if you try to use scenes dependent on GPU acceleration.")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES} ${HIP_SOURCES})

include_directories(${PROJECT_NAME} ${FFMPEG_INCLUDE_DIRS} ${RSVG_INCLUDE_DIRS})
include_directories("/usr/include/glib-2.0")
include_directories("/usr/lib/x86_64-linux-gnu/glib-2.0/include")
include_directories("/usr/include/gdk-pixbuf-2.0")
include_directories("/usr/include/cairo")
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${FFMPEG_LIBRARIES} ${SWSCALE_LIBRARIES} ${RSVG_LIBRARIES} cairo gobject-2.0 ${PNG_LIBRARIES} ${GPU_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE -Wreorder)

