cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
project(swaptube LANGUAGES CXX)

# Pass user-specified values as preprocessor macros
add_definitions(-DPROJECT_NAME_MACRO="${PROJECT_NAME_MACRO}")
add_definitions(-DAUDIO_HINTS=${AUDIO_HINTS})
add_definitions(-DAUDIO_SFX=${AUDIO_SFX})
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
find_package(PNG REQUIRED)
find_package(glm REQUIRED)
find_package(nlohmann_json REQUIRED)

# Initialize the sources list with the main project file. We will append to it soon.
set(SOURCES "src/Core/Main.cpp")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    set(GPU_LANGUAGE CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES native)
    file(GLOB_RECURSE CUDA_SOURCES "src/CUDA/*.cu")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe> $<$<COMPILE_LANGUAGE:CUDA>:"--diag_suppress=20012">)
else()
    message(WARNING "CUDA not found. Please refer to readme for building without GPU support.")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

include_directories(${PROJECT_NAME} ${FFMPEG_INCLUDE_DIRS} ${RSVG_INCLUDE_DIRS} ${GLIB_INCLUDE_DIRS})
include_directories("/usr/include/gdk-pixbuf-2.0")
include_directories("/usr/include/cairo")
include_directories(${PNG_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${GLIB_LIBRARIES} ${FFMPEG_LIBRARIES} ${RSVG_LIBRARIES} ${FDKAAC_LIBRARIES} cairo gobject-2.0 lzma ${PNG_LIBRARIES} ${GPU_LIBRARIES} ${X11_LIBRARIES} nlohmann_json::nlohmann_json)
target_compile_options(${PROJECT_NAME} PRIVATE -Wreorder)
