cmake_minimum_required(VERSION 3.16.3)

project(swaptube LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

message(STATUS "Project '${PROJECT_NAME}' initialized")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")

# =====================
# Preprocessor macros
# =====================
add_definitions(
    -DPROJECT_NAME_MACRO="${PROJECT_NAME_MACRO}"
    -DAUDIO_HINTS=${AUDIO_HINTS}
    -DAUDIO_SFX=${AUDIO_SFX}
)

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# =====================
# Dependencies
# =====================
find_package(PkgConfig REQUIRED)

pkg_check_modules(FFMPEG REQUIRED
    libswscale
    libavcodec
    libavformat
    libavdevice
    libavutil
    libavfilter
)

pkg_check_modules(GLIB REQUIRED glib-2.0)
find_package(Cairo REQUIRED)
find_package(RSVG REQUIRED)
find_package(PNG REQUIRED)
find_package(nlohmann_json REQUIRED)

# =====================
# Sources
# =====================
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "src/Projects/.*")
file(GLOB_RECURSE ACTIVE_PROJECT_SOURCE "src/Projects/.active_project.cpp")

file(GLOB_RECURSE CUDA_SOURCES src/CUDA/*.cu)

# =====================
# Executable
# =====================
add_executable(${PROJECT_NAME}
    ${ACTIVE_PROJECT_SOURCE}
    ${SOURCES}
    ${CUDA_SOURCES}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)

# =====================
# CUDA architecture (CRITICAL FIX)
# =====================
# GeForce 940MX = Maxwell
# Most models are sm_50 or sm_52 — sm_50 is safest

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_50,code=compute_50
        --expt-relaxed-constexpr
        -Xcudafe --diag_suppress=20012
    >
)

# =====================
# Includes
# =====================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${FFMPEG_INCLUDE_DIRS}
    ${RSVG_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIR}
    ${GLM_INCLUDE_DIRS}
    /usr/include/gdk-pixbuf-2.0
    /usr/include/cairo
)

# =====================
# Linking
# =====================
target_link_libraries(${PROJECT_NAME}
    ${GLIB_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    ${RSVG_LIBRARIES}
    cairo
    gobject-2.0
    lzma
    ${PNG_LIBRARIES}
    nlohmann_json::nlohmann_json
)

target_compile_options(${PROJECT_NAME} PRIVATE)

message(STATUS "Configuration complete.")
