cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
message(STATUS "C++ standard set to ${CMAKE_CXX_STANDARD}")
project(swaptube LANGUAGES CXX)
message(STATUS "Project '${PROJECT_NAME}' initialized")

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(PkgConfig REQUIRED)
message(STATUS "Searching for FFMPEG libraries...")
pkg_check_modules(FFMPEG REQUIRED libswscale libavcodec libavformat libavdevice libavutil libavfilter)
message(STATUS "FFMPEG Libraries -> ${FFMPEG_LIBRARIES}")
pkg_check_modules(GLIB REQUIRED glib-2.0)
find_package(Cairo REQUIRED)
find_package(RSVG REQUIRED)
find_package(PNG REQUIRED)
find_package(nlohmann_json REQUIRED)

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

# Add globbed cpp files: all .cpp files in src, including subdirectories.
# Exclude the src/Projects/ directory.
# Include, however, src/Projects/.active_project.cpp
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "src/Projects/.*")
file(GLOB_RECURSE ACTIVE_PROJECT_SOURCE "src/Projects/.active_project.cpp")

include(CheckLanguage)
message(STATUS "Checking for CUDA support...")
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler detected.")
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    set(GPU_LANGUAGE CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES native)
    file(GLOB_RECURSE CUDA_SOURCES "src/CUDA/*.cu")
    set(CMAKE_CONFIGURE_DEPENDS ${CUDA_SOURCES})
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe> $<$<COMPILE_LANGUAGE:CUDA>:"--diag_suppress=20012">)
else()
    message(WARNING "CUDA not found. Please refer to readme for building without GPU support.")
endif()

add_compile_options(-fdiagnostics-color=always)

add_executable(${PROJECT_NAME} ${SOURCES} ${ACTIVE_PROJECT_SOURCE} ${CUDA_SOURCES})
message(STATUS "Executable target ${PROJECT_NAME} added")

# Pass user-specified values as preprocessor macros
target_compile_definitions(${PROJECT_NAME} PRIVATE
    PROJECT_NAME_MACRO="${PROJECT_NAME_MACRO}"
    AUDIO_HINTS=${AUDIO_HINTS}
    AUDIO_SFX=${AUDIO_SFX}
)
message(STATUS "Preprocessor definitions added")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME} ${FFMPEG_INCLUDE_DIRS} ${RSVG_INCLUDE_DIRS} ${GLIB_INCLUDE_DIRS}
    "/usr/include/gdk-pixbuf-2.0"
    "/usr/include/cairo"
    ${PNG_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} ${GLIB_LIBRARIES} ${FFMPEG_LIBRARIES} ${RSVG_LIBRARIES} ${FDKAAC_LIBRARIES} cairo gobject-2.0 lzma ${PNG_LIBRARIES} ${GPU_LIBRARIES} ${X11_LIBRARIES} nlohmann_json::nlohmann_json)
message(STATUS "Link libraries configured for ${PROJECT_NAME}")
target_compile_options(${PROJECT_NAME} PRIVATE -Wreorder)
message(STATUS "Configuration complete.")
