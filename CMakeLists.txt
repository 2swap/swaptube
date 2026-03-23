cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ standard set to ${CMAKE_CXX_STANDARD}")
project(swaptube LANGUAGES CXX)
message(STATUS "Project '${PROJECT_NAME}' initialized")

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libswscale libavcodec libavformat libavdevice libavutil libavfilter)
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
message(STATUS "Checking for CUDA and HIP support...")
check_language(CUDA)
check_language(HIP)
if(CMAKE_CUDA_COMPILER AND NOT ${USE_HIP})
    message(STATUS "CUDA compiler detected.")
    enable_language(CUDA)
    add_definitions(-DUSE_GPU)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    set(GPU_LANGUAGE CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)
    file(GLOB_RECURSE GPU_SOURCES "src/CUDA/*.cu")
    set(CMAKE_CONFIGURE_DEPENDS ${GPU_SOURCES})
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe> $<$<COMPILE_LANGUAGE:CUDA>:"--diag_suppress=20012">)
elseif(CMAKE_HIP_COMPILER)
    message(STATUS "HIP compiler detected.")
    enable_language(HIP)
    add_definitions(-DUSE_GPU)
    message(STATUS "HIP compiler found: ${CMAKE_HIP_COMPILER}")
    set(GPU_LANGUAGE HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_ARCHITECTURES native)
    message(STATUS "Translating CUDA to HIP")
    find_program(HIPIFY_PERL hipify-perl REQUIRED)
    file(GLOB_RECURSE CUDA_TO_TRANSLATE_CU CONFIGURE_DEPENDS "src/CUDA/*.cu")
    file(GLOB_RECURSE CUDA_TO_TRANSLATE_CUH CONFIGURE_DEPENDS "src/CUDA/*.cuh")
    foreach(cuda_file ${CUDA_TO_TRANSLATE_CUH})
        file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}/src/CUDA" "${cuda_file}")
        set(hip_file "${CMAKE_SOURCE_DIR}/src/HIP/${rel_path}")
        add_custom_command(
            OUTPUT ${hip_file}
            COMMAND ${CMAKE_COMMAND} -E copy ${cuda_file} ${hip_file}
            COMMAND ${HIPIFY_PERL} ${hip_file} -inplace
            COMMAND rm "${hip_file}.prehip"
            DEPENDS ${cuda_file}
            COMMAND echo "Hipifying ${rel_path}"
            VERBATIM
        )
        list(APPEND HIP_HEADERS ${hip_file})
    endforeach()
    foreach(cuda_file ${CUDA_TO_TRANSLATE_CU})
        file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}/src/CUDA" "${cuda_file}")
        set(hip_file "${CMAKE_SOURCE_DIR}/src/HIP/${rel_path}")
        add_custom_command(
            OUTPUT ${hip_file}
            COMMAND ${CMAKE_COMMAND} -E copy ${cuda_file} ${hip_file}
            COMMAND ${HIPIFY_PERL} ${hip_file} -inplace
            COMMAND rm "${hip_file}.prehip"
            DEPENDS ${cuda_file} ${HIP_HEADERS}
            COMMAND echo "Hipifying ${rel_path}"
            VERBATIM
        )
        list(APPEND GPU_SOURCES ${hip_file})
    endforeach()
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    set(CMAKE_CONFIGURE_DEPENDS ${GPU_SOURCES})
    # TODO: Check error ouput on CUDA/HIP function calls so we don't have to suppress warnings
    add_compile_options("-Wno-unused-value")
else()
    message(WARNING "Neither CUDA nor HIP found. Please refer to readme for building without GPU support.")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${ACTIVE_PROJECT_SOURCE} ${GPU_SOURCES})
message(STATUS "Executable target ${PROJECT_NAME} added")

# Pass user-specified values as preprocessor macros
target_compile_definitions(${PROJECT_NAME} PRIVATE
    PROJECT_NAME_MACRO="${PROJECT_NAME_MACRO}"
    AUDIO_HINTS=${AUDIO_HINTS}
    AUDIO_SFX=${AUDIO_SFX}
)
message(STATUS "Preprocessor definitions added")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}
    ${FFMPEG_INCLUDE_DIRS} ${RSVG_INCLUDE_DIRS} ${GLIB_INCLUDE_DIRS}
    "/usr/include/gdk-pixbuf-2.0"
    "/usr/include/cairo"
    ${PNG_INCLUDE_DIR}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${GLIB_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    ${RSVG_LIBRARIES}
    ${FDKAAC_LIBRARIES}
    cairo
    gobject-2.0
    lzma
    ${PNG_LIBRARIES}
    ${GPU_LIBRARIES}
    ${X11_LIBRARIES}
    nlohmann_json::nlohmann_json
)

message(STATUS "Link libraries configured for ${PROJECT_NAME}")
target_compile_options(${PROJECT_NAME} PRIVATE -Wreorder)
message(STATUS "Configuration complete.")
