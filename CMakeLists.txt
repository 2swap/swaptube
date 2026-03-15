cmake_minimum_required(VERSION 3.24)
project(swaptube LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
endif()

if(NOT DEFINED PROJECT_NAME_MACRO)
    set(PROJECT_NAME_MACRO "${PROJECT_NAME}" CACHE STRING "Project name macro exposed to the source tree")
endif()
if(NOT DEFINED AUDIO_HINTS)
    set(AUDIO_HINTS 0 CACHE STRING "Enable audio hint streams")
endif()
if(NOT DEFINED AUDIO_SFX)
    set(AUDIO_SFX 0 CACHE STRING "Enable audio SFX streams")
endif()

if(CMAKE_CUDA_COMPILER_VERSION)
    message(STATUS "CUDA toolkit version: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

if(NOT DEFINED SWAPTUBE_CUDA_ARCH OR SWAPTUBE_CUDA_ARCH STREQUAL "")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(_swaptube_default_cuda_arch "60")
    else()
        set(_swaptube_default_cuda_arch "50")
    endif()
    set(SWAPTUBE_CUDA_ARCH "${_swaptube_default_cuda_arch}" CACHE STRING "CUDA compute capability target used for nvcc gencode flags")
endif()

if(NOT SWAPTUBE_CUDA_ARCH MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "SWAPTUBE_CUDA_ARCH must be a numeric compute capability such as 50, 75, or 86."
    )
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND SWAPTUBE_CUDA_ARCH VERSION_LESS 60)
    message(FATAL_ERROR
        "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not support compute capability ${SWAPTUBE_CUDA_ARCH}. "
        "Set SWAPTUBE_CUDA_ARCH to a supported architecture, or build with a CUDA toolkit that still supports Maxwell-era targets."
    )
endif()

message(STATUS "CUDA architecture target: ${SWAPTUBE_CUDA_ARCH}")
message(STATUS "Project '${PROJECT_NAME}' initialized")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

function(swaptube_append_existing_paths out_var)
    set(_paths "${${out_var}}")
    foreach(_path IN LISTS ARGN)
        if(_path AND EXISTS "${_path}")
            file(TO_CMAKE_PATH "${_path}" _cmake_path)
            list(APPEND _paths "${_cmake_path}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _paths)
    set(${out_var} "${_paths}" PARENT_SCOPE)
endfunction()

function(swaptube_define_imported_library target_name library_path)
    if(library_path AND NOT TARGET "${target_name}")
        add_library("${target_name}" UNKNOWN IMPORTED)
        set_target_properties("${target_name}" PROPERTIES IMPORTED_LOCATION "${library_path}")
    endif()
endfunction()

# =====================
# Dependencies
# =====================
if(WIN32)
    set(FFMPEG_ROOT "$ENV{FFMPEG_ROOT}" CACHE PATH "Optional prefix containing FFmpeg headers and libraries")
    set(VCPKG_ROOT "$ENV{VCPKG_ROOT}" CACHE PATH "Optional vcpkg root used for dependency discovery")
    set(VCPKG_INSTALLED_ROOT "$ENV{VCPKG_INSTALLED_DIR}" CACHE PATH "Optional vcpkg installed root or triplet prefix")
    set(MSYS2_ROOT "$ENV{MSYS2_ROOT}" CACHE PATH "Optional MSYS2 root used for dependency discovery")
    set(SWAPTUBE_VCPKG_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "vcpkg triplet used for dependency discovery")
    if(NOT SWAPTUBE_VCPKG_TRIPLET)
        set(SWAPTUBE_VCPKG_TRIPLET "x64-windows")
    endif()

    # Prefer MSVC-friendly import libraries when both static .a and import .dll.a exist.
    if(MSVC)
        list(PREPEND CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll.a")
        list(REMOVE_DUPLICATES CMAKE_FIND_LIBRARY_SUFFIXES)
    endif()

    find_program(FFMPEG_EXECUTABLE NAMES ffmpeg.exe ffmpeg)
    if(FFMPEG_EXECUTABLE)
        get_filename_component(_swaptube_ffmpeg_bin "${FFMPEG_EXECUTABLE}" DIRECTORY)
        get_filename_component(FFMPEG_PATH_ROOT "${_swaptube_ffmpeg_bin}" DIRECTORY)
    endif()

    find_program(VCPKG_EXECUTABLE NAMES vcpkg.exe vcpkg)
    if(VCPKG_EXECUTABLE)
        get_filename_component(VCPKG_PATH_ROOT "${VCPKG_EXECUTABLE}" DIRECTORY)
    endif()

    set(SWAPTUBE_WINDOWS_PREFIXES "")
    swaptube_append_existing_paths(SWAPTUBE_WINDOWS_PREFIXES
        "${FFMPEG_ROOT}"
        "${FFMPEG_PATH_ROOT}"
        "${VCPKG_INSTALLED_ROOT}"
        "${VCPKG_INSTALLED_ROOT}/${SWAPTUBE_VCPKG_TRIPLET}"
        "${VCPKG_ROOT}/installed/${SWAPTUBE_VCPKG_TRIPLET}"
        "${VCPKG_PATH_ROOT}/installed/${SWAPTUBE_VCPKG_TRIPLET}"
        "${MSYS2_ROOT}/mingw64"
        "$ENV{USERPROFILE}/scoop/apps/ffmpeg-shared/current"
        "$ENV{USERPROFILE}/scoop/apps/vcpkg/current/installed/${SWAPTUBE_VCPKG_TRIPLET}"
        "$ENV{USERPROFILE}/scoop/apps/msys2/current/mingw64"
        "C:/vcpkg/installed/${SWAPTUBE_VCPKG_TRIPLET}"
        "C:/msys64/mingw64"
        "$ENV{ProgramFiles}/ffmpeg"
        "$ENV{ProgramFiles}/FFmpeg"
        "$ENV{ProgramData}/chocolatey/lib/ffmpeg-shared/tools/ffmpeg-8.0.1-full_build-shared"
    )
    list(PREPEND CMAKE_PREFIX_PATH ${SWAPTUBE_WINDOWS_PREFIXES})

    find_path(FFMPEG_INCLUDE_DIRS libavcodec/avcodec.h
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES include)
    find_library(AVCODEC_LIBRARY NAMES avcodec libavcodec
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(AVFORMAT_LIBRARY NAMES avformat libavformat
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(AVDEVICE_LIBRARY NAMES avdevice libavdevice
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(AVUTIL_LIBRARY NAMES avutil libavutil
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(AVFILTER_LIBRARY NAMES avfilter libavfilter
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(SWSCALE_LIBRARY NAMES swscale libswscale
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(SWRESAMPLE_LIBRARY NAMES swresample libswresample
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)

    set(FFMPEG_LIBRARIES
        ${SWSCALE_LIBRARY}
        ${AVCODEC_LIBRARY}
        ${AVFORMAT_LIBRARY}
        ${AVDEVICE_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${AVFILTER_LIBRARY}
        ${SWRESAMPLE_LIBRARY}
    )
    list(FILTER FFMPEG_LIBRARIES EXCLUDE REGEX "^$")

    find_path(GLIB_INCLUDE_DIRS glib.h
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES include/glib-2.0 glib-2.0 include)
    find_path(GLIBCONFIG_INCLUDE_DIRS glibconfig.h
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib/glib-2.0/include glib-2.0/include)
    find_library(GLIB_LIBRARY NAMES glib-2.0 libglib-2.0
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    set(GLIB_LIBRARIES ${GLIB_LIBRARY})

    find_path(CAIRO_INCLUDE_DIRS cairo.h
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES include/cairo include)
    find_library(CAIRO_LIBRARY NAMES cairo libcairo
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)

    find_path(RSVG_INCLUDE_DIRS librsvg/rsvg.h
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES include/librsvg-2.0 librsvg-2.0 include)
    find_library(RSVG_LIBRARY NAMES rsvg-2 librsvg-2.0
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    set(RSVG_LIBRARIES ${RSVG_LIBRARY})

    find_path(PNG_INCLUDE_DIR png.h
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES include)
    find_library(PNG_LIBRARIES NAMES libpng16 libpng png
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)

    find_path(GDK_PIXBUF_INCLUDE_DIRS gdk-pixbuf/gdk-pixbuf.h
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES include/gdk-pixbuf-2.0 gdk-pixbuf-2.0 include)
    find_library(GDK_PIXBUF_LIBRARY NAMES gdk_pixbuf-2.0 libgdk_pixbuf-2.0
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)

    find_library(GOBJECT_LIBRARY NAMES gobject-2.0 libgobject-2.0
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(LZMA_LIBRARY NAMES lzma liblzma
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)
    find_library(CAIRO_GOBJECT_LIBRARY NAMES cairo-gobject libcairo-gobject
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES lib)

    find_path(GLM_INCLUDE_DIRS glm/glm.hpp
        HINTS ${SWAPTUBE_WINDOWS_PREFIXES}
        PATH_SUFFIXES include)
    if(NOT GLM_INCLUDE_DIRS)
        set(GLM_INCLUDE_DIRS "")
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/nlohmann/direct_deploy/nlohmann_json/include/nlohmann/json.hpp")
        add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
        set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/deps/nlohmann/direct_deploy/nlohmann_json/include"
        )
    else()
        find_package(nlohmann_json CONFIG QUIET)
        if(NOT nlohmann_json_FOUND)
            include(FetchContent)
            FetchContent_Declare(json
                URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
            )
            FetchContent_MakeAvailable(json)
        endif()
    endif()

    swaptube_define_imported_library(cairo "${CAIRO_LIBRARY}")
    swaptube_define_imported_library(gobject-2.0 "${GOBJECT_LIBRARY}")
    if(LZMA_LIBRARY)
        swaptube_define_imported_library(lzma "${LZMA_LIBRARY}")
    elseif(NOT TARGET lzma)
        add_library(lzma INTERFACE IMPORTED)
    endif()

    if(NOT FFMPEG_INCLUDE_DIRS OR NOT AVCODEC_LIBRARY OR NOT AVFORMAT_LIBRARY OR NOT AVUTIL_LIBRARY OR NOT SWSCALE_LIBRARY
        OR NOT GLIB_INCLUDE_DIRS OR NOT GLIBCONFIG_INCLUDE_DIRS OR NOT GLIB_LIBRARY
        OR NOT CAIRO_INCLUDE_DIRS OR NOT CAIRO_LIBRARY
        OR NOT RSVG_INCLUDE_DIRS OR NOT RSVG_LIBRARY
        OR NOT PNG_INCLUDE_DIR OR NOT PNG_LIBRARIES
        OR NOT GDK_PIXBUF_INCLUDE_DIRS OR NOT GDK_PIXBUF_LIBRARY
        OR NOT GOBJECT_LIBRARY)
        message(STATUS "DEBUG: FFMPEG_INCLUDE_DIRS=${FFMPEG_INCLUDE_DIRS}")
        message(STATUS "DEBUG: AVCODEC_LIBRARY=${AVCODEC_LIBRARY}")
        message(STATUS "DEBUG: RSVG_INCLUDE_DIRS=${RSVG_INCLUDE_DIRS}")
        message(STATUS "DEBUG: CAIRO_INCLUDE_DIRS=${CAIRO_INCLUDE_DIRS}")
        message(STATUS "DEBUG: GLIB_INCLUDE_DIRS=${GLIB_INCLUDE_DIRS}")
        message(FATAL_ERROR "Required Windows dependencies were not found.")
    endif()
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libswscale libavcodec libavformat libavdevice libavutil libavfilter libswresample)
    pkg_check_modules(GLIB REQUIRED glib-2.0)
    find_package(Cairo REQUIRED)
    find_package(RSVG REQUIRED)
    find_package(PNG REQUIRED)
    find_package(nlohmann_json CONFIG QUIET)
    if(NOT nlohmann_json_FOUND)
        include(FetchContent)
        FetchContent_Declare(json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        )
        FetchContent_MakeAvailable(json)
    endif()
endif()

find_package(CUDA REQUIRED)

# =====================
# Sources
# =====================
# Add globbed cpp files: all .cpp files in src, including subdirectories.
# Exclude the src/Projects/ directory.
# Include, however, src/Projects/.active_project.cpp
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
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    PROJECT_NAME_MACRO="${PROJECT_NAME_MACRO}"
    AUDIO_HINTS=${AUDIO_HINTS}
    AUDIO_SFX=${AUDIO_SFX}
    GLM_ENABLE_EXPERIMENTAL
    USE_CUDA
)

# =====================
# CUDA architecture (CRITICAL FIX)
# =====================
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_${SWAPTUBE_CUDA_ARCH},code=sm_${SWAPTUBE_CUDA_ARCH}
        -gencode=arch=compute_${SWAPTUBE_CUDA_ARCH},code=compute_${SWAPTUBE_CUDA_ARCH}
        --expt-relaxed-constexpr
        -Xcudafe --diag_suppress=20012
    >
)

# =====================
# Includes
# =====================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}
    ${FFMPEG_INCLUDE_DIRS}
    ${RSVG_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIR}
    ${GLM_INCLUDE_DIRS}
    /usr/include/gdk-pixbuf-2.0
    /usr/include/cairo
)

if(WIN32)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${GDK_PIXBUF_INCLUDE_DIRS}
        ${GLIBCONFIG_INCLUDE_DIRS}
        ${CAIRO_INCLUDE_DIRS}
    )
endif()

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

if(WIN32)
    target_link_libraries(${PROJECT_NAME}
        ${GDK_PIXBUF_LIBRARY}
        ${CAIRO_GOBJECT_LIBRARY}
    )
    if(MSVC)
        target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:libcmt.lib")
    endif()
endif()

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wreorder>
)

message(STATUS "Configuration complete.")
