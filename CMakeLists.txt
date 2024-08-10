cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 17)
project(swaptube LANGUAGES CXX)

set(ENV{GLIBCXX_FORCE_NEW} 1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(FFmpeg REQUIRED)
find_package(Cairo REQUIRED)
find_package(RSVG REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PNG REQUIRED)
find_package(glm REQUIRED)

# Configure CUDA or ROCm.
find_package(CUDA)
set(GPU_LIBRARIES)

if(CUDA_FOUND)
  message(STATUS "CUDA libraries found. Configuring for CUDA. Libraries -> ${CUDA_LIBRARIES}")
  set(GPU_LANGUAGE CUDA)
  enable_language(CUDA)
  set(GPU_COMPILER ${CMAKE_CUDA_COMPILER})
  set(GPU_ARCHITECTURE ${CMAKE_CUDA_ARCHITECTURES})
  list(APPEND GPU_LIBRARIES ${CUDA_LIBRARIES})
  include_directories(${CUDA_INCLUDE_DIRS})
  file(GLOB_RECURSE CUDA_SOURCES "src/CUDA/*.cu")
  add_executable(${PROJECT_NAME} "src/Projects/.active_project.cpp" ${CUDA_SOURCES})

# ROCm's HIP (Heterogeneous-Compute Interface for Portability) API
# - https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html
elseif(HIP_FOUND)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  find_package(HIP REQUIRED)

  message(STATUS "HIP found. Configuring for HIP. Libraries -> ${HIP_LIBRARIES}")
  set(GPU_LANGUAGE HIP)
  enable_language(HIP)

  set(GPU_COMPILER ${HIP_COMPILER})
  list(APPEND GPU_LIBRARIES hip::host)
  include_directories(${HIP_INCLUDE_DIRS})
  file(GLOB_RECURSE HIP_SOURCES "src/ROCm/*.hip")

  # Set the main entrypoint's source file + update properties for which to compile non '.hip'
  # files as HIP ones.
  set(MAIN_ROCM_SOURCE_FILE "src/projects/rocm_hip_main.cc")
  set_source_files_properties(${MAIN_ROCM_SOURCE_FILE} PROPERTIES LANGUAGE HIP)
  add_executable(${PROJECT_NAME} "src/Projects/.active_project.cpp" ${HIP_SOURCES})
else()
  message(WARNING "Neither CUDA nor HIP was found. A linker error may occur if you try to use scenes dependent on GPU acceleration.")
  add_executable(${PROJECT_NAME} "src/Projects/.active_project.cpp")
endif()

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

