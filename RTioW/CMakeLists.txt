cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(RTioW LANGUAGES C CXX CUDA)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
endif()

set(owl_dir ${PROJECT_SOURCE_DIR}/../owl)
add_subdirectory(${owl_dir} buildOwl/ EXCLUDE_FROM_ALL)
embed_ptx(
  OUTPUT_TARGET
    RTioW-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    shader.cu
)

add_executable(RTioW main.cpp
                     viewer.cpp

                     materials.h
                     shader.h
                     viewer.h)

target_link_libraries(RTioW
    PRIVATE
    RTioW-ptx
    owl::owl
    owl_viewer
    stb_image
)

set_property(TARGET RTioW PROPERTY CXX_STANDARD 17)
