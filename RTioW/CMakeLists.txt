cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(RTioW LANGUAGES C CXX CUDA)

#enable_language(CUDA)

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

file(COPY data DESTINATION ${PROJECT_BINARY_DIR})
