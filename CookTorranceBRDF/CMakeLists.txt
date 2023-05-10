cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CookTorranceBRDF LANGUAGES C CXX CUDA)

set(owl_dir ${PROJECT_SOURCE_DIR}/../owl)
add_subdirectory(${owl_dir} buildOwl/ EXCLUDE_FROM_ALL)

embed_ptx(
  OUTPUT_TARGET
    CookTorranceBRDF-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    shader.cu
)

add_executable(CookTorranceBRDF main.cpp
                                objUtils.cpp
                                viewer.cpp

                                cudaBuffer.h
                                geometriesData.h
                                objUtils.h
                                optix7.h
                                shader.h
                                shaderMaterials.h
                                viewer.h)

include_directories(".")

target_link_libraries(CookTorranceBRDF
    CookTorranceBRDF-ptx
    owl::owl
    owl_viewer
    stb_image
)

set_property(TARGET CookTorranceBRDF PROPERTY CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
endif()

