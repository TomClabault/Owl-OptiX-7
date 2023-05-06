cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(OwlRayTracer LANGUAGES C CXX CUDA)

set(owl_dir ${PROJECT_SOURCE_DIR}/../owl)
add_subdirectory(${owl_dir} buildOwl/ EXCLUDE_FROM_ALL)
embed_ptx(
  OUTPUT_TARGET
    owlRayTracer-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)

add_executable(OwlRayTracer hostCode.cpp
                            objUtils.cpp
                            scene.cpp
                            viewer.cpp

                            deviceCode.h
                            objUtils.h
                            scene.h
                            viewer.h)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

target_link_libraries(OwlRayTracer
    PRIVATE
    owlRayTracer-ptx
    owl::owl
    owl_viewer
    stb_image
)

set_property(TARGET OwlRayTracer PROPERTY CXX_STANDARD 17)

file(COPY data DESTINATION ${PROJECT_BINARY_DIR})
