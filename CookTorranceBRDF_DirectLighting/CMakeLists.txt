cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CookTorranceBRDF_DirectLighting LANGUAGES C CXX CUDA)

set(owl_dir ${PROJECT_SOURCE_DIR}/../owl)
add_subdirectory(${owl_dir} buildOwl/ EXCLUDE_FROM_ALL)

embed_ptx(
  OUTPUT_TARGET
    CookTorranceBRDF_DirectLighting-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    shader.cu
)

add_executable(CookTorranceBRDF_DirectLighting main.cpp
                                utils.cpp

                                cookTorrance.h
                                cudaBuffer.h
                                geometriesData.h
                                utils.h
                                optix7.h
                                shader.h
                                shaderMaterials.h

                                imGuiViewer.h
                                imGuiViewer.cpp

                                ../imgui/imgui.cpp
                                ../imgui/imgui_draw.cpp
                                ../imgui/imgui_tables.cpp
                                ../imgui/imgui_widgets.cpp
                                ../imgui/imgui_demo.cpp
                                ../imgui/backends/imgui_impl_glfw.cpp
                                ../imgui/backends/imgui_impl_opengl3.cpp
                                ../imgui/backends/imgui_impl_win32.cpp)

include_directories(".")
include_directories("../imgui")
include_directories("../imgui/backends")

target_link_libraries(CookTorranceBRDF_DirectLighting
    CookTorranceBRDF_DirectLighting-ptx
    owl::owl
    owl_viewer
    stb_image
)

set_property(TARGET CookTorranceBRDF_DirectLighting PROPERTY CXX_STANDARD 17)

set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use-fast-math --gpu-architecture=compute_50 --gpu-code=compute_50")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
endif()

