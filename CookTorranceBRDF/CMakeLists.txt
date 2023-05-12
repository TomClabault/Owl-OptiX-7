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
                                utils.cpp
                                viewer.cpp

                                cookTorrance.h
                                cudaBuffer.h
                                geometriesData.h
                                utils.h
                                optix7.h
                                shader.h
                                shaderMaterials.h
                                viewer.h

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

