#ifndef OBJ_UTILS_H
#define OBJ_UTILS_H

#include "rapidobj/rapidobj.hpp"
#include "texture_types.h"

#include <owl/common/math/vec.h>

using namespace owl;

class Utils
{
public:
    /**
     * @brief read_obj
     * @param filepath
     * @param indices
     * @param vertices
     * @param vertex_normals Vertex normals are stored in the same order as the vertices (in the \param vertices vector).
     * The correspondance between vertex normals and vertices is thus simple
     * @param vertex_normal_indices This is the exact same vector as \param indices.
     * @param materials
     * @param materials_indices
     */
    static void read_obj(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<vec3i>& vertex_normal_indices, std::vector<vec2f>& vertex_texcoords, std::vector<vec3i>& vertex_texcoords_indices, std::vector<rapidobj::Material>& materials, std::vector<int>& materials_indices);

    static void read_obj_with_vertex_normals(const char* filepath, rapidobj::Result& mesh_data, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<vec3i>& vertex_normals_indices, std::vector<vec2f>& vertex_texcoords, std::vector<vec3i>& vertex_texcoords_indices, std::vector<rapidobj::Material>& materials, std::vector<int>& materials_indices);

    static float* read_image( const char *filename, int& width, int& height, const bool flipY);

    /**
     * @brief read_image_rgba_4x8
     * @param texture_path
     * @param [out] width The width of the texture read
     * @param [out] height The height of the texture read
     * @return The data of the image. 4 bytes per pixel. Order: R, G, B, A
     */
    static unsigned char* read_image_rgba_4x8(const char* texture_path, int& width, int& height);

    /**
     * @brief Returns a pointer to device memory containing the texture created with the given data
     * @param data The pixels of the texture. Format 4x8 RGBA
     * @param width The width of the texture in pixels
     * @param height The height of the texure in pixels
     * @return
     */
    static cudaTextureObject_t create_simple_cuda_texture(unsigned char* data, int width, int height);
};

#endif
