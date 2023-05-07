#ifndef OBJ_UTILS_H
#define OBJ_UTILS_H

#include "owl/common/math/vec.h"

#include "shader.h"

#include <vector>

using namespace owl;

class OBJUtils
{
public:
    /**
     * @brief The vertex normal indices array is the exact same as indices.
     * So use \param indices as your vertex normals indices array
     * @param filepath
     * @param indices
     * @param vertices
     * @param vertex_normals Vertex normals are stored in the same order as the vertices (in the \param vertices vector).
     * The correspondance between vertex normals and vertices is thus simple
     * @param materials
     * @param materials_indices
     */
    static void read_obj(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<Material>& materials, std::vector<int>& materials_indices);

    static void read_obj_no_vertex_normals(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<Material>& materials, std::vector<int>& materials_indices);
};

#endif
