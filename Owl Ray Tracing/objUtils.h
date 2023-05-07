#ifndef OBJ_UTILS_H
#define OBJ_UTILS_H

#include "owl/common/math/vec.h"

#include "rapidobj/rapidobj.hpp"

#include "deviceCode.h"

using namespace owl;

class OBJUtils
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
    static void read_obj(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<vec3i>& vertex_normal_indices, std::vector<Material>& materials, std::vector<int>& materials_indices);

    static void read_obj_no_vertex_normals(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<Material>& materials, std::vector<int>& materials_indices);
};

#endif
