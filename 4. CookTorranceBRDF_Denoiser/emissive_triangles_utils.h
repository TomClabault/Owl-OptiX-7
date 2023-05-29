#ifndef EMISSIVE_TRIANGLES_UTILS_H
#define EMISSIVE_TRIANGLES_UTILS_H

#include "rapidobj/rapidobj.hpp"
#include "shaderMaterials.h"

#include <owl/common/math/vec.h>

using namespace owl;

struct EmissiveTrianglesInfo
{
    vec3i* triangles_indices;
    vec3f* triangles_vertices;
    int* triangles_materials_indices;
    SimpleObjMaterial* triangles_materials;

    unsigned int count;
    int* triangles_primitive_indices;
};

class EmissiveTrianglesUtils
{
public:
    static EmissiveTrianglesInfo extract_emissive_triangles(std::vector<vec3i>& indices,
                                                       std::vector<vec3f>& vertices,
                                                       std::vector<vec3f>& vertex_normals,
                                                       std::vector<vec3i>& vertex_normals_indices,
                                                       std::vector<rapidobj::Material>& obj_materials,
                                                       std::vector<int>& materials_indices);
};

#endif
