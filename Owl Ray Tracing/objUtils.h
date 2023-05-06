#ifndef OBJ_UTILS_H
#define OBJ_UTILS_H

#include "owl/common/math/vec.h"

#include "rapidobj/rapidobj.hpp"

#include "deviceCode.h"

using namespace owl;

class OBJUtils
{
public:
    static void read_obj(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<vec3i>& vertex_normal_indices, std::vector<Material>& materials, std::vector<int>& materials_indices);
};

#endif
