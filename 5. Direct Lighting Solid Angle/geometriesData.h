#ifndef GEOMETRIES_DATA_H
#define GEOMETRIES_DATA_H

#include "rapidobj/rapidobj.hpp"
#include "shaderMaterials.h"
#include "texture_types.h"

#include <owl/common/math/vec.h>
#include <vector>

using namespace owl;

struct NoNormalsTriangleData
{
    vec3i* indices;
    vec3f* vertices;
};

struct SmoothNormalsTriangleData
{
    vec3i* indices;
    vec3f* vertices;

    vec3f* vertex_normals;
    vec3i* vertex_normals_indices;
};

struct SmoothNormalsUVTriangleData
{
    vec3i* indices;
    vec3f* vertices;

    vec3f* vertex_normals;
    vec3i* vertex_normals_indices;

    vec2f* vertex_uvs;
    vec3i* vertex_uvs_indices;
};

struct CookTorranceTriangleData
{
    SmoothNormalsTriangleData triangle_data;
};

struct SimpleObjTriangleData
{
    SmoothNormalsUVTriangleData triangle_data;
    
    SimpleObjMaterial* materials;
    int* materials_indices;
};

struct EmissiveTriangleData
{
    NoNormalsTriangleData triangle_data;

    vec3f emissive;
};


#endif
