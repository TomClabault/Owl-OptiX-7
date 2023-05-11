#ifndef GEOMETRIES_DATA_H
#define GEOMETRIES_DATA_H

#include "shaderMaterials.h"

#include <owl/common/math/vec.h>

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

struct CookTorranceTriangleData
{
    SmoothNormalsTriangleData triangle_data;

    int* materials_indices;
    CookTorranceMaterial* materials;
};

struct DiffuseTriangleData
{
    NoNormalsTriangleData triangle_data;

    vec3f albedo;
};

#endif
