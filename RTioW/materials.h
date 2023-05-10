#ifndef MATERIALS_H
#define MATERIALS_H

#include <owl/owl.h>
#include <owl/common/math/vec.h>

using namespace owl;

struct DiffuseTriangleGeomData
{
    vec3i* indices;
    vec3f* vertices;

    vec3f albedo;
};

//Perfectly diffuse
struct FloorTriangleGeomData
{
    vec3i* indices;
    vec3f* vertices;

    vec3f albedo;
};

struct LambertianMaterial
{
    vec3f albedo;
};

struct DielectricMaterial
{
    float ior;

    vec3f color_attenuation = vec3f(1.0f, 1.0f, 1.0f);
};

struct MetalMaterial
{
    vec3f albedo;

    float roughness;
};

#endif
