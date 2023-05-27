#ifndef SHADER_MATERIALS_H
#define SHADER_MATERIALS_H

#include <owl/common/math/vec.h>

using namespace owl;

struct SimpleObjMaterial
{
    vec3f albedo;
    vec3f emissive;

    float ns;
};

struct CookTorranceMaterial
{
    vec3f albedo;

    float metallic;
    float roughness;
    float reflectance;
};

#endif
