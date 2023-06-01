#ifndef SHADER_MATERIALS_H
#define SHADER_MATERIALS_H

#include "texture_types.h"

#include <owl/common/math/vec.h>

using namespace owl;

struct SimpleObjMaterial
{
    vec3f albedo = vec3f(0.0f);
    vec3f emissive = vec3f(0.0f);

    float ns = 0.0f;

    cudaTextureObject_t diffuse_texture;
};

struct CookTorranceMaterial
{
    vec3f albedo;

    float metallic;
    float roughness;
    float reflectance;
};

#endif
