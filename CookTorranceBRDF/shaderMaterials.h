#ifndef SHADER_MATERIALS_H
#define SHADER_MATERIALS_H

#include "optix_types.h"
#include "owl/common/math/vec.h"

using namespace owl;

struct CookTorranceMaterial
{
    vec3f albedo;

    float shininess;
    float roughness;
};

#endif
