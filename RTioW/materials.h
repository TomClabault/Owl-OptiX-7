#ifndef MATERIALS_H
#define MATERIALS_H

#include <owl/owl.h>
#include <owl/common/math/vec.h>

using namespace owl;

struct LambertianMaterial
{
    vec3f albedo;
};

#endif
