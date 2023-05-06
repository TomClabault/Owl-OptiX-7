#ifndef SHADER_H
#define SHADER_H

#include <owl/owl.h>
#include <owl/common/math/vec.h>

#include "materials.h"

using namespace owl;

struct Sphere
{
    vec3f center;
    float radius;
};

struct LambertianSphere
{
    Sphere sphere;
    LambertianMaterial material;
};

struct LambertianSpheresGeometryData
{
    LambertianSphere* primitives;
};

struct RayGenData
{
    OptixTraversableHandle scene;

    vec2i frame_buffer_size;

    uint32_t* frame_buffer;

    struct
    {
        vec3f position;
        vec3f direction_00;
        vec3f direction_dx, direction_dy;
    } camera ;
};

struct MissProgData
{
    vec3f background_color;
};

#endif
