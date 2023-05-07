#ifndef SHADER_H
#define SHADER_H

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>

#include "materials.h"
#include "texture_types.h"

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

struct TriangleGeomData
{
    vec3i* indices;
    vec3f* vertices;
};

struct RayGenData
{
    OptixTraversableHandle scene;

    vec2i frame_buffer_size;

    uint32_t* frame_buffer;

    uint32_t frame_number;
    vec3f* accumulation_buffer;

    struct
    {
        vec3f position;
        vec3f direction_00;
        vec3f direction_dx, direction_dy;
    } camera;
};

enum ScatterState
{
    BOUNCED,//Bounced/reflected/refracted off of object
    TERMINATED, //Completely absorbed and not bounced
    MISSED  //Hit the sky
};

struct PerRayData
{
    owl::common::LCG<4> random;

    struct
    {
        vec3f origin, target;

        ScatterState state;
    } scatter;

    vec3f color;
};

struct MissProgData
{
    cudaTextureObject_t skysphere;
};

#endif
