#ifndef SHADER_H
#define SHADER_H

#include <owl/owl.h>
#include <owl/common/math/vec.h>

#include "shaderMaterials.h"
#include "texture_types.h"

#define RADIANCE_RAY 0
#define SHADOW_RAY 1

using namespace owl;

struct LaunchParams
{
    OptixTraversableHandle scene;

    vec3f* accumulation_buffer;
    unsigned int frame_number;

    CookTorranceMaterial obj_material;
};

struct RayGenData
{
    vec2i frame_buffer_size;
    uint32_t* frame_buffer;

    struct
    {
        vec3f position;
        vec3f direction_00;
        vec3f direction_dx;
        vec3f direction_dy;
    } camera;
};

struct MissProgData
{
    cudaTextureObject_t skysphere;
};

#endif
