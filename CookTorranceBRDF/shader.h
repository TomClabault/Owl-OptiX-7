#ifndef SHADER_H
#define SHADER_H

#include "optix_types.h"
#include "owl/common/math/vec.h"
#include "texture_types.h"

using namespace owl;

struct RayGenData
{
    OptixTraversableHandle scene;

    vec2i frame_buffer_size;
    uint32_t* frame_buffer;
    vec3f* accumulation_buffer;
    unsigned int frame_number;

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
