#ifndef SHADER_H
#define SHADER_H

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/AffineSpace.h>

#include "emissiveTrianglesUtils.h"
#include "owl/common/math/random.h"
#include "shaderMaterials.h"
#include "texture_types.h"

#define RADIANCE_RAY 0
#define SHADOW_RAY 1

using namespace owl;

struct LaunchParams
{
    OptixTraversableHandle scene;

    vec3f* accumulation_buffer;
    float4* float4_frame_buffer;
    float4* normal_buffer;
    float4* albedo_buffer;
    unsigned int frame_number;

    bool mouse_moving;

    //Max recursion depth of the rays
    int max_bounces;

    //Material controling the apperance of the CookTorrance triangles
    //of the scene
    CookTorranceMaterial obj_material;

    //Information about the emissives triangles of the scene (how many,
    //indices, vertices, ...). Used for direct lighting in the ray_gen program
    EmissiveTrianglesInfo emissive_triangles_info;
};

enum ScatterState
{
    BOUNCED,
    EMITTED,
    MISSED,
    TERMINATED
};

struct PerRayData
{
    vec3f attenuation;
    vec3f emissive;

    struct
    {
        vec3f origin;
        vec3f direction;
        vec3f normal;
        vec3f albedo;
        float pdf;

        ScatterState state;

    } scatter;

    owl::common::LCG<4> random;
};

struct RayGenData
{
    vec2i frame_buffer_size;

    struct
    {
        vec3f position;
        vec3f direction_00;
        vec3f direction_dx;
        vec3f direction_dy;
        AffineSpace3f view_matrix;
    } camera;
};

struct MissProgData
{
    cudaTextureObject_t skysphere;
};

#endif
