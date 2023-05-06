#pragma once

#include "owl/common/math/random.h"
#include "texture_types.h"
#include <owl/owl.h>
#include <owl/common/math/vec.h>

extern "C" char deviceCode_ptx[];

using namespace owl;

enum RAY_SCATTER_STATE
{
    RAY_SCATTER_TERMINATED,//Ray absorbed by a material
    RAY_SCATTER_BOUNCE,
    RAY_SCATTER_MISS,
};

struct PerRayData
{
    vec3f color;

    owl::common::LCG<4> random;

    struct
    {
        vec3f origin;
        vec3f direction;

        RAY_SCATTER_STATE state;
    } scatter;
};

struct RayGenData
{
    OptixTraversableHandle scene;

    unsigned int frame_number;
    vec3f* frame_accumulation_buffer;
    uint32_t* frame_buffer_ptr;
    vec2i fb_size;

    struct {
        vec3f position;
        vec3f direction_00;
        vec3f du;
        vec3f dv;
    } camera;
};

struct MissProgData
{
    cudaTextureObject_t skysphere;
};

struct Material
{
    vec3f ambient       = { 0, 0, 0 }; // Ka
    vec3f diffuse       = { 0, 0, 0 }; // Kd
    vec3f specular      = { 0, 0, 0 }; // Ks
    vec3f transmittance = { 0, 0, 0 }; // Kt
    vec3f emission      = { 0, 0, 0 }; // Ke
    float reflection_coefficient = 0.0f;
    float  shininess     = 1.0f;        // Ns
    float  ior           = 1.0f;        // Ni
    float  dissolve      = 1.0f;        // d
    int    illum         = 0;           // illum
};

struct TriangleGeomData
{
    vec3f* vertices;
    vec3i* indices;
    vec3f* normals;
    vec3i* normals_indices;

    int* materials_indices;
    Material* materials;
};
