#ifndef SHADER_H
#define SHADER_H

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>

#include "materials.h"
#include "texture_types.h"

using namespace owl;

//Material from OBJ read
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

struct ObjTriangleGeomData
{
    vec3i* indices;
    vec3f* vertices;

    vec3i* vertex_normals_indices;
    vec3f* vertex_normals;

    Material* materials;
    int* materials_indices;
};

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

struct DiffuseTriangleGeomData
{
    vec3i* indices;
    vec3f* vertices;

    vec3f albedo;
};

struct MetalTriangleGeomData
{
    vec3i* indices;
    vec3f* vertices;

    vec3f albedo;
    float roughness;
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

struct ScatterInfo
{
    vec3f origin, target;

    vec3f attenuation;

    ScatterState state;
};

struct PerRayData
{
    owl::common::LCG<4> random;

    ScatterInfo scatter;
};

struct MissProgData
{
    cudaTextureObject_t skysphere;
};

#endif
