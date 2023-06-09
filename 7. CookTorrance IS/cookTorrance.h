#ifndef COOK_TORRANCE_H
#define COOK_TORRANCE_H

#include "shader.h"
#include "shaderMaterials.h"

#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

using namespace owl;

inline vec3f __device__ schlick_approximation(float cos_theta, const vec3f& F0)
{
    return F0 + (1.0f - F0) * powf(clamp((1.0f - cos_theta), 0.0f, 1.0f), 5.0f);
}

//inline float __device__ GGX_NDF(float NdotH, float roughness)
//{
//    float roughness2 = roughness * roughness;

//    float denom = NdotH * NdotH * (roughness2 - 1.0f) + 1.0f;

//    return roughness2 / max(((float)M_PI * denom * denom), 0.00001f);
//}

inline float __device__ GGX_NDF(float NoH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NoH * NoH;

    float num = alpha2;
    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    denom = (float)M_PI * denom * denom;

    return num / denom;
}

//inline float __device__ G1_GGX_schlick(float dot, float k)
//{
//    //max() to avoid division by zero
//    return dot / max((dot * (1.0f - k) + k), 0.0001f);
//}

//inline float __device__ geometry_Smith(float NdotV, float NdotL, float roughness)
//{
//    float alpha = roughness * roughness;
//    float k = alpha * 0.5f;

//    return G1_GGX_schlick(NdotL, k) * G1_GGX_schlick(NdotV, k);
//}

inline float __device__ GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

inline float __device__ GeometrySmith(vec3f N, vec3f V, vec3f L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

inline vec3f __device__ cook_torrance_brdf(const CookTorranceMaterial& material, const vec3f& view_dir, const vec3f& outgoing_light_dir, const vec3f& normal)
{
    //    vec3f normal_distribution_term = DistributionGGX(normal, halfway_vector, material.roughness);
    //    //vec3f normal_distribution_term = GGX_NDF(NoH, material.roughness);
    //    //vec3f geometry_term = geometry_Smith(NoV, NoL, material.roughness);
    //    vec3f geometry_term = GeometrySmith(normal, view_dir, outgoing_light_dir, material.roughness);

    //    //max() to prevent division by zero
    //    float denominator = 4.0 * max(dot(normal, view_dir), 0.0f) * max(dot(normal, outgoing_light_dir), 0.0f)  + 0.0001f;
    //    vec3f specular_term = (fresnel_term * normal_distribution_term * geometry_term) / denominator;// max((4.0f * NoV * NoL), 0.00001f);
    //    vec3f diffuse_term = material.albedo / (float)M_PI;
    //    //No diffuse part for the metals
    //    diffuse_term *= (1.0f - material.metallic);
    //    //Only the transmitted light contributes to the diffuse term
    //    diffuse_term *= vec3f(1.0f) - fresnel_term;

    //    return specular_term + diffuse_term;

    vec3f halfway_vector = normalize(view_dir + outgoing_light_dir);

    float NoV = max(dot(normal, view_dir),            0.0f);
    float NoL = max(dot(normal, outgoing_light_dir),  0.0f);
    float NoH = max(dot(normal, halfway_vector),      0.0f);
    float VoH = max(dot(view_dir, halfway_vector),    0.0f);

    vec3f F0 = (0.16f * (material.reflectance * material.reflectance));
    //F0 for metals is equal to the albedo.
    //We're going to lerp between the albedo and 0.04
    //(previous value of F0) depending on
    //the metalness of the material. A fully metalic material
    //will thus have a F0 equal to its albedo
    F0 = (1.0f - material.metallic) * F0 + material.metallic * material.albedo;

    //Reflected portion of the light (1 - transmitted)
    vec3f F = schlick_approximation(VoH, F0);
    float NDF = GGX_NDF(NoH, material.roughness);
    float G = GeometrySmith(normal, view_dir, outgoing_light_dir, material.roughness);

    vec3f kS = F;
    vec3f kD = vec3f(1.0f) - kS;
    kD *= 1.0 - material.metallic;

    vec3f numerator = NDF * G * F;
    //+0.00001f to avoid dividng by zero if the dot products are 0
    float denominator = 4.0f * NoV * NoL + 0.0001f;
    vec3f specular = numerator / denominator;

    // add to outgoing radiance Lo
    return (kD * material.albedo / (float)M_PI + specular);

//    vec3f halfway_vector = normalize(view_dir + outgoing_light_dir);

//    float NoV = clamp(dot(normal, view_dir),            0.0f, 1.0f);
//    float NoL = clamp(dot(normal, outgoing_light_dir),  0.0f, 1.0f);
//    float NoH = clamp(dot(normal, halfway_vector),      0.0f, 1.0f);
//    float VoH = clamp(dot(view_dir, halfway_vector),    0.0f, 1.0f);

//    //Fresnel reflectance when perpendicular to the surface for dielectrics.
//    //Goes from 0.04 to 0.16 which is 4% to 16% depending on the reflectance
//    vec3f F0 = (0.16f * (material.reflectance * material.reflectance));

//    //F0 for metals is equal to the albedo.
//    //We're going to lerp between the albedo and 0.04
//    //(previous value of F0) depending on
//    //the metalness of the material. A fully metalic material
//    //will thus have a F0 equal to its albedo
//    F0 = (1.0f - material.metallic) * F0 + material.metallic * material.albedo;

//    vec3f fresnel_term = schlick_approximation(VoH, F0);//Reflected portion of the light (1 - transmitted)
//    vec3f normal_distribution_term = DistributionGGX(normal, halfway_vector, material.roughness);
//    //vec3f normal_distribution_term = GGX_NDF(NoH, material.roughness);
//    //vec3f geometry_term = geometry_Smith(NoV, NoL, material.roughness);
//    vec3f geometry_term = GeometrySmith(normal, view_dir, outgoing_light_dir, material.roughness);

//    //max() to prevent division by zero
//    float denominator = 4.0 * max(dot(normal, view_dir), 0.0f) * max(dot(normal, outgoing_light_dir), 0.0f)  + 0.0001f;
//    vec3f specular_term = (fresnel_term * normal_distribution_term * geometry_term) / denominator;// max((4.0f * NoV * NoL), 0.00001f);
//    vec3f diffuse_term = material.albedo / (float)M_PI;
//    //No diffuse part for the metals
//    diffuse_term *= (1.0f - material.metallic);
//    //Only the transmitted light contributes to the diffuse term
//    diffuse_term *= vec3f(1.0f) - fresnel_term;

//    return specular_term + diffuse_term;
}

/**
 * @brief Transforms a vector (not a normal) from a local space oriented around
 * normal n to world space
 * Implementation from: Frisvad, "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization",
 * @param normal The normal representing the z-axis of the local space
 * @param local_to_transform The local direction to transform to world space
 * @return The world space direction
 */
inline vec3f __device__ local_to_world(vec3f& n, vec3f& local)
{
    vec3f b1, b2;

    if(n.z < -0.9999999f) // Handle the singularity
    {
        b1 = vec3f ( 0.0f , -1.0f , 0.0f );
        b2 = vec3f ( -1.0f , 0.0f , 0.0f );
    }
    else
    {
        const float a = 1.0f /(1.0f + n.z );
        const float b = -n.x*n .y*a ;
        b1 = vec3f (1.0f - n .x*n. x*a , b , -n .x );
        b2 = vec3f (b , 1.0f - n .y*n. y*a , -n .y );
    }

    return local.x * b1 + local.y * b2 + local.z * n;
}

inline vec3f __device__ world_to_local(vec3f& n, vec3f& world)
{
    vec3f b1, b2;

    if(n.z < -0.9999999f) // Handle the singularity
    {
        b1 = vec3f ( 0.0f , -1.0f , 0.0f );
        b2 = vec3f ( -1.0f , 0.0f , 0.0f );
    }
    else
    {
        const float a = 1.0f /(1.0f + n.z );
        const float b = -n.x*n .y*a ;
        b1 = vec3f (1.0f - n .x*n. x*a , b , -n .x );
        b2 = vec3f (b , 1.0f - n .y*n. y*a , -n .y );
    }

    return vec3f(dot(world, b1), dot(world, b2), dot(world, n));
}

////Following https://agraphicsguynotes.com/posts/sample_microfacet_brdf/
inline vec3f __device__ cook_torrance_sample_direction(PerRayData& prd, vec3f& view_direction, vec3f& surface_normal, float roughness, float& pdf)
{
    vec2i index = getLaunchIndex();

    float roughness2 = roughness * roughness;

    float rand_theta = prd.random();

    float theta = atan(roughness * sqrtf(rand_theta / (1.0f - rand_theta)));
    float phi = prd.random() * 2.0f * M_PI;//Isotropic case

    float cos_theta = clamp(cos(theta), 0.0f, 1.0f);
    float sin_theta = clamp(sin(theta), 0.0f, 1.0f);
    float cos_phi = clamp(cos(phi), 0.0f, 1.0f);
    float sin_phi = clamp(sin(phi), 0.0f, 1.0f);

    float denom = (roughness2 - 1.0f) * cos_theta * cos_theta + 1.0f;
    float pdf_half_vector = (2 * roughness2 * cos_theta * sin_theta) / (denom * denom) + 0.0001f;

    vec3f sampled_normal_local = vec3f( cos_phi * sin_theta,
                                        sin_phi * sin_theta,
                                        cos_theta);

    vec3f view_direction_local = world_to_local(surface_normal, view_direction);

    pdf = pdf_half_vector / (4 * fabs(dot(view_direction_local, sampled_normal_local))) + 0.00001f;

    vec3f surface_normal_local = vec3f(0.0f, 0.0f, 1.0f);
    vec3f reflected_direction_local = normalize(view_direction_local - 2 * dot(view_direction_local, sampled_normal_local) * sampled_normal_local);
    if (dot(reflected_direction_local, surface_normal_local) < 0)
        reflected_direction_local = -reflected_direction_local;

    //TODO normalize ?
    return local_to_world(surface_normal, reflected_direction_local);
}

#endif
