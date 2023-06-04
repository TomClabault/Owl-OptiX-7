#ifndef COOK_TORRANCE_H
#define COOK_TORRANCE_H

#include "shader.h"
#include "shaderMaterials.h"

#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

using namespace owl;

inline vec3f __device__ schlick_approximation(float cos_theta, const vec3f& F0)
{
    return F0 + (1.0f - F0) * powf((1.0f - cos_theta), 5.0f);
}

inline float __device__ D_GGX(float NdotH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float denom = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;

    return alpha2 / max(((float)M_PI * denom * denom), 0.00001f);
}

inline float __device__ G1_GGX_schlick(float dot, float k)
{
    //max() to avoid division by zero
    return dot / max((dot * (1.0f - k) + k), 0.0001f);
}

inline float __device__ G2_Smith(float NoL, float NoV, float alpha)
{
    float NoL2 = NoL * NoL;
    float NoV2 = NoV * NoV;

    float numerator = 2 * NoL *NoV;
    float denom = NoV * sqrtf(alpha + (1.0f - alpha) * NoL2)
                + NoL * sqrtf(alpha + (1.0f - alpha) * NoV2);

    return numerator / max(denom, 0.0001f);
}

inline float __device__ g_smith(float NdotV, float NdotL, float roughness)
{
    float alpha = roughness * roughness;
    float k = alpha * 0.5f;

    //return G1_GGX_schlick(NdotL, k) * G1_GGX_schlick(NdotV, k);
    return G2_Smith(NdotL, NdotV, alpha);
}

inline vec3f __device__ cook_torrance_brdf(const CookTorranceMaterial& material, const vec3f& view_dir, const vec3f& outgoing_light_dir, const vec3f& normal)
{
    vec3f halfway_vector = normalize(view_dir + outgoing_light_dir);

    float NoV = clamp(dot(normal, view_dir),            0.0f, 1.0f);
    float NoL = clamp(dot(normal, outgoing_light_dir),  0.0f, 1.0f);
    float NoH = clamp(dot(normal, halfway_vector),      0.0f, 1.0f);
    float VoH = clamp(dot(view_dir, halfway_vector),    0.0f, 1.0f);

    //Fresnel reflectance when perpendicular to the surface for dielectrics.
    //Goes from 0.04 to 0.16 which is 4% to 16% depending on the reflectance
    vec3f F0 = (0.16f * (material.reflectance * material.reflectance));

    //F0 for metals is equal to the albedo.
    //We're going to lerp between F0 and 1 depending on
    //the metalness of the material. A fully metalic material
    //will thus have a F0 equal to its base color
    F0 = (1.0f - material.metallic) * F0 + material.metallic * material.albedo;

    vec3f fresnel_term = schlick_approximation(VoH, F0);//Reflected portion of the light (1 - transmitted)
    vec3f normal_distribution_term = D_GGX(NoH, material.roughness);
    vec3f geometry_term = g_smith(NoV, NoL, material.roughness);

    //max() to prevent division by zero
    vec3f specular_term = (fresnel_term * normal_distribution_term * geometry_term) / max((4.0f * NoV * NoL), 0.00001f);
    //if (specular_term.x > 16.0f || specular_term.y > 16.0f || specular_term.z > 16.0f)
        //printf("fresnel_term(%f %f %f), normal_distribution_term (%f %f %f), geometry (%f %f %f), NoH: %f\n", fresnel_term.x, fresnel_term.y, fresnel_term.z, normal_distribution_term.x, normal_distribution_term.y, normal_distribution_term.z, geometry_term.x, geometry_term.y, geometry_term.z, NoH);
    vec3f diffuse_term = material.albedo / (float)M_PI;
    //No diffuse part for the metals
    diffuse_term *= (1.0f - material.metallic);
    //Only the transmitted light contributes to the diffuse term
    diffuse_term *= vec3f(1.0f) - fresnel_term;

    return specular_term + diffuse_term;
}

inline vec3f __device__ cook_torrance_sample_direction(PerRayData& prd, float& pdf)
{

}

#endif
