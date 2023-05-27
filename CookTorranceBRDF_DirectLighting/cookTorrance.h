#ifndef COOK_TORRANCE_H
#define COOK_TORRANCE_H

#include "shaderMaterials.h"

#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

using namespace owl;

inline vec3f __device__ schlick_approximation(float cos_theta, const vec3f& F0)
{
    return F0 + (1.0f - F0) * powf((1.0f - cos_theta), 5.0f);
}

inline float __device__ d_GGX(float NdotH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    //alpha2 = max(alpha2, 0.05f);

    float denom = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;

    return alpha2 / max(((float)M_PI * denom * denom), 0.0001f);
}

inline float __device__ G1_GGX_schlick(float dot, float k)
{
    //max() to avoid division by zero
    return dot / max((dot * (1.0f - k) + k), 0.0001f);
}

inline float __device__ g_smith(float NdotV, float NdotL, float roughness)
{
    float alpha = roughness * roughness;
    float k = alpha * 0.5f;

    return G1_GGX_schlick(NdotL, k) * G1_GGX_schlick(NdotV, k);
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
    vec3f normal_distribution_term = d_GGX(NoH, material.roughness);
    vec3f geometry_term = g_smith(NoV, NoL, material.roughness);

    //max() to prevent division by zero
    vec3f specular_term = (fresnel_term * normal_distribution_term * geometry_term) / max((4.0f * NoV * NoL), 0.00001f);
    vec3f diffuse_term = material.albedo / (float)M_PI;
    diffuse_term *= (1.0f - material.metallic);//No diffuse part for the metals
    diffuse_term *= vec3f(1.0f) - fresnel_term;//Only the transmitted light contributes to the diffuse term

    return specular_term + diffuse_term;
}

//inline vec3f __device__ cook_torrance_brdf(const CookTorranceMaterial& material, vec3f lightDir, vec3f viewDir, vec3f normal)
//{
//    float k = 0.2f;
//    float F0 = 0.8f;

//    float NdotL = max(0.0f, dot(normal, lightDir));
//    float Rs = 0.0f;
//    if (NdotL > 0.0f)
//    {
//        vec3f H = normalize(lightDir + viewDir);
//        float NdotH = max(0.0f, dot(normal, H));
//        float NdotV = max(0.0f, dot(normal, viewDir));
//        float VdotH = max(0.0f, dot(lightDir, H));

//        // Fresnel reflectance
//        float F = pow(1.0f - VdotH, 5.0f);
//        F *= (1.0f - F0);
//        F += F0;

//        // Microfacet distribution by Beckmann
//        float m_squared = material.roughness * material.roughness;
//        float r1 = 1.0f / (4.0f * m_squared * pow(NdotH, 4.0f));
//        float r2 = (NdotH * NdotH - 1.0f) / (m_squared * NdotH * NdotH);
//        float D = r1 * exp(r2);

//        // Geometric shadowing
//        float two_NdotH = 2.0f * NdotH;
//        float g1 = (two_NdotH * NdotV) / VdotH;
//        float g2 = (two_NdotH * NdotL) / VdotH;
//        float G = min(1.0f, min(g1, g2));

//        Rs = (F * D * G) / ((float)M_PI * NdotL * NdotV);
//    }

//    vec3f specular_color = material.metallic * material.albedo + (1.0f - material.metallic) * vec3f(1.0f);
//    return material.albedo * vec3f(1.0f) * NdotL + vec3f(1.0f) * specular_color * NdotL * (k + Rs * (1.0f - k));
//}

#endif
