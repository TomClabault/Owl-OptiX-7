#include <device_launch_parameters.h>
#include <owl/owl_device.h>
#include <optix.h>

#include "cookTorrance.h"
#include "geometriesData.h"
#include "optix_device.h"
#include "owl/common/math/random.h"
#include "shader.h"

#define NUM_SAMPLE_PER_PIXEL 1
#define MAX_RECURSION_DEPTH 1500

using namespace owl;

__constant__ LaunchParams optixLaunchParams;

enum ScatterState
{
    BOUNCED,
    MISSED,
    TERMINATED
};

struct PerRayData
{
    vec3f attenuation;

    struct
    {
        vec3f origin;
        vec3f direction;

        ScatterState state;
    } scatter;

    owl::common::LCG<4> random;
};

vec3f __device__ random_in_unit_sphere(PerRayData& prd)
{
    return normalize(vec3f(prd.random(), prd.random(), prd.random()) * 2 - 1);
}

vec3f __device__ random_in_hemisphere(PerRayData& prd, const vec3f& normal)
{
    vec3f random_in_sphere = random_in_unit_sphere(prd);
    if (dot(random_in_sphere, normal) < 0)//Below the surface
        return -random_in_sphere;
    else
        return random_in_sphere;
}

vec3f __device__ perfect_reflect_direction(const vec3f& incident_ray, const vec3f& normal)
{
    return normalize(incident_ray - 2 * dot(incident_ray, normal) * normal);
}

void inline __device__ metal_scatter(PerRayData& prd, const vec3f& hit_point, const vec3f& ray_direction, const vec3f& normal, float roughness, const vec3f& albedo)
{
    prd.scatter.origin = hit_point + normal * 1.0e-4f;
    vec3f perfect_reflection_target = perfect_reflect_direction(ray_direction, normal) + hit_point;
    //Lerping between fuzzy target direction and perfect reflection
    prd.scatter.direction = normalize((1.0f - roughness) * perfect_reflection_target + (roughness) * (random_in_hemisphere(prd, normal) + hit_point) - hit_point);
    prd.attenuation = albedo;

    prd.scatter.state = ScatterState::BOUNCED;
}

OPTIX_RAYGEN_PROGRAM(ray_gen)()
{
    vec2i pixel_ID = getLaunchIndex();
    RayGenData ray_gen_data = getProgramData<RayGenData>();
    unsigned int pixel_index = pixel_ID.y * ray_gen_data.frame_buffer_size.x + pixel_ID.x;

    PerRayData prd;
    prd.random.init(pixel_ID.x * optixLaunchParams.frame_number * ray_gen_data.frame_buffer_size.x * NUM_SAMPLE_PER_PIXEL,
                    pixel_ID.y * optixLaunchParams.frame_number * ray_gen_data.frame_buffer_size.y * NUM_SAMPLE_PER_PIXEL);

    vec3f sum_samples_color = vec3f(0.0f);
    vec3f ray_origin = ray_gen_data.camera.position;
    vec3f ray_direction = normalize(ray_gen_data.camera.direction_00
                + ray_gen_data.camera.direction_dx * (prd.random() + pixel_ID.x) / (float)ray_gen_data.frame_buffer_size.x
                + ray_gen_data.camera.direction_dy * (prd.random() + pixel_ID.y) / (float)ray_gen_data.frame_buffer_size.y);
    for (int sample = 0; sample < NUM_SAMPLE_PER_PIXEL; sample++)
    {
        vec3f sample_color = vec3f(1.0f);
        for (int depth = 0; depth < MAX_RECURSION_DEPTH; depth++)
        {
            Ray ray(ray_origin, ray_direction, 1.0e-3f, 1.0e10f);
            traceRay(optixLaunchParams.scene, ray, prd);

            sample_color *= prd.attenuation;

            if (prd.scatter.state == ScatterState::BOUNCED)
            {
                ray_origin = prd.scatter.origin;
                ray_direction = prd.scatter.direction;
            }
            else if (prd.scatter.state == ScatterState::MISSED || prd.scatter.state == ScatterState::TERMINATED)
                break;
        }

        sum_samples_color += sample_color;
    }

    vec3f averaged_color = clamp(sum_samples_color / (float)NUM_SAMPLE_PER_PIXEL, vec3f(0.0f), vec3f(1.0f));

    if (optixLaunchParams.frame_number == 1)
        optixLaunchParams.accumulation_buffer[pixel_index] = vec3f(0.0f);
    optixLaunchParams.accumulation_buffer[pixel_index] += averaged_color;

    vec3f accumulated_color = optixLaunchParams.accumulation_buffer[pixel_index] / (float)optixLaunchParams.frame_number;
    vec3f gamma_corrected = vec3f(sqrtf(accumulated_color.x),
                                  sqrtf(accumulated_color.y),
                                  sqrtf(accumulated_color.z));

    ray_gen_data.frame_buffer[pixel_index] = make_rgba(gamma_corrected);
}

OPTIX_CLOSEST_HIT_PROGRAM(cook_torrance_obj_triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();
    const CookTorranceTriangleData& triangle_data = getProgramData<CookTorranceTriangleData>();

    int primitive_index = optixGetPrimitiveIndex();
    float2 uv = optixGetTriangleBarycentrics();
    float u = uv.x, v = uv.y;

    vec3i normal_indices = triangle_data.triangle_data.vertex_normals_indices[primitive_index];
    vec3f normal_a = triangle_data.triangle_data.vertex_normals[normal_indices.x];
    vec3f normal_b = triangle_data.triangle_data.vertex_normals[normal_indices.y];
    vec3f normal_c = triangle_data.triangle_data.vertex_normals[normal_indices.z];
    vec3f smooth_normal = normalize(u * normal_b
                                  + v * normal_c
                                    + (1 - - u - v) * normal_a);

    CookTorranceMaterial material = optixLaunchParams.obj_material;

    vec3f ray_origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();

    float hit_t = optixGetRayTmax();
    vec3f hit_point = ray_origin + hit_t * ray_direction;

    metal_scatter(prd, hit_point, ray_direction, smooth_normal, material.roughness, material.albedo);
    prd.attenuation = cook_torrance_brdf(material, -normalize(ray_direction), prd.scatter.direction, smooth_normal);//BRDF
    prd.scatter.state = ScatterState::BOUNCED;
}

OPTIX_CLOSEST_HIT_PROGRAM(floor_triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();

    //This the floor, we know the normal
    vec3f normal = vec3f(0.0f, 1.0f, 0.0f);

    float hit_t = optixGetRayTmax();
    vec3f ray_origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();
    vec3f hit_point = ray_origin + hit_t * ray_direction;

    prd.attenuation = vec3f(0.8f, 0.8f, 0.8f);
    prd.scatter.origin = hit_point + 1.0e-5f * normal;
    prd.scatter.direction = normalize(random_in_hemisphere(prd, normal));
    prd.scatter.state = ScatterState::BOUNCED;
}

OPTIX_MISS_PROGRAM(miss)()
{
    PerRayData& prd = getPRD<PerRayData>();
    MissProgData miss_data = getProgramData<MissProgData>();

    vec3f ray_direction = optixGetWorldRayDirection();

    float u, v;
    u = 0.5 + atan2(ray_direction.z, ray_direction.x) / (2.0f * (float)M_PI);
    v = 0.5 + asin(ray_direction.y) / (float)M_PI;

    float4 texel = tex2D<float4>(miss_data.skysphere, u, v);
    vec3f skysphere_color = vec3f(texel.x, texel.y, texel.z);

    prd.attenuation = skysphere_color;
    prd.scatter.state = ScatterState::MISSED;
}
