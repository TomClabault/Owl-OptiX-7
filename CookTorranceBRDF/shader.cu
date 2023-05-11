#include <cuda_device_launch_parameters.h>

#include <owl/owl_device.h>

#include "cookTorrance.h"
#include "geometriesData.h"
#include "owl/common/math/random.h"
#include "shader.h"

#define NUM_SAMPLE_PER_PIXEL 1
#define MAX_RECURSION_DEPTH 15

using namespace owl;

enum ScatterState
{
    BOUNCED,
    MISSED,
    TERMINATED
};

struct PerRayData
{
    OptixTraversableHandle scene;

    vec3f color;

    int current_depth = 0;

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

vec3f __device__ trace_path(OptixTraversableHandle scene, PerRayData& prd, const Ray& ray)
{
    vec2i launch_index = getLaunchIndex();
    if (launch_index.x == 0 && launch_index.y == 0)
        printf("current depth: %d\n", prd.current_depth);
    if (prd.current_depth == MAX_RECURSION_DEPTH)
        return vec3f(1.0f);//Returning 1.0f because this is going to be multiplied so we don't want to multiply by vec3f(0.0f)

    prd.current_depth = prd.current_depth + 1;
    traceRay(scene, ray, prd);

    return prd.color;
}

OPTIX_RAYGEN_PROGRAM(ray_gen)()
{
    vec2i pixel_ID = getLaunchIndex();
    RayGenData ray_gen_data = getProgramData<RayGenData>();
    unsigned int pixel_index = pixel_ID.y * ray_gen_data.frame_buffer_size.x + pixel_ID.x;

    PerRayData prd;
    prd.random.init(pixel_ID.x * ray_gen_data.frame_number * ray_gen_data.frame_buffer_size.x * NUM_SAMPLE_PER_PIXEL,
                    pixel_ID.y * ray_gen_data.frame_number * ray_gen_data.frame_buffer_size.y * NUM_SAMPLE_PER_PIXEL);
    prd.scene = ray_gen_data.scene;

    vec3f ray_origin = ray_gen_data.camera.position;
    vec3f sum_samples_color = vec3f(0.0f);
    for (int sample = 0; sample < NUM_SAMPLE_PER_PIXEL; sample++)
    {
        vec3f ray_direction = normalize(ray_gen_data.camera.direction_00
                                        + ray_gen_data.camera.direction_dx * (prd.random() + pixel_ID.x) / (float)ray_gen_data.frame_buffer_size.x
                                        + ray_gen_data.camera.direction_dy * (prd.random() + pixel_ID.y) / (float)ray_gen_data.frame_buffer_size.y);

        Ray ray(ray_origin, ray_direction, 1.0e-3f, 1.0e10f);
        sum_samples_color += trace_path(ray_gen_data.scene, prd, ray);
    }

    vec3f averaged_color = clamp(sum_samples_color / (float)NUM_SAMPLE_PER_PIXEL, vec3f(0.0f), vec3f(1.0f));
//    vec3f gamma_corrected = vec3f(sqrtf(averaged_color.x),
//                                  sqrtf(averaged_color.y),
//                                  sqrtf(averaged_color.z));
    vec3f gamma_corrected = averaged_color;

    if (ray_gen_data.frame_number == 1)
        ray_gen_data.accumulation_buffer[pixel_index] = vec3f(0.0f);
    ray_gen_data.accumulation_buffer[pixel_index] += gamma_corrected;

    ray_gen_data.frame_buffer[pixel_index] = make_rgba(ray_gen_data.accumulation_buffer[pixel_index] / (float)ray_gen_data.frame_number);
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

    CookTorranceMaterial material = triangle_data.materials[triangle_data.materials_indices[primitive_index]];

    vec3f ray_origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();

    float hit_t = optixGetRayTmax();
    vec3f hit_point = ray_origin + hit_t * ray_direction;

    vec3f next_origin = hit_point + 1.0e-5f * smooth_normal;
    vec3f next_direction = normalize(random_in_hemisphere(prd, smooth_normal));

    Ray next_ray(next_origin, next_direction, 1.0e-3f, 1.0e10f);

    prd.color *= cook_torrance_brdf(material, -ray_direction, next_direction, smooth_normal);//BRDF
    prd.color *= trace_path(prd.scene, prd, next_ray);//Incoming light Li
    //prd.color *= dot(smooth_normal, next_direction);//Cosine weight
}

OPTIX_CLOSEST_HIT_PROGRAM(floor_triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();

    prd.color = vec3f(0.0f);
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

    prd.color = skysphere_color;
}
