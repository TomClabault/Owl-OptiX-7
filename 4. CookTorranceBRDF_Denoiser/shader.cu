#include <device_launch_parameters.h>
#include <owl/owl_device.h>
#include <optix.h>

#include "cookTorrance.h"
#include "geometriesData.h"
#include "optix_device.h"
#include "owl/common/math/random.h"
#include "shader.h"

#define NUM_SAMPLE_PER_PIXEL 1

using namespace owl;

__constant__ LaunchParams optixLaunchParams;

typedef owl::RayT<SHADOW_RAY, 2> ShadowRay;

enum ScatterState
{
    BOUNCED,
    EMITTED,
    MISSED,
    TERMINATED
};

struct PerRayData
{
    vec3f normal;//TODO remove

    vec3f attenuation;
    vec3f emissive;

    struct
    {
        vec3f origin;
        vec3f direction;
        vec3f normal;

        ScatterState state;
    } scatter;

    owl::common::LCG<4> random;
};

struct ShadowRayPrd
{
    bool obstructed = true;
};

inline vec3f __device__ uniform_point_in_triangle(PerRayData& prd, const vec3f& A, const vec3f& B, const vec3f& C)
{
    float u = prd.random();//[0.0, 1.0]
    float v = prd.random() * (1.0f - u);//[0.0, 1.0 - u] (because we don't want the sum of u + v to be > 1.0f
    float w = 1 - u - v;

    return u * A + v * B + w * C;
}

inline vec3f __device__ random_in_unit_sphere(PerRayData& prd)
{
    return normalize(vec3f(prd.random(), prd.random(), prd.random()) * 2 - 1);
}

inline vec3f __device__ random_in_hemisphere(PerRayData& prd, const vec3f& normal)
{
    vec3f random_in_sphere = random_in_unit_sphere(prd);
    if (dot(random_in_sphere, normal) < 0)//Below the surface
        return -random_in_sphere;
    else
        return random_in_sphere;
}

inline vec3f __device__ perfect_reflect_direction(const vec3f& incident_ray, const vec3f& normal)
{
    return normalize(incident_ray - 2 * dot(incident_ray, normal) * normal);
}

vec3f inline __device__ roughness_scatter(PerRayData& prd, const vec3f& hit_point, const vec3f& ray_direction, const vec3f& normal, float roughness)
{
    vec3f perfect_reflection = perfect_reflect_direction(ray_direction, normal);
    vec3f fuzzy_reflection = random_in_hemisphere(prd, normal);
    //Lerping between fuzzy target direction and perfect reflection
    return normalize((1.0f - roughness) * perfect_reflection + roughness * fuzzy_reflection);
}

vec3f inline __device__ ns_scatter(PerRayData& prd, const vec3f& hit_point, const vec3f& ray_direction, const vec3f& normal, float ns)
{
    vec3f perfect_reflection = perfect_reflect_direction(ray_direction, normal);
    vec3f fuzzy_reflection = random_in_hemisphere(prd, normal);
    //Lerping between fuzzy target direction and perfect reflection
    ns /= 1000.0f;
    return normalize((1.0f - ns) * fuzzy_reflection + ns * perfect_reflection);
}

vec3f inline __device__ direct_lighting(PerRayData& prd)
{
    //We're multiplying a uniform random number [0.0, 1.0] by (nb_emissive_triangles + 1) because
    //if we don't add +1 here, there's an infinitely small chance that we're going to sample
    //the very last emissive triangle (this would require the random number to be exactly
    //1.0 and this is not going to happen).
    int random_emissive_triangle_index = prd.random() * (optixLaunchParams.emissive_triangles_info.count + 1);

    const vec3i& triangle_indices = optixLaunchParams.emissive_triangles_info.triangles_indices[random_emissive_triangle_index];
    const vec3f& triangle_A = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.x];
    const vec3f& triangle_B = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.y];
    const vec3f& triangle_C = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.z];

    vec3f point_on_light = uniform_point_in_triangle(prd, triangle_A, triangle_B, triangle_C);
    vec3f shadow_ray_origin = prd.scatter.origin;
    vec3f light_direction = normalize(point_on_light - prd.scatter.origin);

    float dist = length(point_on_light - prd.scatter.origin);
    ShadowRay shadow_ray(prd.scatter.origin, normalize(point_on_light - shadow_ray_origin), 1.0e-3f, dist -1.0e-4f);
    ShadowRayPrd shadow_prd;
    traceRay(optixLaunchParams.scene, shadow_ray, shadow_prd, OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                                                                  | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

    int triangle_mat_index = optixLaunchParams.emissive_triangles_info.triangles_materials_indices[random_emissive_triangle_index];
    vec3f light_color = optixLaunchParams.emissive_triangles_info.triangles_materials[triangle_mat_index].emissive;
    float light_angle = dot(prd.scatter.normal, normalize(point_on_light - shadow_ray_origin));
    return vec3f(!shadow_prd.obstructed) * light_angle * light_color;
}

OPTIX_RAYGEN_PROGRAM(ray_gen)()
{
    vec2i pixel_ID = getLaunchIndex();
    RayGenData ray_gen_data = getProgramData<RayGenData>();
    unsigned int pixel_index = pixel_ID.y * ray_gen_data.frame_buffer_size.x + pixel_ID.x;

    //This is going to be useful to get the triangles when sampling direct lighting
    const SimpleObjTriangleData& lambertian_triangle_data = getProgramData<SimpleObjTriangleData>();

    PerRayData prd;
    prd.random.init(pixel_ID.x * optixLaunchParams.frame_number * ray_gen_data.frame_buffer_size.x * NUM_SAMPLE_PER_PIXEL,
                    pixel_ID.y * optixLaunchParams.frame_number * ray_gen_data.frame_buffer_size.y * NUM_SAMPLE_PER_PIXEL);
    prd.emissive = vec3f(0.0f);

    vec3f sum_samples_color = vec3f(0.0f);
    vec3f ray_origin = ray_gen_data.camera.position;
    vec3f ray_direction = normalize(ray_gen_data.camera.direction_00
                + ray_gen_data.camera.direction_dx * (prd.random() + pixel_ID.x) / (float)ray_gen_data.frame_buffer_size.x
                + ray_gen_data.camera.direction_dy * (prd.random() + pixel_ID.y) / (float)ray_gen_data.frame_buffer_size.y);
    for (int sample = 0; sample < NUM_SAMPLE_PER_PIXEL; sample++)
    {
        vec3f ray_attenuation = vec3f(1.0f);
        vec3f ray_color = vec3f(0.0f);
        for (int depth = 0; depth < optixLaunchParams.max_bounces; depth++)
        {
            Ray ray(ray_origin, ray_direction, 1.0e-3f, 1.0e10f);//Radiance ray
            traceRay(optixLaunchParams.scene, ray, prd, OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES | OPTIX_RAY_FLAG_DISABLE_ANYHIT);

            if (prd.scatter.state == ScatterState::BOUNCED)
            {
                ray_attenuation *= prd.attenuation;

                if (depth == 0)
                    ray_color += prd.emissive;
                ray_color += ray_attenuation * direct_lighting(prd);

                ray_origin = prd.scatter.origin;
                ray_direction = prd.scatter.direction;
            }
            else if (prd.scatter.state == ScatterState::MISSED)
                break;
        }

        sum_samples_color += ray_color;
    }

    vec3f averaged_color = sum_samples_color / (float)NUM_SAMPLE_PER_PIXEL;

    if (optixLaunchParams.frame_number == 1)
        optixLaunchParams.accumulation_buffer[pixel_index] = vec3f(0.0f);
    optixLaunchParams.accumulation_buffer[pixel_index] += averaged_color;

    vec3f accumulated_color = optixLaunchParams.accumulation_buffer[pixel_index] / (float)optixLaunchParams.frame_number;
    accumulated_color = clamp(accumulated_color, vec3f(0.0f), vec3f(1.0f));

    //Gamma correction will be applied later, after the denoising, when the denoised buffer
    //is converted to a uint32 buffer
    float4 accumulated_color_float4 = make_float4(accumulated_color.x, accumulated_color.y, accumulated_color.z, 1.0f);

    optixLaunchParams.float4_frame_buffer[pixel_index] = accumulated_color_float4;
}

OPTIX_CLOSEST_HIT_PROGRAM(cook_torrance_obj_triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();
    const CookTorranceTriangleData& triangle_data = getProgramData<CookTorranceTriangleData>();

    int primitive_index = optixGetPrimitiveIndex();
    float2 uv = optixGetTriangleBarycentrics();
    float u = uv.x, v = uv.y;

    //Smooth normal
    vec3i normal_indices = triangle_data.triangle_data.vertex_normals_indices[primitive_index];
    vec3f normal_a = triangle_data.triangle_data.vertex_normals[normal_indices.x];
    vec3f normal_b = triangle_data.triangle_data.vertex_normals[normal_indices.y];
    vec3f normal_c = triangle_data.triangle_data.vertex_normals[normal_indices.z];
    vec3f smooth_normal = normalize(u * normal_b
                                  + v * normal_c
                                  + (1 - u - v) * normal_a);

    CookTorranceMaterial material = optixLaunchParams.obj_material;

    vec3f ray_origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();

    float hit_t = optixGetRayTmax();
    vec3f hit_point = ray_origin + hit_t * ray_direction;

    prd.scatter.origin = hit_point + smooth_normal * 1.0e-5f;
    prd.scatter.direction = roughness_scatter(prd, hit_point, ray_direction, smooth_normal, material.roughness);
    prd.attenuation = cook_torrance_brdf(material, -normalize(ray_direction), prd.scatter.direction, smooth_normal);
    prd.emissive = vec3f(0.0f);
    prd.scatter.state = ScatterState::BOUNCED;

    prd.scatter.normal = smooth_normal;
}

OPTIX_CLOSEST_HIT_PROGRAM(obj_triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();
    SimpleObjTriangleData triangle_data = getProgramData<SimpleObjTriangleData>();

    int primitive_index = optixGetPrimitiveIndex();
    int material_index = triangle_data.materials_indices[primitive_index];
    SimpleObjMaterial mat = triangle_data.materials[material_index];

    float hit_t = optixGetRayTmax();
    vec3f ray_origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();
    vec3f hit_point = ray_origin + hit_t * ray_direction;

    float2 uv = optixGetTriangleBarycentrics();
    float u = uv.x, v = uv.y;

    vec3i normal_indices = triangle_data.triangle_data.vertex_normals_indices[primitive_index];
    vec3f normal_a = triangle_data.triangle_data.vertex_normals[normal_indices.x];
    vec3f normal_b = triangle_data.triangle_data.vertex_normals[normal_indices.y];
    vec3f normal_c = triangle_data.triangle_data.vertex_normals[normal_indices.z];
    vec3f smooth_normal = normalize(u * normal_b
                                    + v * normal_c
                                    + (1 - u - v) * normal_a);

    prd.emissive = mat.emissive;
    prd.attenuation = mat.albedo;
    prd.scatter.origin = hit_point + 1.0e-5f * smooth_normal;
    prd.scatter.direction = ns_scatter(prd, hit_point, ray_direction, smooth_normal, mat.ns);
    prd.scatter.state = ScatterState::BOUNCED;
    prd.scatter.normal = smooth_normal;
}

OPTIX_MISS_PROGRAM(shadow_ray_miss)()
{
    ShadowRayPrd& prd = getPRD<ShadowRayPrd>();
    prd.obstructed = false;
}

OPTIX_MISS_PROGRAM(miss)()
{
    PerRayData& prd = getPRD<PerRayData>();
    prd.attenuation = vec3f(0.0f);
    prd.emissive = vec3f(0.0f);
    prd.scatter.state = ScatterState::MISSED;

    return;//No skysphere

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
