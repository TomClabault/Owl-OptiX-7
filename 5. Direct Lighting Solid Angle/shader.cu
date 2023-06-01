#include <device_launch_parameters.h>
#include <owl/owl_device.h>
#include <optix.h>

#include "cookTorrance.h"
#include "geometriesData.h"
#include "optix_device.h"
#include "owl/common/math/random.h"
#include "shader.h"

#define NUM_SAMPLE_PER_PIXEL 1

#define ENABLE_SKYBOX 1

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
    vec3f attenuation;
    vec3f emissive;

    struct
    {
        vec3f origin;
        vec3f direction;
        vec3f normal;
        vec3f albedo;
        vec3f specular_intensity;

        ScatterState state;

    } scatter;

    owl::common::LCG<4> random;
};

struct ShadowRayPrd
{
    bool obstructed = true;
};

inline vec3f __device__ uniform_point_on_triangle(PerRayData& prd, const vec3f& A, const vec3f& B, const vec3f& C)
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
//    float solid_angle_sum = 0.0f;
//    for (int i = 0; i < optixLaunchParams.emissive_triangles_info.count; i++)
//    {
//        int global_triangle_index = optixLaunchParams.emissive_triangles_info.emissive_triangles_indices[i];

//        vec3i& indices = optixLaunchParams.emissive_triangles_info.triangles_indices[global_triangle_index];

//        vec3f& A = optixLaunchParams.emissive_triangles_info.triangles_vertices[indices.x];
//        vec3f& B = optixLaunchParams.emissive_triangles_info.triangles_vertices[indices.y];
//        vec3f& C = optixLaunchParams.emissive_triangles_info.triangles_vertices[indices.z];

//        vec3f norm_A = normalize(A);
//        vec3f norm_B = normalize(B);
//        vec3f norm_C = normalize(C);

//        vec3f AB = norm_B - norm_A;
//        vec3f BC = norm_C - norm_B;
//        vec3f CA = norm_A - norm_C;

//        float solid_angle = dot(AB, -CA) + dot(BC, -AB) + dot(CA, -BC) - (float)M_PI;
//        solid_angle_sum += solid_angle;
//    }

//    int global_triangle_index_chosen;
//    float random_01 = prd.random();
//    float sum_so_far = 0.0f;
//    for (int i = 0; i < optixLaunchParams.emissive_triangles_info.count; i++)
//    {
//        int global_triangle_index = optixLaunchParams.emissive_triangles_info.emissive_triangles_indices[i];

//        vec3i& indices = optixLaunchParams.emissive_triangles_info.triangles_indices[global_triangle_index];

//        vec3f& A = optixLaunchParams.emissive_triangles_info.triangles_vertices[indices.x];
//        vec3f& B = optixLaunchParams.emissive_triangles_info.triangles_vertices[indices.y];
//        vec3f& C = optixLaunchParams.emissive_triangles_info.triangles_vertices[indices.z];

//        vec3f norm_A = normalize(A);
//        vec3f norm_B = normalize(B);
//        vec3f norm_C = normalize(C);

//        vec3f AB = norm_B - norm_A;
//        vec3f BC = norm_C - norm_B;
//        vec3f CA = norm_A - norm_C;

//        float solid_angle = dot(AB, -CA) + dot(BC, -AB) + dot(CA, -BC) - (float)M_PI;
//        solid_angle /= solid_angle_sum;//Normalized

//        sum_so_far += solid_angle;

//        if (sum_so_far > random_01)
//        {
//            global_triangle_index_chosen = global_triangle_index;
//            pdf = 1.0f / solid_angle;

//            break;
//        }
//    }

//    const vec3i& triangle_indices = optixLaunchParams.emissive_triangles_info.triangles_indices[global_triangle_index_chosen];
//    const vec3f& triangle_A = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.x];
//    const vec3f& triangle_B = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.y];
//    const vec3f& triangle_C = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.z];

//    vec3f point_on_light = uniform_point_on_triangle(prd, triangle_A, triangle_B, triangle_C);
//    vec3f shadow_ray_origin = prd.scatter.origin;
//    vec3f light_direction = normalize(point_on_light - prd.scatter.origin);

//    float dist = length(point_on_light - prd.scatter.origin);
//    ShadowRay shadow_ray(prd.scatter.origin, normalize(point_on_light - shadow_ray_origin), 1.0e-3f, dist -1.0e-4f);
//    ShadowRayPrd shadow_prd;
//    traceRay(optixLaunchParams.scene, shadow_ray, shadow_prd, OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
//                                                                  | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

//    int triangle_mat_index = optixLaunchParams.emissive_triangles_info.triangles_materials_indices[global_triangle_index_chosen];
//    vec3f light_color = optixLaunchParams.emissive_triangles_info.triangles_materials[triangle_mat_index].emissive;
//    float light_angle = dot(prd.scatter.normal, normalize(point_on_light - shadow_ray_origin));
//    return vec3f(!shadow_prd.obstructed) * light_angle * light_color;

    int random_emissive_triangle_index = prd.random() * optixLaunchParams.emissive_triangles_info.count;

    int global_triangle_index = optixLaunchParams.emissive_triangles_info.emissive_triangles_indices[random_emissive_triangle_index];

    const vec3i& triangle_indices = optixLaunchParams.emissive_triangles_info.triangles_indices[global_triangle_index];
    const vec3f& triangle_A = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.x];
    const vec3f& triangle_B = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.y];
    const vec3f& triangle_C = optixLaunchParams.emissive_triangles_info.triangles_vertices[triangle_indices.z];

    vec3f point_on_light = uniform_point_on_triangle(prd, triangle_A, triangle_B, triangle_C);
    vec3f shadow_ray_origin = prd.scatter.origin;
    vec3f light_direction = normalize(point_on_light - prd.scatter.origin);

    float dist = length(point_on_light - prd.scatter.origin);
    ShadowRay shadow_ray(prd.scatter.origin, normalize(point_on_light - shadow_ray_origin), 1.0e-3f, dist -1.0e-4f);
    ShadowRayPrd shadow_prd;
    traceRay(optixLaunchParams.scene, shadow_ray, shadow_prd, OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                                                                  | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

    int triangle_mat_index = optixLaunchParams.emissive_triangles_info.triangles_materials_indices[global_triangle_index];
    vec3f light_color = optixLaunchParams.emissive_triangles_info.triangles_materials[triangle_mat_index].emissive;
    float light_angle = max(0.0f, dot(prd.scatter.normal, normalize(point_on_light - shadow_ray_origin)));
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

    vec3f ray_origin = ray_gen_data.camera.position;
    vec3f ray_direction = normalize(ray_gen_data.camera.direction_00
                                    + ray_gen_data.camera.direction_dx * (prd.random() + pixel_ID.x) / (float)ray_gen_data.frame_buffer_size.x
                                    + ray_gen_data.camera.direction_dy * (prd.random() + pixel_ID.y) / (float)ray_gen_data.frame_buffer_size.y);

    ///// ----- Ray tracing ----- /////
    vec3f primary_hit_normal = vec3f(0.0f);//Used for the denoiser only
    vec3f primary_hit_albedo = vec3f(0.0f);//Used for the denoiser only

    vec3f ray_throughput = vec3f(1.0f);
    vec3f ray_color = vec3f(0.0f);
    int max_bounces = optixLaunchParams.mouse_moving ? 2 : optixLaunchParams.max_bounces;
    for (int depth = 0; depth < max_bounces; depth++)
    {
        Ray ray(ray_origin, ray_direction, 1.0e-3f, 1.0e10f);//Radiance ray
        traceRay(optixLaunchParams.scene, ray, prd, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

        if (prd.scatter.state == ScatterState::BOUNCED)
        {
            //Only settings the normal for the primary hit --> depth == 0
            if (depth == 0)
            {
                //Transforming the normal from world space to view space as this is
                //required by the denoiser specification
                primary_hit_normal = normalize(xfmNormal(ray_gen_data.camera.view_matrix, prd.scatter.normal));

                primary_hit_albedo = prd.scatter.albedo;
            }


            if (depth == 0)
                ray_color += prd.emissive;
            else //We allow emissive when there has been a specular bounce. If the bounce
                //was specular, 'specular_intensity' will not be vec3f(0.0f) and the emissive
                //will be accounted for
                ray_color += prd.emissive * prd.scatter.specular_intensity;

            ray_throughput *= prd.attenuation;

            //At least one light in the scene to sample
            if (optixLaunchParams.emissive_triangles_info.count > 0)
                ray_color += ray_throughput * direct_lighting(prd);

            ray_origin = prd.scatter.origin;
            ray_direction = prd.scatter.direction;
        }
        else if (prd.scatter.state == ScatterState::MISSED)
        {
            ray_color += prd.emissive * ray_throughput;

            break;
        }
    }
    ///// ----- Ray tracing ----- /////

    if (optixLaunchParams.frame_number == 1)
        optixLaunchParams.accumulation_buffer[pixel_index] = vec3f(0.0f);
    optixLaunchParams.accumulation_buffer[pixel_index] += ray_color;//clamp(ray_color, vec3f(0.0f), vec3f(1.0f));

    vec3f accumulated_color = optixLaunchParams.accumulation_buffer[pixel_index] / (float)optixLaunchParams.frame_number;
    accumulated_color = clamp(accumulated_color, vec3f(0.0f), vec3f(1.0f));

    //Gamma correction will be applied later, after the denoising, when the denoised buffer
    //is converted to a uint32 buffer
    float4 accumulated_color_float4 = make_float4(accumulated_color.x, accumulated_color.y, accumulated_color.z, 1.0f);
    vec3f primary_hit_normal_clamped = abs(primary_hit_normal);
    float4 primary_hit_normal_float4 = make_float4(primary_hit_normal_clamped.x, primary_hit_normal_clamped.y, primary_hit_normal_clamped.z, 1.0f);
    float4 primary_hit_albedo_float4 = make_float4(primary_hit_albedo.x, primary_hit_albedo.y, primary_hit_albedo.z, 1.0f);

    optixLaunchParams.float4_frame_buffer[pixel_index] = accumulated_color_float4;
    optixLaunchParams.normal_buffer[pixel_index] = primary_hit_normal_float4;
    optixLaunchParams.albedo_buffer[pixel_index] = primary_hit_albedo_float4;

//    float4 value = optixLaunchParams.float4_frame_buffer[pixel_index];
//    if (value.x < 0 || value.y < 0 || value.z < 0)
//        printf("INF ZERO FLOAT4: pixel (%d, %d) = %f %f %f\n", pixel_ID.x, pixel_ID.y, value.x, value.y, value.z);
//    else if (value.w < 0 || value.w > 1)
//        printf("WRONG ALPHA FLOAT4: pixel (%d, %d) = %f\n", pixel_ID.x, pixel_ID.y, value.w);
//    else if (value.x > 1 || value.y > 1 || value.z > 1)
//        printf("SUP ONE FLOAT4: pixel (%d, %d) = %f %f %f\n", pixel_ID.x, pixel_ID.y, value.x, value.y, value.z);

//    value = optixLaunchParams.normal_buffer[pixel_index];
//    if (value.x < 0 || value.y < 0 || value.z < 0)
//        printf("INF ZERO NORMAL: pixel (%d, %d) = %f %f %f\n", pixel_ID.x, pixel_ID.y, value.x, value.y, value.z);
//    else if (value.w < 0 || value.w > 1)
//        printf("WRONG ALPHA NORMAL: pixel (%d, %d) = %f\n", pixel_ID.x, pixel_ID.y, value.w);
//    else if (value.x > 1 || value.y > 1 || value.z > 1)
//        printf("SUP ONE NORMAL: pixel (%d, %d) = %f %f %f\n", pixel_ID.x, pixel_ID.y, value.x, value.y, value.z);

//    value = optixLaunchParams.albedo_buffer[pixel_index];
//    if (value.x < 0 || value.y < 0 || value.z < 0)
//        printf("INF ZERO ALBEDO: pixel (%d, %d) = %f %f %f\n", pixel_ID.x, pixel_ID.y, value.x, value.y, value.z);
//    else if (value.w < 0 || value.w > 1)
//        printf("WRONG ALPHA ALBEDO: pixel (%d, %d) = %f\n", pixel_ID.x, pixel_ID.y, value.w);
//    else if (value.x > 1 || value.y > 1 || value.z > 1)
//        printf("SUP ONE ALBEDO: pixel (%d, %d) = %f %f %f\n", pixel_ID.x, pixel_ID.y, value.x, value.y, value.z);

    //if (pixel_index == 0)
//        printf("(%f %f %f)\n", primary_hit_normal_float4.x, primary_hit_normal_float4.y, primary_hit_normal_float4.z);
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
    prd.scatter.albedo = material.albedo;
    prd.scatter.specular_intensity = prd.attenuation;
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

    //DEBUG
//    vec2i index = getLaunchIndex();
//    if (index.x == 0 && index.y == 0)
//    {
//        printf("Normal a: %f %f %f\n", normal_a.x, normal_a.y, normal_a.z);
//        printf("Normal b: %f %f %f\n", normal_b.x, normal_b.y, normal_b.z);
//        printf("Normal c: %f %f %f\n\n", normal_c.x, normal_c.y, normal_c.z);
//    }

    vec3f smooth_normal = normalize(u * normal_b
                                  + v * normal_c
                                  + (1 - u - v) * normal_a);

    vec3f albedo;
    //If there is a diffuse texture, we're going to sample it
    if (mat.diffuse_texture != 0)
    {
        vec3i uvs_indices = triangle_data.triangle_data.vertex_uvs_indices[primitive_index];
        vec2f texcoords_a = triangle_data.triangle_data.vertex_uvs[uvs_indices.x];
        vec2f texcoords_b = triangle_data.triangle_data.vertex_uvs[uvs_indices.y];
        vec2f texcoords_c = triangle_data.triangle_data.vertex_uvs[uvs_indices.z];

        vec2f tex_uv = u * texcoords_b
                     + v * texcoords_c
                     + (1 - u - v) * texcoords_a;

        albedo = tex2D<float4>(mat.diffuse_texture, tex_uv.u, tex_uv.v);
    }
    //No diffuse texture, taking the albedo of the material
    else
        albedo = mat.albedo;

    prd.emissive = mat.emissive;
    prd.attenuation = albedo;
    prd.scatter.origin = hit_point + 1.0e-5f * smooth_normal;
    prd.scatter.direction = ns_scatter(prd, hit_point, ray_direction, smooth_normal, mat.ns);
    prd.scatter.normal = smooth_normal;
    prd.scatter.albedo = albedo;
    prd.scatter.specular_intensity = mat.ns / 1000.0f;
    prd.scatter.state = ScatterState::BOUNCED;
}

OPTIX_MISS_PROGRAM(shadow_ray_miss)()
{
    ShadowRayPrd& prd = getPRD<ShadowRayPrd>();
    prd.obstructed = false;
}

OPTIX_MISS_PROGRAM(miss)()
{
    PerRayData& prd = getPRD<PerRayData>();

#if ENABLE_SKYBOX
    MissProgData miss_data = getProgramData<MissProgData>();

    vec3f ray_direction = optixGetWorldRayDirection();

    float u, v;
    u = 0.5 + atan2(ray_direction.z, ray_direction.x) / (2.0f * (float)M_PI);
    v = 0.5 + asin(ray_direction.y) / (float)M_PI;

    float4 texel = tex2D<float4>(miss_data.skysphere, u, v);
    vec3f skysphere_color = vec3f(texel.x, texel.y, texel.z);

    prd.emissive = skysphere_color;
    prd.scatter.state = ScatterState::MISSED;
#else
    prd.attenuation = vec3f(0.0f);
    prd.emissive = vec3f(0.0f);
    prd.scatter.state = ScatterState::MISSED;

    return;//No skysphere
#endif
}
