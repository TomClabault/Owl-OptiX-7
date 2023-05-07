#include "deviceCode.h"
#include "owl/owl_device.h"
#include "owl/common/math/LinearSpace.h"

#include <stdio.h>
#include <optix_device.h>

#define MAX_RECURSION_DEPTH 10

inline __device__ vec3f trace_ray(const Ray& ray, const RayGenData& rg_data, PerRayData& prd)
{
    traceRay(rg_data.scene, ray, prd);

    return prd.color;
}

extern "C" __global__ void __raygen__simpleRayGen()
{
    // Under the hood, OptiX maps rays generated in CUDA thread blocks to a pixel ID,
    // where the ID is a 2D vector, 0 to frame buffer width-1, 0 to height-1
    const vec2i pixel_id = getLaunchIndex();

    const RayGenData &ray_gen_data = getProgramData<RayGenData>();
    PerRayData prd;
    prd.random.init(pixel_id.x + ray_gen_data.fb_size.x * ray_gen_data.frame_number,
                    pixel_id.y + ray_gen_data.fb_size.y * ray_gen_data.frame_number);

    vec3f ray_origin = ray_gen_data.camera.position;
    vec3f ray_direction = normalize(ray_gen_data.camera.direction_00
                                  + ray_gen_data.camera.du * (pixel_id.x + prd.random()) / static_cast<float>(ray_gen_data.fb_size.x)
                                  + ray_gen_data.camera.dv * (pixel_id.y + prd.random()) / static_cast<float>(ray_gen_data.fb_size.y));

    vec3f pixel_color = 0.0f;

    Ray ray(ray_origin, ray_direction, 1.0e-3f, 1.0e10f);

    for (int i = 0; i < MAX_RECURSION_DEPTH; i++)
    {
        vec3f ray_color = trace_ray(ray, ray_gen_data, prd);

        if (prd.scatter.state == RAY_SCATTER_STATE::RAY_SCATTER_MISS || prd.scatter.state == RAY_SCATTER_STATE::RAY_SCATTER_TERMINATED)
        {
            //The ray missed or got absorbed by a material
            pixel_color = ray_color;
            break;
        }
        else if (prd.scatter.state == RAY_SCATTER_STATE::RAY_SCATTER_BOUNCE)
        {
            ray.origin = prd.scatter.origin;
            ray.direction = prd.scatter.direction;
        }
    }

    // find the frame buffer location (x + width * y)
    const int pixel_index = pixel_id.x + ray_gen_data.fb_size.x * pixel_id.y;

    if (ray_gen_data.frame_number == 1)
        ray_gen_data.frame_accumulation_buffer[pixel_index] = pixel_color;
    else
        ray_gen_data.frame_accumulation_buffer[pixel_index] += pixel_color;

    ray_gen_data.frame_buffer_ptr[pixel_index] = owl::make_rgba(ray_gen_data.frame_accumulation_buffer[pixel_index] / static_cast<float>(ray_gen_data.frame_number));
}

OPTIX_CLOSEST_HIT_PROGRAM(Triangle)()
{
    const TriangleGeomData& prog_data = getProgramData<TriangleGeomData>();
    PerRayData& prd = getPRD<PerRayData>();
    prd.scatter.state = RAY_SCATTER_STATE::RAY_SCATTER_TERMINATED;

    int primitive_index = optixGetPrimitiveIndex();

    float2 uv_coordinates = optixGetTriangleBarycentrics();
    float u = uv_coordinates.x, v = uv_coordinates.y;

    vec3f normal;
    if (prog_data.normals != nullptr)
    {
        normal = normalize(u * prog_data.normals[prog_data.normals_indices[primitive_index].y]
                         + v * prog_data.normals[prog_data.normals_indices[primitive_index].z]
                         + (1 - u - v) * prog_data.normals[prog_data.normals_indices[primitive_index].x]);
    }
    else
    {
        const vec3f A = prog_data.vertices[prog_data.indices[primitive_index].x];
        const vec3f B = prog_data.vertices[prog_data.indices[primitive_index].y];
        const vec3f C = prog_data.vertices[prog_data.indices[primitive_index].z];

        normal = normalize(cross(B - A, C - A));
    }

    vec3f ray_direction = optixGetWorldRayDirection();
    const vec3f view_direction = -ray_direction;

    vec3f& pixel_color = getPRD<vec3f>();

    unsigned int mat_index = prog_data.materials_indices[primitive_index];

    vec3f triangle_diffuse_color = prog_data.materials[mat_index].diffuse;
    // -------------- Diffuse -------------- //
    pixel_color = triangle_diffuse_color * (0.5f + 1.0f * dot(normal, view_direction));
    //if (triangle_specular_color.x != 0 || triangle_specular_color.y != 0 || triangle_specular_color.z != 0)
        //printf("%f, %f, %f | %f\n", triangle_specular_color.x, triangle_specular_color.y, triangle_specular_color.z, triangle_shininess);

    // -------------- Specular -------------- //
    vec3f triangle_specular_color = prog_data.materials[mat_index].specular;
    float triangle_shininess = prog_data.materials[mat_index].shininess;
    vec3f reflection_direction = normalize(ray_direction - 2.0f * dot(normal, ray_direction) * normal);
    pixel_color += triangle_specular_color * pow(max(dot(reflection_direction, view_direction), 0.0f), triangle_shininess);

    // -------------- Reflections -------------- //
    float triangle_reflection = prog_data.materials[mat_index].reflection_coefficient;
    if (triangle_reflection > 0.0f)
    {
        //We're declaring the ray as reflected, the 'recursion' will handle it
        prd.scatter.state = RAY_SCATTER_STATE::RAY_SCATTER_BOUNCE;

        //We don't need to 'push' the intersection point along the normal to avoid self intersection
        //here because we have a t_min = 1.0e-3 which limits the distance of intersection to be at
        //least 1.0e-3. Self intersection will not be counted this way
        float hit_t = optixGetRayTmax();
        vec3f ray_origin = optixGetWorldRayOrigin();
        vec3f inter_point = ray_origin + ray_direction * hit_t;
        prd.scatter.origin = inter_point;
        prd.scatter.direction = reflection_direction;
    }
}

OPTIX_MISS_PROGRAM(miss)()
{    
    MissProgData miss_prog_data = getProgramData<MissProgData>();

    PerRayData& prd = getPRD<PerRayData>();
    prd.scatter.state = RAY_SCATTER_STATE::RAY_SCATTER_MISS;

    vec3f ray_direction = optixGetWorldRayDirection();
    ray_direction = normalize(ray_direction);

    float u = 0.5 + atan2(ray_direction.z, ray_direction.x) / (2.0f * static_cast<float>(M_PI));
    float v = 0.5 + asin(ray_direction.y) / static_cast<float>(M_PI);

    vec4f skysphere_color = tex2D<float4>(miss_prog_data.skysphere, u, v);

    prd.color = vec3f(skysphere_color.x, skysphere_color.y, skysphere_color.z);
}
