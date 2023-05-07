#include "shader.h"
#include "owl/owl_device.h"
#include "owl/common/math/LinearSpace.h"

#include <stdio.h>
#include <optix_device.h>

#define NUM_SAMPLE_PER_PIXEL 1
#define MAX_RECURSION_DEPTH 16

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

template <typename SphereGeomType>
void __device__ bounds_program(const void* primitive_data, box3f& bounds, int primitive_index)
{
    SphereGeomType& geometry_data = *(SphereGeomType*)primitive_data;
    Sphere sphere = geometry_data.primitives[primitive_index].sphere;

    bounds.extend(sphere.center - sphere.radius);
    bounds.extend(sphere.center + sphere.radius);
}

OPTIX_BOUNDS_PROGRAM(lambertian_spheres)(const void* primitive_data, box3f& bounds, int primitive_index)
{ bounds_program<LambertianSpheresGeometryData>(primitive_data, bounds, primitive_index); }

template <typename SphereGeomType>
void __device__ sphere_intersect()
{
    int primitive_index = optixGetPrimitiveIndex();
    const auto& sphere_data = getProgramData<SphereGeomType>().primitives[primitive_index];

    const vec3f org  = optixGetObjectRayOrigin();
    const vec3f dir  = optixGetObjectRayDirection();
    float hit_t      = optixGetRayTmax();
    const float tmin = optixGetRayTmin();

    const vec3f oc = org - sphere_data.sphere.center;
    const float a = dot(dir,dir);
    const float b = dot(oc, dir);
    const float c = dot(oc, oc) - sphere_data.sphere.radius * sphere_data.sphere.radius;
    const float discriminant = b * b - a * c;

    if (discriminant < 0.f)
        return;

    float temp;

    temp = (-b - sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin)
        hit_t = temp;

    temp = (-b + sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin)
        hit_t = temp;

    if (hit_t < optixGetRayTmax())
        optixReportIntersection(hit_t, 0);
}

OPTIX_INTERSECT_PROGRAM(lambertian_spheres)()
{ sphere_intersect<LambertianSpheresGeometryData>(); }

OPTIX_CLOSEST_HIT_PROGRAM(lambertian_spheres)()
{
    int primitive_index = optixGetPrimitiveIndex();
    const LambertianSphere& sphere_data = getProgramData<LambertianSpheresGeometryData>().primitives[primitive_index];

    PerRayData& prd = getPRD<PerRayData>();

    float t_hit = optixGetRayTmax();
    vec3f origin = optixGetWorldRayOrigin();
    vec3f direction = optixGetWorldRayDirection();
    vec3f hit_point = origin + t_hit * direction;
    vec3f normal = normalize(hit_point - sphere_data.sphere.center);

    prd.scatter.origin = hit_point + normal * 1.0e-4f;//Shifting the hit_point to avoid self intersections
    prd.scatter.target = random_in_hemisphere(prd, normal) + hit_point;
    prd.scatter.state = ScatterState::BOUNCED;
}

OPTIX_CLOSEST_HIT_PROGRAM(triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();

    float t_hit = optixGetRayTmax();
    vec3f origin = optixGetWorldRayOrigin();
    vec3f direction = optixGetWorldRayDirection();
    vec3f hit_point = origin + t_hit * direction;
    vec3f normal = vec3f(0.0, 1.0f, 0.0f);//This is the floor so we know the normal

    prd.scatter.origin = hit_point + normal * 1.0e-4f;
    prd.scatter.target = random_in_hemisphere(prd, normal) + hit_point;
    prd.scatter.state = ScatterState::BOUNCED;
}

OPTIX_MISS_PROGRAM(miss)()
{
    const MissProgData& miss_prog_data = getProgramData<MissProgData>();
    PerRayData& prd = getPRD<PerRayData>();

    vec3f ray_direction = optixGetWorldRayDirection();
    ray_direction = normalize(ray_direction);

    float u = 0.5 + atan2(ray_direction.z, ray_direction.x) / (2 * static_cast<float>(M_PI));
    float v = 0.5 + asin(ray_direction.y) / static_cast<float>(M_PI);

    float4 skysphere_color = tex2D<float4>(miss_prog_data.skysphere, u, v);
    prd.color = vec3f(skysphere_color.x, skysphere_color.y, skysphere_color.z);
    prd.scatter.state = ScatterState::MISSED;
}

extern "C" __global__ void __raygen__ray_gen()
{
    const RayGenData& ray_gen_data = getProgramData<RayGenData>();

    const vec2i pixel_ID = getLaunchIndex();

    PerRayData prd;
    prd.random.init(pixel_ID.x * ray_gen_data.frame_buffer_size.x * NUM_SAMPLE_PER_PIXEL * ray_gen_data.frame_number,
                    pixel_ID.y * ray_gen_data.frame_buffer_size.y * NUM_SAMPLE_PER_PIXEL * ray_gen_data.frame_number);


    vec3f final_color = vec3f(0.0f);

    Ray ray;
    for (int sample = 0; sample < NUM_SAMPLE_PER_PIXEL; sample++)
    {
        vec3f direction = normalize(ray_gen_data.camera.direction_00 + ray_gen_data.camera.direction_dx * (pixel_ID.x + prd.random()) / (float)ray_gen_data.frame_buffer_size.x
                                  + ray_gen_data.camera.direction_dy * (pixel_ID.y + prd.random()) / (float)ray_gen_data.frame_buffer_size.y);
        ray.origin = ray_gen_data.camera.position;
        ray.direction = direction;

        float attenuation = 1.0f;

        for (int recurse = 0; recurse < MAX_RECURSION_DEPTH; recurse++)
        {
            traceRay(ray_gen_data.scene, ray, prd);

            if (prd.scatter.state == ScatterState::MISSED || prd.scatter.state == ScatterState::TERMINATED)
            {
                final_color += prd.color * attenuation;

                break;
            }
            else if (prd.scatter.state == ScatterState::BOUNCED)
            {
                ray.origin = prd.scatter.origin;
                ray.direction = normalize(prd.scatter.target - prd.scatter.origin);

                attenuation *= 0.75f;
            }
        }

    }

    vec3f color_average = final_color / (float)NUM_SAMPLE_PER_PIXEL;

    unsigned int pixel_index = pixel_ID.x + pixel_ID.y * ray_gen_data.frame_buffer_size.x;

    if (ray_gen_data.frame_number == 1)
        ray_gen_data.accumulation_buffer[pixel_index] = vec3f(0.0f);
    ray_gen_data.accumulation_buffer[pixel_index] += color_average;

    vec3f accumulated_color = ray_gen_data.accumulation_buffer[pixel_index];
    vec3f averaged_color = accumulated_color / (float)ray_gen_data.frame_number;
    vec3f gamma_corrected = vec3f(sqrtf(averaged_color.x), sqrtf(averaged_color.y), sqrtf(averaged_color.z));

    ray_gen_data.frame_buffer[pixel_index] = make_rgba(gamma_corrected);
}
