#include "shader.h"
#include "owl/owl_device.h"
#include "owl/common/math/LinearSpace.h"

#include <stdio.h>
#include <optix_device.h>

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
    printf("intersect\n");

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
    printf("closest hit\n");

    int primitive_index = optixGetPrimitiveIndex();
    const LambertianSphere& sphere_data = getProgramData<LambertianSpheresGeometryData>().primitives[primitive_index];

    float t_hit = optixGetRayTmax();
    vec3f origin = optixGetWorldRayOrigin();
    vec3f direction = optixGetWorldRayDirection();

    vec3f normal = normalize(origin + t_hit * direction - sphere_data.sphere.center);

    vec3f& color = getPRD<vec3f>();
    //color = make_rgba((normal + 1.0f) * 0.5f);
    color = make_rgba(vec3f(1.0f, 0.0f, 0.0f));
}

extern "C" __global__ void __raygen__ray_gen()
{
    const vec2i pixel_ID = getLaunchIndex();
    const RayGenData& ray_gen_data = getProgramData<RayGenData>();

    vec3f origin = ray_gen_data.camera.position;
    vec3f direction = ray_gen_data.camera.direction_00 + ray_gen_data.camera.direction_dx * pixel_ID.x / ray_gen_data.frame_buffer_size.x
                                                       + ray_gen_data.camera.direction_dy * pixel_ID.y / ray_gen_data.frame_buffer_size.y;

    Ray ray(origin, direction, 1.0e-3, 1.0e10);

    vec3f prd_color;
    traceRay(ray_gen_data.scene, ray, prd_color);

    ray_gen_data.frame_buffer[pixel_ID.x + pixel_ID.y * ray_gen_data.frame_buffer_size.x] = make_rgba(prd_color);
}

OPTIX_MISS_PROGRAM(miss)()
{
    const MissProgData& miss_prog_data = getProgramData<MissProgData>();

    vec3f& color = getPRD<vec3f>();

    color = miss_prog_data.background_color;
}
