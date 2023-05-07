#include "shader.h"
#include "owl/owl_device.h"
#include "owl/common/math/LinearSpace.h"

#include <stdio.h>
#include <optix_device.h>

#define NUM_SAMPLE_PER_PIXEL 1
#define MAX_RECURSION_DEPTH 10

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

void inline __device__ diffuse_scatter(PerRayData& prd, const vec3f& hit_point, const vec3f& normal, const vec3f& albedo)
{
    prd.scatter.origin = hit_point + normal * 1.0e-4f;//Shifting the hit_point to avoid self intersections
    prd.scatter.target = random_in_hemisphere(prd, normal) + hit_point;
    prd.scatter.attenuation = albedo;
    prd.scatter.state = ScatterState::BOUNCED;
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
    vec3f ray_direction = optixGetWorldRayDirection();
    vec3f hit_point = origin + t_hit * ray_direction;
    vec3f normal = normalize(hit_point - sphere_data.sphere.center);

    diffuse_scatter(prd, hit_point, normal, sphere_data.material.albedo);
}

template <typename TriangleGeomDataType>
void __device__ triangle_closest_hit() {}

template <>
void __device__ triangle_closest_hit<MetalTriangleGeomData>()
{
    PerRayData& prd = getPRD<PerRayData>();
    MetalTriangleGeomData triangle_data = getProgramData<MetalTriangleGeomData>();

    int primitive_index = optixGetPrimitiveIndex();
    vec3i triangle_vertices_indices = triangle_data.indices[primitive_index];
    const vec3f& A = triangle_data.vertices[triangle_vertices_indices.x];
    const vec3f& B = triangle_data.vertices[triangle_vertices_indices.y];
    const vec3f& C = triangle_data.vertices[triangle_vertices_indices.z];
    vec3f normal = normalize(cross(B - A, C - A));

    float t_hit = optixGetRayTmax();
    vec3f origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();
    vec3f hit_point = origin + t_hit * ray_direction;

    if (dot(normal, ray_direction) > 0)
        normal = -normal;

    prd.scatter.origin = hit_point + normal * 1.0e-4f;
    prd.scatter.target = perfect_reflect_direction(ray_direction, normal) + hit_point;
    //Lerping between fuzzy target direction and perfect reflection
    prd.scatter.target = (1.0f - triangle_data.roughness) * prd.scatter.target + (triangle_data.roughness) * (random_in_hemisphere(prd, normal) + hit_point);
    prd.scatter.attenuation = triangle_data.albedo;

    prd.scatter.state = ScatterState::BOUNCED;
}

OPTIX_CLOSEST_HIT_PROGRAM(metal_triangles)()
{
    triangle_closest_hit<MetalTriangleGeomData>();
}

OPTIX_CLOSEST_HIT_PROGRAM(obj_triangle)()
{
    const ObjTriangleGeomData& triangle_data = getProgramData<ObjTriangleGeomData>();
    PerRayData& prd = getPRD<PerRayData>();

    float hit_t = optixGetRayTmax();
    vec3f origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();
    vec3f hit_point = origin + hit_t * ray_direction;

    int primitive_index = optixGetPrimitiveIndex();
    vec3i triangle_indices = triangle_data.indices[primitive_index];

    float2 uv = optixGetTriangleBarycentrics();
    float u = uv.x, v = uv.y;
    const vec3f& A = triangle_data.vertices[triangle_indices.x];
    const vec3f& B = triangle_data.vertices[triangle_indices.y];
    const vec3f& C = triangle_data.vertices[triangle_indices.z];

    vec3i vertex_normals_indices = triangle_data.vertex_normals_indices[primitive_index];
//    vec3f normal = normalize(u * triangle_data.vertex_normals[triangle_data.vertex_normals_indices[primitive_index].x]
//                       + v * triangle_data.vertex_normals[triangle_data.vertex_normals_indices[primitive_index].y]
//                       + (1 - u - v) * triangle_data.vertex_normals[triangle_data.vertex_normals_indices[primitive_index].z]);
    vec3f normal = triangle_data.vertex_normals[vertex_normals_indices.x] * uv.x
                 + triangle_data.vertex_normals[vertex_normals_indices.y] * uv.y
                 + triangle_data.vertex_normals[vertex_normals_indices.z] * (1 - uv.x - uv.y);
    normal = normalize(normal);

    vec3f albedo = triangle_data.materials[primitive_index].diffuse;

    diffuse_scatter(prd, hit_point, normal, albedo);
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
    prd.scatter.attenuation = vec3f(skysphere_color.x, skysphere_color.y, skysphere_color.z);
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
        //Non jittered direction below
//        vec3f direction = normalize(ray_gen_data.camera.direction_00 + ray_gen_data.camera.direction_dx * pixel_ID.x / (float)ray_gen_data.frame_buffer_size.x
//                                  + ray_gen_data.camera.direction_dy * pixel_ID.y / (float)ray_gen_data.frame_buffer_size.y);
        ray.origin = ray_gen_data.camera.position;
        ray.direction = direction;

        vec3f attenuation = 1.0f;
        int recurse = 0;
        for (recurse = 0; recurse < MAX_RECURSION_DEPTH; recurse++)
        {
            traceRay(ray_gen_data.scene, ray, prd);
            attenuation *= prd.scatter.attenuation;

            if (prd.scatter.state == ScatterState::MISSED)
                break;
            else if (prd.scatter.state == ScatterState::TERMINATED)
            {
                //The ray got fully absorbed
                attenuation = vec3f(0.0f);

                break;
            }
            else if (prd.scatter.state == ScatterState::BOUNCED)
            {
                ray.origin = prd.scatter.origin;
                ray.direction = normalize(prd.scatter.target - prd.scatter.origin);
            }
        }

        //If we didn't not exceed the maximum recursion depth. Otherwise we do not add the attenuation which correspond to a black color
        if (recurse < MAX_RECURSION_DEPTH)
            final_color += attenuation;
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
