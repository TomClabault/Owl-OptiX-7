#include "shader.h"
#include "owl/owl_device.h"
#include "owl/common/math/LinearSpace.h"

#include <device_launch_parameters.h>

#include <stdio.h>
#include <optix_device.h>

#define NUM_SAMPLE_PER_PIXEL 1
#define MAX_RECURSION_DEPTH 20 //This is the maximum number of bounces

enum ScatterState
{
    BOUNCED,//Bounced off of object
    //The distinction between bounced and refracted and reflected is necessary for the denoiser to get the proper albedo
    REFLECTED,//Reflected off of object
    REFRACTED,//Refracted off of object
    TERMINATED, //Completely absorbed and not bounced
    MISSED  //Hit the sky
};

struct ScatterInfo
{
    vec3f origin, target;
    vec3f attenuation;
    vec3f normal;
    vec3f albedo;

    ScatterState state;
};

struct PerRayData
{
    owl::common::LCG<4> random;

    ScatterInfo scatter;
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

vec3f __device__ perfect_reflect_target_direction(const vec3f& incident_ray, const vec3f& hit_point, const vec3f& normal)
{
    return incident_ray - 2 * dot(incident_ray, normal) * normal;
}

vec3f __device__ perfect_reflect_direction(const vec3f& incident_ray, const vec3f& normal)
{
    return normalize(incident_ray - 2 * dot(incident_ray, normal) * normal);
}

float inline __device__ schlick_approximation(float cos_incident_angle, float ior)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;

    return r0 + (1.0f - r0)*powf((1.0f - cos_incident_angle), 5.0f);
}

void inline __device__ diffuse_scatter(PerRayData& prd, const vec3f& hit_point, const vec3f& normal, const vec3f& albedo)
{
    prd.scatter.origin = hit_point + normal * 1.0e-4f;//Shifting the hit_point to avoid self intersections
    prd.scatter.target = random_in_hemisphere(prd, normal) + hit_point;
    prd.scatter.attenuation = albedo;
    prd.scatter.normal = normal;
    prd.scatter.albedo = albedo;
    prd.scatter.state = ScatterState::BOUNCED;
}

void inline __device__ metal_scatter(PerRayData& prd, const vec3f& hit_point, const vec3f& ray_direction, const vec3f& normal, float roughness, const vec3f& albedo)
{
    prd.scatter.origin = hit_point + normal * 1.0e-4f;
    vec3f perfect_reflection_target = perfect_reflect_direction(ray_direction, normal) + hit_point;
    //Lerping between fuzzy target direction and perfect reflection
    prd.scatter.target = (1.0f - roughness) * perfect_reflection_target + (roughness) * (random_in_hemisphere(prd, normal) + hit_point);
    prd.scatter.attenuation = albedo;
    prd.scatter.normal = normal;
    prd.scatter.albedo = albedo;

    if (roughness < 0.1)//The object is polished enough to be considered 'reflective'
        prd.scatter.state = ScatterState::REFLECTED;
    else
        prd.scatter.state = ScatterState::BOUNCED;
}

void inline __device__ dielectric_scatter(PerRayData& prd, const vec3f hit_point, const vec3f& ray_direction, const vec3f& normal, const vec3f& color, float ior)
{
    float cosi = clamp(dot(ray_direction, normal), -1.0f, 1.0f);
    float etai = 1, etat = ior;
    vec3f n = normal;
    if (cosi < 0)
        cosi = -cosi;
    else
    {
        float temp = etai;
        etai = etat;
        etat = temp;
        n = -normal;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);

    bool total_internal_reflection = k < 0.0f;

    vec3f new_ray_target;
    vec3f refract_dir;
    if (total_internal_reflection)//We're going to reflect the ray rather than refract it
    {
        new_ray_target = perfect_reflect_target_direction(ray_direction, hit_point, normal);
        prd.scatter.state = ScatterState::REFLECTED;
    }
    else
    {
        float reflect_probability = schlick_approximation(cosi, ior);

        vec2i idx = getLaunchIndex();

        if (prd.random() < reflect_probability)//The ray is reflected
        {
            new_ray_target = perfect_reflect_target_direction(ray_direction, hit_point, normal);
            prd.scatter.state = ScatterState::REFLECTED;
        }
        else//It is refracted
        {
            new_ray_target = k < 0 ? 0 : eta * ray_direction + (eta * cosi - sqrtf(k)) * n;
            prd.scatter.state = ScatterState::REFRACTED;
        }
    }

    prd.scatter.origin = hit_point - n * 1.0e-4f;
    prd.scatter.target = prd.scatter.origin + new_ray_target;
    prd.scatter.attenuation = color;
    prd.scatter.normal = normal;
    prd.scatter.albedo = color;
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

OPTIX_BOUNDS_PROGRAM(dielectric_spheres)(const void* primitive_data, box3f& bounds, int primitive_index)
{ bounds_program<DielectricSpheresGeometryData>(primitive_data, bounds, primitive_index); }

OPTIX_BOUNDS_PROGRAM(metal_spheres)(const void* primitive_data, box3f& bounds, int primitive_index)
{ bounds_program<MetalSpheresGeometryData>(primitive_data, bounds, primitive_index); }

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

OPTIX_INTERSECT_PROGRAM(dielectric_spheres)()
{ sphere_intersect<DielectricSpheresGeometryData>(); }

OPTIX_INTERSECT_PROGRAM(metal_spheres)()
{ sphere_intersect<MetalSpheresGeometryData>(); }

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
    prd.scatter.attenuation *= dot(normal, normalize(prd.scatter.target - prd.scatter.origin));
}

OPTIX_CLOSEST_HIT_PROGRAM(dielectric_spheres)()
{
    int primitive_index = optixGetPrimitiveIndex();
    const DielectricSphere& sphere_data = getProgramData<DielectricSpheresGeometryData>().primitives[primitive_index];

    PerRayData& prd = getPRD<PerRayData>();

    float t_hit = optixGetRayTmax();
    vec3f origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();
    vec3f hit_point = origin + t_hit * ray_direction;
    vec3f normal = normalize(hit_point - sphere_data.sphere.center);

    dielectric_scatter(prd, hit_point, ray_direction, normal, sphere_data.material.color_attenuation, sphere_data.material.ior);
}

OPTIX_CLOSEST_HIT_PROGRAM(metal_spheres)()
{
    int primitive_index = optixGetPrimitiveIndex();
    const MetalSphere& sphere_data = getProgramData<MetalSpheresGeometryData>().primitives[primitive_index];

    PerRayData& prd = getPRD<PerRayData>();

    float t_hit = optixGetRayTmax();
    vec3f origin = optixGetWorldRayOrigin();
    vec3f ray_direction = optixGetWorldRayDirection();
    vec3f hit_point = origin + t_hit * ray_direction;
    vec3f normal = normalize(hit_point - sphere_data.sphere.center);

    metal_scatter(prd, hit_point, ray_direction, normal, sphere_data.material.roughness, sphere_data.material.albedo);
}

template <typename TriangleGeomDataType>
void __device__ triangle_closest_hit() {}

template <>
void __device__ triangle_closest_hit<FloorTriangleGeomData>()
{
    PerRayData& prd = getPRD<PerRayData>();
    FloorTriangleGeomData triangle_data = getProgramData<FloorTriangleGeomData>();

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

    diffuse_scatter(prd, hit_point, normal, triangle_data.albedo);
}

OPTIX_CLOSEST_HIT_PROGRAM(floor_triangles)()
{
    triangle_closest_hit<FloorTriangleGeomData>();
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
    vec3f normal = normalize(u * triangle_data.vertex_normals[triangle_data.vertex_normals_indices[primitive_index].y]
                           + v * triangle_data.vertex_normals[triangle_data.vertex_normals_indices[primitive_index].z]
                           + (1 - u - v) * triangle_data.vertex_normals[triangle_data.vertex_normals_indices[primitive_index].x]);

    vec3f albedo = triangle_data.materials[triangle_data.materials_indices[primitive_index]].diffuse;

    dielectric_scatter(prd, hit_point, ray_direction, normal, albedo, 1.5);
    //diffuse_scatter(prd, hit_point, normal, albedo);
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
    prd.scatter.normal = vec3f(0.0f);
    prd.scatter.albedo = prd.scatter.attenuation;
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
    vec3f sum_normal = vec3f(0.0f);//Used by the denoiser
    vec3f sum_albedo = vec3f(0.0f);//Used by the denoiser

    Ray ray;
    for (int sample = 0; sample < NUM_SAMPLE_PER_PIXEL; sample++)
    {
        vec3f direction = normalize(ray_gen_data.camera.direction_00 + ray_gen_data.camera.direction_dx * (pixel_ID.x + prd.random()) / (float)ray_gen_data.frame_buffer_size.x
                                  + ray_gen_data.camera.direction_dy * (pixel_ID.y + prd.random()) / (float)ray_gen_data.frame_buffer_size.y);

        ray.origin = ray_gen_data.camera.position;
        ray.direction = direction;

        vec3f attenuation = 1.0f;
        int recurse = 0;
        bool albedo_set = false;//This boolean is used to get the albedo of the pixel only once because the denoiser only wants the first albedo encountered
        for (recurse = 0; recurse < MAX_RECURSION_DEPTH; recurse++)
        {
            traceRay(ray_gen_data.scene, ray, prd);
            attenuation *= prd.scatter.attenuation;
            if (recurse == 0)//We only want to consider the normal of the primary hits for the denoiser
                sum_normal += prd.scatter.normal;

            //We're only going to add the albedo if we hit a non
            //reflective/refractive material (or missed completely)
            //and if the albedo hasn't been added yet
            if (recurse == 0)
                sum_albedo = prd.scatter.albedo;

            if (prd.scatter.state == ScatterState::MISSED)
                break;
            else if (prd.scatter.state == ScatterState::TERMINATED)
            {
                //The ray got fully absorbed
                attenuation = vec3f(0.0f);

                break;
            }
            else if (prd.scatter.state == ScatterState::BOUNCED || prd.scatter.state == ScatterState::REFLECTED || prd.scatter.state == ScatterState::REFRACTED)
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

    vec3f sRGB_corrected = vec3f(pow(averaged_color.x, 1.0f / 2.2f),
                                 pow(averaged_color.y, 1.0f / 2.2f),
                                 pow(averaged_color.z, 1.0f / 2.2f));

    float4 float4_val;
    float4_val.x = sRGB_corrected.x;
    float4_val.y = sRGB_corrected.y;
    float4_val.z = sRGB_corrected.z;
    float4_val.w = 1.0f;

    float4 final_normal_f4;
    final_normal_f4.x = sum_normal.x / (float)NUM_SAMPLE_PER_PIXEL;
    final_normal_f4.y = sum_normal.y / (float)NUM_SAMPLE_PER_PIXEL;
    final_normal_f4.z = sum_normal.z / (float)NUM_SAMPLE_PER_PIXEL;
    final_normal_f4.w = 1.0f;

    float4 normal_remapped = final_normal_f4;
    if (normal_remapped.x != 0.0f && normal_remapped.y != 0.0f && normal_remapped.z != 0.0f)
    {
        normal_remapped.x = (normal_remapped.x + 1.0f) * 0.5;
        normal_remapped.y = (normal_remapped.y + 1.0f) * 0.5;
        normal_remapped.z = (normal_remapped.z + 1.0f) * 0.5;
    }

    float4 final_albedo_f4;
    final_albedo_f4.x = sum_albedo.x / (float)NUM_SAMPLE_PER_PIXEL;
    final_albedo_f4.y = sum_albedo.y / (float)NUM_SAMPLE_PER_PIXEL;
    final_albedo_f4.z = sum_albedo.z / (float)NUM_SAMPLE_PER_PIXEL;
    final_albedo_f4.w = 1.0f;

    ray_gen_data.float_frame_buffer[pixel_index] = float4_val;
    ray_gen_data.normal_buffer[pixel_index] = normal_remapped;
    ray_gen_data.albedo_buffer[pixel_index] = final_albedo_f4;
}
