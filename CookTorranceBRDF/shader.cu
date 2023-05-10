#include <device_launch_parameters.h>

#include "optix_device.h"
#include "owl/common/math/random.h"
#include "owl/owl_device.h"
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
    vec3f attenuation;

    struct
    {
        vec3f origin;//Scatter origin
        vec3f direction;//Scatter direction NORMALIZED

        ScatterState state;
    } scatter;

    owl::common::LCG<4> random;
};

OPTIX_RAYGEN_PROGRAM(ray_gen)()
{
    vec2i pixel_ID = getLaunchIndex();
    RayGenData ray_gen_data = getProgramData<RayGenData>();
    unsigned int pixel_index = pixel_ID.y * ray_gen_data.frame_buffer_size.x + pixel_ID.x;

    PerRayData prd;
    prd.random.init(pixel_ID.x * ray_gen_data.frame_number * ray_gen_data.frame_buffer_size.x * NUM_SAMPLE_PER_PIXEL,
                    pixel_ID.y * ray_gen_data.frame_number * ray_gen_data.frame_buffer_size.y * NUM_SAMPLE_PER_PIXEL);

    vec3f ray_origin = ray_gen_data.camera.position;
    vec3f sum_samples_color = vec3f(0.0f);
    for (int sample = 0; sample < NUM_SAMPLE_PER_PIXEL; sample++)
    {
        vec3f ray_direction = normalize(ray_gen_data.camera.direction_00
                                        + ray_gen_data.camera.direction_dx * (prd.random() + pixel_ID.x) / (float)ray_gen_data.frame_buffer_size.x
                                        + ray_gen_data.camera.direction_dy * (prd.random() + pixel_ID.y) / (float)ray_gen_data.frame_buffer_size.y);

        vec3f current_color = vec3f(1.0f);
        for (int depth = 0; depth < MAX_RECURSION_DEPTH; depth++)
        {
            Ray ray(ray_origin, ray_direction, 1.0e-3f, 1.0e30f);

            traceRay(ray_gen_data.scene, ray, prd);
            current_color *= prd.attenuation;

            if (prd.scatter.state == MISSED)
                break;
            else if (prd.scatter.state == BOUNCED)
            {
                ray_origin = prd.scatter.origin;
                ray_direction = prd.scatter.direction;
            }
        }

        sum_samples_color += current_color;
    }

    vec3f averaged_color = clamp(sum_samples_color / (float)NUM_SAMPLE_PER_PIXEL, vec3f(0.0f), vec3f(1.0f));
    vec3f gamma_corrected = sqrt(averaged_color);

    if (ray_gen_data.frame_number == 1)
        ray_gen_data.accumulation_buffer[pixel_index] = vec3f(0.0f);
    ray_gen_data.accumulation_buffer[pixel_index] += gamma_corrected;

    ray_gen_data.frame_buffer[pixel_index] = make_rgba(ray_gen_data.accumulation_buffer[pixel_index] / (float)ray_gen_data.frame_number);
}

OPTIX_CLOSEST_HIT_PROGRAM(cook_torrance_obj_triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();

    prd.attenuation = vec3f(1.0f);
    prd.scatter.state = MISSED;
}

OPTIX_CLOSEST_HIT_PROGRAM(floor_triangle)()
{
    PerRayData& prd = getPRD<PerRayData>();

    prd.attenuation = vec3f(0.0f);
    prd.scatter.state = MISSED;
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
