#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>

//For vec3f and uint32_t
#include <owl/common/math/vec.h>

using namespace owl;

inline float __device__ luminance(vec3f rgb)
{
    return 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
}

inline vec3f __device__ change_luminance(vec3f c_in, float l_out)
{
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

inline vec3f __device__ uncharted2_tonemap_partial(vec3f x)
{
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

void __global__ hdr_uncharted2_filmic_tone_mapping_kernel(float4* in, uint32_t* out, vec2i input_size)
{
    unsigned int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    //Thread out of bounds
    if (pixel_x >= input_size.x || pixel_y >= input_size.y)
        return;

    unsigned int pixel_index = pixel_y * input_size.x + pixel_x;

    float4 f4 = in[pixel_index];
    vec3f vec3_pixel_value = (vec3f(f4.x, f4.y, f4.z));

    //Uncharted 2 filmic tone-mapping
    //https://64.github.io/tonemapping/
    float exposure_bias = 2.0f;
    vec3f curr = uncharted2_tonemap_partial(vec3_pixel_value * exposure_bias);

    vec3f W = vec3f(11.2f);
    vec3f white_scale = vec3f(1.0f) / uncharted2_tonemap_partial(W);
    vec3f tone_mapped = curr * white_scale;

    vec3f gamma_corrected = clamp(sqrt(tone_mapped), vec3f(0.0f), vec3f(1.0f));
//    vec3f gamma_corrected = vec3f(clamp(sqrtf(f4.x), 1.0f),
//                                  clamp(sqrtf(f4.y), 1.0f),
//                                  clamp(sqrtf(f4.z), 1.0f));

    uint32_t value = 0;
    value |= (uint32_t)(gamma_corrected.x * 255.9f) <<  0;
    value |= (uint32_t)(gamma_corrected.y * 255.9f) <<  8;
    value |= (uint32_t)(gamma_corrected.z * 255.9f) << 16;
    value |= (uint32_t)255             << 24;

    out[pixel_index] = value;
}

void hdr_tone_mapping(float4* float4_input, unsigned int width, unsigned int height, uint32_t* uint32_output)
{
    vec2i block_size = 32;
    vec2i nb_blocks = divRoundUp(vec2i(width, height), block_size);

    hdr_uncharted2_filmic_tone_mapping_kernel<<<dim3(nb_blocks.x, nb_blocks.y), dim3(block_size.x, block_size.y)>>>((float4*)float4_input, uint32_output, vec2i(width, height));
}
