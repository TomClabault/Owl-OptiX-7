#include "easy_denoiser.h"

#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>

void __global__ float4_to_uint32_kernel(float4* in, uint32_t* out, vec2i input_size)
{
    unsigned int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    //Thread out of bounds
    if (pixel_x >= input_size.x || pixel_y >= input_size.y)
        return;

    unsigned int pixel_index = pixel_y * input_size.x + pixel_x;

    float4 f4 = in[pixel_index];
    vec3f gamma_corrected = vec3f(clamp(sqrtf(f4.x), 1.0f),
                                  clamp(sqrtf(f4.y), 1.0f),
                                  clamp(sqrtf(f4.z), 1.0f));

    uint32_t value = 0;
    value |= (uint32_t)(gamma_corrected.x * 255.9f) <<  0;
    value |= (uint32_t)(gamma_corrected.y * 255.9f) <<  8;
    value |= (uint32_t)(gamma_corrected.z * 255.9f) << 16;
    value |= (uint32_t)255             << 24;

    out[pixel_index] = value;
}

void cuda_float4_to_uint32(float4* float4_input, unsigned int width, unsigned int height, uint32_t* uint32_output)
{
    vec2i block_size = 32;
    vec2i nb_blocks = divRoundUp(vec2i(width, height), block_size);

    float4_to_uint32_kernel<<<dim3(nb_blocks.x, nb_blocks.y), dim3(block_size.x, block_size.y)>>>((float4*)float4_input, uint32_output, vec2i(width, height));
}
