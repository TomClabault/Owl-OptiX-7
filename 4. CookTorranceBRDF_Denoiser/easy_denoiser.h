#ifndef EASY_DENOISER_H
#define EASY_DENOISER_H

#include "cudaBuffer.h"

#include <owl/common/math/vec.h>
#include <owl/owl_host.h>

using namespace owl;

class EasyDenoiser
{
public:
    void convert_denoised_to_rgb_uint32(uint32_t* output_uint32);

    void denoise_float4(CUDABuffer input, unsigned int frame_number);
    void denoise_float4_to_uint32(CUDABuffer input, uint32_t* output, unsigned int frame_number);

    void setup(OWLContext& owl_context, const vec2i& newSize);

private:
    OptixDenoiser m_denoiser = nullptr;

    CUDABuffer m_denoised_buffer_float4;

    CUDABuffer m_denoiser_scratch;
    CUDABuffer m_denoiser_state;
    CUDABuffer m_denoiser_intensity;

    unsigned int m_buffer_width, m_buffer_height;
};

#endif
