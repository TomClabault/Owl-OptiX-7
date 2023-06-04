#ifndef EASY_DENOISER_H
#define EASY_DENOISER_H

#include "cudaBuffer.h"

#include <owl/common/math/vec.h>
#include <owl/owl_host.h>

using namespace owl;

class OptiXDenoiserWrapper
{
public:
    void convert_denoised_to_rgb_uint32(uint32_t* output_uint32);

    void denoise(CUDABuffer input_buffer, CUDABuffer normal_buffer, CUDABuffer albedo_buffer, float4** output, unsigned int frame_number);

    void setup_ldr(OWLContext& owl_context, const vec2i& newSize);
    void setup_hdr(OWLContext& owl_context, const vec2i& newSize);

    float m_blend_factor = 24.0f;

private:
    void setup(OWLContext& owl_context, const vec2i& newSize, OptixDenoiserModelKind model_kind);

private:
    OptixDenoiser m_denoiser = nullptr;

    CUDABuffer m_denoised_buffer_float4;
    CUDABuffer m_denoiser_scratch;
    CUDABuffer m_denoiser_state;
    CUDABuffer m_denoiser_intensity;

    unsigned int m_buffer_width, m_buffer_height;
};

#endif
