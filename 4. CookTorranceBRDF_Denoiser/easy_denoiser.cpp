#include "easy_denoiser.h"
#include "float4ToUint32.h"
#include "optix.h"
#include "vector_types.h"

void EasyDenoiser::denoise_float4_to_uint32(CUDABuffer input_buffer, uint32_t* output, unsigned int frame_number)
{
    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
    denoiserParams.hdrIntensity = (CUdeviceptr)0;// m_denoiser_intensity.d_pointer();
    denoiserParams.blendFactor = 1.0f / (frame_number / 2.0f);

    // -------------------------------------------------------
    OptixImage2D inputLayer;
    inputLayer.data = input_buffer.d_pointer();
    /// Width of the image (in pixels)
    inputLayer.width = m_buffer_width;
    /// Height of the image (in pixels)
    inputLayer.height = m_buffer_height;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = m_buffer_width * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = m_denoised_buffer_float4.d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = m_buffer_width;
    /// Height of the image (in pixels)
    outputLayer.height = m_buffer_height;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = m_buffer_width * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
//    OPTIX_CHECK(optixDenoiserComputeIntensity
//                (m_denoiser,
//                 /*stream*/0,
//                 &inputLayer,
//                 (CUdeviceptr)m_denoiser_intensity.d_pointer(),
//                 (CUdeviceptr)m_denoiser_scratch.d_pointer(),
//                 m_denoiser_scratch.size()));

    OptixDenoiserGuideLayer denoiser_guide_layer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser,
                                    /*stream*/0,
                                    &denoiserParams,
                                    m_denoiser_state.d_pointer(),
                                    m_denoiser_state.size(),
                                    &denoiser_guide_layer,
                                    &denoiserLayer,1,
                                    /*inputOffsetX*/0,
                                    /*inputOffsetY*/0,
                                    m_denoiser_scratch.d_pointer(),
                                    m_denoiser_scratch.size()));

    cuda_float4_to_uint32((float4*)m_denoised_buffer_float4.d_pointer(), m_buffer_width, m_buffer_height, output);
}

void EasyDenoiser::setup(OWLContext& owl_context, const vec2i& newSize)
{
    m_buffer_width = newSize.x;
    m_buffer_height = newSize.y;

    if (m_denoiser)
        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};

    OPTIX_CHECK(optixDenoiserCreate(owlContextGetOptixContext(owl_context, 0), OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &m_denoiser));

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, newSize.x, newSize.y, &denoiserReturnSizes));

    m_denoiser_scratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                                       denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
    m_denoiser_state.resize(denoiserReturnSizes.stateSizeInBytes);

    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    m_denoised_buffer_float4.resize(newSize.x * newSize.y * sizeof(float4));

    // ------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(m_denoiser, 0, newSize.x, newSize.y,
                                   m_denoiser_state.d_pointer(), m_denoiser_state.size(),
                                   m_denoiser_scratch.d_pointer(), m_denoiser_scratch.size()));
}
