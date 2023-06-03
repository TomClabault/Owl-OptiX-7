    #include "easy_denoiser.h"
#include "float4ToUint32.h"
#include "optix.h"
#include "vector_types.h"

void EasyDenoiser::denoise_float4_to_uint32(CUDABuffer input_buffer, CUDABuffer normal_buffer, CUDABuffer albedo_buffer, uint32_t* output, unsigned int frame_number)
{
    OptixDenoiserParams denoiserParams = {};
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
    denoiserParams.hdrIntensity = (CUdeviceptr)0;

    //This blends the output of the denoiser between the denoised image and
    //the noisy image (what we give as the input to the denoiser). The point
    //of this is to avoid displaying on screen the output of the denoiser when it's
    //been given a very noisy image (such as the very first frame rendered for example
    //so that's why the formula below returns 1 when the frame number is 1 and the blend
    //factor 1) because it in such cases, it tends to return deformed images
    //The minimum to 1.0f is used to avoid overshoothing the blending range over
    //1 (because the blending range is supposed to be [0.0, 1.0]) when the frame
    //number is < blend_factor and thus frame_number / m_blend_factor < 0 which in turn
    //means that 1.0f / (frame_number / m_blend_factor) > 1.0 -> overshooting the range
    denoiserParams.blendFactor = min(1.0f, 1.0f / ((float)frame_number / m_blend_factor));

    // -------------------------------------------------------
    OptixImage2D inputLayer[3];
    // -------------- Beauty layer -------------- //
    inputLayer[0].data = input_buffer.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[0].width = m_buffer_width;
    /// Height of the image (in pixels)
    inputLayer[0].height = m_buffer_height;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[0].rowStrideInBytes = m_buffer_width * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[0].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------- Normal layer -------------- //
    inputLayer[1].data = normal_buffer.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[1].width = m_buffer_width;
    /// Height of the image (in pixels)
    inputLayer[1].height = m_buffer_height;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[1].rowStrideInBytes = m_buffer_width * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[1].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------- Albedo layer -------------- //
    inputLayer[2].data = albedo_buffer.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[2].width = m_buffer_width;
    /// Height of the image (in pixels)
    inputLayer[2].height = m_buffer_height;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[2].rowStrideInBytes = m_buffer_width * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[2].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

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

    OptixDenoiserGuideLayer denoiser_guide_layer = {};
    denoiser_guide_layer.normal = inputLayer[1];
    denoiser_guide_layer.albedo = inputLayer[2];

    OptixDenoiserLayer denoiser_layer = {};
    denoiser_layer.input = inputLayer[0];
    denoiser_layer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser,
                                    /*stream*/0,
                                    &denoiserParams,
                                    m_denoiser_state.d_pointer(),
                                    m_denoiser_state.size(),
                                    &denoiser_guide_layer,
                                    &denoiser_layer, 1,
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
    denoiserOptions.guideNormal = 1;
    denoiserOptions.guideAlbedo = 1;

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
