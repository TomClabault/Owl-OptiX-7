#ifndef VIEWER_H
#define VIEWER_H

#include <chrono>

#include <owl/owl.h>
#include "owlViewer/OWLViewer.h"

#include "cudaBuffer.h"
#include "shader.h"

using namespace owl;

class Viewer : public viewer::OWLViewer
{
public:
    Viewer();

    OWLGroup create_floor_group();
    OWLGroup create_obj_group(const char* obj_file_path);

    void render() override;
    void setup_denoiser(const vec2i& newSize);
    void denoise_render();
    void cuda_float4_to_rgb();

    void resize(const vec2i& new_size) override;
    void cameraChanged() override;

    void mouseButtonLeft(const vec2i &where, bool pressed) override;
    void mouseButtonRight(const vec2i& where, bool pressed) override;
    void mouseButtonCenter(const vec2i& where, bool pressed) override;

    void load_skysphere(const char* filepath);
    void print_frame_time(std::chrono::time_point<std::chrono::steady_clock>& start, std::chrono::time_point<std::chrono::steady_clock>& stop);

private:
    vec3f* m_accumulation_buffer = nullptr;
    uint32_t m_frame_number = 0;

    bool m_sbt_dirty = true;

    std::vector<vec4uc> m_skysphere;
    int m_skysphere_width, m_skysphere_height;

    std::vector<LambertianSphere> m_lambertian_spheres;

    OWLContext m_owl_context;
    OWLModule m_module;

    OWLRayGen m_ray_gen_program;

    //Float4 frame buffer needed as input to the denoiser
    CUDABuffer m_float_frame_buffer;
    CUDABuffer m_normal_buffer;
    CUDABuffer m_albedo_buffer;
    CUDABuffer m_denoiserIntensity;
    OptixDenoiser denoiser = nullptr;
    CUDABuffer m_denoiserScratch;
    CUDABuffer m_denoiserState;
    CUDABuffer m_converted_buffer;//Buffer that will hold the converted data from the float4 denoised_buffer in the uint32_t format, ready to be displayed

    bool denoiser_on = true;
    CUDABuffer m_denoised_buffer;
};

#endif
