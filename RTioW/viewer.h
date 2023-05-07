#ifndef VIEWER_H
#define VIEWER_H

#include <owl/owl.h>

#include "owlViewer/OWLViewer.h"

#include <chrono>

#include "shader.h"

using namespace owl;

class Viewer : public viewer::OWLViewer
{
public:
    Viewer();

    void render() override;

    void resize(const vec2i& new_size) override;
    void cameraChanged() override;

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
};

#endif
