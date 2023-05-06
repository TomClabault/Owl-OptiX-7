#ifndef VIEWER_H
#define VIEWER_H

#include <owl/owl.h>

#include "owlViewer/OWLViewer.h"

#include <chrono>

using namespace owl;

class Viewer : public viewer::OWLViewer
{
public:
    static const vec3f BACKGROUND_COLOR;

    Viewer();

    void render() override;

    void resize(const vec2i& new_size) override;
    void cameraChanged() override;

    void print_frame_time(std::chrono::time_point<std::chrono::steady_clock>& start, std::chrono::time_point<std::chrono::steady_clock>& stop);

private:
    unsigned int m_frame_number = 0;

    bool m_sbt_dirty = true;

    OWLContext m_owl_context;
    OWLModule m_module;

    OWLRayGen m_ray_gen_program;
};

#endif
