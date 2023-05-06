#include "owl/owl.h"

#define STB_IMAGE_IMPLEMENTATION 1
#include "owlViewer/OWLViewer.h"

#include <chrono>

#include "scene.h"

class Viewer : public owl::viewer::OWLViewer
{
public:
    static const owl::vec3f INIT_CAMERA_POSITION;
    static const owl::vec3f INIT_CAMERA_LOOKAT;
    static const owl::vec3f INIT_CAMERA_UP;
    static const float INIT_CAMERA_FOVY;

    Viewer(const Scene& scene);
    ~Viewer();

    void cameraChanged() override;

    void resize(const owl::vec2i& new_size) override;
    void render() override;

    void print_frame_time(std::chrono::time_point<std::chrono::steady_clock>& start, std::chrono::time_point<std::chrono::steady_clock>& stop);

private:
    const Scene& m_scene;

    vec3f* m_frame_accumulation_buffer_ptr = nullptr;
    unsigned int m_frame_number = 0;

    bool m_sbtDirty = true;

    OWLContext m_owl;
    OWLModule m_module;
    OWLRayGen m_rayGen;
};
