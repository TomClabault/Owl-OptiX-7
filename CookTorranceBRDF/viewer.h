#include "cudaBuffer.h"
#include "owl/owl.h"
#include "owlViewer/OWLViewer.h"

#include <chrono>

using namespace owl;

class Viewer : public viewer::OWLViewer
{
public:
    static const owl::vec3f INIT_CAMERA_POSITION;
    static const owl::vec3f INIT_CAMERA_LOOKAT;
    static const owl::vec3f INIT_CAMERA_UP;
    static const float INIT_CAMERA_FOVY;

    Viewer();
    ~Viewer();

    void cameraChanged() override;

    void resize(const owl::vec2i& new_size) override;
    void render() override;

    void load_skysphere(const char* filepath);
    OWLGroup create_cook_torrance_obj_group(const char* obj_file_path);
    OWLGroup create_floor_group();

    void print_frame_time(std::chrono::time_point<std::chrono::steady_clock>& start, std::chrono::time_point<std::chrono::steady_clock>& stop);

private:
    CUDABuffer m_accumulation_buffer;
    unsigned int m_frame_number = 0;

    int m_skysphere_width, m_skysphere_height;
    std::vector<vec4uc> m_skysphere;

    bool m_sbtDirty = true;

    OWLContext m_owl;
    OWLModule m_module;
    OWLRayGen m_rayGen;
};
