#include "cudaBuffer.h"
#include "easy_denoiser.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "owl/owl.h"
#include "owlViewer/OWLViewer.h"
#include "shaderMaterials.h"

#include "emissive_triangles_utils.h"

#include <chrono>

using namespace owl;

class ImGuiViewer : public viewer::OWLViewer
{
public:

    static const owl::vec3f INIT_CAMERA_POSITION;
    static const owl::vec3f INIT_CAMERA_LOOKAT;
    static const owl::vec3f INIT_CAMERA_UP;
    static const float INIT_CAMERA_FOVY;

    ImGuiViewer();
    ~ImGuiViewer();

    void showAndRun();
    void showAndRun(std::function<bool()> keepgoing);

    void setupImGUI();

    void cameraChanged() override;
    void update_obj_material();
    void update_max_bounces();

    void resize(const owl::vec2i& new_size) override;
    void render() override;
    void imgui_render();
    void draw() override;

    void mouseButtonLeft(const vec2i &where, bool pressed) override;
    void mouseButtonRight(const vec2i& where, bool pressed) override;
    void mouseButtonCenter(const vec2i& where, bool pressed) override;

    void load_skysphere(const char* filepath);
    OWLGroup create_obj_group(const char* obj_file_path, EmissiveTrianglesInfo& emissive_triangles, OWLBuffer* triangles_indices, OWLBuffer* triangles_vertices, OWLBuffer* triangles_materials_indices, OWLBuffer* triangles_materials);
    OWLGroup create_cook_torrance_obj_group(const char* obj_file_path);
    OWLGroup create_floor_group();
    OWLGroup create_emissive_triangles_group();

    void print_frame_time(std::chrono::time_point<std::chrono::steady_clock>& start, std::chrono::time_point<std::chrono::steady_clock>& stop);

private:
    static void glfwindow_reshape_cb(GLFWwindow* window, int width, int height )
    {
        OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
        assert(gw);
        gw->resize(vec2i(width,height));
    }

    /*! callback for a key press */
    static void glfwindow_char_cb(GLFWwindow *window,
                                  unsigned int key)
    {
        ImGui_ImplGlfw_CharCallback(window, key);

        ImGuiIO io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard)
        {
            OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
            assert(gw);
            gw->key(key,gw->getMousePos());
        }
    }

    /*! callback for a key press */
    static void glfwindow_key_cb(GLFWwindow *window,
                                 int key,
                                 int scancode,
                                 int action,
                                 int mods)
    {
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard)
        {
            OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
            assert(gw);
            if (action == GLFW_PRESS) {
                gw->special(key,mods,gw->getMousePos());
            }
        }
    }

    /*! callback for _moving_ the mouse to a new position */
    static void glfwindow_mouseMotion_cb(GLFWwindow *window, double x, double y)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddMousePosEvent(x, y);

        if (!io.WantSetMousePos)
        {
            OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
            assert(gw);
            gw->mouseMotion(vec2i((int)x, (int)y));
        }
    }

    /*! callback for pressing _or_ releasing a mouse button*/
    static void glfwindow_mouseButton_cb(GLFWwindow *window,
                                         int button,
                                         int action,
                                         int mods)
    {
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            OWLViewer *gw = static_cast<OWLViewer*>(glfwGetWindowUserPointer(window));
            assert(gw);
            gw->mouseButton(button,action,mods);
        }
    }

private:
    CUDABuffer m_float4_frame_buffer;
    CUDABuffer m_normal_buffer;
    CUDABuffer m_albedo_buffer;
    CUDABuffer m_accumulation_buffer;
    unsigned int m_frame_number = 0;

    //Maximum recursion depth of the rays
    //Kept here so that ImGUI can modify it
    int m_max_bounces = 1;
    //Max bounces parameter at the last frame. Used
    //to determine when the user has changed the max bounces
    //using the ImGUI slider
    int m_previous_max_bounces;

    //This is the buffer that is going to hold the primitives
    //indices of the emissive triangles of the scene.
    //This buffer will be used to sample direct lighting
    EmissiveTrianglesInfo m_emissive_triangles_info;

    //We're storing the geom of the OBJ object here
    //because we're going to need it to update its materials
    //when ImGui modifies the materials
    OWLGeom m_obj_triangle_geom;
    //We're storing the materials so that ImGui can
    //access and modify it interactively
    CookTorranceMaterial m_obj_material;
    //We also need to store the material of the previous frame
    //to know when the material changed and be able to stop accumulating
    CookTorranceMaterial m_previous_obj_material;

    int m_skysphere_width, m_skysphere_height;
    std::vector<vec4uc> m_skysphere;

    bool m_sbtDirty = true;

    //Whether or not the mouse is moving
    //(useful to avoid denoising when moving the camera is moving
    //for example -> this allows for improved frametes while
    //moving. We're also reducing the number of bounces while the
    //camera is moving)
    bool m_mouse_moving = false;
    EasyDenoiser m_denoiser;

    OWLContext m_owl;
    OWLModule m_module;
    OWLRayGen m_ray_gen;
    OWLLaunchParams m_launch_params;
};
