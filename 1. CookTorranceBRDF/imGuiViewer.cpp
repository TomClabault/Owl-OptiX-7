#include "geometriesData.h"
#include "utils.h"
#include "shader.h"
#include "imGuiViewer.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <thread>

extern "C" char shader_ptx[];

ImGuiViewer::ImGuiViewer()
{
    setupImGUI();

    setTitle("CookTorranceBRDF");

    m_owl = owlContextCreate(nullptr,1);
    m_module = owlModuleCreate(m_owl, shader_ptx);

    OWLVarDecl rayGenVars[] = {
        { "frame_buffer",         OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, frame_buffer) },
        { "frame_buffer_size",    OWL_INT2,           OWL_OFFSETOF(RayGenData, frame_buffer_size) },
        { "camera.position",      OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.position) },
        { "camera.direction_00",  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_00) },
        { "camera.direction_dx",  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dx) },
        { "camera.direction_dy",  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dy) },
        { /* sentinel: */ }
    };

    m_rayGen = owlRayGenCreate(m_owl, m_module, "ray_gen", sizeof(RayGenData), rayGenVars, -1);

    OWLGroup triangle_group = create_cook_torrance_obj_group("../../common_data/bunny_translated.obj");
    //OWLGroup triangle_group = create_cook_torrance_obj_group("D:\\Bureau\\Repos\\M1\\m-1-synthese\\tp2\\data\\xyzrgb_dragon.obj");
    OWLGroup floor_group = create_floor_group();

    OWLGroup scene = owlInstanceGroupCreate(m_owl, 2);
    owlInstanceGroupSetChild(scene, 0, triangle_group);
    owlInstanceGroupSetChild(scene, 1, floor_group);
    owlGroupBuildAccel(scene);

    OWLVarDecl miss_prog_vars[] = {
        { "skysphere", OWL_TEXTURE, OWL_OFFSETOF(MissProgData, skysphere) },
        { /* sentinel */ }
    };

    OWLMissProg miss_program = owlMissProgCreate(m_owl, m_module, "miss", sizeof(MissProgData), miss_prog_vars, 1);

    load_skysphere("../../common_data/industrial_sunset_puresky_bright.png");
    OWLTexture skysphere = owlTexture2DCreate(m_owl, OWL_TEXEL_FORMAT_RGBA8, m_skysphere_width, m_skysphere_height, m_skysphere.data());
    owlMissProgSetTexture(miss_program, "skysphere", skysphere);

    OWLVarDecl launch_params_vars[] = {
        { "scene", OWL_GROUP, OWL_OFFSETOF(LaunchParams, scene) },
        { "accumulation_buffer", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams, accumulation_buffer) },
        { "frame_number", OWL_UINT, OWL_OFFSETOF(LaunchParams, frame_number) },
        { "obj_material", OWL_USER_TYPE(m_obj_material), OWL_OFFSETOF(LaunchParams, obj_material) },
        { /* sentinel */ }
    };

    m_launch_params = owlParamsCreate(m_owl, sizeof(LaunchParams), launch_params_vars, -1);

    m_accumulation_buffer.resize(sizeof(vec3f) * fbSize.x * fbSize.y);
    owlParamsSet1ui(m_launch_params, "frame_number", 1);
    owlParamsSet1ul(m_launch_params, "accumulation_buffer", (uint64_t)m_accumulation_buffer.d_pointer());
    owlParamsSetGroup(m_launch_params, "scene", scene);

    owlBuildPrograms(m_owl);
    owlBuildPipeline(m_owl);
    owlBuildSBT(m_owl);
}

void ImGuiViewer::showAndRun()
{
    showAndRun([]() {return true; }); // run until closed manually
}

void ImGuiViewer::showAndRun(std::function<bool()> keepgoing)
{
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width,height));

    glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
    glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
    glfwSetKeyCallback(handle, glfwindow_key_cb);
    glfwSetCharCallback(handle, glfwindow_char_cb);
    glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);

    while (!glfwWindowShouldClose(handle) && keepgoing()) {
        static double lastCameraUpdate = -1.f;
        if (camera.lastModified != lastCameraUpdate) {
            cameraChanged();
            lastCameraUpdate = camera.lastModified;
        }
        render();
        draw();

        glfwSwapBuffers(handle);
        glfwPollEvents();
    }

    glfwDestroyWindow(handle);
    glfwTerminate();
}

void ImGuiViewer::setupImGUI()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(viewer::OWLViewer::handle, true);
    ImGui_ImplOpenGL3_Init("#version 130");
}

OWLGroup ImGuiViewer::create_cook_torrance_obj_group(const char* obj_file_path)
{
    std::vector<vec3i> indices;
    std::vector<vec3f> vertices;
    std::vector<vec3f> vertex_normals;
    std::vector<vec3i> vertex_normals_indices;
    std::vector<CookTorranceMaterial> materials;
    std::vector<int> materials_indices;

    Utils::read_obj(obj_file_path, indices, vertices, vertex_normals, vertex_normals_indices, materials, materials_indices);
    m_obj_material = materials[0];

    OWLVarDecl triangleGeometryVars[] = {
        { "triangle_data.indices",                  OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.indices)},
        { "triangle_data.vertices",                 OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertices)},
        { "triangle_data.vertex_normals",           OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertex_normals)},
        { "triangle_data.vertex_normals_indices",   OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertex_normals_indices)},
        { /* sentinel */ }
    };

    OWLGeomType triangle_geometry_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(CookTorranceTriangleData), triangleGeometryVars, -1);
    owlGeomTypeSetClosestHit(triangle_geometry_type, 0, m_module, "cook_torrance_obj_triangle");

    OWLBuffer triangles_indices_buffer = owlDeviceBufferCreate(m_owl,               OWL_INT3, indices.size(), indices.data());
    OWLBuffer triangles_vertices_buffer = owlDeviceBufferCreate(m_owl,              OWL_FLOAT3, vertices.size(), vertices.data());
    OWLBuffer triangles_normals_buffer = owlDeviceBufferCreate(m_owl,               OWL_FLOAT3, vertex_normals.size(), vertex_normals.data());
    OWLBuffer triangles_normals_indices_buffer = owlDeviceBufferCreate(m_owl,       OWL_INT3, vertex_normals_indices.size(), vertex_normals_indices.data());

    m_obj_triangle_geom = owlGeomCreate(m_owl, triangle_geometry_type);

    owlTrianglesSetIndices(m_obj_triangle_geom, triangles_indices_buffer, indices.size(), sizeof(vec3i), 0);
    owlTrianglesSetVertices(m_obj_triangle_geom, triangles_vertices_buffer, vertices.size(), sizeof(vec3f), 0);

    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.indices", triangles_indices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertices", triangles_vertices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_normals", triangles_normals_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_normals_indices", triangles_normals_indices_buffer);

    OWLGroup triangle_group = owlTrianglesGeomGroupCreate(m_owl, 1, &m_obj_triangle_geom);
    owlGroupBuildAccel(triangle_group);

    return triangle_group;
}

OWLGroup ImGuiViewer::create_floor_group()
{
    vec3i indices[] =
    {
        vec3i(0, 1, 2),
        vec3i(1, 3, 2)
    };

    vec3f vertices[] =
    {
        vec3f(-100.0f, -1.0f, 100.0f),
        vec3f(100.0f, -1.0f, 100.0f),
        vec3f(-100.0f, -1.0f, -100.0f),
        vec3f(100.0f, -1.0f, -100.0f),
    };

    OWLVarDecl floor_vars[] =
    {
       { "triangle_data.indices",                   OWL_BUFPTR, OWL_OFFSETOF(DiffuseTriangleData, triangle_data.indices)},
       { "triangle_data.vertices",                  OWL_BUFPTR, OWL_OFFSETOF(DiffuseTriangleData, triangle_data.vertices)},
       { "albedo",                                  OWL_FLOAT3, OWL_OFFSETOF(DiffuseTriangleData, albedo)},
       { /*sentinel */ }
    };

    OWLGeomType floor_geom_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(DiffuseTriangleData), floor_vars, -1);
    owlGeomTypeSetClosestHit(floor_geom_type, 0, m_module, "floor_triangle");
    OWLGeom floor_geom = owlGeomCreate(m_owl, floor_geom_type);

    OWLBuffer indices_buffer =  owlDeviceBufferCreate(m_owl, OWL_INT3,   2, indices);
    OWLBuffer vertices_buffer = owlDeviceBufferCreate(m_owl, OWL_FLOAT3, 4, vertices);
    owlTrianglesSetIndices(floor_geom, indices_buffer, 2, sizeof(vec3i), 0);
    owlTrianglesSetVertices(floor_geom, vertices_buffer, 4, sizeof(vec3f), 0);

    owlGeomSetBuffer(floor_geom, "triangle_data.indices", indices_buffer);
    owlGeomSetBuffer(floor_geom, "triangle_data.vertices", vertices_buffer);
    owlGeomSet3f(floor_geom, "albedo", 0.9f, 0.9f, 0.9f);

    OWLGroup floor_group = owlTrianglesGeomGroupCreate(m_owl, 1, &floor_geom);
    owlGroupBuildAccel(floor_group);

    return floor_group;
}

void ImGuiViewer::load_skysphere(const char* filepath)
{
    float* data = Utils::read_image(filepath, m_skysphere_width, m_skysphere_height, true);

    m_skysphere.resize(m_skysphere_width * m_skysphere_height);

    for (int i = 0; i < m_skysphere_width * m_skysphere_height; i++)
    {
        vec4uc pixel_val = vec4uc(data[i * 4 + 0] * 255,
                                  data[i * 4 + 1] * 255,
                                  data[i * 4 + 2] * 255,
                                  data[i * 4 + 3] * 255);

        m_skysphere[i] = pixel_val;
    }

    stbi_image_free(data);
}

ImGuiViewer::~ImGuiViewer()
{
    owlModuleRelease(m_module);
    owlRayGenRelease(m_rayGen);
    owlContextDestroy(m_owl);
}

void ImGuiViewer::cameraChanged()
{
    vec3f camera_direction_00 = normalize(camera.getAt() - camera.getFrom());
    vec3f norm_camera_up = normalize(camera.getUp());

    vec3f camera_du = camera.getCosFovy() * camera.aspect * normalize(cross(camera_direction_00, norm_camera_up));
    vec3f camera_dv = camera.getCosFovy() * normalize(cross(camera_du, camera_direction_00));
    //Moves the 0, 0 to the lower left corner
    camera_direction_00 -= 0.5f * camera_du;
    camera_direction_00 -= 0.5f * camera_dv;
    camera_direction_00 = normalize(camera_direction_00);

    m_accumulation_buffer.resize(sizeof(vec3f) * fbSize.x * fbSize.y);
    m_frame_number = 0;

    owlRayGenSet2i(m_rayGen, "frame_buffer_size", fbSize.x, fbSize.y);
    owlRayGenSet1ul(m_rayGen, "frame_buffer", (uint64_t) fbPointer);
    owlRayGenSet3f(m_rayGen, "camera.position", (const owl3f&)camera.position);
    owlRayGenSet3f(m_rayGen, "camera.direction_00", (const owl3f&)camera_direction_00);
    owlRayGenSet3f(m_rayGen, "camera.direction_dx", (const owl3f&)camera_du);
    owlRayGenSet3f(m_rayGen, "camera.direction_dy", (const owl3f&)camera_dv);

    owlParamsSet1ul(m_launch_params, "accumulation_buffer", (uint64_t)m_accumulation_buffer.d_pointer());
    owlParamsSet1ui(m_launch_params, "frame_number", m_frame_number);

    m_sbtDirty = true;
}

void ImGuiViewer::resize(const owl::vec2i& new_size)
{
    viewer::OWLViewer::resize(new_size);
    cameraChanged();
}

void ImGuiViewer::imgui_render()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    //if (imgui_state.show_demo_window)
        //ImGui::ShowDemoWindow(&imgui_state.show_demo_window);

    {
        ImGui::Begin("Cook Torrance BRDF");

        ImGui::ColorPicker3("Albedo", (float*)&m_obj_material.albedo);
        ImGui::SliderFloat("Metallic", &m_obj_material.metallic, 0.0f, 1.0f);
        ImGui::SliderFloat("Roughness", &m_obj_material.roughness, 0.0f, 1.0f);
        ImGui::SliderFloat("Reflectance", &m_obj_material.reflectance, 0.0f, 1.0f);

        ImGui::End();
    }

    // Rendering
    ImGui::Render();
}

void ImGuiViewer::update_frame_number()
{
    owlParamsSet1ui(m_launch_params, "frame_number", ++m_frame_number);
}

void ImGuiViewer::update_obj_material()
{
    owlParamsSetRaw(m_launch_params, "obj_material", &m_obj_material);

    //If the material changed, we need to stop accumulating
    if (m_obj_material.metallic != m_previous_obj_material.metallic
        || m_obj_material.roughness != m_previous_obj_material.roughness
        || m_obj_material.reflectance != m_previous_obj_material.reflectance
        || m_obj_material.albedo.x != m_previous_obj_material.albedo.x
        || m_obj_material.albedo.y != m_previous_obj_material.albedo.y
        || m_obj_material.albedo.z != m_previous_obj_material.albedo.z)
    {
        m_frame_number = 0;
        update_frame_number();
    }

    m_previous_obj_material = m_obj_material;
}

void ImGuiViewer::render()
{
    imgui_render();

    update_frame_number();
    update_obj_material();

    if (m_sbtDirty)
    {
        owlBuildSBT(m_owl);
        m_sbtDirty = false;
    }

    // Normally launching without a hit or miss shader causes OptiX to trigger warnings.
    // Owl's wrapper call here will set up fake hit and miss records into the SBT to avoid these.
    if (fbSize.x == 0 || fbSize.y == 0)//Happens when the window is minimized
    {
        fbSize.x = 1;
        fbSize.y = 1;
        resize(fbSize);

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        return;
    }

    owlLaunch2D(m_rayGen, fbSize.x, fbSize.y, m_launch_params);
}

void ImGuiViewer::draw()
{
    owl::viewer::OWLViewer::draw();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
