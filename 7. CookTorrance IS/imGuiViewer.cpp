#include "toneMapping.h"
#include "geometriesData.h"
#include "utils.h"
#include "shader.h"
#include "imGuiViewer.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define TINYEXR_USE_OPENMP 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr/tinyexr.h"

#include <chrono>
#include <thread>
#include <owl/common/math/AffineSpace.h>

extern "C" char shader_ptx[];

ImGuiViewer::ImGuiViewer()
{
    setupImGUI();

    setTitle("CookTorranceBRDF");

    m_owl = owlContextCreate(nullptr,1);
    owlContextSetRayTypeCount(m_owl, 2);
    m_module = owlModuleCreate(m_owl, shader_ptx);

    OWLVarDecl rayGenVars[] = {
        { "frame_buffer_size",      OWL_INT2,                       OWL_OFFSETOF(RayGenData, frame_buffer_size) },
        { "camera.position",        OWL_FLOAT3,                     OWL_OFFSETOF(RayGenData, camera.position) },
        { "camera.direction_00",    OWL_FLOAT3,                     OWL_OFFSETOF(RayGenData, camera.direction_00) },
        { "camera.direction_dx",    OWL_FLOAT3,                     OWL_OFFSETOF(RayGenData, camera.direction_dx) },
        { "camera.direction_dy",    OWL_FLOAT3,                     OWL_OFFSETOF(RayGenData, camera.direction_dy) },
        { "camera.view_matrix",     OWL_USER_TYPE(AffineSpace3f),   OWL_OFFSETOF(RayGenData, camera.view_matrix) },
        { /* sentinel: */ }
    };

    m_ray_gen = owlRayGenCreate(m_owl, m_module, "ray_gen", sizeof(RayGenData), rayGenVars, -1);

    OWLGroup bunny_group = create_cook_torrance_obj_group("../../common_data/bunny_for_cornell.obj");
    OWLGroup dragon_group = create_cook_torrance_obj_group("../../common_data/dragon_for_cornell.obj");

    OWLBuffer obj_indices, obj_vertices, obj_mat_indices, obj_mats;
    EmissiveTrianglesInfo emissive_triangles_info;
    //OWLGroup main_obj = create_obj_group("../../common_data/cornell_blocked_double_light.obj", emissive_triangles_info, &obj_indices, &obj_vertices, &obj_mat_indices, &obj_mats);
    //OWLGroup main_obj = create_obj_group("../../common_data/cornell_blocked.obj", emissive_triangles_info, &obj_indices, &obj_vertices, &obj_mat_indices, &obj_mats);
    OWLGroup main_obj = create_obj_group("../../common_data/cornell-box.obj", emissive_triangles_info, &obj_indices, &obj_vertices, &obj_mat_indices, &obj_mats);
    //OWLGroup main_obj = create_obj_group("../../common_data/InteriorTest.obj", emissive_triangles_info, &obj_indices, &obj_vertices, &obj_mat_indices, &obj_mats);
    //OWLGroup main_obj = create_obj_group("../../common_data/indoor/Namas_x_scaled.obj", emissive_triangles_info, &obj_indices, &obj_vertices, &obj_mat_indices, &obj_mats);
    //OWLGroup main_obj = create_obj_group("../../common_data/villa/CG Villa Scene Packed.obj", emissive_triangles_info, &obj_indices, &obj_vertices, &obj_mat_indices, &obj_mats);
    //OWLGroup main_obj = create_obj_group("../../common_data/geometry.obj", emissive_triangles_info, &obj_indices, &obj_vertices, &obj_mat_indices, &obj_mats);
    m_emissive_triangles_info = emissive_triangles_info;

    OWLGroup scene = owlInstanceGroupCreate(m_owl, 3);
    owlInstanceGroupSetChild(scene, 0, bunny_group);
    owlInstanceGroupSetChild(scene, 1, dragon_group);
    owlInstanceGroupSetChild(scene, 2, main_obj);
    owlGroupBuildAccel(scene);

    OWLVarDecl miss_prog_vars[] = {
        { "skysphere", OWL_TEXTURE, OWL_OFFSETOF(MissProgData, skysphere) },
        { /* sentinel */ }
    };

    OWLMissProg miss_program = owlMissProgCreate(m_owl, m_module, "miss", sizeof(MissProgData), miss_prog_vars, 1);

    OWLTexture skysphere = create_ldr_skysphere("../../common_data/industrial_sunset_puresky_bright.png");
    //OWLTexture skysphere = create_hdr_skysphere("../../common_data/Villa/textures/hilly_terrain_01_1k.exr");
    //OWLTexture skysphere = create_hdr_skysphere("../../common_data/HDRIs/OutdoorHDRI024_1K-HDR.exr");
    //OWLTexture skysphere = create_hdr_skysphere("../../common_data/HDRIs/christmas_photo_studio_01_1k.exr");
    //OWLTexture skysphere = create_hdr_skysphere("../../common_data/HDRIs/moonless_golf_1k.exr");
    owlMissProgSetTexture(miss_program, "skysphere", skysphere);

    //Creating the miss program for the shadow rays. This program doesn't have variables or data.
    OWLMissProg shadow_ray_miss_prog = owlMissProgCreate(m_owl, m_module, "shadow_ray_miss", 0, nullptr, 0);
    owlMissProgSet(m_owl, SHADOW_RAY, shadow_ray_miss_prog);

    OWLVarDecl launch_params_vars[] = {
        { "scene",                                               OWL_GROUP,                                  OWL_OFFSETOF(LaunchParams, scene) },
        { "accumulation_buffer",                                 OWL_RAW_POINTER,                            OWL_OFFSETOF(LaunchParams, accumulation_buffer) },
        { "float4_frame_buffer",                                 OWL_RAW_POINTER,                            OWL_OFFSETOF(LaunchParams, float4_frame_buffer) },
        { "normal_buffer",                                       OWL_RAW_POINTER,                            OWL_OFFSETOF(LaunchParams, normal_buffer) },
        { "albedo_buffer",                                       OWL_RAW_POINTER,                            OWL_OFFSETOF(LaunchParams, albedo_buffer) },
        { "frame_number",                                        OWL_UINT,                                   OWL_OFFSETOF(LaunchParams, frame_number) },
        { "mouse_moving",                                        OWL_BOOL,                                   OWL_OFFSETOF(LaunchParams, mouse_moving) },
        { "max_bounces",                                         OWL_INT,                                    OWL_OFFSETOF(LaunchParams, max_bounces) },
        { "obj_material",                                        OWL_USER_TYPE(m_obj_material),              OWL_OFFSETOF(LaunchParams, obj_material) },
        { "emissive_triangles_info",                             OWL_USER_TYPE(m_emissive_triangles_info),   OWL_OFFSETOF(LaunchParams, emissive_triangles_info) },
        { "emissive_triangles_info.triangles_indices",           OWL_BUFPTR,                                 OWL_OFFSETOF(LaunchParams, emissive_triangles_info.triangles_indices) },
        { "emissive_triangles_info.triangles_vertices",          OWL_BUFPTR,                                 OWL_OFFSETOF(LaunchParams, emissive_triangles_info.triangles_vertices) },
        { "emissive_triangles_info.triangles_materials_indices", OWL_BUFPTR,                                 OWL_OFFSETOF(LaunchParams, emissive_triangles_info.triangles_materials_indices) },
        { "emissive_triangles_info.triangles_materials",         OWL_BUFPTR,                                 OWL_OFFSETOF(LaunchParams, emissive_triangles_info.triangles_materials) },
        { /* sentinel */ }
    };

    m_launch_params = owlParamsCreate(m_owl, sizeof(LaunchParams), launch_params_vars, -1);

    owlParamsSetGroup(m_launch_params, "scene", scene);
    owlParamsSet1ui(m_launch_params, "frame_number", 1);
    owlParamsSet1b(m_launch_params, "mouse_moving", m_mouse_moving);
    owlParamsSet1i(m_launch_params, "max_bounces", m_max_bounces);
    owlParamsSetRaw(m_launch_params, "emissive_triangles_info", &m_emissive_triangles_info);
    owlParamsSetBuffer(m_launch_params, "emissive_triangles_info.triangles_indices", obj_indices);
    owlParamsSetBuffer(m_launch_params, "emissive_triangles_info.triangles_vertices", obj_vertices);
    owlParamsSetBuffer(m_launch_params, "emissive_triangles_info.triangles_materials_indices", obj_mat_indices);
    owlParamsSetBuffer(m_launch_params, "emissive_triangles_info.triangles_materials", obj_mats);

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

OWLGroup ImGuiViewer::create_obj_group(const char* obj_file_path, EmissiveTrianglesInfo& emissive_triangles, OWLBuffer* triangles_indices, OWLBuffer* triangles_vertices, OWLBuffer* triangles_materials_indices, OWLBuffer* triangles_materials)
{
    std::vector<vec3i> indices;
    std::vector<vec3f> vertices;
    std::vector<vec3f> vertex_normals;
    std::vector<vec3i> vertex_normals_indices;
    std::vector<rapidobj::Material> obj_materials;
    std::vector<int> materials_indices;
    std::vector<vec2f> vertex_texcoords;
    std::vector<vec3i> vertex_texcoords_indices;

    Utils::read_obj(obj_file_path, indices, vertices, vertex_normals, vertex_normals_indices, vertex_texcoords, vertex_texcoords_indices, obj_materials, materials_indices);

    //This is basically the 'obj_file_path' argument but without the obj file name at the end.
    //This path ends with a '/'
    std::string obj_file_path_str = std::string(obj_file_path);
    std::string path_obj_folder = obj_file_path_str.substr(0,  obj_file_path_str.rfind('/') + 1);
    std::vector<SimpleObjMaterial> simple_obj_materials;
    for (rapidobj::Material& mat : obj_materials)
    {
        SimpleObjMaterial obj_mat;
        obj_mat.albedo = *((vec3f*)&mat.diffuse);
        obj_mat.emissive = *((vec3f*)&mat.emission);
        obj_mat.ns = mat.shininess;

        //There is a diffuse texture
        if (mat.diffuse_texname.length() > 0)
        {
            int tex_width, tex_height;
            unsigned char* diffuse_texture_data = Utils::read_image_rgba_4x8((path_obj_folder + mat.diffuse_texname).c_str(), tex_width, tex_height);
            obj_mat.diffuse_texture = Utils::create_simple_cuda_texture(diffuse_texture_data, tex_width, tex_height);

            stbi_image_free(diffuse_texture_data);
        }
        else
            obj_mat.diffuse_texture = 0;

        simple_obj_materials.push_back(obj_mat);
    }

    OWLVarDecl triangle_geometry_vars[] = {
        { "triangle_data.indices",                  OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.indices)},
        { "triangle_data.vertices",                 OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertices)},
        { "triangle_data.vertex_normals",           OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertex_normals)},
        { "triangle_data.vertex_normals_indices",   OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertex_normals_indices)},
        { "triangle_data.vertex_uvs",               OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertex_uvs)},
        { "triangle_data.vertex_uvs_indices",       OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertex_uvs_indices)},
        { "materials",                              OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, materials)},
        { "materials_indices",                      OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, materials_indices)},
        { /* sentinel */ }
    };

    OWLGeomType triangle_geometry_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(SimpleObjTriangleData), triangle_geometry_vars, -1);
    owlGeomTypeSetClosestHit(triangle_geometry_type, RADIANCE_RAY, m_module, "obj_triangle");

    OWLBuffer triangles_indices_buffer = owlDeviceBufferCreate(m_owl,           OWL_INT3, indices.size(), indices.data());
    OWLBuffer triangles_vertices_buffer = owlDeviceBufferCreate(m_owl,          OWL_FLOAT3, vertices.size(), vertices.data());
    OWLBuffer triangles_normals_buffer = owlDeviceBufferCreate(m_owl,           OWL_FLOAT3, vertex_normals.size(), vertex_normals.data());
    OWLBuffer triangles_normals_indices_buffer = owlDeviceBufferCreate(m_owl,   OWL_INT3, vertex_normals_indices.size(), vertex_normals_indices.data());
    OWLBuffer triangles_uvs_buffer = owlDeviceBufferCreate(m_owl,               OWL_FLOAT2, vertex_texcoords.size(), vertex_texcoords.data());
    OWLBuffer triangles_uvs_indices_buffer = owlDeviceBufferCreate(m_owl,       OWL_INT3, vertex_texcoords_indices.size(), vertex_texcoords_indices.data());
    OWLBuffer triangles_materials_buffer = owlDeviceBufferCreate(m_owl,         OWL_USER_TYPE(simple_obj_materials[0]), simple_obj_materials.size(), simple_obj_materials.data());
    OWLBuffer triangles_materials_indices_buffer = owlDeviceBufferCreate(m_owl, OWL_INT, materials_indices.size(), materials_indices.data());

    m_obj_triangle_geom = owlGeomCreate(m_owl, triangle_geometry_type);

    owlTrianglesSetIndices(m_obj_triangle_geom, triangles_indices_buffer, indices.size(), sizeof(vec3i), 0);
    owlTrianglesSetVertices(m_obj_triangle_geom, triangles_vertices_buffer, vertices.size(), sizeof(vec3f), 0);

    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.indices", triangles_indices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertices", triangles_vertices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_normals", triangles_normals_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_normals_indices", triangles_normals_indices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_uvs", triangles_uvs_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_uvs_indices", triangles_uvs_indices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "materials", triangles_materials_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "materials_indices", triangles_materials_indices_buffer);

    emissive_triangles = EmissiveTrianglesUtils::extract_emissive_triangles(indices, vertices, vertex_normals, vertex_normals_indices, obj_materials, materials_indices);
    *triangles_indices = triangles_indices_buffer;
    *triangles_vertices = triangles_vertices_buffer;
    *triangles_materials_indices = triangles_materials_indices_buffer;
    *triangles_materials = triangles_materials_buffer;

    OWLGroup triangle_group = owlTrianglesGeomGroupCreate(m_owl, 1, &m_obj_triangle_geom);
    owlGroupBuildAccel(triangle_group);

    return triangle_group;
}

OWLGroup ImGuiViewer::create_cook_torrance_obj_group(const char* obj_file_path)
{
    std::vector<vec3i> indices;
    std::vector<vec3f> vertices;
    std::vector<vec3f> vertex_normals;
    std::vector<vec3i> vertex_normals_indices;
    std::vector<vec2f> vertex_uvs;
    std::vector<vec3i> vertex_uvs_indices;
    std::vector<rapidobj::Material> obj_materials;
    std::vector<int> materials_indices;

    Utils::read_obj(obj_file_path, indices, vertices, vertex_normals, vertex_normals_indices, vertex_uvs, vertex_uvs_indices, obj_materials, materials_indices);

    std::vector<CookTorranceMaterial> cook_torrance_materials;
    for (rapidobj::Material& mat : obj_materials)
    {
        CookTorranceMaterial cook_torrance_mat;
        cook_torrance_mat.albedo = *((vec3f*)&mat.diffuse);
        cook_torrance_mat.roughness = 0.5f;
        cook_torrance_mat.metallic = 0.5f;
        cook_torrance_mat.reflectance = 0.5f;

        cook_torrance_materials.push_back(cook_torrance_mat);
    }
    m_obj_material = cook_torrance_materials[0];

    OWLVarDecl triangle_geometry_vars[] = {
        { "triangle_data.indices",                  OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.indices)},
        { "triangle_data.vertices",                 OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertices)},
        { "triangle_data.vertex_normals",           OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertex_normals)},
        { "triangle_data.vertex_normals_indices",   OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertex_normals_indices)},
        { /* sentinel */ }
    };

    OWLGeomType triangle_geometry_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(CookTorranceTriangleData), triangle_geometry_vars, -1);
    owlGeomTypeSetClosestHit(triangle_geometry_type, RADIANCE_RAY, m_module, "cook_torrance_obj_triangle");

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
    std::vector<vec3i> triangles_indices =
        {
            vec3i(0, 1, 2),
            vec3i(1, 3, 2)
        };

    std::vector<vec3f> triangles_vertices =
        {
            vec3f(-100.0f, -1.0f, 100.0f),
            vec3f(100.0f, -1.0f, 100.0f),
            vec3f(-100.0f, -1.0f, -100.0f),
            vec3f(100.0f, -1.0f, -100.0f),
        };

    std::vector<vec3f> vertex_normals =
        {
            vec3f(0.0f, 1.0f, 0.0f),
        };

    std::vector<vec3i> vertex_normals_indices =
        {
            vec3i(0, 0, 0),
            vec3i(0, 0, 0),
        };

    std::vector<SimpleObjMaterial> lambertian_materials =
        {
            { vec3f(0.8f, 0.8f, 0.8f), vec3f(0.0f, 0.0f, 0.0f), 0.0f }
        };

    std::vector<int> materials_indices =
        {
            0, 0
        };

    OWLVarDecl floor_vars[] =
        {
            { "triangle_data.indices",                  OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.indices)},
            { "triangle_data.vertices",                 OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertices)},
            { "triangle_data.vertex_normals",           OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertex_normals)},
            { "triangle_data.vertex_normals_indices",   OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, triangle_data.vertex_normals_indices)},
            { "materials",                              OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, materials)},
            { "materials_indices",                      OWL_BUFPTR, OWL_OFFSETOF(SimpleObjTriangleData, materials_indices)},
            { /*sentinel */ }
        };

    OWLGeomType floor_geom_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(SimpleObjTriangleData), floor_vars, -1);
    owlGeomTypeSetClosestHit(floor_geom_type, RADIANCE_RAY, m_module, "obj_triangle");
    OWLGeom floor_geom = owlGeomCreate(m_owl, floor_geom_type);

    OWLBuffer triangles_indices_buffer =  owlDeviceBufferCreate(m_owl,          OWL_INT3,                               triangles_indices.size(),       triangles_indices.data());
    OWLBuffer triangles_vertices_buffer = owlDeviceBufferCreate(m_owl,          OWL_FLOAT3,                             triangles_vertices.size(),      triangles_vertices.data());
    OWLBuffer triangles_normals_buffer = owlDeviceBufferCreate(m_owl,           OWL_FLOAT3,                             vertex_normals.size(),          vertex_normals.data());
    OWLBuffer triangles_normals_indices_buffer = owlDeviceBufferCreate(m_owl,   OWL_INT3,                               vertex_normals_indices.size(),  vertex_normals_indices.data());
    OWLBuffer triangles_materials_buffer = owlDeviceBufferCreate(m_owl,         OWL_USER_TYPE(lambertian_materials[0]), lambertian_materials.size(),    lambertian_materials.data());
    OWLBuffer triangles_materials_indices_buffer = owlDeviceBufferCreate(m_owl, OWL_INT,                                materials_indices.size(),       materials_indices.data());

    owlTrianglesSetIndices(floor_geom, triangles_indices_buffer, triangles_indices.size(), sizeof(vec3i), 0);
    owlTrianglesSetVertices(floor_geom, triangles_vertices_buffer, triangles_vertices.size(), sizeof(vec3f), 0);

    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.indices", triangles_indices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertices", triangles_vertices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_normals", triangles_normals_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "triangle_data.vertex_normals_indices", triangles_normals_indices_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "materials", triangles_materials_buffer);
    owlGeomSetBuffer(m_obj_triangle_geom, "materials_indices", triangles_materials_indices_buffer);

    OWLGroup floor_group = owlTrianglesGeomGroupCreate(m_owl, 1, &floor_geom);
    owlGroupBuildAccel(floor_group);

    return floor_group;
}

OWLGroup ImGuiViewer::create_emissive_triangles_group()
{
    vec3i indices[] =
        {
            vec3i(0, 1, 2),
            vec3i(1, 3, 2)
        };

    vec3f vertices[] =
        {
            vec3f(-1.0f, 2.0f, 2.0f),
            vec3f(1.0f, 2.0f, 2.0f),
            vec3f(-1.0f, 4.0f, 2.0f),
            vec3f(1.0f, 4.0f, 2.0f),
        };

    OWLVarDecl floor_vars[] =
        {
            { "triangle_data.indices",  OWL_BUFPTR, OWL_OFFSETOF(EmissiveTriangleData, triangle_data.indices)},
            { "triangle_data.vertices", OWL_BUFPTR, OWL_OFFSETOF(EmissiveTriangleData, triangle_data.vertices)},
            { "emissive",               OWL_FLOAT3, OWL_OFFSETOF(EmissiveTriangleData, emissive)},
            { /*sentinel */ }
        };

    OWLGeomType emsisive_geom_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(EmissiveTriangleData), floor_vars, -1);
    owlGeomTypeSetClosestHit(emsisive_geom_type, RADIANCE_RAY, m_module, "emissive_triangle");
    OWLGeom emissive_geom = owlGeomCreate(m_owl, emsisive_geom_type);

    OWLBuffer indices_buffer =  owlDeviceBufferCreate(m_owl, OWL_INT3,   2, indices);
    OWLBuffer vertices_buffer = owlDeviceBufferCreate(m_owl, OWL_FLOAT3, 4, vertices);
    owlTrianglesSetIndices(emissive_geom, indices_buffer, 2, sizeof(vec3i), 0);
    owlTrianglesSetVertices(emissive_geom, vertices_buffer, 4, sizeof(vec3f), 0);

    owlGeomSetBuffer(emissive_geom, "triangle_data.indices", indices_buffer);
    owlGeomSetBuffer(emissive_geom, "triangle_data.vertices", vertices_buffer);
    owlGeomSet3f(emissive_geom, "emissive", 4.0f, 4.0f, 4.0f);

    OWLGroup emissive_group = owlTrianglesGeomGroupCreate(m_owl, 1, &emissive_geom);
    owlGroupBuildAccel(emissive_group);

    return emissive_group;
}

OWLTexture ImGuiViewer::create_ldr_skysphere(const char* filepath)
{
    int width, height;
    float* data = Utils::read_image(filepath, width, height, true);

    std::vector<vec4uc> uchar4_skypshere;
    uchar4_skypshere.reserve(width * height);

    for (int i = 0; i < width * height; i++)
    {
        vec4uc pixel_val = vec4uc(data[i * 4 + 0] * 255,
                                  data[i * 4 + 1] * 255,
                                  data[i * 4 + 2] * 255,
                                  data[i * 4 + 3] * 255);

        uchar4_skypshere[i] = pixel_val;
    }

    OWLTexture skysphere_texture = owlTexture2DCreate(m_owl, OWL_TEXEL_FORMAT_RGBA8, width, height, uchar4_skypshere.data());
    stbi_image_free(data);

    return skysphere_texture;
}

OWLTexture ImGuiViewer::create_hdr_skysphere(const char* path_to_exr)
{
    int width;
    int height;

    float* data; // width * height * RGBA

    const char* err = nullptr;

    int ret = LoadEXR(&data, &width, &height, path_to_exr, &err);

    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "ERR : %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.

        }

        std::exit(0);
    } else {
        OWLTexture skysphere_texture = owlTexture2DCreate(m_owl, OWL_TEXEL_FORMAT_RGBA32F, width, height, data);

        free(data); // release memory of image data

        return skysphere_texture;
    }
}

ImGuiViewer::~ImGuiViewer()
{
    owlModuleRelease(m_module);
    owlRayGenRelease(m_ray_gen);
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

    AffineSpace3f camera_view_matrix = AffineSpace3f::lookat(camera.position, camera.getAt(), camera.getUp());

    m_frame_number = 0;

    owlRayGenSet2i(m_ray_gen, "frame_buffer_size", fbSize.x, fbSize.y);
    owlRayGenSet3f(m_ray_gen, "camera.position", (const owl3f&)camera.position);
    owlRayGenSet3f(m_ray_gen, "camera.direction_00", (const owl3f&)camera_direction_00);
    owlRayGenSet3f(m_ray_gen, "camera.direction_dx", (const owl3f&)camera_du);
    owlRayGenSet3f(m_ray_gen, "camera.direction_dy", (const owl3f&)camera_dv);
    owlRayGenSetRaw(m_ray_gen, "camera.view_matrix", &camera_view_matrix);

    owlParamsSet1ui(m_launch_params, "frame_number", m_frame_number);

    m_sbtDirty = true;
}

void ImGuiViewer::resize(const owl::vec2i& new_size)
{
    viewer::OWLViewer::resize(new_size);

    m_accumulation_buffer.resize(sizeof(vec3f) * fbSize.x * fbSize.y);
    m_float4_frame_buffer.resize(sizeof(float4) * fbSize.x * fbSize.y);
    m_normal_buffer.resize(sizeof(float4) * fbSize.x * fbSize.y);
    m_albedo_buffer.resize(sizeof(float4) * fbSize.x * fbSize.y);

    owlParamsSet1ul(m_launch_params, "accumulation_buffer", (uint64_t)m_accumulation_buffer.d_pointer());
    owlParamsSet1ul(m_launch_params, "float4_frame_buffer", (uint64_t)m_float4_frame_buffer.d_pointer());
    owlParamsSet1ul(m_launch_params, "normal_buffer", (uint64_t)m_normal_buffer.d_pointer());
    owlParamsSet1ul(m_launch_params, "albedo_buffer", (uint64_t)m_albedo_buffer.d_pointer());

    m_denoiser.setup_hdr(m_owl, fbSize);

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
        ImGuiIO& io = ImGui::GetIO();

        ImGui::Begin("Application Settings");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::Text("Sample count: %d", m_frame_number);

        ImGui::Separator();

        ImGui::Text("Cook Torrance BRDF");
        ImGui::ColorPicker3("Albedo", (float*)&m_obj_material.albedo);
        ImGui::SliderFloat("Metallic", &m_obj_material.metallic, 0.0f, 1.0f);
        ImGui::SliderFloat("Roughness", &m_obj_material.roughness, 0.0f, 1.0f);
        ImGui::SliderFloat("Reflectance", &m_obj_material.reflectance, 0.0f, 1.0f);

        ImGui::Separator();

        ImGui::Text("General settings");
        ImGui::SliderInt("Max bounces", &m_max_bounces, 0, 64);

        ImGui::Separator();

        ImGui::Text("OptiX AI Denoiser");
        ImGui::Checkbox("Enable denoiser", &m_denoiser_on);
        ImGui::SliderFloat("Blend factor", &m_denoiser.m_blend_factor, 1.0f, 32.0f);

        ImGui::End();
    }

    // Rendering
    ImGui::Render();
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
        m_frame_number = 0;

    m_previous_obj_material = m_obj_material;
}

void ImGuiViewer::update_max_bounces()
{
    //The max bounces count has changed
    if (m_max_bounces != m_previous_max_bounces)
    {
        m_frame_number = 0;

        owlParamsSet1i(m_launch_params, "max_bounces", m_max_bounces);
    }

    m_previous_max_bounces = m_max_bounces;
}

void ImGuiViewer::render()
{
    imgui_render();

    owlParamsSet1b(m_launch_params, "mouse_moving", m_mouse_moving);
    if (m_mouse_moving)
        m_frame_number = 0;

    update_max_bounces();
    update_obj_material();
    owlParamsSet1ui(m_launch_params, "frame_number", ++m_frame_number);

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

    owlLaunch2D(m_ray_gen, fbSize.x, fbSize.y, m_launch_params);

    if (!m_mouse_moving)
    {

        //We're going to denoise unless the denoiser has been un checked by the user
        if (m_denoiser_on)
        {
            float4* float4_hdr_output;

            m_denoiser.denoise(m_float4_frame_buffer, m_normal_buffer, m_albedo_buffer, &float4_hdr_output, m_frame_number);
            hdr_tone_mapping(float4_hdr_output, fbSize.x, fbSize.y, fbPointer);
        }
        else
            hdr_tone_mapping((float4*)m_float4_frame_buffer.d_pointer(), fbSize.x, fbSize.y, fbPointer);
    }
    else
        hdr_tone_mapping((float4*)m_normal_buffer.d_pointer(), fbSize.x, fbSize.y, fbPointer);
}

void ImGuiViewer::draw()
{
    owl::viewer::OWLViewer::draw();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

/**
    * Disabling the denoiser while we're moving the camera
    */
void ImGuiViewer::mouseButtonLeft(const vec2i &where, bool pressed) { m_mouse_moving = pressed; }
void ImGuiViewer::mouseButtonRight(const vec2i& where, bool pressed) { m_mouse_moving = pressed; }
void ImGuiViewer::mouseButtonCenter(const vec2i& where, bool pressed) { m_mouse_moving = pressed; }
