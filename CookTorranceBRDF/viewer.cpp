#include "geometriesData.h"
#include "objUtils.h"
#include "shader.h"
#include "viewer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <thread>

extern "C" char shader_ptx[];

Viewer::Viewer()
{
    setTitle("CookTorranceBRDF");

    m_owl = owlContextCreate(nullptr,1);
    m_module = owlModuleCreate(m_owl, shader_ptx);

    OWLVarDecl rayGenVars[] = {
        { "scene",                OWL_GROUP,          OWL_OFFSETOF(RayGenData, scene) },
        { "frame_number",         OWL_UINT,           OWL_OFFSETOF(RayGenData, frame_number) },
        { "accumulation_buffer",  OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, accumulation_buffer) },
        { "frame_buffer",         OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, frame_buffer) },
        { "frame_buffer_size",    OWL_INT2,           OWL_OFFSETOF(RayGenData, frame_buffer_size) },
        { "camera.position",      OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.position) },
        { "camera.direction_00",  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_00) },
        { "camera.direction_dx",  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dx) },
        { "camera.direction_dy",  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dy) },
        { /* sentinel: */ }
    };

    m_rayGen = owlRayGenCreate(m_owl, m_module, "ray_gen", sizeof(RayGenData), rayGenVars, -1);

    m_accumulation_buffer.resize(sizeof(vec3f) * fbSize.x * fbSize.y);
    owlRayGenSet1ui(m_rayGen, "frame_number", 1);
    owlRayGenSet1ul(m_rayGen, "accumulation_buffer", (uint64_t)m_accumulation_buffer.d_pointer());

    OWLGroup triangle_group = create_cook_torrance_obj_group("../../common_data/bunny_translated.obj");
    OWLGroup floor_group = create_floor_group();

    OWLGroup scene = owlInstanceGroupCreate(m_owl, 2);
    owlInstanceGroupSetChild(scene, 0, triangle_group);
    owlInstanceGroupSetChild(scene, 1, floor_group);
    owlGroupBuildAccel(scene);

    owlRayGenSetGroup(m_rayGen, "scene", scene);

    OWLVarDecl miss_prog_vars[] = {
        { "skysphere", OWL_TEXTURE, OWL_OFFSETOF(MissProgData, skysphere) },
        { /* sentinel */ }
    };

    OWLMissProg miss_program = owlMissProgCreate(m_owl, m_module, "miss", sizeof(MissProgData), miss_prog_vars, 1);

    load_skysphere("../../common_data/industrial_sunset_puresky_bright.png");
    OWLTexture skysphere = owlTexture2DCreate(m_owl, OWL_TEXEL_FORMAT_RGBA8, m_skysphere_width, m_skysphere_height, m_skysphere.data());
    owlMissProgSetTexture(miss_program, "skysphere", skysphere);

    owlBuildPrograms(m_owl);
    owlBuildPipeline(m_owl);
    owlBuildSBT(m_owl);
}

OWLGroup Viewer::create_cook_torrance_obj_group(const char* obj_file_path)
{
    std::vector<vec3i> indices;
    std::vector<vec3f> vertices;
    std::vector<vec3f> vertex_normals;
    std::vector<vec3i> vertex_normals_indices;
    std::vector<CookTorranceMaterial> materials;
    std::vector<int> materials_indices;

    OBJUtils::read_obj(obj_file_path, indices, vertices, vertex_normals, vertex_normals_indices, materials, materials_indices);

    OWLVarDecl triangleGeometryVars[] = {
        { "triangle_data.indices",                  OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.indices)},
        { "triangle_data.vertices",                 OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertices)},
        { "triangle_data.vertex_normals",           OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertex_normals)},
        { "triangle_data.vertex_normals_indices",   OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, triangle_data.vertex_normals_indices)},
        { "materials",                              OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, materials)},
        { "materials_indices",                      OWL_BUFPTR, OWL_OFFSETOF(CookTorranceTriangleData, materials_indices)},
        { /* sentinel */ }
    };

    OWLGeomType triangle_geometry_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(CookTorranceTriangleData), triangleGeometryVars, -1);
    owlGeomTypeSetClosestHit(triangle_geometry_type, 0, m_module, "cook_torrance_obj_triangle");

    OWLBuffer triangles_indices_buffer = owlDeviceBufferCreate(m_owl,               OWL_INT3, indices.size(), indices.data());
    OWLBuffer triangles_vertices_buffer = owlDeviceBufferCreate(m_owl,              OWL_FLOAT3, vertices.size(), vertices.data());
    OWLBuffer triangles_normals_buffer = owlDeviceBufferCreate(m_owl,               OWL_FLOAT3, vertex_normals.size(), vertex_normals.data());
    OWLBuffer triangles_normals_indices_buffer = owlDeviceBufferCreate(m_owl,       OWL_INT3, vertex_normals_indices.size(), vertex_normals_indices.data());
    OWLBuffer triangles_materials_buffer = owlDeviceBufferCreate(m_owl,             OWL_USER_TYPE(materials[0]), materials.size(), materials.data());
    OWLBuffer triangles_materials_indices_buffers = owlDeviceBufferCreate(m_owl,    OWL_INT, materials_indices.size(), materials_indices.data());

    OWLGeom triangle_geom = owlGeomCreate(m_owl, triangle_geometry_type);

    owlTrianglesSetIndices(triangle_geom, triangles_indices_buffer, indices.size(), sizeof(vec3i), 0);
    owlTrianglesSetVertices(triangle_geom, triangles_vertices_buffer, vertices.size(), sizeof(vec3f), 0);

    owlGeomSetBuffer(triangle_geom, "triangle_data.indices", triangles_indices_buffer);
    owlGeomSetBuffer(triangle_geom, "triangle_data.vertices", triangles_vertices_buffer);
    owlGeomSetBuffer(triangle_geom, "triangle_data.vertex_normals", triangles_normals_buffer);
    owlGeomSetBuffer(triangle_geom, "triangle_data.vertex_normals_indices", triangles_normals_indices_buffer);
    owlGeomSetBuffer(triangle_geom, "materials", triangles_materials_buffer);
    owlGeomSetBuffer(triangle_geom, "materials_indices", triangles_materials_indices_buffers);

    OWLGroup triangle_group = owlTrianglesGeomGroupCreate(m_owl, 1, &triangle_geom);
    owlGroupBuildAccel(triangle_group);

    return triangle_group;
}

OWLGroup Viewer::create_floor_group()
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

float* read_image( const char *filename, int& width, int& height, const bool flipY )
{
    stbi_set_flip_vertically_on_load(flipY);

    int channels;
    float *data= stbi_loadf(filename, &width, &height, &channels, 4);

    if(!data)
    {
        printf("[error] loading '%s'...\n%s\n", filename, stbi_failure_reason());
        return {};
    }

    return data;
}

void Viewer::load_skysphere(const char* filepath)
{
    float* data = read_image(filepath, m_skysphere_width, m_skysphere_height, true);

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

Viewer::~Viewer()
{
    owlModuleRelease(m_module);
    owlRayGenRelease(m_rayGen);
    owlContextDestroy(m_owl);
}

void Viewer::cameraChanged()
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
    owlRayGenSet1ui(m_rayGen, "frame_number", m_frame_number);
    owlRayGenSet1ul(m_rayGen, "accumulation_buffer", (uint64_t)m_accumulation_buffer.d_pointer());
    owlRayGenSet1ul(m_rayGen, "frame_buffer", (uint64_t) fbPointer);
    owlRayGenSet3f(m_rayGen, "camera.position", (const owl3f&)camera.position);
    owlRayGenSet3f(m_rayGen, "camera.direction_00", (const owl3f&)camera_direction_00);
    owlRayGenSet3f(m_rayGen, "camera.direction_dx", (const owl3f&)camera_du);
    owlRayGenSet3f(m_rayGen, "camera.direction_dy", (const owl3f&)camera_dv);

    m_sbtDirty = true;
}

void Viewer::resize(const owl::vec2i& new_size)
{
    viewer::OWLViewer::resize(new_size);
    cameraChanged();
}

void Viewer::render()
{
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::high_resolution_clock::now();

    m_frame_number++;
    owlRayGenSet1ui(m_rayGen, "frame_number", m_frame_number);
    owlBuildSBT(m_owl, OWLBuildSBTFlags::OWL_SBT_RAYGENS);

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

    owlRayGenLaunch2D(m_rayGen, fbSize.x, fbSize.y);

    auto stop = std::chrono::high_resolution_clock::now();

    print_frame_time(start, stop);
}

void Viewer::print_frame_time(std::chrono::time_point<std::chrono::steady_clock>& start, std::chrono::time_point<std::chrono::steady_clock>& stop)
{
    long long int micro_count = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << "[" << fbSize.x << "x" << fbSize.y << "] ";
    if (micro_count < 1000)
        std::cout << "Frame time: " << micro_count << " microseconds" << std::endl;
    else
    {
        if (micro_count < 1000000)
            std::cout << "Frame time: " << micro_count / 1000.0f << " ms" << std::endl;
        else
            std::cout << "Frame time: " << micro_count / 1000000.0f << " s" << std::endl;
    }
}
