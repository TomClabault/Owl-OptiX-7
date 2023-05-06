#include "deviceCode.h"
#include "viewer.h"

#include <thread>

const vec3f Viewer::INIT_CAMERA_POSITION = vec3f(0.0f, 0.0f, 0.0f);
const vec3f Viewer::INIT_CAMERA_LOOKAT = vec3f(0.0f, 0.0f, -1.0f);
const vec3f Viewer::INIT_CAMERA_UP = vec3f(0.0f, 1.0f, 0.0f);
const float Viewer::INIT_CAMERA_FOVY = 0.66f;

Viewer::Viewer(const Scene& rt_scene) : m_scene(rt_scene)
{
    setTitle("OptiX Ray Tracer");

    // Initialize CUDA and OptiX 7, and create an "owl device," a context to hold the
    // ray generation shader and output buffer. The "1" is the number of devices requested.
    m_owl = owlContextCreate(nullptr,1);
    // PTX is the intermediate code that the CUDA deviceCode.cu shader program is converted into.
    // You can see the machine-centric PTX code in
    // build\samples\s00-rayGenOnly\cuda_compile_ptx_1_generated_deviceCode.cu.ptx_embedded.c
    // This PTX intermediate code representation is then compiled into an OptiX module.
    // See https://devblogs.nvidia.com/how-to-get-started-with-optix-7/ for more information.
    m_module = owlModuleCreate(m_owl, deviceCode_ptx);

    OWLVarDecl rayGenVars[] = {
        { "scene",                      OWL_GROUP,          OWL_OFFSETOF(RayGenData, scene) },
        { "frame_number",               OWL_UINT,           OWL_OFFSETOF(RayGenData, frame_number) },
        { "frame_accumulation_buffer",  OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, frame_accumulation_buffer) },
        { "frame_buffer_ptr",           OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, frame_buffer_ptr) },
        { "fb_size",                    OWL_INT2,           OWL_OFFSETOF(RayGenData, fb_size) },
        { "camera.position",            OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.position) },
        { "camera.direction_00",        OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_00) },
        { "camera.du",                  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.du) },
        { "camera.dv",                  OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.dv) },
        { /* sentinel: */ nullptr }
    };

    // Allocate room for one RayGen shader, create it, and
    // hold on to it with the "owl" context
    m_rayGen = owlRayGenCreate(m_owl, m_module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);

    cudaMallocManaged(&m_frame_accumulation_buffer_ptr, sizeof(vec3f) * fbSize.x * fbSize.y);

    owlRayGenSet2i(m_rayGen, "fb_size", fbSize.x, fbSize.y);
    owlRayGenSet1ui(m_rayGen, "frame_number", 1);
    owlRayGenSet1ul(m_rayGen, "frame_accumulation_buffer", (uint64_t) m_frame_accumulation_buffer_ptr);
    owlRayGenSet1ul(m_rayGen, "frame_buffer_ptr", (uint64_t) fbPointer);

    OWLVarDecl triangleGeometryVars[] = {
        { "indices",            OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, indices)},
        { "vertices",           OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, vertices)},
        { "normals",            OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, normals)},
        { "normals_indices",    OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, normals_indices)},
        { "materials",          OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, materials)},
        { "materials_indices",  OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, materials_indices)},
        { /* sentinel */ }
    };

    OWLGeomType triangle_geometry_type = owlGeomTypeCreate(m_owl, OWL_TRIANGLES, sizeof(TriangleGeomData), triangleGeometryVars, -1);
    owlGeomTypeSetClosestHit(triangle_geometry_type, 0, m_module, "Triangle");

    OWLBuffer triangles_indices_buffer = owlDeviceBufferCreate(m_owl,               OWL_INT3, m_scene.m_indices.size(), m_scene.m_indices.data());
    OWLBuffer triangles_vertices_buffer = owlDeviceBufferCreate(m_owl,              OWL_FLOAT3, m_scene.m_vertices.size(), m_scene.m_vertices.data());
    OWLBuffer triangles_normals_buffer = owlDeviceBufferCreate(m_owl,               OWL_FLOAT3, m_scene.m_vertex_normals.size(), m_scene.m_vertex_normals.data());
    OWLBuffer triangles_normals_indices_buffer = owlDeviceBufferCreate(m_owl,       OWL_INT3, m_scene.m_vertex_normals_indices.size(), m_scene.m_vertex_normals_indices.data());
    OWLBuffer triangles_materials_buffer = owlDeviceBufferCreate(m_owl,             OWL_USER_TYPE(m_scene.m_materials[0]), m_scene.m_materials.size(), m_scene.m_materials.data());
    OWLBuffer triangles_materials_indices_buffers = owlDeviceBufferCreate(m_owl,    OWL_INT, m_scene.m_materials_indices.size(), m_scene.m_materials_indices.data());

    OWLGeom triangle_geom = owlGeomCreate(m_owl, triangle_geometry_type);

    owlTrianglesSetIndices(triangle_geom, triangles_indices_buffer, m_scene.m_indices.size(), sizeof(vec3i), 0);
    owlTrianglesSetVertices(triangle_geom, triangles_vertices_buffer, m_scene.m_vertices.size(), sizeof(vec3f), 0);

    owlGeomSetBuffer(triangle_geom, "indices", triangles_indices_buffer);
    owlGeomSetBuffer(triangle_geom, "vertices", triangles_vertices_buffer);
    owlGeomSetBuffer(triangle_geom, "normals", triangles_normals_buffer);
    owlGeomSetBuffer(triangle_geom, "normals_indices", triangles_normals_indices_buffer);
    owlGeomSetBuffer(triangle_geom, "materials", triangles_materials_buffer);
    owlGeomSetBuffer(triangle_geom, "materials_indices", triangles_materials_indices_buffers);

    OWLGroup triangle_group = owlTrianglesGeomGroupCreate(m_owl, 1, &triangle_geom);
    owlGroupBuildAccel(triangle_group);
    OWLGroup scene = owlInstanceGroupCreate(m_owl, 1, &triangle_group);
    owlGroupBuildAccel(scene);
    owlRayGenSetGroup(m_rayGen, "scene", scene);

    OWLVarDecl miss_prog_vars[] = {
        { "skysphere", OWL_TEXTURE, OWL_OFFSETOF(MissProgData, skysphere) },
        { /* sentinel */ }
    };

    OWLMissProg miss_program = owlMissProgCreate(m_owl, m_module, "miss", sizeof(MissProgData), miss_prog_vars, 1);

    OWLTexture skysphere = owlTexture2DCreate(m_owl, OWL_TEXEL_FORMAT_RGBA8, rt_scene.m_skysphere_width, rt_scene.m_skysphere_height, rt_scene.m_skysphere.data());

    owlMissProgSetTexture(miss_program, "skysphere", skysphere);

    owlBuildPrograms(m_owl);
    owlBuildPipeline(m_owl);
    owlBuildSBT(m_owl);
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

    if (m_frame_accumulation_buffer_ptr)
        cudaFree(m_frame_accumulation_buffer_ptr);
    cudaMallocManaged(&m_frame_accumulation_buffer_ptr, sizeof(vec3f) * fbSize.x * fbSize.y);

    m_frame_number = 0;

    owlRayGenSet2i(m_rayGen, "fb_size", fbSize.x, fbSize.y);
    owlRayGenSet1ui(m_rayGen, "frame_number", m_frame_number);
    owlRayGenSet1ul(m_rayGen, "frame_accumulation_buffer", (uint64_t) m_frame_accumulation_buffer_ptr);
    owlRayGenSet1ul(m_rayGen, "frame_buffer_ptr", (uint64_t) fbPointer);
    owlRayGenSet3f(m_rayGen, "camera.position", (const owl3f&)camera.position);
    owlRayGenSet3f(m_rayGen, "camera.direction_00", (const owl3f&)camera_direction_00);
    owlRayGenSet3f(m_rayGen, "camera.du", (const owl3f&)camera_du);
    owlRayGenSet3f(m_rayGen, "camera.dv", (const owl3f&)camera_dv);

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
