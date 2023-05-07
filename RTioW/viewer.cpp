#include "shader.h"
#include "viewer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

extern "C" char shader_ptx[];

std::vector<LambertianSphere> create_lambertian_spheres()
{
    std::vector<LambertianSphere> spheres;

    LambertianSphere sphere;
    sphere.sphere.center = vec3f(0.0f, 0.0f, -3.0f);
    sphere.sphere.radius = 1.0f;
    sphere.material.albedo = vec3f(1.0f, 0.0f, 0.0f);

    spheres.push_back(sphere);

    return spheres;
}

vec3i plane_indices[] = {
    { 0, 1, 2 },
    { 1, 3, 2 }
};

vec3f plane_vertices[] = {
    vec3f(-100.0f, -1.0f, 100.0f),
    vec3f(100.0f, -1.0f, 100.0f),
    vec3f(-100.0f, -1.0f, -100.0f),

    vec3f(100.0f, -1.0f, -100.0f),
};

Viewer::Viewer()
{
    setTitle("Ray tracing in one weekend");

    m_owl_context = owlContextCreate(nullptr, 1);
    m_module = owlModuleCreate(m_owl_context, shader_ptx);

    OWLVarDecl lambertian_sphere_geometry_vars[] = {
        { "primitives",  OWL_BUFPTR, OWL_OFFSETOF(LambertianSpheresGeometryData, primitives)},
        { /* sentinel to mark end of list */ }
    };

    OWLGeomType lambertian_sphere_geometry_type = owlGeomTypeCreate(m_owl_context, OWL_GEOMETRY_USER, sizeof(LambertianSpheresGeometryData), lambertian_sphere_geometry_vars, -1);
    owlGeomTypeSetClosestHit(lambertian_sphere_geometry_type, 0, m_module, "lambertian_spheres");
    owlGeomTypeSetIntersectProg(lambertian_sphere_geometry_type, 0, m_module, "lambertian_spheres");
    owlGeomTypeSetBoundsProg(lambertian_sphere_geometry_type, m_module, "lambertian_spheres");

    owlBuildPrograms(m_owl_context);

    m_lambertian_spheres = create_lambertian_spheres();

    OWLGeom lambertian_sphere_geometry = owlGeomCreate(m_owl_context, lambertian_sphere_geometry_type);
    OWLBuffer lambertian_spheres_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(m_lambertian_spheres[0]), m_lambertian_spheres.size(), m_lambertian_spheres.data());
    owlGeomSetPrimCount(lambertian_sphere_geometry, m_lambertian_spheres.size());
    owlGeomSetBuffer(lambertian_sphere_geometry, "primitives", lambertian_spheres_buffer);

    OWLGeom spheres_geoms[] = {
        lambertian_sphere_geometry
    };

    OWLVarDecl triangle_vars[] = {
        { "indices", OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, indices)},
        { "vertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleGeomData, vertices)},
        { /* sentinel */ }
    };

    OWLBuffer triangle_indices_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(vec3i), 2, plane_indices);
    OWLBuffer triangle_vertices_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(vec3f), 4, plane_vertices);

    OWLGeomType triangle_geometry_type = owlGeomTypeCreate(m_owl_context, OWL_TRIANGLES, sizeof(TriangleGeomData), triangle_vars, -1);
    OWLGeom triangle_geom = owlGeomCreate(m_owl_context, triangle_geometry_type);

    owlGeomTypeSetClosestHit(triangle_geometry_type, 0, m_module, "triangle");
    owlTrianglesSetIndices(triangle_geom, triangle_indices_buffer, 2, sizeof(vec3i), 0);
    owlTrianglesSetVertices(triangle_geom, triangle_vertices_buffer, 4, sizeof(vec3f), 0);

    owlGeomSetBuffer(triangle_geom, "indices", triangle_indices_buffer);
    owlGeomSetBuffer(triangle_geom, "vertices", triangle_vertices_buffer);

    OWLGroup triangle_group = owlTrianglesGeomGroupCreate(m_owl_context, 1, &triangle_geom);
    owlGroupBuildAccel(triangle_group);

    OWLGroup spheres_group = owlUserGeomGroupCreate(m_owl_context, 1, spheres_geoms);
    owlGroupBuildAccel(spheres_group);

    OWLGroup scene = owlInstanceGroupCreate(m_owl_context, 2);
    owlInstanceGroupSetChild(scene, 0, spheres_group);
    owlInstanceGroupSetChild(scene, 1, triangle_group);
    owlGroupBuildAccel(scene);

    OWLVarDecl ray_gen_variables[] = {
        { "scene",                  OWL_GROUP,          OWL_OFFSETOF(RayGenData, scene) },
        { "frame_buffer_size",      OWL_INT2,           OWL_OFFSETOF(RayGenData, frame_buffer_size) },
        { "frame_buffer",           OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, frame_buffer)},
        { "accumulation_buffer",    OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, accumulation_buffer)},
        { "frame_number",           OWL_UINT,           OWL_OFFSETOF(RayGenData, frame_number)},
        { "camera.position",        OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.position) },
        { "camera.direction_00",    OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_00) },
        { "camera.direction_dx",    OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dx) },
        { "camera.direction_dy",    OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dy) },
        { /* sentinel */ }
    };

    m_ray_gen_program = owlRayGenCreate(m_owl_context, m_module, "ray_gen", sizeof(RayGenData), ray_gen_variables, -1);

    cudaMallocManaged(&m_accumulation_buffer, sizeof(vec3f) * fbSize.x * fbSize.y);

    camera.position = vec3f(0.0f, 0.0f, 1.0f);
    owlRayGenSet2i(m_ray_gen_program,     "frame_buffer_size",    fbSize.x, fbSize.y);
    owlRayGenSet1ul(m_ray_gen_program,    "frame_buffer",         (uint64_t)fbPointer);
    owlRayGenSet1ul(m_ray_gen_program,    "accumulation_buffer",  (uint64_t)m_accumulation_buffer);
    owlRayGenSet1ui(m_ray_gen_program,    "frame_number",         m_frame_number);
    owlRayGenSetGroup(m_ray_gen_program,  "scene",                scene);


    OWLVarDecl miss_prog_vars[] = {
        { "skysphere", OWL_TEXTURE, OWL_OFFSETOF(MissProgData, skysphere) },
        { /* sentinel */ }
    };


    OWLMissProg miss_program = owlMissProgCreate(m_owl_context, m_module, "miss", sizeof(MissProgData), miss_prog_vars, -1);

    load_skysphere("../../common_data/industrial_sunset_puresky.jpg");
    OWLTexture skysphere_tex = owlTexture2DCreate(m_owl_context, OWL_TEXEL_FORMAT_RGBA8, m_skysphere_width, m_skysphere_height, m_skysphere.data());
    owlMissProgSetTexture(miss_program, "skysphere", skysphere_tex);

    owlBuildPrograms(m_owl_context);
    owlBuildPipeline(m_owl_context);
    owlBuildSBT(m_owl_context);
}

void Viewer::resize(const vec2i& new_size)
{
    viewer::OWLViewer::resize(new_size);
    cameraChanged();
}

void Viewer::cameraChanged()
{
    vec3f look_at = camera.getAt();
    vec3f up = camera.getUp();
    vec3f position = camera.getFrom();
    float cos_fov_y = camera.getCosFovy();
    float aspect = fbSize.x / (float)fbSize.y;

    vec3f camera_direction = normalize(look_at - position);
    vec3f camera_dx = cos_fov_y * aspect * normalize(cross(camera_direction, up));
    vec3f camera_dy = cos_fov_y * normalize(cross(camera_dx, camera_direction));

    vec3f camera_direction_00 = camera_direction;
    camera_direction_00 -= 0.5f * camera_dx;
    camera_direction_00 -= 0.5f * camera_dy;

    if (m_accumulation_buffer)
        cudaFree(m_accumulation_buffer);

    cudaMallocManaged(&m_accumulation_buffer, sizeof(vec3f) * fbSize.x * fbSize.y);

    owlRayGenSet2i(m_ray_gen_program,   "frame_buffer_size",   fbSize.x, fbSize.y);
    owlRayGenSet1ul(m_ray_gen_program,  "frame_buffer",       (uint64_t)fbPointer);
    owlRayGenSet1ul(m_ray_gen_program,  "accumulation_buffer",  (uint64_t)m_accumulation_buffer);
    owlRayGenSet3f(m_ray_gen_program,   "camera.position",     (const owl3f&) position);
    owlRayGenSet3f(m_ray_gen_program,   "camera.direction_00", (const owl3f&) camera_direction_00);
    owlRayGenSet3f(m_ray_gen_program,   "camera.direction_dx", (const owl3f&) camera_dx);
    owlRayGenSet3f(m_ray_gen_program,   "camera.direction_dy", (const owl3f&) camera_dy);

    m_frame_number = 0;
    m_sbt_dirty = true;
}

void Viewer::render()
{
    m_frame_number++;
    owlRayGenSet1ui(m_ray_gen_program, "frame_number", m_frame_number);
    owlBuildSBT(m_owl_context, OWLBuildSBTFlags::OWL_SBT_RAYGENS);

    auto start = std::chrono::high_resolution_clock::now();
    if (m_sbt_dirty)
    {
        owlBuildSBT(m_owl_context);
        m_sbt_dirty = false;
    }

    if (fbSize.x == 0 && fbSize.y == 0)
    {
        fbSize = vec2i(1, 1);
        resize(fbSize);

        owlBuildSBT(m_owl_context, OWLBuildSBTFlags::OWL_SBT_RAYGENS);
    }

    owlRayGenLaunch2D(m_ray_gen_program, fbSize.x, fbSize.y);

    auto stop = std::chrono::high_resolution_clock::now();

    print_frame_time(start, stop);
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
