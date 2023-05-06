#include "shader.h"
#include "viewer.h"

const vec3f Viewer::BACKGROUND_COLOR = vec3f(135.0f / 255.0f, 206.0f / 255.0f, 235.0f / 255.0f);

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

    std::vector<LambertianSphere> lambertian_spheres = create_lambertian_spheres();

    OWLGeom lambertian_sphere_geometry = owlGeomCreate(m_owl_context, lambertian_sphere_geometry_type);
    OWLBuffer lambertian_spheres_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(lambertian_spheres[0]), lambertian_spheres.size(), lambertian_spheres.data());
    owlGeomSetPrimCount(lambertian_sphere_geometry, lambertian_spheres.size());
    owlGeomSetBuffer(lambertian_sphere_geometry, "primitives", lambertian_spheres_buffer);

    OWLGeom spheres_geoms[] = {
        lambertian_sphere_geometry
    };
    OWLGroup spheres_group = owlUserGeomGroupCreate(m_owl_context, 1, spheres_geoms);
    owlGroupBuildAccel(spheres_group);

    OWLGroup scene = owlInstanceGroupCreate(m_owl_context, 1, &spheres_group);
    owlGroupBuildAccel(scene);


    OWLVarDecl ray_gen_variables[] = {
        { "scene",                  OWL_GROUP,          OWL_OFFSETOF(RayGenData, scene) },
        { "frame_buffer_size",      OWL_INT2,           OWL_OFFSETOF(RayGenData, frame_buffer_size) },
        { "frame_buffer",           OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, frame_buffer)},
        { "camera.position",        OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.position) },
        { "camera.direction_00",    OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_00) },
        { "camera.direction_dx",    OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dx) },
        { "camera.direction_dy",    OWL_FLOAT3,         OWL_OFFSETOF(RayGenData, camera.direction_dy) },
        { /* sentinel */ }
    };

    m_ray_gen_program = owlRayGenCreate(m_owl_context, m_module, "ray_gen", sizeof(RayGenData), ray_gen_variables, -1);

    owlRayGenSet2i(m_ray_gen_program,     "frame_buffer_size",    fbSize.x, fbSize.y);
    owlRayGenSet1ul(m_ray_gen_program,    "frame_buffer",         (uint64_t)fbPointer);
    owlRayGenSetGroup(m_ray_gen_program, "scene", scene);

    OWLVarDecl miss_prog_vars[] = {
        { "background_color", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, background_color) },
        { /* sentinel */ }
    };

    OWLMissProg miss_program = owlMissProgCreate(m_owl_context, m_module, "miss", sizeof(MissProgData), miss_prog_vars, -1);

    owlMissProgSet3f(miss_program, "background_color", Viewer::BACKGROUND_COLOR.x, Viewer::BACKGROUND_COLOR.y, Viewer::BACKGROUND_COLOR.z);

    owlBuildPrograms(m_owl_context);
    owlBuildPipeline(m_owl_context);

    OWLViewer::cameraChanged();
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
    float aspect = fbSize.x / fbSize.y;

    vec3f camera_direction = normalize(look_at - position);
    vec3f camera_dx = cos_fov_y * aspect * normalize(cross(camera_direction, up));
    vec3f camera_dy = cos_fov_y * normalize(cross(camera_dx, camera_direction));

    vec3f camera_direction_00 = camera_direction - 0.5f * camera_dx - 0.5f * camera_dy;

    owlRayGenSet2i(m_ray_gen_program, "frame_buffer_size", fbSize.x, fbSize.y);
    owlRayGenSet1ul(m_ray_gen_program, "frame_buffer", (uint64_t)fbPointer);
    owlRayGenSet3f(m_ray_gen_program, "camera.direction_00", (const owl3f&) camera_direction_00);
    owlRayGenSet3f(m_ray_gen_program, "camera.direction_dx", (const owl3f&) camera_dx);
    owlRayGenSet3f(m_ray_gen_program, "camera.direction_dy", (const owl3f&) camera_dy);

    m_sbt_dirty = true;
}

void Viewer::render()
{
    auto start = std::chrono::high_resolution_clock::now();
    if (m_sbt_dirty)
    {
        owlBuildSBT(m_owl_context);
        m_sbt_dirty = false;
    }

    owlRayGenLaunch2D(m_ray_gen_program, fbSize.x, fbSize.y);

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
