#include "objUtils.h"
#include "shader.h"
#include "viewer.h"

#include "owl/helper/optix.h"

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

std::vector<DielectricSphere> create_dielectric_spheres()
{
    std::vector<DielectricSphere> spheres;

    DielectricSphere sphere;
    sphere.sphere.center = vec3f(0.0f, 0.0f, -5.0f);
    sphere.sphere.radius = 1.0f;
    sphere.material.ior = 1.5;

    spheres.push_back(sphere);

    return spheres;
}

OWLGroup Viewer::create_floor_group()
{
    vec3i floor_indices[] = {
        { 0, 1, 2 },
        { 1, 3, 2 }
    };

    vec3f floor_vertices[] = {
        vec3f(-100.0f, -1.0f, 100.0f),
        vec3f(100.0f, -1.0f, 100.0f),
        vec3f(-100.0f, -1.0f, -100.0f),

        vec3f(100.0f, -1.0f, -100.0f),
    };

    OWLVarDecl floor_vars[] = {
        { "indices",    OWL_BUFPTR, OWL_OFFSETOF(MetalTriangleGeomData, indices)},
        { "vertices",   OWL_BUFPTR, OWL_OFFSETOF(MetalTriangleGeomData, vertices)},
        { "albedo",     OWL_FLOAT3, OWL_OFFSETOF(MetalTriangleGeomData, albedo)},
        { "roughness",  OWL_FLOAT,  OWL_OFFSETOF(MetalTriangleGeomData, roughness)},
        { /* sentinel */ }
    };

    OWLBuffer floor_indices_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(vec3i), 2, floor_indices);
    OWLBuffer flooor_vertices_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(vec3f), 4, floor_vertices);

    OWLGeomType floor_geometry_type = owlGeomTypeCreate(m_owl_context, OWL_TRIANGLES, sizeof(MetalTriangleGeomData), floor_vars, -1);
    OWLGeom floor_geom = owlGeomCreate(m_owl_context, floor_geometry_type);

    owlGeomTypeSetClosestHit(floor_geometry_type, 0, m_module, "metal_triangles");
    owlTrianglesSetIndices(floor_geom, floor_indices_buffer, 2, sizeof(vec3i), 0);
    owlTrianglesSetVertices(floor_geom, flooor_vertices_buffer, 4, sizeof(vec3f), 0);

    owlGeomSetBuffer(floor_geom, "indices", floor_indices_buffer);
    owlGeomSetBuffer(floor_geom, "vertices", flooor_vertices_buffer);
    owlGeomSet3f(floor_geom, "albedo", 0.9f, 0.9f, 0.9f);
    owlGeomSet1f(floor_geom, "roughness", 0.025f);

    OWLGroup floor_group = owlTrianglesGeomGroupCreate(m_owl_context, 1, &floor_geom);

    return floor_group;
}


OWLGroup Viewer::create_obj_group(const char* obj_file_path)
{
    std::vector<vec3i> indices;
    std::vector<vec3f> vertices;
    std::vector<vec3f> vertex_normals;
    std::vector<vec3i> vertex_normals_indices;
    std::vector<Material> materials;
    std::vector<int> materials_indices;

    OBJUtils::read_obj(obj_file_path, indices, vertices, vertex_normals, vertex_normals_indices, materials, materials_indices);

    OWLVarDecl triangleGeometryVars[] = {
        { "indices",            OWL_BUFPTR, OWL_OFFSETOF(ObjTriangleGeomData, indices)},
        { "vertices",           OWL_BUFPTR, OWL_OFFSETOF(ObjTriangleGeomData, vertices)},
        { "vertex_normals",            OWL_BUFPTR, OWL_OFFSETOF(ObjTriangleGeomData, vertex_normals)},
        { "vertex_normals_indices",    OWL_BUFPTR, OWL_OFFSETOF(ObjTriangleGeomData, vertex_normals_indices)},
        { "materials",          OWL_BUFPTR, OWL_OFFSETOF(ObjTriangleGeomData, materials)},
        { "materials_indices",  OWL_BUFPTR, OWL_OFFSETOF(ObjTriangleGeomData, materials_indices)},
        { /* sentinel */ }
    };

    OWLGeomType triangle_geometry_type = owlGeomTypeCreate(m_owl_context, OWL_TRIANGLES, sizeof(ObjTriangleGeomData), triangleGeometryVars, -1);
    owlGeomTypeSetClosestHit(triangle_geometry_type, 0, m_module, "obj_triangle");

    OWLBuffer triangles_indices_buffer = owlDeviceBufferCreate(m_owl_context,               OWL_INT3, indices.size(), indices.data());
    OWLBuffer triangles_vertices_buffer = owlDeviceBufferCreate(m_owl_context,              OWL_FLOAT3, vertices.size(), vertices.data());
    OWLBuffer triangles_normals_buffer = owlDeviceBufferCreate(m_owl_context,               OWL_FLOAT3, vertex_normals.size(), vertex_normals.data());
    OWLBuffer triangles_normals_indices_buffer = owlDeviceBufferCreate(m_owl_context,       OWL_INT3, vertex_normals_indices.size(), vertex_normals_indices.data());
    OWLBuffer triangles_materials_buffer = owlDeviceBufferCreate(m_owl_context,             OWL_USER_TYPE(materials[0]), materials.size(), materials.data());
    OWLBuffer triangles_materials_indices_buffers = owlDeviceBufferCreate(m_owl_context,    OWL_INT, materials_indices.size(), materials_indices.data());

    OWLGeom triangle_geom = owlGeomCreate(m_owl_context, triangle_geometry_type);

    owlTrianglesSetIndices(triangle_geom, triangles_indices_buffer, indices.size(), sizeof(vec3i), 0);
    owlTrianglesSetVertices(triangle_geom, triangles_vertices_buffer, vertices.size(), sizeof(vec3f), 0);

    owlGeomSetBuffer(triangle_geom, "indices", triangles_indices_buffer);
    owlGeomSetBuffer(triangle_geom, "vertices", triangles_vertices_buffer);
    owlGeomSetBuffer(triangle_geom, "vertex_normals", triangles_normals_buffer);
    owlGeomSetBuffer(triangle_geom, "vertex_normals_indices", triangles_normals_indices_buffer);
    owlGeomSetBuffer(triangle_geom, "materials", triangles_materials_buffer);
    owlGeomSetBuffer(triangle_geom, "materials_indices", triangles_materials_indices_buffers);

    OWLGroup triangle_group = owlTrianglesGeomGroupCreate(m_owl_context, 1, &triangle_geom);

    return triangle_group;
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

    m_lambertian_spheres = create_lambertian_spheres();

    OWLGeom lambertian_sphere_geometry = owlGeomCreate(m_owl_context, lambertian_sphere_geometry_type);
    OWLBuffer lambertian_spheres_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(m_lambertian_spheres[0]), m_lambertian_spheres.size(), m_lambertian_spheres.data());
    owlGeomSetPrimCount(lambertian_sphere_geometry, m_lambertian_spheres.size());
    owlGeomSetBuffer(lambertian_sphere_geometry, "primitives", lambertian_spheres_buffer);

    OWLVarDecl dieletric_spheres_geometry_vars[] = {
        { "primitives", OWL_BUFPTR, OWL_OFFSETOF(DielectricSpheresGeometryData, primitives)},
        { /* sentinel */ }
    };
    OWLGeomType dieletric_sphere_geometry_type = owlGeomTypeCreate(m_owl_context, OWL_GEOMETRY_USER, sizeof(DielectricSpheresGeometryData), dieletric_spheres_geometry_vars, -1);
    owlGeomTypeSetBoundsProg(dieletric_sphere_geometry_type, m_module, "dielectric_spheres");
    owlGeomTypeSetIntersectProg(dieletric_sphere_geometry_type, 0, m_module, "dielectric_spheres");
    owlGeomTypeSetClosestHit(dieletric_sphere_geometry_type, 0, m_module, "dielectric_spheres");

    OWLGeom dielectric_sphere_geom = owlGeomCreate(m_owl_context, dieletric_sphere_geometry_type);
    std::vector<DielectricSphere> dielectric_spheres = create_dielectric_spheres();
    OWLBuffer dielectric_spheres_buffer = owlDeviceBufferCreate(m_owl_context, OWL_USER_TYPE(DielectricSphere), dielectric_spheres.size(), dielectric_spheres.data());
    owlGeomSetPrimCount(dielectric_sphere_geom, dielectric_spheres.size());
    owlGeomSetBuffer(dielectric_sphere_geom, "primitives", dielectric_spheres_buffer);

    OWLGeom spheres_geoms[] = {
        lambertian_sphere_geometry,
        dielectric_sphere_geom
    };

    owlBuildPrograms(m_owl_context);

    OWLGroup floor_group = create_floor_group();
    owlGroupBuildAccel(floor_group);

    OWLGroup obj_group = create_obj_group("../../common_data/stanford_bunny.obj");
    owlGroupBuildAccel(obj_group);

    OWLGroup spheres_group = owlUserGeomGroupCreate(m_owl_context, 2, spheres_geoms);
    owlGroupBuildAccel(spheres_group);

    OWLGroup scene = owlInstanceGroupCreate(m_owl_context, 3);
    owlInstanceGroupSetChild(scene, 0, spheres_group);
    owlInstanceGroupSetChild(scene, 1, floor_group);
    owlInstanceGroupSetChild(scene, 2, obj_group);
    owlGroupBuildAccel(scene);

    OWLVarDecl ray_gen_variables[] = {
        { "scene",                  OWL_GROUP,          OWL_OFFSETOF(RayGenData, scene) },
        { "frame_buffer_size",      OWL_INT2,           OWL_OFFSETOF(RayGenData, frame_buffer_size) },
        { "frame_buffer",           OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, frame_buffer)},
        { "float_frame_buffer",           OWL_RAW_POINTER,    OWL_OFFSETOF(RayGenData, float_frame_buffer)},
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
    cudaMallocManaged(&m_float_frame_buffer,  sizeof(float4) * fbSize.x * fbSize.y);

    camera.position = vec3f(0.0f, 0.0f, 1.0f);
    owlRayGenSet2i(m_ray_gen_program,     "frame_buffer_size",    fbSize.x, fbSize.y);
    owlRayGenSet1ul(m_ray_gen_program,    "frame_buffer",         (uint64_t)fbPointer);
    owlRayGenSet1ul(m_ray_gen_program,    "float_frame_buffer",   (uint64_t)m_float_frame_buffer);
    owlRayGenSet1ul(m_ray_gen_program,    "accumulation_buffer",  (uint64_t)m_accumulation_buffer);
    owlRayGenSet1ui(m_ray_gen_program,    "frame_number",         m_frame_number);
    owlRayGenSetGroup(m_ray_gen_program,  "scene",                scene);


    OWLVarDecl miss_prog_vars[] = {
        { "skysphere", OWL_TEXTURE, OWL_OFFSETOF(MissProgData, skysphere) },
        { /* sentinel */ }
    };


    OWLMissProg miss_program = owlMissProgCreate(m_owl_context, m_module, "miss", sizeof(MissProgData), miss_prog_vars, -1);

    load_skysphere("../../common_data/industrial_sunset_puresky_bright.png");
    OWLTexture skysphere_tex = owlTexture2DCreate(m_owl_context, OWL_TEXEL_FORMAT_RGBA8, m_skysphere_width, m_skysphere_height, m_skysphere.data());
    owlMissProgSetTexture(miss_program, "skysphere", skysphere_tex);

    owlBuildPrograms(m_owl_context);
    owlBuildPipeline(m_owl_context);
    owlBuildSBT(m_owl_context);
}

void Viewer::setup_denoiser(const vec2i& newSize)
{
    if (denoiser)
        OPTIX_CHECK(optixDenoiserDestroy(denoiser));

    //The float frame buffer for the denoiser
    if (m_float_frame_buffer)
        cudaFree(m_float_frame_buffer);
    cudaMallocManaged(&m_float_frame_buffer, sizeof(float4) * newSize.x * newSize.y);

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};

    OPTIX_CHECK(optixDenoiserCreate(owlContextGetOptixContext(m_owl_context, 0) ,OPTIX_DENOISER_MODEL_KIND_LDR,&denoiserOptions,&denoiser));

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, newSize.x, newSize.y, &denoiserReturnSizes));

    denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
                                    denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    m_denoised_buffer.resize(newSize.x*newSize.y*sizeof(float4));

    // ------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(denoiser,0,
                                   newSize.x,newSize.y,
                                   denoiserState.d_pointer(),
                                   denoiserState.size(),
                                   denoiserScratch.d_pointer(),
                                   denoiserScratch.size()));
}

void Viewer::resize(const vec2i& new_size)
{
    viewer::OWLViewer::resize(new_size);

    //Resizing the accumulation buffer
    if (m_accumulation_buffer)
        cudaFree(m_accumulation_buffer);//Freeing it if it was already allocated
    cudaMallocManaged(&m_accumulation_buffer, sizeof(vec3f) * fbSize.x * fbSize.y);//Allocating the new buffer

    setup_denoiser(new_size);

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


    owlRayGenSet2i(m_ray_gen_program,   "frame_buffer_size",   fbSize.x, fbSize.y);
    owlRayGenSet1ul(m_ray_gen_program,  "frame_buffer",        (uint64_t)fbPointer);
    owlRayGenSet1ul(m_ray_gen_program,  "float_frame_buffer",  (uint64_t)m_float_frame_buffer);
    owlRayGenSet1ul(m_ray_gen_program,  "accumulation_buffer", (uint64_t)m_accumulation_buffer);
    owlRayGenSet3f(m_ray_gen_program,   "camera.position",     (const owl3f&) position);
    owlRayGenSet3f(m_ray_gen_program,   "camera.direction_00", (const owl3f&) camera_direction_00);
    owlRayGenSet3f(m_ray_gen_program,   "camera.direction_dx", (const owl3f&) camera_dx);
    owlRayGenSet3f(m_ray_gen_program,   "camera.direction_dy", (const owl3f&) camera_dy);

    m_frame_number = 0;
    m_sbt_dirty = true;
}

void Viewer::denoise_render()
{
    //setup_denoiser(fbSize);

    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    denoiserParams.blendFactor = 1.f / (float)m_frame_number;

    // -------------------------------------------------------
    OptixImage2D inputLayer;
    inputLayer.data = (CUdeviceptr) m_float_frame_buffer;
    /// Width of the image (in pixels)
    inputLayer.width = fbSize.x;
    /// Height of the image (in pixels)
    inputLayer.height = fbSize.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = fbSize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = m_denoised_buffer.d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = fbSize.x;
    /// Height of the image (in pixels)
    outputLayer.height = fbSize.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = fbSize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                    /*stream*/0,
                                    &denoiserParams,
                                    denoiserState.d_pointer(),
                                    denoiserState.size(),
                                    &denoiserGuideLayer,
                                    &denoiserLayer,1,
                                    /*inputOffsetX*/0,
                                    /*inputOffsetY*/0,
                                    denoiserScratch.d_pointer(),
                                    denoiserScratch.size()));

    cuda_float4_to_rgba<<<>>>();
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

    if (denoiser_on)
        denoise_render();

    auto stop = std::chrono::high_resolution_clock::now();

    print_frame_time(start, stop);
}

/**
 * Disabling the denoiser while we're moving the camera
 */
void Viewer::mouseButtonLeft(const vec2i &where, bool pressed) { denoiser_on = !pressed; }
void Viewer::mouseButtonRight(const vec2i& where, bool pressed) { denoiser_on = !pressed; }
void Viewer::mouseButtonCenter(const vec2i& where, bool pressed) { denoiser_on = !pressed; }

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
