#include "utils.h"

//#define STB_IMAGE_IMPELMENTATION already defined in imGuiViewer.cpp
#include "stb/stb_image.h"

#include "optix7.h"

float angle(const vec3f& vec_a, const vec3f& vec_b)
{
    return std::acos(dot(normalize(vec_a), normalize(vec_b)));
}

std::vector<vec3f> compute_smooth_normals(rapidobj::Result& mesh_data, const std::vector<vec3i>& indices, const std::vector<vec3f>& vertices)
{
    std::vector<vec3f> smooth_normals;

    for (int index = 0; index < indices.size(); index++)
    {
        const vec3i& triangle_indices = indices[index];

        int index_vertex_A = triangle_indices.x;
        int index_vertex_B = triangle_indices.y;
        int index_vertex_C = triangle_indices.z;

        const vec3f A = vertices[index_vertex_A];
        const vec3f B = vertices[index_vertex_B];
        const vec3f C = vertices[index_vertex_C];

        vec3f crossed = cross(B - A, C - A);
        const vec3f face_normal = normalize(crossed);

        //Smooth normal for Vertex A
        smooth_normals[index_vertex_A] += face_normal * angle(B - A, C - A);

        //Smooth normal for Vertex B
        smooth_normals[index_vertex_B] += face_normal * angle(C - B, A - B);

        //Smooth normal for Vertex C
        smooth_normals[index_vertex_C] += face_normal * angle(A - C, B - C);

        if(std::isnan(smooth_normals[index_vertex_A].x) || std::isnan(smooth_normals[index_vertex_B].x) || std::isnan(smooth_normals[index_vertex_C].x))
            std::cout << index << std::endl;
    }

    for (vec3f& smooth_normal : smooth_normals)
        smooth_normal = normalize(smooth_normal);

    return smooth_normals;
}

void Utils::read_obj_with_vertex_normals(const char* filepath, rapidobj::Result& mesh_data, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<vec3i>& vertex_normals_indices, std::vector<vec2f>& vertex_texcoords, std::vector<vec3i>& vertex_texcoords_indices, std::vector<rapidobj::Material>& materials, std::vector<int>& materials_indices)
{
    mesh_data = rapidobj::ParseFile(filepath);
    rapidobj::Triangulate(mesh_data);

    if (mesh_data.error.code.value() != 0)
    {
        std::cout << mesh_data.error.code.message() << std::endl;
        std::exit(0);
    }

    //Just counting the number of triangles
    unsigned int triangle_count = 0;
    for (int i = 0; i < mesh_data.shapes.size(); i++)
    {
        const rapidobj::Shape& shape = mesh_data.shapes[i];
        triangle_count += shape.mesh.indices.size() / 3;
    }

    //We don't have 3 times as many indices as there are triangles because 1
    //'index' in OptiX is already the indices of the three vertices of a triangle
    unsigned int indices_count = triangle_count;
    unsigned int vertices_count = mesh_data.attributes.positions.size();
    unsigned int materials_count = mesh_data.materials.size();
    unsigned int materials_indices_count = triangle_count;

    indices.reserve(indices_count);
    vertices.reserve(vertices_count);
    materials.reserve(materials_count);
    materials_indices.reserve(materials_indices_count);

    const rapidobj::Array<float>& mesh_positions = mesh_data.attributes.positions;
    const rapidobj::Array<float>& mesh_normals = mesh_data.attributes.normals;
    const rapidobj::Array<float>& mesh_texcoords = mesh_data.attributes.texcoords;
    materials = mesh_data.materials;


    for (int i = 0; i < mesh_positions.size(); i += 3)
        vertices.push_back(vec3f(mesh_positions[i + 0],
                                 mesh_positions[i + 1],
                                 mesh_positions[i + 2]));

    for (int i = 0; i < mesh_texcoords.size(); i += 2)
        vertex_texcoords.push_back(vec2f(mesh_texcoords[i + 0],
                                         mesh_texcoords[i + 1]));

    //If the vertex normals of the first shape are available,
    //we assume they are available for the whole OBJ
    if (mesh_data.shapes[0].mesh.indices[0].normal_index != -1)
        for (int i = 0; i < mesh_normals.size(); i += 3)
            vertex_normals.push_back(vec3f(mesh_normals[i + 0],
                                           mesh_normals[i + 1],
                                           mesh_normals[i + 2]));

    for (int shape_index = 0; shape_index < mesh_data.shapes.size(); shape_index++)
    {
        const rapidobj::Array<rapidobj::Index>& mesh_indices = mesh_data.shapes[shape_index].mesh.indices;
        const rapidobj::Array<int32_t>& mesh_materials_indices = mesh_data.shapes[shape_index].mesh.material_ids;

        for (int i = 0; i < mesh_indices.size(); i += 3)
        {
            indices.push_back(vec3i(mesh_indices[i + 0].position_index,
                                    mesh_indices[i + 1].position_index,
                                    mesh_indices[i + 2].position_index));

            vertex_texcoords_indices.push_back(vec3i(mesh_indices[i + 0].texcoord_index,
                                                     mesh_indices[i + 1].texcoord_index,
                                                     mesh_indices[i + 2].texcoord_index));

            //If the vertex normals are available
            if (mesh_indices[0].normal_index != -1)
                vertex_normals_indices.push_back(vec3i(mesh_indices[i + 0].normal_index,
                                                       mesh_indices[i + 1].normal_index,
                                                       mesh_indices[i + 2].normal_index));
        }

        for (int i = 0; i < mesh_materials_indices.size(); i++)
            materials_indices.push_back(mesh_materials_indices[i]);
    }
}

void Utils::read_obj(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<vec3i>& vertex_normal_indices, std::vector<vec2f>& vertex_texcoords, std::vector<vec3i>& vertex_texcoords_indices, std::vector<rapidobj::Material>& materials, std::vector<int>& materials_indices)
{
    rapidobj::Result mesh_data;
    read_obj_with_vertex_normals(filepath, mesh_data, indices, vertices, vertex_normals, vertex_normal_indices, vertex_texcoords, vertex_texcoords_indices, materials, materials_indices);

    //No vertex normals were read from the OBJ, we're going to have to compute
    //them ourselves
    if (vertex_normals.size() == 0)
    {
        vertex_normals = compute_smooth_normals(mesh_data, indices, vertices);
        vertex_normal_indices = indices;
    }
}

float* Utils::read_image( const char *filename, int& width, int& height, const bool flipY)
{
    stbi_set_flip_vertically_on_load(flipY);

    int channels;
    float *data= stbi_loadf(filename, &width, &height, &channels, STBI_rgb_alpha);

    if(!data)
    {
        printf("[error] loading '%s'...\n%s\n", filename, stbi_failure_reason());
        return {};
    }

    return data;
}

unsigned char* Utils::read_image_rgba_4x8(const char* texture_path, int& width, int& height)
{
    stbi_set_flip_vertically_on_load(true);

    int channels;
    unsigned char* data= stbi_load(texture_path, &width, &height, &channels, STBI_rgb_alpha);

    if(!data)
    {
        printf("[error] loading '%s'...\n%s\n", texture_path, stbi_failure_reason());
        return {};
    }

    return data;
}

cudaTextureObject_t Utils::create_simple_cuda_texture(unsigned char* data, int width, int height)
{
    cudaResourceDesc res_desc = {};

    cudaChannelFormatDesc channel_desc;
    int32_t numComponents = 4;
    int32_t pitch  = width*numComponents*sizeof(uint8_t);
    channel_desc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t pixelArray;
    CUDA_CHECK(MallocArray(&pixelArray,
                           &channel_desc,
                           width,height));

    CUDA_CHECK(Memcpy2DToArray(pixelArray,
                               /* offset */0, 0,
                               data,
                               pitch, pitch, height,
                               cudaMemcpyHostToDevice));

    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = pixelArray;

    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = cudaAddressModeWrap;
    tex_desc.addressMode[1]      = cudaAddressModeWrap;
    tex_desc.filterMode          = cudaFilterModeLinear;
    tex_desc.readMode            = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 1.0f;
    tex_desc.sRGB                = 1;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));

    return cuda_tex;
}
