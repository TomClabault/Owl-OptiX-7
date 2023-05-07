#include "scene.h"

#include "objUtils.h"

#define STBI_SSE2
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

void Scene::load_obj_into_scene(const char* filepath)
{
    OBJUtils::read_obj(filepath, m_indices, m_vertices, m_vertex_normals, m_vertex_normals_indices, m_materials, m_materials_indices);
}

void Scene::load_obj_into_scene_no_vertex_normals(const char* filepath)
{
    OBJUtils::read_obj_no_vertex_normals(filepath, m_indices, m_vertices, m_materials, m_materials_indices);
}

float* read_image( const char *filename, int& width, int& height, const bool flipY )
{
    stbi_set_flip_vertically_on_load(flipY);

    int channels;
    stbi_ldr_to_hdr_scale(1.0f);
    stbi_ldr_to_hdr_gamma(2.2f);
    float *data= stbi_loadf(filename, &width, &height, &channels, 4);

    if(!data)
    {
        printf("[error] loading '%s'...\n%s\n", filename, stbi_failure_reason());
        return {};
    }

    return data;
}

void Scene::load_skysphere(const char* filepath)
{
    float* data = read_image(filepath, m_skysphere_width, m_skysphere_height, true);

    m_skysphere.resize(m_skysphere_width * m_skysphere_height);

    for (int i = 0; i < m_skysphere_width * m_skysphere_height; i++)
        m_skysphere[i] = vec4uc(0);

    for (int i = 0; i < m_skysphere_width * m_skysphere_height; i++)
    {
        //Gamma correction included
        vec4uc pixel_val = vec4uc(std::pow(data[i * 4 + 0], 1.0f / 2.2f) * 255,
                                  std::pow(data[i * 4 + 1], 1.0f / 2.2f) * 255,
                                  std::pow(data[i * 4 + 2], 1.0f / 2.2f) * 255,
                                  data[i * 4 + 3] * 255);

        m_skysphere[i] = pixel_val;
    }

    stbi_image_free(data);
}
