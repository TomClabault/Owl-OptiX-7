#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "owl/owl.h"

#include "deviceCode.h"

using namespace owl;

class Scene
{
public:
    void load_obj_into_scene(const char* file_path);
    void load_skysphere(const char* file_path);

    std::vector<vec4uc> m_skysphere;
    int m_skysphere_width;
    int m_skysphere_height;

    std::vector<vec3i> m_indices;
    std::vector<vec3f> m_vertices;
    std::vector<vec3f> m_vertex_normals;
    std::vector<vec3i> m_vertex_normals_indices;
    std::vector<Material> m_materials;
    std::vector<int> m_materials_indices;
};

#endif
