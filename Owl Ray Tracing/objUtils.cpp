#include "objUtils.h"

void OBJUtils::read_obj(const char* filepath, std::vector<vec3i>& indices, std::vector<vec3f>& vertices, std::vector<vec3f>& vertex_normals, std::vector<vec3i>& vertex_normal_indices, std::vector<Material>& materials, std::vector<int>& materials_indices)
{
    rapidobj::Result mesh_data = rapidobj::ParseFile(filepath);
    rapidobj::Triangulate(mesh_data);

    if (mesh_data.error.code.value() != 0)
    {
        std::cout << mesh_data.error.code.message() << std::endl;
        std::exit(0);
    }

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
    unsigned int vertex_normals_count = mesh_data.attributes.normals.size() / 3;
    unsigned int vertex_normal_indices_count = triangle_count;
    unsigned int materials_count = mesh_data.materials.size();
    unsigned int materials_indices_count = triangle_count;

    indices.reserve(indices_count);
    vertices.reserve(vertices_count);
    vertex_normals.reserve(vertex_normals_count);
    vertex_normal_indices.reserve(vertex_normal_indices_count);
    materials.reserve(materials_count);
    materials_indices.reserve(materials_indices_count);

    const rapidobj::Array<float>& mesh_positions = mesh_data.attributes.positions;
    const rapidobj::Array<float>& mesh_normals = mesh_data.attributes.normals;
    const std::vector<rapidobj::Material>& mesh_materials = mesh_data.materials;

    for (int i = 0; i < mesh_materials.size(); i++)
    {
        const rapidobj::Material& material = mesh_materials[i];

        Material ray_tracer_material;
        ray_tracer_material.ambient = vec3f(material.ambient[0], material.ambient[1], material.ambient[2]);
        ray_tracer_material.diffuse = vec3f(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
        ray_tracer_material.dissolve = material.dissolve;
        ray_tracer_material.emission = vec3f(material.emission[0], material.emission[1], material.emission[2]);
        ray_tracer_material.illum = material.illum;
        ray_tracer_material.ior = material.ior;
        ray_tracer_material.reflection_coefficient = 1.0;
        ray_tracer_material.shininess = material.shininess;
        ray_tracer_material.specular = vec3f(material.specular[0], material.specular[1], material.specular[2]);
        ray_tracer_material.transmittance = vec3f(material.transmittance[0], material.transmittance[1], material.transmittance[2]);

        materials.push_back(ray_tracer_material);
    }

    for (int i = 0; i < mesh_positions.size(); i += 3)
        vertices.push_back(vec3f(mesh_positions[i + 0],
                                 mesh_positions[i + 1],
                                 mesh_positions[i + 2]));

    for (int i = 0; i < mesh_positions.size(); i += 3)
        vertex_normals.push_back(vec3f(mesh_normals[i + 0],
                                       mesh_normals[i + 1],
                                       mesh_normals[i + 2]));

    for (int shape_index = 0; shape_index < mesh_data.shapes.size(); shape_index++)
    {
        const rapidobj::Array<rapidobj::Index>& mesh_indices = mesh_data.shapes[shape_index].mesh.indices;
        const rapidobj::Array<int32_t>& mesh_materials_indices = mesh_data.shapes[shape_index].mesh.material_ids;

        for (int i = 0; i < mesh_indices.size(); i += 3)
            indices.push_back(vec3i(mesh_indices[i + 0].position_index,
                              mesh_indices[i + 1].position_index,
                              mesh_indices[i + 2].position_index));

        for (int i = 0; i < mesh_indices.size(); i += 3)
            vertex_normal_indices.push_back(vec3i(mesh_indices[i + 0].normal_index,
                                                  mesh_indices[i + 1].normal_index,
                                                  mesh_indices[i + 2].normal_index));

        for (int i = 0; i < mesh_materials_indices.size(); i++)
            materials_indices.push_back(mesh_materials_indices[i]);
    }
}
