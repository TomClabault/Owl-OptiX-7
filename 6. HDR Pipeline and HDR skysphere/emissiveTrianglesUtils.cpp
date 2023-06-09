#include "cudaBuffer.h"
#include "emissiveTrianglesUtils.h"

#include <unordered_set>

EmissiveTrianglesInfo EmissiveTrianglesUtils::extract_emissive_triangles(std::vector<vec3i>& indices,
                                                                    std::vector<vec3f>& vertices,
                                                                    std::vector<vec3f>& vertex_normals,
                                                                    std::vector<vec3i>& vertex_normals_indices,
                                                                    std::vector<rapidobj::Material>& obj_materials,
                                                                    std::vector<int>& materials_indices)
{
    EmissiveTrianglesInfo emissive_triangles_info;

    std::vector<int> emissive_triangles_indices;
    std::unordered_set<int> emissive_materials_indices;

    //Gathering all the indices of emissive materials
    int index = 0;
    for (rapidobj::Material& mat : obj_materials)
    {
        //If the material is emissive
        if (mat.emission[0] != 0.0f || mat.emission[1] != 0.0f || mat.emission[2] != 0.0f)
            emissive_materials_indices.insert(index);

        index++;
    }

    //We're going to loop through all materials indices. Each time we find a material
    //index that is in the emissive_materials_indices array, this means
    //that the corresponding triangle is emissive and thus, we're going to store
    //its indes in the output vector
    int triangle_primitive_index = 0;
    for (int mat_index : materials_indices)
    {
        if (emissive_materials_indices.find(mat_index) != emissive_materials_indices.end())
            //We found an emissive triangle so we're adding
            //its primitive ID to the output vector
            emissive_triangles_indices.push_back(triangle_primitive_index);

        triangle_primitive_index++;
    }

    if (emissive_materials_indices.size() > 0)
    {
        CUDABuffer primitive_indices_buffer;
        primitive_indices_buffer.resize(emissive_triangles_indices.size() * sizeof(int));
        primitive_indices_buffer.upload(emissive_triangles_indices.data(), emissive_triangles_indices.size());

        emissive_triangles_info.emissive_triangles_indices = (int*)primitive_indices_buffer.d_pointer();
    }

    emissive_triangles_info.count = emissive_triangles_indices.size();

    return emissive_triangles_info;
}
