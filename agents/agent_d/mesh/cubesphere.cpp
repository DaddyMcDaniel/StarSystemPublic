/**
 * Cube-Sphere Primitive Generator for Agent D - T02
 * ================================================
 * 
 * C++ implementation for high-performance cube-sphere generation.
 * Generates uniform cube-sphere with shared vertices and seam-aware UVs.
 * 
 * Features:
 * - 6 face grids at configurable resolution (N×N)
 * - Shared edge vertices for seamless geometry  
 * - Seam-safe UV mapping per face (no polar pinch)
 * - Triangle indices (two triangles per quad)
 * - Export to manifest format with binary buffers
 * 
 * Compile:
 *   g++ -std=c++17 -O3 cubesphere.cpp -o cubesphere
 * 
 * Usage:
 *   ./cubesphere --face_res 32 --output sphere_manifest.json
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>

struct Vec3 {
    float x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
    
    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    Vec3 normalized() const {
        float len = length();
        return Vec3(x/len, y/len, z/len);
    }
};

struct Vec2 {
    float u, v;
    
    Vec2() : u(0), v(0) {}
    Vec2(float u, float v) : u(u), v(v) {}
};

class CubeSphereGenerator {
private:
    int face_res;
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::vector<uint32_t> indices;
    
    // For shared vertex detection
    std::unordered_map<uint64_t, uint32_t> vertex_map;
    
    uint64_t get_vertex_key(const Vec3& pos, float epsilon = 1e-6f) {
        // Quantize position for sharing detection
        int64_t x = static_cast<int64_t>(pos.x / epsilon);
        int64_t y = static_cast<int64_t>(pos.y / epsilon);
        int64_t z = static_cast<int64_t>(pos.z / epsilon);
        
        // Pack into 64-bit key
        return (static_cast<uint64_t>(x & 0x1FFFFF) << 42) |
               (static_cast<uint64_t>(y & 0x1FFFFF) << 21) |
               (static_cast<uint64_t>(z & 0x1FFFFF));
    }
    
    Vec2 compute_face_uv(int face_id, float u, float v) {
        // Simple face-based UV mapping to avoid seams
        float face_u_offset = (face_id % 3) / 3.0f;
        float face_v_offset = (face_id / 3) / 2.0f;
        
        float final_u = face_u_offset + u / 3.0f;
        float final_v = face_v_offset + v / 2.0f;
        
        return Vec2(final_u, final_v);
    }
    
    void generate_face(int face_id, const Vec3& normal, const Vec3& up, const Vec3& right) {
        int n = face_res;
        std::vector<uint32_t> face_vertices;
        
        // Generate grid vertices for this face
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                // UV coordinates in [0,1] range for this face
                float u = static_cast<float>(i) / (n - 1);
                float v = static_cast<float>(j) / (n - 1);
                
                // Convert to [-1,1] cube coordinates
                float cube_u = u * 2.0f - 1.0f;
                float cube_v = v * 2.0f - 1.0f;
                
                // Calculate position on cube face
                Vec3 pos = normal + right * cube_u + up * cube_v;
                
                // Project to unit sphere
                Vec3 sphere_pos = pos.normalized();
                
                // Check if this vertex is shared with another face
                uint64_t vertex_key = get_vertex_key(sphere_pos);
                
                auto it = vertex_map.find(vertex_key);
                if (it != vertex_map.end()) {
                    // Reuse existing vertex
                    face_vertices.push_back(it->second);
                } else {
                    // Create new vertex
                    uint32_t vertex_idx = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(sphere_pos);
                    normals.push_back(sphere_pos); // For sphere, normal = position
                    
                    // Compute seam-safe UV for this face
                    Vec2 face_uv = compute_face_uv(face_id, u, v);
                    uvs.push_back(face_uv);
                    
                    vertex_map[vertex_key] = vertex_idx;
                    face_vertices.push_back(vertex_idx);
                }
            }
        }
        
        // Generate triangle indices for this face
        generate_face_indices(face_vertices, n);
    }
    
    void generate_face_indices(const std::vector<uint32_t>& face_vertices, int n) {
        for (int j = 0; j < n - 1; j++) {
            for (int i = 0; i < n - 1; i++) {
                // Quad corners
                uint32_t bl = face_vertices[j * n + i];           // bottom-left
                uint32_t br = face_vertices[j * n + (i + 1)];     // bottom-right
                uint32_t tl = face_vertices[(j + 1) * n + i];     // top-left
                uint32_t tr = face_vertices[(j + 1) * n + (i + 1)]; // top-right
                
                // Two triangles per quad
                // Triangle 1: bl -> br -> tl
                indices.push_back(bl);
                indices.push_back(br);
                indices.push_back(tl);
                
                // Triangle 2: br -> tr -> tl
                indices.push_back(br);
                indices.push_back(tr);
                indices.push_back(tl);
            }
        }
    }
    
public:
    CubeSphereGenerator(int face_res) : face_res(face_res) {}
    
    void generate() {
        std::cout << "🔮 Generating cube-sphere with face resolution " 
                  << face_res << "×" << face_res << std::endl;
        
        // Define cube face configurations (normal, up, right vectors)
        struct FaceConfig {
            Vec3 normal, up, right;
        };
        
        FaceConfig faces[6] = {
            // +X face (right)
            {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, -1)},
            // -X face (left)
            {Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)},
            // +Y face (top)
            {Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(1, 0, 0)},
            // -Y face (bottom)
            {Vec3(0, -1, 0), Vec3(0, 0, -1), Vec3(1, 0, 0)},
            // +Z face (front)
            {Vec3(0, 0, 1), Vec3(0, 1, 0), Vec3(1, 0, 0)},
            // -Z face (back)
            {Vec3(0, 0, -1), Vec3(0, 1, 0), Vec3(-1, 0, 0)}
        };
        
        // Generate all faces
        for (int face_id = 0; face_id < 6; face_id++) {
            const auto& face = faces[face_id];
            generate_face(face_id, face.normal, face.up, face.right);
        }
        
        std::cout << "✅ Generated " << vertices.size() << " vertices, " 
                  << indices.size() / 3 << " triangles" << std::endl;
        std::cout << "📊 Shared vertices: " 
                  << (face_res * face_res * 6) - vertices.size() 
                  << " vertices saved" << std::endl;
    }
    
    void export_manifest(const std::string& output_path, const std::string& buffer_dir = "") {
        std::string actual_buffer_dir = buffer_dir.empty() ? 
            output_path.substr(0, output_path.find_last_of("/\\") + 1) : 
            buffer_dir + "/";
        
        // Extract filename without extension
        size_t last_slash = output_path.find_last_of("/\\");
        size_t last_dot = output_path.find_last_of(".");
        std::string base_name = output_path.substr(
            last_slash + 1, 
            last_dot - last_slash - 1
        );
        
        // Write binary buffers
        std::string positions_file = actual_buffer_dir + base_name + "_positions.bin";
        std::string normals_file = actual_buffer_dir + base_name + "_normals.bin";
        std::string uvs_file = actual_buffer_dir + base_name + "_uvs.bin";
        std::string indices_file = actual_buffer_dir + base_name + "_indices.bin";
        
        // Write positions
        std::ofstream pos_out(positions_file, std::ios::binary);
        pos_out.write(reinterpret_cast<const char*>(vertices.data()), 
                     vertices.size() * sizeof(Vec3));
        pos_out.close();
        
        // Write normals
        std::ofstream norm_out(normals_file, std::ios::binary);
        norm_out.write(reinterpret_cast<const char*>(normals.data()), 
                      normals.size() * sizeof(Vec3));
        norm_out.close();
        
        // Write UVs
        std::ofstream uv_out(uvs_file, std::ios::binary);
        uv_out.write(reinterpret_cast<const char*>(uvs.data()), 
                    uvs.size() * sizeof(Vec2));
        uv_out.close();
        
        // Write indices
        std::ofstream idx_out(indices_file, std::ios::binary);
        idx_out.write(reinterpret_cast<const char*>(indices.data()), 
                     indices.size() * sizeof(uint32_t));
        idx_out.close();
        
        // Calculate bounds
        Vec3 center(0, 0, 0);
        for (const auto& vertex : vertices) {
            center = center + vertex;
        }
        center = center * (1.0f / vertices.size());
        
        float max_dist = 0;
        for (const auto& vertex : vertices) {
            Vec3 diff = vertex + center * (-1);
            float dist = diff.length();
            if (dist > max_dist) max_dist = dist;
        }
        
        // Write manifest JSON
        std::ofstream manifest(output_path);
        manifest << std::fixed << std::setprecision(6);
        manifest << "{\n";
        manifest << "  \"mesh\": {\n";
        manifest << "    \"primitive_topology\": \"triangles\",\n";
        manifest << "    \"positions\": \"buffer://" << base_name << "_positions.bin\",\n";
        manifest << "    \"normals\": \"buffer://" << base_name << "_normals.bin\",\n";
        manifest << "    \"uv0\": \"buffer://" << base_name << "_uvs.bin\",\n";
        manifest << "    \"indices\": \"buffer://" << base_name << "_indices.bin\",\n";
        manifest << "    \"bounds\": {\n";
        manifest << "      \"center\": [" << center.x << ", " << center.y << ", " << center.z << "],\n";
        manifest << "      \"radius\": " << max_dist << "\n";
        manifest << "    }\n";
        manifest << "  },\n";
        manifest << "  \"metadata\": {\n";
        manifest << "    \"generator\": \"cubesphere.cpp\",\n";
        manifest << "    \"face_resolution\": " << face_res << ",\n";
        manifest << "    \"vertex_count\": " << vertices.size() << ",\n";
        manifest << "    \"triangle_count\": " << indices.size() / 3 << "\n";
        manifest << "  }\n";
        manifest << "}\n";
        manifest.close();
        
        std::cout << "📁 Exported manifest: " << output_path << std::endl;
        std::cout << "📁 Binary buffers in: " << actual_buffer_dir << std::endl;
    }
};

int main(int argc, char* argv[]) {
    int face_res = 32;
    std::string output = "cubesphere_manifest.json";
    std::string buffer_dir = "";
    
    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--face_res" && i + 1 < argc) {
            face_res = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--buffer_dir" && i + 1 < argc) {
            buffer_dir = argv[++i];
        }
    }
    
    // Generate cube-sphere
    CubeSphereGenerator generator(face_res);
    generator.generate();
    
    // Export to manifest format
    generator.export_manifest(output, buffer_dir);
    
    std::cout << "✅ Cube-sphere generation complete!" << std::endl;
    std::cout << "📋 Face resolution: " << face_res << "×" << face_res << std::endl;
    
    return 0;
}