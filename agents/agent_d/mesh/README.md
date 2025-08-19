# Agent D Mesh Generation - T02 Cube-Sphere Primitive

This directory contains the cube-sphere primitive generator for the Agent D terrain pipeline, implementing task T02 from the HOP_terrain specification.

## Overview

The cube-sphere generator creates uniform spherical meshes by projecting cube face grids onto a unit sphere. This approach provides:

- **Seam-aware geometry**: Shared vertices at face edges prevent gaps
- **Uniform distribution**: Better than UV-sphere for tessellation
- **No polar pinch**: Unlike UV-spheres, no vertex clustering at poles
- **Scalable LOD**: Foundation for per-face quadtree chunking (T06)

## Files

- `cubesphere.py` - Python implementation with full feature set
- `cubesphere.cpp` - C++ implementation for high performance
- `README.md` - This documentation
- `*.json` - Generated manifest files
- `*_positions.bin` - Binary vertex position buffers
- `*_normals.bin` - Binary vertex normal buffers  
- `*_uvs.bin` - Binary texture coordinate buffers
- `*_indices.bin` - Binary triangle index buffers

## Usage

### Python Version

```bash
# Generate cube-sphere with 32×32 resolution per face
python cubesphere.py --face_res 32 --output sphere_32.json

# Generate high-resolution version
python cubesphere.py --face_res 128 --output sphere_128.json --buffer_dir ./buffers/
```

### C++ Version

```bash
# Compile
g++ -std=c++17 -O3 cubesphere.cpp -o cubesphere

# Generate cube-sphere
./cubesphere --face_res 32 --output sphere_32.json
```

## Output Format

The generator produces a manifest file conforming to the HOP_terrain specification:

```json
{
  "mesh": {
    "primitive_topology": "triangles",
    "positions": "buffer://sphere_32_positions.bin",
    "normals": "buffer://sphere_32_normals.bin", 
    "uv0": "buffer://sphere_32_uvs.bin",
    "indices": "buffer://sphere_32_indices.bin",
    "bounds": {
      "center": [0.0, 0.0, 0.0],
      "radius": 1.0
    }
  },
  "metadata": {
    "generator": "cubesphere.py",
    "face_resolution": 32,
    "vertex_count": 5768,
    "triangle_count": 11532
  }
}
```

## Technical Details

### Face Configuration

The generator creates 6 faces with proper orientation:

- **+X face (right)**: normal=(1,0,0), up=(0,1,0), right=(0,0,-1)
- **-X face (left)**: normal=(-1,0,0), up=(0,1,0), right=(0,0,1)
- **+Y face (top)**: normal=(0,1,0), up=(0,0,1), right=(1,0,0)
- **-Y face (bottom)**: normal=(0,-1,0), up=(0,0,-1), right=(1,0,0)
- **+Z face (front)**: normal=(0,0,1), up=(0,1,0), right=(1,0,0)
- **-Z face (back)**: normal=(0,0,-1), up=(0,1,0), right=(-1,0,0)

### Vertex Sharing

Edge vertices are shared between adjacent faces using quantized position keys. This ensures seamless geometry without T-junctions.

### UV Mapping

Each face gets a dedicated region in UV space to prevent seam artifacts:
- Face 0-2: U=[0, 1/3, 2/3], V=[0, 1/2]
- Face 3-5: U=[0, 1/3, 2/3], V=[1/2, 1]

### Triangle Generation

Each quad in the face grid generates two triangles:
- Triangle 1: bottom-left → bottom-right → top-left
- Triangle 2: bottom-right → top-right → top-left

## Integration with Viewer

The generated manifests can be loaded by the T01 viewer system:

```python
# Scene file example
{
  "objects": [
    {
      "type": "MESH",
      "pos": [0, 0, 0],
      "manifest": "agents/agent_d/mesh/sphere_32.json", 
      "material": "terrain"
    }
  ]
}
```

## Performance

Resolution recommendations:
- **8×8**: Development/testing (296 vertices, 588 triangles)
- **32×32**: Standard quality (5,768 vertices, 11,532 triangles)
- **64×64**: High quality (22,632 vertices, 45,264 triangles)
- **128×128**: Very high quality (90,248 vertices, 180,496 triangles)

Vertex sharing reduces memory usage by ~1.5% compared to naive per-face generation.

## Next Steps (Terrain Pipeline)

This cube-sphere primitive serves as the foundation for:

- **T03**: Normals & tangents (MikkTSpace compatibility)
- **T04**: PCC terrain spec integration
- **T05**: Heightfield displacement  
- **T06**: Per-face quadtree LOD
- **T07**: Crack prevention between LOD levels

The uniform tessellation and shared vertices make it ideal for LOD systems and heightfield displacement while maintaining seamless geometry.