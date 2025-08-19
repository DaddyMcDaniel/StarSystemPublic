#!/usr/bin/env python3
"""
T05 Heightfield Displacement Verification
=========================================

Tests and validates the heightfield displacement implementation without requiring OpenGL.
Analyzes mesh properties, displacement effectiveness, and visual characteristics.
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'terrain'))

from mesh.cubesphere import CubeSphereGenerator
from terrain.heightfield import create_heightfield_from_pcc


def load_mesh_from_manifest(manifest_path: Path):
    """Load mesh data from manifest file"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    mesh_info = manifest["mesh"]
    buffer_dir = manifest_path.parent
    
    # Load binary buffers
    positions_file = buffer_dir / mesh_info["positions"].replace("buffer://", "")
    normals_file = buffer_dir / mesh_info["normals"].replace("buffer://", "")
    indices_file = buffer_dir / mesh_info["indices"].replace("buffer://", "")
    
    positions = np.fromfile(positions_file, dtype=np.float32).reshape(-1, 3)
    normals = np.fromfile(normals_file, dtype=np.float32).reshape(-1, 3)
    indices = np.fromfile(indices_file, dtype=np.uint32)
    
    return {
        "positions": positions,
        "normals": normals, 
        "indices": indices,
        "manifest": manifest
    }


def analyze_displacement_effectiveness(displaced_mesh, reference_mesh):
    """Analyze how effective the displacement was"""
    print("🔍 Displacement Effectiveness Analysis:")
    print("-" * 40)
    
    displaced_pos = displaced_mesh["positions"]
    reference_pos = reference_mesh["positions"]
    
    # Compute radius for each vertex
    displaced_radii = np.linalg.norm(displaced_pos, axis=1)
    reference_radii = np.linalg.norm(reference_pos, axis=1)
    
    # Displacement magnitude
    displacement = displaced_radii - reference_radii
    
    print(f"   Reference radius: mean={np.mean(reference_radii):.6f}, std={np.std(reference_radii):.6f}")
    print(f"   Displaced radius: mean={np.mean(displaced_radii):.6f}, std={np.std(displaced_radii):.6f}")
    print(f"   Displacement: min={np.min(displacement):.6f}, max={np.max(displacement):.6f}")
    print(f"   Displacement: mean={np.mean(displacement):.6f}, std={np.std(displacement):.6f}")
    print(f"   Displacement range: {np.max(displacement) - np.min(displacement):.6f}")
    
    # Check if displacement actually varied the surface
    if np.std(displacement) > 1e-6:
        print("   Status: ✅ Displacement is working (surface variation detected)")
    else:
        print("   Status: ❌ No displacement variation detected")
    
    return displacement


def analyze_mesh_quality(mesh_data, name="Mesh"):
    """Analyze mesh quality and properties"""
    print(f"📊 {name} Quality Analysis:")
    print("-" * 40)
    
    positions = mesh_data["positions"]
    normals = mesh_data["normals"]
    indices = mesh_data["indices"]
    
    # Basic stats
    vertex_count = len(positions)
    triangle_count = len(indices) // 3
    
    print(f"   Vertices: {vertex_count}")
    print(f"   Triangles: {triangle_count}")
    
    # Bounding sphere
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    radius = np.max(distances)
    
    print(f"   Bounding sphere center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    print(f"   Bounding sphere radius: {radius:.6f}")
    
    # Normal validation
    normal_magnitudes = np.linalg.norm(normals, axis=1)
    normal_errors = np.abs(normal_magnitudes - 1.0)
    max_normal_error = np.max(normal_errors)
    
    print(f"   Normal magnitude error: max={max_normal_error:.8f}")
    if max_normal_error < 1e-5:
        print("   Normal status: ✅ All normals properly normalized")
    else:
        print("   Normal status: ⚠️ Some normals not normalized")
    
    # Triangle area analysis
    triangles = indices.reshape(-1, 3)
    areas = []
    
    for tri in triangles[:min(100, len(triangles))]:  # Sample first 100 triangles
        v0, v1, v2 = positions[tri]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)
    
    areas = np.array(areas)
    print(f"   Triangle areas (sample): min={np.min(areas):.8f}, max={np.max(areas):.8f}")
    print(f"   Triangle area variation: {np.std(areas)/np.mean(areas):.4f}")


def test_displacement_variations():
    """Test displacement with different parameters"""
    print("🧪 Testing Displacement Parameter Variations:")
    print("=" * 50)
    
    # Create heightfield
    terrain_spec_path = Path(__file__).parent / "terrain" / "example_terrain_spec.json"
    heightfield = create_heightfield_from_pcc(str(terrain_spec_path), global_seed=42)
    
    # Test different displacement scales
    scales = [0.0, 0.1, 0.2, 0.5]
    face_res = 16
    
    for scale in scales:
        print(f"\n🔬 Testing displacement scale: {scale}")
        print("-" * 30)
        
        generator = CubeSphereGenerator(
            face_res=face_res,
            base_radius=1.0,
            heightfield=heightfield if scale > 0 else None,
            displacement_scale=scale
        )
        
        mesh_data = generator.generate()
        
        # Analyze mesh properties
        positions = mesh_data["positions"]
        radii = np.linalg.norm(positions, axis=1)
        
        print(f"   Vertex radii: min={np.min(radii):.6f}, max={np.max(radii):.6f}")
        print(f"   Radius variation: {np.std(radii):.6f}")
        print(f"   Expected range: {1.0 - scale * 0.25:.3f} to {1.0 + scale * 0.25:.3f}")


def main():
    """Main test function"""
    print("🚀 T05 Heightfield Displacement Verification")
    print("=" * 60)
    
    # Test parameter variations
    test_displacement_variations()
    
    # Analyze existing displaced mesh if available
    displaced_manifest = Path(__file__).parent / "mesh" / "displaced_planet_manifest.json"
    
    if displaced_manifest.exists():
        print(f"\n📁 Analyzing Displaced Mesh: {displaced_manifest}")
        print("=" * 60)
        
        # Load displaced mesh
        displaced_mesh = load_mesh_from_manifest(displaced_manifest)
        analyze_mesh_quality(displaced_mesh, "Displaced Planet")
        
        # Generate reference mesh for comparison
        print("\n🔄 Generating Reference Mesh for Comparison...")
        ref_generator = CubeSphereGenerator(face_res=16, base_radius=1.0)
        ref_mesh_data = ref_generator.generate()
        
        analyze_mesh_quality({
            "positions": ref_mesh_data["positions"],
            "normals": ref_mesh_data["normals"],
            "indices": ref_mesh_data["indices"]
        }, "Reference Sphere")
        
        # Compare displacement effectiveness
        print("\n")
        displacement = analyze_displacement_effectiveness(displaced_mesh, ref_mesh_data)
        
        # Statistics about displacement
        print(f"\n📈 Displacement Statistics:")
        print("-" * 40)
        print(f"   Vertices with positive displacement: {np.sum(displacement > 0)}")
        print(f"   Vertices with negative displacement: {np.sum(displacement < 0)}")
        print(f"   Displacement efficiency: {np.std(displacement)/np.mean(np.abs(displacement)):.4f}")
        
    else:
        print(f"\n⚠️ Displaced mesh not found at {displaced_manifest}")
        print("   Run cubesphere.py with displacement parameters first")
    
    print("\n✅ T05 Displacement Verification Complete")


if __name__ == "__main__":
    main()