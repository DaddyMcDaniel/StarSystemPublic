#!/usr/bin/env python3
"""
T10 Marching Cubes System Test Suite
====================================

Comprehensive tests for Marching Cubes polygonization, cave generation, and viewer integration:
- Marching Cubes algorithm correctness
- SDF to mesh conversion validation
- Cave mesh generation and export
- Manifest creation and loading
- Normal generation from SDF gradients
- Integration with viewer system

Usage:
    python test_marching_cubes.py
"""

import numpy as np
import math
import time
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))

from marching_cubes import MarchingCubes, CaveMeshGenerator
from chunk_cave_generator import ChunkCaveGenerator
from cave_manifest import CaveManifestManager
from sdf_evaluator import VoxelGrid, ChunkBounds, SDFEvaluator
from sdf_primitives import SDFSphere, SDFBox, SDFGyroid, SDFUnion, SDFSubtract, sdf_gradient


def test_marching_cubes_algorithm():
    """Test the Marching Cubes algorithm with known SDFs"""
    print("\n🔍 Testing Marching Cubes Algorithm...")
    
    success = True
    
    # Test with simple sphere
    sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    bounds = ChunkBounds(
        min_point=np.array([-1.5, -1.5, -1.5]),
        max_point=np.array([1.5, 1.5, 1.5])
    )
    
    print("   Testing sphere SDF polygonization:")
    
    # Test different resolutions
    resolutions = [8, 16, 32]
    for resolution in resolutions:
        voxel_grid = VoxelGrid(bounds, resolution)
        evaluator = SDFEvaluator()
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, sphere)
        
        mc = MarchingCubes(iso_value=0.0)
        vertices, triangles = mc.polygonize(voxel_grid, scalar_field, sphere)
        
        print(f"     Resolution {resolution}^3: {len(vertices)} vertices, {len(triangles)} triangles")
        
        # Validate mesh properties
        if len(vertices) == 0:
            print(f"     ❌ No vertices generated at resolution {resolution}")
            success = False
            continue
        
        if len(triangles) == 0:
            print(f"     ❌ No triangles generated at resolution {resolution}")
            success = False
            continue
        
        # Check vertex properties
        for i, vertex in enumerate(vertices[:5]):  # Check first 5 vertices
            # Position should be within bounds
            if not (np.all(vertex.position >= bounds.min_point) and 
                   np.all(vertex.position <= bounds.max_point)):
                print(f"     ❌ Vertex {i} position out of bounds: {vertex.position}")
                success = False
                break
            
            # Normal should be unit length
            normal_length = np.linalg.norm(vertex.normal)
            if abs(normal_length - 1.0) > 1e-3:
                print(f"     ❌ Vertex {i} normal not unit length: {normal_length}")
                success = False
                break
        
        # Check triangle indices
        max_vertex_index = len(vertices) - 1
        for i, triangle in enumerate(triangles[:5]):  # Check first 5 triangles
            if np.any(triangle < 0) or np.any(triangle > max_vertex_index):
                print(f"     ❌ Triangle {i} has invalid indices: {triangle}")
                success = False
                break
        
        if len(vertices) > 0 and len(triangles) > 0:
            print(f"     ✅ Resolution {resolution}^3: Valid mesh generated")
    
    if success:
        print("   ✅ All Marching Cubes algorithm tests passed")
    else:
        print("   ❌ Some Marching Cubes algorithm tests failed")
    
    return success


def test_sdf_gradient_normals():
    """Test normal generation from SDF gradients"""
    print("\n🔍 Testing SDF Gradient Normal Generation...")
    
    success = True
    tolerance = 1e-3
    
    # Test with sphere (known analytical gradients)
    sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    
    test_points = [
        np.array([1.0, 0.0, 0.0]),    # On surface, should point outward
        np.array([0.0, 1.0, 0.0]),    # On surface, should point upward
        np.array([0.5, 0.5, 0.0]),    # Near surface
        np.array([1.5, 0.0, 0.0])     # Outside sphere
    ]
    
    print("   Testing gradient calculation:")
    for i, point in enumerate(test_points):
        # Compute gradient using finite differences
        epsilon = 1e-4
        gradient = sdf_gradient(sphere, point, epsilon)
        
        # Normalize to get normal
        length = np.linalg.norm(gradient)
        if length < 1e-8:
            print(f"     ❌ Point {i}: Zero gradient at {point}")
            success = False
            continue
        
        normal = gradient / length
        
        # For sphere, analytical normal should point away from center
        analytical_normal = (point - sphere.center) / np.linalg.norm(point - sphere.center)
        
        # Compare with analytical normal
        error = np.linalg.norm(normal - analytical_normal)
        if error > tolerance:
            print(f"     ❌ Point {i}: Normal error {error:.6f} > {tolerance}")
            print(f"         Computed: {normal}")
            print(f"         Expected: {analytical_normal}")
            success = False
        else:
            print(f"     ✅ Point {i}: Normal error {error:.6f}")
    
    # Test with box SDF
    box = SDFBox(center=[0, 0, 0], size=[2, 2, 2], seed=42)
    
    box_test_points = [
        np.array([1.0, 0.0, 0.0]),    # On face, should point in +X
        np.array([0.0, 1.0, 0.0]),    # On face, should point in +Y
        np.array([-1.0, 0.0, 0.0])    # On face, should point in -X
    ]
    
    print("   Testing box gradient calculation:")
    for i, point in enumerate(box_test_points):
        gradient = sdf_gradient(box, point, epsilon)
        length = np.linalg.norm(gradient)
        
        if length < 1e-8:
            print(f"     ❌ Box point {i}: Zero gradient at {point}")
            success = False
            continue
        
        normal = gradient / length
        print(f"     ✅ Box point {i}: Normal {normal} (length {length:.6f})")
    
    if success:
        print("   ✅ All SDF gradient normal tests passed")
    else:
        print("   ❌ Some SDF gradient normal tests failed")
    
    return success


def test_cave_mesh_generation():
    """Test cave mesh generation for chunks"""
    print("\n🔍 Testing Cave Mesh Generation...")
    
    success = True
    
    # Create test cave generator
    cave_generator = CaveMeshGenerator(material_id=1)
    
    # Test bounds
    test_bounds = [
        ChunkBounds(np.array([-2, -2, -2]), np.array([2, 2, 2])),      # Medium chunk
        ChunkBounds(np.array([-1, -1, -1]), np.array([1, 1, 1])),      # Small chunk
        ChunkBounds(np.array([-4, -4, -4]), np.array([4, 4, 4]))       # Large chunk
    ]
    
    # Test different SDF types
    sdf_configs = [
        ("sphere", SDFSphere(center=[0, 0, 0], radius=1.5, seed=42)),
        ("box", SDFBox(center=[0, 0, 0], size=[2, 2, 2], seed=42)),
        ("gyroid", SDFGyroid(scale=2.0, thickness=0.3, seed=42)),
        ("union", SDFUnion(
            SDFSphere(center=[-0.5, 0, 0], radius=1.0, seed=42),
            SDFSphere(center=[0.5, 0, 0], radius=1.0, seed=43),
            seed=44
        ))
    ]
    
    print("   Testing cave mesh generation:")
    for bounds_idx, bounds in enumerate(test_bounds):
        for sdf_name, sdf_tree in sdf_configs:
            chunk_id = f"test_{sdf_name}_{bounds_idx}"
            
            # Generate cave mesh
            cave_mesh = cave_generator.generate_cave_mesh(
                bounds, sdf_tree, resolution=16, chunk_id=chunk_id
            )
            
            if cave_mesh is None:
                print(f"     ⚠️ No mesh generated for {chunk_id} (may be valid if no surface)")
                continue
            
            # Validate mesh
            vertex_count = len(cave_mesh.vertices)
            triangle_count = len(cave_mesh.triangles)
            
            if vertex_count == 0 or triangle_count == 0:
                print(f"     ❌ Empty mesh for {chunk_id}")
                success = False
                continue
            
            # Check mesh properties
            material_id = cave_mesh.material_id
            if material_id != 1:
                print(f"     ❌ Wrong material ID for {chunk_id}: {material_id}")
                success = False
                continue
            
            # Check vertex normals
            normal_errors = 0
            for vertex in cave_mesh.vertices:
                normal_length = np.linalg.norm(vertex.normal)
                if abs(normal_length - 1.0) > 1e-3:
                    normal_errors += 1
            
            if normal_errors > vertex_count * 0.1:  # Allow 10% error tolerance
                print(f"     ❌ Too many invalid normals in {chunk_id}: {normal_errors}/{vertex_count}")
                success = False
                continue
            
            print(f"     ✅ {chunk_id}: {vertex_count} verts, {triangle_count} tris, mat_id {material_id}")
    
    if success:
        print("   ✅ All cave mesh generation tests passed")
    else:
        print("   ❌ Some cave mesh generation tests failed")
    
    return success


def test_chunk_cave_generator():
    """Test the chunk-based cave generation system"""
    print("\n🔍 Testing Chunk Cave Generator...")
    
    success = True
    
    # Create test directory
    test_dir = Path("test_t10_caves")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    try:
        # Create test terrain manifest
        test_terrain_manifest = {
            "name": "Test Terrain T10",
            "chunks": [
                {
                    "chunk_id": "test_chunk_0_0",
                    "aabb": {
                        "min": [-2.0, -2.0, -2.0],
                        "max": [2.0, 2.0, 2.0]
                    }
                },
                {
                    "chunk_id": "test_chunk_1_0",
                    "aabb": {
                        "min": [2.0, -2.0, -2.0],
                        "max": [6.0, 2.0, 2.0]
                    }
                }
            ]
        }
        
        # Save test terrain manifest
        terrain_manifest_path = test_dir / "test_terrain.json"
        with open(terrain_manifest_path, 'w') as f:
            json.dump(test_terrain_manifest, f, indent=2)
        
        # Generate cave chunks
        generator = ChunkCaveGenerator(resolution=16, cave_material_id=2)
        cave_manifest = generator.generate_cave_chunks(
            str(terrain_manifest_path),
            str(test_dir),
            cave_types=["caves", "overhangs"]
        )
        
        # Validate cave manifest
        if not cave_manifest:
            print("     ❌ No cave manifest generated")
            success = False
        else:
            chunk_count = cave_manifest.get("chunk_count", 0)
            print(f"     ✅ Generated {chunk_count} cave chunks")
            
            if chunk_count == 0:
                print("     ⚠️ No cave chunks generated (may be valid)")
            
            # Check manifest structure
            required_fields = ["name", "type", "chunks", "statistics"]
            for field in required_fields:
                if field not in cave_manifest:
                    print(f"     ❌ Missing field in manifest: {field}")
                    success = False
            
            # Check generated files
            manifest_file = test_dir / "cave_chunks.json"
            if not manifest_file.exists():
                print("     ❌ Cave manifest file not created")
                success = False
            else:
                print("     ✅ Cave manifest file created")
        
        # Test cave manifest loading
        manager = CaveManifestManager()
        if manifest_file.exists():
            loaded_chunks = manager.load_cave_manifest(str(manifest_file))
            print(f"     ✅ Loaded {len(loaded_chunks)} cave chunks from manifest")
        
    except Exception as e:
        print(f"     ❌ Chunk cave generator test failed: {e}")
        success = False
    
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
    
    if success:
        print("   ✅ All chunk cave generator tests passed")
    else:
        print("   ❌ Some chunk cave generator tests failed")
    
    return success


def test_cave_manifest_system():
    """Test cave manifest creation and loading"""
    print("\n🔍 Testing Cave Manifest System...")
    
    success = True
    
    # Test manifest manager
    manager = CaveManifestManager()
    
    # Test material system
    print("   Testing material system:")
    for mat_id in [0, 1, 2]:
        color = manager.get_material_color(mat_id)
        print(f"     Material {mat_id}: {color}")
        
        if len(color) != 3:
            print(f"     ❌ Invalid color format for material {mat_id}")
            success = False
    
    # Test spatial queries (with dummy data)
    from cave_manifest import CaveChunkInfo
    
    test_chunks = {
        "test_chunk_1": CaveChunkInfo(
            chunk_id="test_chunk_1",
            material_id=1,
            vertex_count=100,
            triangle_count=50,
            index_count=150,
            aabb_min=np.array([-1, -1, -1]),
            aabb_max=np.array([1, 1, 1]),
            buffers={},
            lod_levels=[0]
        ),
        "test_chunk_2": CaveChunkInfo(
            chunk_id="test_chunk_2",
            material_id=2,
            vertex_count=200,
            triangle_count=100,
            index_count=300,
            aabb_min=np.array([3, -1, -1]),
            aabb_max=np.array([5, 1, 1]),
            buffers={},
            lod_levels=[0]
        )
    }
    
    manager.cave_chunks = test_chunks
    
    # Test spatial queries
    print("   Testing spatial queries:")
    center = np.array([0, 0, 0])
    
    # Query within radius
    chunks_in_radius = manager.get_chunks_in_radius(center, 2.0, "cave")
    expected_chunks = ["test_chunk_1"]  # Only chunk 1 should be within radius 2
    
    if set(chunks_in_radius) != set(expected_chunks):
        print(f"     ❌ Spatial query failed: got {chunks_in_radius}, expected {expected_chunks}")
        success = False
    else:
        print(f"     ✅ Spatial query: {len(chunks_in_radius)} chunks within radius")
    
    # Test bounds calculations
    for chunk_id, chunk_info in test_chunks.items():
        center = chunk_info.bounds_center
        size = chunk_info.bounds_size
        
        # Validate center calculation
        expected_center = (chunk_info.aabb_min + chunk_info.aabb_max) * 0.5
        if not np.allclose(center, expected_center):
            print(f"     ❌ Bounds center calculation error for {chunk_id}")
            success = False
        else:
            print(f"     ✅ {chunk_id}: center {center}, size {size}")
    
    if success:
        print("   ✅ All cave manifest system tests passed")
    else:
        print("   ❌ Some cave manifest system tests failed")
    
    return success


def test_viewer_integration():
    """Test viewer integration components"""
    print("\n🔍 Testing Viewer Integration...")
    
    success = True
    
    try:
        # Test cave viewer extension import
        from cave_viewer_extension import InitializeCaveSystem, HandleCaveDebugKeys
        
        print("   Testing cave viewer extension:")
        
        # Test initialization
        InitializeCaveSystem()
        print("     ✅ Cave system initialization")
        
        # Test debug key handling
        test_keys = ['M', 'B', 'C']
        for key in test_keys:
            try:
                HandleCaveDebugKeys(key)
                print(f"     ✅ Debug key '{key}' handled")
            except Exception as e:
                print(f"     ❌ Debug key '{key}' failed: {e}")
                success = False
        
        print("     ✅ Cave viewer extension functional")
        
    except ImportError as e:
        print(f"     ❌ Cave viewer extension import failed: {e}")
        success = False
    except Exception as e:
        print(f"     ❌ Viewer integration test failed: {e}")
        success = False
    
    if success:
        print("   ✅ All viewer integration tests passed")
    else:
        print("   ❌ Some viewer integration tests failed")
    
    return success


def benchmark_marching_cubes_performance():
    """Benchmark Marching Cubes performance"""
    print("\n🚀 Marching Cubes Performance Benchmarks...")
    
    # Test sphere SDF
    sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    bounds = ChunkBounds(
        min_point=np.array([-1.5, -1.5, -1.5]),
        max_point=np.array([1.5, 1.5, 1.5])
    )
    
    resolutions = [8, 16, 32, 64]
    
    for resolution in resolutions:
        voxel_grid = VoxelGrid(bounds, resolution)
        evaluator = SDFEvaluator()
        
        # Time SDF sampling
        start_time = time.time()
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, sphere)
        sampling_time = time.time() - start_time
        
        # Time Marching Cubes
        mc = MarchingCubes(iso_value=0.0)
        start_time = time.time()
        vertices, triangles = mc.polygonize(voxel_grid, scalar_field, sphere)
        mc_time = time.time() - start_time
        
        total_time = sampling_time + mc_time
        voxel_count = resolution ** 3
        
        print(f"   Resolution {resolution}^3 ({voxel_count:,} voxels):")
        print(f"     SDF sampling: {sampling_time*1000:.2f} ms")
        print(f"     Marching Cubes: {mc_time*1000:.2f} ms")
        print(f"     Total: {total_time*1000:.2f} ms")
        print(f"     Output: {len(vertices)} vertices, {len(triangles)} triangles")
        
        if total_time > 0:
            voxels_per_sec = voxel_count / total_time
            print(f"     Performance: {voxels_per_sec:,.0f} voxels/sec")
    
    print("   ✅ Performance benchmarks completed")


def run_comprehensive_t10_tests():
    """Run all T10 Marching Cubes system tests"""
    print("🚀 T10 Marching Cubes System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Marching Cubes Algorithm", test_marching_cubes_algorithm),
        ("SDF Gradient Normals", test_sdf_gradient_normals),
        ("Cave Mesh Generation", test_cave_mesh_generation),
        ("Chunk Cave Generator", test_chunk_cave_generator),
        ("Cave Manifest System", test_cave_manifest_system),
        ("Viewer Integration", test_viewer_integration),
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Run performance benchmarks
    benchmark_marching_cubes_performance()
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"   Total tests: {len(tests)}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {len(tests) - passed}")
    print(f"   Success rate: {passed/len(tests)*100:.1f}%")
    
    print(f"\n📋 Individual Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if passed == len(tests):
        print(f"\n🎉 All T10 Marching Cubes tests passed!")
        return True
    else:
        print(f"\n⚠️ Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    success = run_comprehensive_t10_tests()
    sys.exit(0 if success else 1)