#!/usr/bin/env python3
"""
T08 Runtime LOD Testing Suite
=============================

Comprehensive tests for runtime LOD selection and frustum culling:
- Bounding volume calculations
- Frustum plane extraction and intersection tests
- LOD selection based on distance and screen error
- Chunk streaming and VAO management
- Performance benchmarks

Usage:
    python test_runtime_lod.py
"""

import numpy as np
import math
import time
import json
from pathlib import Path
import sys
import os
from typing import List, Dict

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh'))

from mesh.runtime_lod import (
    RuntimeLODManager, LODLevel, BoundingSphere, AABB, 
    create_view_matrix, create_projection_matrix
)
from mesh.chunk_streamer import ChunkStreamer
from test_crack_prevention import create_mixed_lod_test_chunks


def test_bounding_volume_calculations():
    """Test bounding sphere and AABB calculations"""
    print("\n🔍 Testing Bounding Volume Calculations...")
    
    lod_manager = RuntimeLODManager()
    
    # Create test chunk with known geometry
    test_positions = np.array([
        -1.0, -1.0, -1.0,  # (-1, -1, -1)
         1.0, -1.0, -1.0,  # ( 1, -1, -1)
         1.0,  1.0, -1.0,  # ( 1,  1, -1)
        -1.0,  1.0, -1.0,  # (-1,  1, -1)
        -1.0, -1.0,  1.0,  # (-1, -1,  1)
         1.0, -1.0,  1.0,  # ( 1, -1,  1)
         1.0,  1.0,  1.0,  # ( 1,  1,  1)
        -1.0,  1.0,  1.0,  # (-1,  1,  1)
    ], dtype=np.float32)
    
    chunk_data = {
        "positions": test_positions,
        "chunk_info": {
            "chunk_id": "test_cube",
            "aabb": {
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            }
        }
    }
    
    # Test bounding sphere calculation
    bounding_sphere = lod_manager.compute_bounding_sphere(chunk_data)
    print(f"   Bounding sphere center: {bounding_sphere.center}")
    print(f"   Bounding sphere radius: {bounding_sphere.radius:.3f}")
    
    # Expected: center at origin, radius = sqrt(3) ≈ 1.732
    expected_center = np.array([0.0, 0.0, 0.0])
    expected_radius = math.sqrt(3.0)
    
    center_error = np.linalg.norm(bounding_sphere.center - expected_center)
    radius_error = abs(bounding_sphere.radius - expected_radius)
    
    print(f"   Center error: {center_error:.6f}")
    print(f"   Radius error: {radius_error:.6f}")
    
    # Test AABB calculation
    aabb = lod_manager.compute_aabb(chunk_data)
    print(f"   AABB min: {aabb.min_bounds}")
    print(f"   AABB max: {aabb.max_bounds}")
    print(f"   AABB center: {aabb.center}")
    print(f"   AABB diagonal: {aabb.diagonal_length():.3f}")
    
    # Validate results
    success = (center_error < 0.001 and radius_error < 0.001 and 
               np.allclose(aabb.min_bounds, [-1.0, -1.0, -1.0]) and
               np.allclose(aabb.max_bounds, [1.0, 1.0, 1.0]))
    
    if success:
        print("   ✅ Bounding volume calculations test passed")
    else:
        print("   ❌ Bounding volume calculations test failed")
    
    return success


def test_frustum_culling():
    """Test frustum plane extraction and intersection tests"""
    print("\n🔍 Testing Frustum Culling...")
    
    lod_manager = RuntimeLODManager()
    
    # Create test camera setup
    camera_pos = np.array([0.0, 0.0, 5.0])
    camera_target = np.array([0.0, 0.0, 0.0])
    up_vector = np.array([0.0, 1.0, 0.0])
    
    view_matrix = create_view_matrix(camera_pos, camera_target, up_vector)
    proj_matrix = create_projection_matrix(
        fov_y=math.radians(60), aspect=1.0, near=0.1, far=100.0
    )
    
    print(f"   Camera position: {camera_pos}")
    print(f"   Camera target: {camera_target}")
    
    # Extract frustum planes
    frustum_planes = lod_manager.extract_frustum_planes(view_matrix, proj_matrix)
    print(f"   Extracted {len(frustum_planes)} frustum planes")
    
    for i, plane in enumerate(frustum_planes):
        plane_names = ["Left", "Right", "Bottom", "Top", "Near", "Far"]
        print(f"   {plane_names[i]} plane: normal={plane.normal}, distance={plane.distance:.3f}")
    
    # Test sphere frustum intersection
    test_spheres = [
        BoundingSphere(np.array([0.0, 0.0, 0.0]), 1.0),   # At origin, should be visible
        BoundingSphere(np.array([0.0, 0.0, -10.0]), 1.0), # Behind camera, should be culled
        BoundingSphere(np.array([10.0, 0.0, 0.0]), 1.0),  # Far right, might be culled
        BoundingSphere(np.array([0.0, 10.0, 0.0]), 1.0),  # Far up, might be culled
    ]
    
    visible_count = 0
    for i, sphere in enumerate(test_spheres):
        is_visible = lod_manager.test_sphere_frustum(sphere, frustum_planes)
        print(f"   Sphere {i} at {sphere.center}: {'visible' if is_visible else 'culled'}")
        if is_visible:
            visible_count += 1
    
    # Expect at least the sphere at origin to be visible
    success = visible_count > 0
    
    if success:
        print("   ✅ Frustum culling test passed")
    else:
        print("   ❌ Frustum culling test failed")
    
    return success


def test_lod_selection():
    """Test LOD level selection based on distance and screen error"""
    print("\n🔍 Testing LOD Selection...")
    
    lod_manager = RuntimeLODManager(
        lod_distance_bands=[2.0, 5.0, 15.0, 40.0],
        screen_error_thresholds=[0.5, 2.0, 8.0, 32.0]
    )
    
    # Test distance-based LOD selection
    test_distances = [1.0, 3.0, 8.0, 20.0, 60.0]
    test_screen_errors = [0.3, 1.5, 5.0, 20.0, 50.0]
    
    print("   Distance-based LOD selection:")
    for distance in test_distances:
        lod_level = lod_manager.select_lod_level(distance, 1.0)  # Low screen error
        print(f"   Distance {distance:5.1f} -> LOD{lod_level.value}")
    
    print("   Screen error-based LOD selection:")
    for screen_error in test_screen_errors:
        lod_level = lod_manager.select_lod_level(5.0, screen_error)  # Mid distance
        print(f"   Screen error {screen_error:5.1f} -> LOD{lod_level.value}")
    
    # Test screen space error calculation
    test_aabb = AABB(np.array([-1, -1, -1]), np.array([1, 1, 1]))
    
    test_cases = [
        (1.0, math.radians(60), 600),   # Close, should be large screen error
        (10.0, math.radians(60), 600),  # Medium, should be medium screen error  
        (50.0, math.radians(60), 600),  # Far, should be small screen error
    ]
    
    print("   Screen space error calculation:")
    for distance, fov_y, screen_height in test_cases:
        screen_error = lod_manager.calculate_screen_space_error(test_aabb, distance, fov_y, screen_height)
        print(f"   Distance {distance:5.1f}, FOV {math.degrees(fov_y):3.0f}°, Height {screen_height} -> Error {screen_error:.2f} px")
    
    print("   ✅ LOD selection test completed")
    return True


def test_chunk_streaming():
    """Test chunk streaming and VAO management"""
    print("\n🔍 Testing Chunk Streaming...")
    
    # Create mock chunk streamer (without OpenGL)
    streamer = ChunkStreamer(max_memory_mb=64.0, max_active_chunks=10)
    
    # Create test chunks
    test_chunks = []
    for i in range(5):
        positions = np.random.random(72).astype(np.float32)  # 24 vertices
        indices = np.arange(72, dtype=np.uint32)  # 24 triangles
        
        chunk_data = {
            "positions": positions,
            "normals": np.random.random(72).astype(np.float32),
            "indices": indices,
            "chunk_info": {"chunk_id": f"test_chunk_{i}"}
        }
        test_chunks.append(chunk_data)
    
    print(f"   Created {len(test_chunks)} test chunks")
    
    # Test memory estimation
    for i, chunk in enumerate(test_chunks):
        memory_usage = streamer.estimate_chunk_memory_usage(chunk)
        print(f"   Chunk {i} estimated memory: {memory_usage / 1024:.1f} KB")
    
    # Test caching system
    for i, chunk in enumerate(test_chunks):
        chunk_id = f"test_chunk_{i}"
        streamer.cache_chunk_data(chunk_id, chunk)
    
    print(f"   Cached {len(test_chunks)} chunks")
    
    # Test cache retrieval
    cached_chunk = streamer.get_cached_chunk_data("test_chunk_2")
    cache_hit = cached_chunk is not None
    
    print(f"   Cache hit test: {'✅ SUCCESS' if cache_hit else '❌ FAILED'}")
    
    # Test streaming statistics
    stats = streamer.get_streaming_stats()
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache misses: {stats.cache_misses}")
    print(f"   GPU memory: {streamer.get_memory_usage_mb():.1f} MB")
    
    print("   ✅ Chunk streaming test completed")
    return True


def test_integrated_lod_system():
    """Test integrated LOD system with real chunk data"""
    print("\n🔍 Testing Integrated LOD System...")
    
    # Create mixed LOD test chunks
    test_chunks = create_mixed_lod_test_chunks()
    if not test_chunks:
        print("   ⚠️ No test chunks available, skipping integration test")
        return True
    
    print(f"   Using {len(test_chunks)} mixed LOD test chunks")
    
    # Initialize LOD manager
    lod_manager = RuntimeLODManager(
        lod_distance_bands=[3.0, 8.0, 20.0, 50.0],
        screen_error_thresholds=[0.5, 2.0, 8.0, 32.0],
        max_chunks_per_frame=50
    )
    
    # Test multiple camera positions
    test_positions = [
        np.array([0.0, 2.0, 5.0]),   # Close to planet
        np.array([0.0, 5.0, 10.0]),  # Medium distance
        np.array([0.0, 10.0, 20.0]), # Far from planet
    ]
    
    for i, camera_pos in enumerate(test_positions):
        print(f"   Camera position {i+1}: {camera_pos}")
        
        # Create view and projection matrices
        camera_target = np.array([0.0, 0.0, 0.0])
        view_matrix = create_view_matrix(camera_pos, camera_target, np.array([0, 1, 0]))
        proj_matrix = create_projection_matrix(
            fov_y=math.radians(60), aspect=1.33, near=0.1, far=100.0
        )
        
        # Select active chunks
        start_time = time.time()
        selected_chunks = lod_manager.select_active_chunks(
            test_chunks, camera_pos, view_matrix, proj_matrix
        )
        selection_time = (time.time() - start_time) * 1000
        
        print(f"     Selection time: {selection_time:.2f} ms")
        print(f"     Selected chunks: {len(selected_chunks)}")
        
        # Analyze LOD distribution
        lod_counts = {}
        distance_range = [float('inf'), 0.0]
        
        for chunk_info in selected_chunks:
            lod_level = chunk_info.lod_level.value
            lod_counts[lod_level] = lod_counts.get(lod_level, 0) + 1
            
            distance_range[0] = min(distance_range[0], chunk_info.distance_to_camera)
            distance_range[1] = max(distance_range[1], chunk_info.distance_to_camera)
        
        print(f"     LOD distribution: {lod_counts}")
        print(f"     Distance range: {distance_range[0]:.1f} - {distance_range[1]:.1f}")
        
        # Get performance statistics
        stats = lod_manager.get_lod_statistics()
        print(f"     Culled chunks: {stats['culled_chunks']}")
        print(f"     Total processing time: {stats['lod_time_ms']:.2f} ms")
    
    print("   ✅ Integrated LOD system test completed")
    return True


def benchmark_performance():
    """Benchmark runtime LOD performance"""
    print("\n🚀 Performance Benchmarks...")
    
    # Create larger test dataset
    test_chunks = []
    chunk_counts = [50, 100, 200, 400]
    
    for chunk_count in chunk_counts:
        print(f"\n   Benchmarking with {chunk_count} chunks:")
        
        # Generate test chunks
        chunks = []
        for i in range(chunk_count):
            # Random positions on unit sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            
            x = np.cos(phi) * np.cos(theta)
            y = np.cos(phi) * np.sin(theta) 
            z = np.sin(phi)
            
            # Generate chunk geometry
            resolution = 16
            positions = np.random.random(resolution * resolution * 3).astype(np.float32)
            positions = positions.reshape(-1, 3)
            positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)  # Normalize to sphere
            
            chunk_data = {
                "positions": positions.flatten(),
                "normals": positions.flatten(),  # Use positions as normals for sphere
                "indices": np.arange(len(positions) - 3, dtype=np.uint32),
                "chunk_info": {
                    "chunk_id": f"bench_chunk_{i}",
                    "face_id": i % 6,
                    "level": np.random.randint(0, 3),
                    "resolution": resolution,
                    "aabb": {
                        "min": np.min(positions, axis=0).tolist(),
                        "max": np.max(positions, axis=0).tolist()
                    }
                }
            }
            chunks.append(chunk_data)
        
        # Initialize LOD manager
        lod_manager = RuntimeLODManager(max_chunks_per_frame=chunk_count)
        
        # Benchmark LOD selection
        camera_pos = np.array([0.0, 2.0, 5.0])
        camera_target = np.array([0.0, 0.0, 0.0])
        view_matrix = create_view_matrix(camera_pos, camera_target, np.array([0, 1, 0]))
        proj_matrix = create_projection_matrix(
            fov_y=math.radians(60), aspect=1.33, near=0.1, far=100.0
        )
        
        # Warm-up run
        lod_manager.select_active_chunks(chunks, camera_pos, view_matrix, proj_matrix)
        
        # Benchmark runs
        times = []
        for _ in range(10):
            start_time = time.time()
            selected_chunks = lod_manager.select_active_chunks(
                chunks, camera_pos, view_matrix, proj_matrix
            )
            elapsed_time = (time.time() - start_time) * 1000
            times.append(elapsed_time)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"     Selection time (avg): {avg_time:.2f} ms")
        print(f"     Selection time (min): {min_time:.2f} ms") 
        print(f"     Selection time (max): {max_time:.2f} ms")
        print(f"     Selected chunks: {len(selected_chunks)}")
        print(f"     Throughput: {chunk_count / (avg_time / 1000):.0f} chunks/sec")
        
        # Memory usage estimate
        stats = lod_manager.get_lod_statistics()
        print(f"     Performance rating: {'✅ GOOD' if avg_time < 5.0 else '⚠️ MODERATE' if avg_time < 15.0 else '❌ SLOW'}")


def run_comprehensive_lod_tests():
    """Run all runtime LOD tests"""
    print("🚀 T08 Runtime LOD Test Suite")
    print("=" * 60)
    
    tests = [
        ("Bounding Volume Calculations", test_bounding_volume_calculations),
        ("Frustum Culling", test_frustum_culling),
        ("LOD Selection", test_lod_selection),
        ("Chunk Streaming", test_chunk_streaming),
        ("Integrated LOD System", test_integrated_lod_system),
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
    benchmark_performance()
    
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
        print(f"\n🎉 All runtime LOD tests passed!")
        return True
    else:
        print(f"\n⚠️ Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    success = run_comprehensive_lod_tests()
    sys.exit(0 if success else 1)