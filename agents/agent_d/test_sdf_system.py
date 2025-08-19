#!/usr/bin/env python3
"""
T09 SDF System Testing Suite
============================

Comprehensive tests for SDF primitives, evaluation, and voxelization:
- Known primitive distance validation
- Boolean operation correctness
- Noise displacement determinism
- PCC composition parsing
- Voxel grid sampling accuracy
- Chunk-local deterministic sampling

Usage:
    python test_sdf_system.py
"""

import numpy as np
import math
import time
import json
from pathlib import Path
import sys
import os
from typing import List, Dict, Tuple

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf'))

from sdf.sdf_primitives import *
from sdf.sdf_evaluator import *


def test_primitive_distances():
    """Test that primitive SDFs produce expected distances for known points"""
    print("\n🔍 Testing SDF Primitive Distance Calculations...")
    
    success = True
    tolerance = 1e-6
    
    # Test sphere SDF
    sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    
    test_cases_sphere = [
        (np.array([0, 0, 0]), -1.0),      # Center: inside by radius
        (np.array([1, 0, 0]), 0.0),       # On surface
        (np.array([2, 0, 0]), 1.0),       # Outside by 1 unit
        (np.array([0.5, 0, 0]), -0.5),    # Inside by 0.5
        (np.array([1, 1, 0]), math.sqrt(2) - 1.0),  # Diagonal point
    ]
    
    print("   Sphere SDF tests:")
    for point, expected in test_cases_sphere:
        actual = sphere.evaluate(point)
        error = abs(actual - expected)
        
        if error > tolerance:
            print(f"   ❌ Point {point}: expected {expected:.6f}, got {actual:.6f} (error: {error:.6f})")
            success = False
        else:
            print(f"   ✅ Point {point}: distance {actual:.6f}")
    
    # Test capsule SDF
    capsule = SDFCapsule(point_a=[0, -1, 0], point_b=[0, 1, 0], radius=0.5, seed=42)
    
    test_cases_capsule = [
        (np.array([0, 0, 0]), -0.5),      # Center: inside by radius
        (np.array([0.5, 0, 0]), 0.0),     # On surface at middle
        (np.array([1, 0, 0]), 0.5),       # Outside at middle
        (np.array([0, 1.5, 0]), 0.0),     # On cap surface
        (np.array([0, 2, 0]), 0.5),       # Outside cap by 0.5
    ]
    
    print("   Capsule SDF tests:")
    for point, expected in test_cases_capsule:
        actual = capsule.evaluate(point)
        error = abs(actual - expected)
        
        if error > tolerance:
            print(f"   ❌ Point {point}: expected {expected:.6f}, got {actual:.6f} (error: {error:.6f})")
            success = False
        else:
            print(f"   ✅ Point {point}: distance {actual:.6f}")
    
    # Test box SDF
    box = SDFBox(center=[0, 0, 0], size=[2, 2, 2], seed=42)
    
    test_cases_box = [
        (np.array([0, 0, 0]), -1.0),      # Center: inside by half-size
        (np.array([1, 0, 0]), 0.0),       # On face
        (np.array([1.5, 0, 0]), 0.5),     # Outside face
        (np.array([1, 1, 0]), 0.0),       # On edge
        (np.array([1, 1, 1]), 0.0),       # On corner
    ]
    
    print("   Box SDF tests:")
    for point, expected in test_cases_box:
        actual = box.evaluate(point)
        error = abs(actual - expected)
        
        if error > tolerance:
            print(f"   ❌ Point {point}: expected {expected:.6f}, got {actual:.6f} (error: {error:.6f})")
            success = False
        else:
            print(f"   ✅ Point {point}: distance {actual:.6f}")
    
    # Test gyroid SDF (basic properties)
    gyroid = SDFGyroid(scale=1.0, thickness=0.1, offset=0.0, seed=42)
    
    print("   Gyroid SDF tests (property validation):")
    test_points = [
        np.array([0, 0, 0]),
        np.array([math.pi, 0, 0]),
        np.array([0, math.pi, 0]),
        np.array([math.pi, math.pi, math.pi])
    ]
    
    for point in test_points:
        distance = gyroid.evaluate(point)
        print(f"   ✅ Point {point}: distance {distance:.6f}")
        
        # Check that gyroid produces reasonable values (not NaN, not infinite)
        if not np.isfinite(distance):
            print(f"   ❌ Invalid distance at {point}: {distance}")
            success = False
    
    if success:
        print("   ✅ All primitive distance tests passed")
    else:
        print("   ❌ Some primitive distance tests failed")
    
    return success


def test_boolean_operations():
    """Test Boolean operation correctness"""
    print("\n🔍 Testing SDF Boolean Operations...")
    
    success = True
    tolerance = 1e-6
    
    # Create test primitives
    sphere_a = SDFSphere(center=[-0.5, 0, 0], radius=0.8, seed=42)
    sphere_b = SDFSphere(center=[0.5, 0, 0], radius=0.8, seed=43)
    
    # Test union
    union = SDFUnion(sphere_a, sphere_b, seed=42)
    subtract = SDFSubtract(sphere_a, sphere_b, seed=42)
    intersect = SDFIntersect(sphere_a, sphere_b, seed=42)
    
    test_points = [
        np.array([-1.0, 0, 0]),  # Clearly in sphere_a only
        np.array([1.0, 0, 0]),   # Clearly in sphere_b only
        np.array([0, 0, 0]),     # In intersection region
        np.array([0, 2, 0]),     # Outside both spheres
    ]
    
    print("   Boolean operation tests:")
    for point in test_points:
        dist_a = sphere_a.evaluate(point)
        dist_b = sphere_b.evaluate(point)
        
        union_dist = union.evaluate(point)
        subtract_dist = subtract.evaluate(point)
        intersect_dist = intersect.evaluate(point)
        
        # Validate union: should be minimum of both
        expected_union = min(dist_a, dist_b)
        if abs(union_dist - expected_union) > tolerance:
            print(f"   ❌ Union at {point}: expected {expected_union:.6f}, got {union_dist:.6f}")
            success = False
        else:
            print(f"   ✅ Union at {point}: {union_dist:.6f}")
        
        # Validate subtraction: should be max(A, -B)
        expected_subtract = max(dist_a, -dist_b)
        if abs(subtract_dist - expected_subtract) > tolerance:
            print(f"   ❌ Subtract at {point}: expected {expected_subtract:.6f}, got {subtract_dist:.6f}")
            success = False
        else:
            print(f"   ✅ Subtract at {point}: {subtract_dist:.6f}")
        
        # Validate intersection: should be maximum of both
        expected_intersect = max(dist_a, dist_b)
        if abs(intersect_dist - expected_intersect) > tolerance:
            print(f"   ❌ Intersect at {point}: expected {expected_intersect:.6f}, got {intersect_dist:.6f}")
            success = False
        else:
            print(f"   ✅ Intersect at {point}: {intersect_dist:.6f}")
    
    # Test smooth operations
    smooth_union = SDFSmoothUnion(sphere_a, sphere_b, blend_radius=0.2, seed=42)
    smooth_subtract = SDFSmoothSubtract(sphere_a, sphere_b, blend_radius=0.2, seed=42)
    
    print("   Smooth operation tests:")
    for point in test_points:
        smooth_union_dist = smooth_union.evaluate(point)
        smooth_subtract_dist = smooth_subtract.evaluate(point)
        
        print(f"   ✅ Smooth union at {point}: {smooth_union_dist:.6f}")
        print(f"   ✅ Smooth subtract at {point}: {smooth_subtract_dist:.6f}")
        
        # Check for finite values
        if not np.isfinite(smooth_union_dist) or not np.isfinite(smooth_subtract_dist):
            print(f"   ❌ Invalid smooth operation result at {point}")
            success = False
    
    if success:
        print("   ✅ All boolean operation tests passed")
    else:
        print("   ❌ Some boolean operation tests failed")
    
    return success


def test_noise_displacement():
    """Test noise displacement determinism and properties"""
    print("\n🔍 Testing Noise Displacement...")
    
    success = True
    
    # Create base sphere
    base_sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    
    # Create displaced versions with same parameters
    displaced_a = SDFNoiseDisplace(base_sphere, displacement_scale=0.1, 
                                  noise_frequency=2.0, octaves=3, seed=42)
    displaced_b = SDFNoiseDisplace(base_sphere, displacement_scale=0.1, 
                                  noise_frequency=2.0, octaves=3, seed=42)
    
    # Test determinism
    test_points = [
        np.array([0.5, 0.5, 0.5]),
        np.array([-0.3, 0.7, -0.2]),
        np.array([1.2, -0.8, 0.9]),
        np.array([0, 0, 0])
    ]
    
    print("   Determinism tests:")
    for point in test_points:
        dist_a = displaced_a.evaluate(point)
        dist_b = displaced_b.evaluate(point)
        
        if abs(dist_a - dist_b) > 1e-10:
            print(f"   ❌ Non-deterministic at {point}: {dist_a:.10f} vs {dist_b:.10f}")
            success = False
        else:
            print(f"   ✅ Deterministic at {point}: {dist_a:.6f}")
    
    # Test that displacement actually changes the distance
    print("   Displacement effect tests:")
    for point in test_points:
        base_dist = base_sphere.evaluate(point)
        displaced_dist = displaced_a.evaluate(point)
        
        print(f"   Point {point}: base={base_dist:.6f}, displaced={displaced_dist:.6f}")
        
        # For most points, displacement should change the distance
        # (though it's possible for some points to have zero displacement)
    
    # Test different seeds produce different results
    displaced_c = SDFNoiseDisplace(base_sphere, displacement_scale=0.1, 
                                  noise_frequency=2.0, octaves=3, seed=123)
    
    print("   Seed variation tests:")
    differences_found = 0
    for point in test_points:
        dist_a = displaced_a.evaluate(point)
        dist_c = displaced_c.evaluate(point)
        
        if abs(dist_a - dist_c) > 1e-6:
            differences_found += 1
        
        print(f"   Point {point}: seed42={dist_a:.6f}, seed123={dist_c:.6f}")
    
    if differences_found == 0:
        print("   ⚠️ Different seeds produced identical results (unexpected)")
        success = False
    else:
        print(f"   ✅ Different seeds produced different results at {differences_found}/{len(test_points)} points")
    
    if success:
        print("   ✅ All noise displacement tests passed")
    else:
        print("   ❌ Some noise displacement tests failed")
    
    return success


def test_pcc_composition():
    """Test PCC-style SDF composition"""
    print("\n🔍 Testing PCC Composition...")
    
    success = True
    
    # Create evaluator
    evaluator = SDFEvaluator(seed=42)
    
    # Define complex SDF specification
    sdf_spec = {
        "type": "subtract",
        "seed": 42,
        "sdf_a": {
            "type": "box",
            "center": [0, 0, 0],
            "size": [2, 2, 2],
            "seed": 42
        },
        "sdf_b": {
            "type": "union",
            "seed": 43,
            "sdf_a": {
                "type": "sphere",
                "center": [0, 0, 0],
                "radius": 0.8,
                "seed": 44
            },
            "sdf_b": {
                "type": "noise_displace",
                "displacement_scale": 0.1,
                "noise_frequency": 3.0,
                "octaves": 2,
                "seed": 45,
                "base": {
                    "type": "capsule",
                    "point_a": [-0.5, -0.5, 0],
                    "point_b": [0.5, 0.5, 0],
                    "radius": 0.3,
                    "seed": 46
                }
            }
        }
    }
    
    print("   Building SDF from PCC specification...")
    try:
        sdf_tree = evaluator.build_sdf_from_pcc(sdf_spec)
        print(f"   ✅ SDF tree built: {type(sdf_tree).__name__}")
        
        # Test evaluation
        test_points = [
            np.array([0, 0, 0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([-0.8, 0.3, -0.2]),
            np.array([1.5, 1.5, 1.5])
        ]
        
        print("   Evaluating composed SDF:")
        for point in test_points:
            distance = evaluator.evaluate_sdf(point, sdf_tree)
            print(f"   Point {point}: distance = {distance:.6f}")
            
            if not np.isfinite(distance):
                print(f"   ❌ Invalid distance at {point}: {distance}")
                success = False
        
    except Exception as e:
        print(f"   ❌ Failed to build/evaluate SDF: {e}")
        success = False
    
    # Test transformation
    transform_spec = {
        "type": "transform",
        "translation": [1, 0, 0],
        "rotation": [0, 0, math.pi/4],
        "scale": [2, 1, 1],
        "seed": 42,
        "base": {
            "type": "sphere",
            "center": [0, 0, 0],
            "radius": 0.5,
            "seed": 42
        }
    }
    
    print("   Testing transformation SDF:")
    try:
        transform_sdf = evaluator.build_sdf_from_pcc(transform_spec)
        
        # Test that transformation affects the SDF appropriately
        point = np.array([1, 0, 0])  # Should be closer to transformed sphere
        distance = evaluator.evaluate_sdf(point, transform_sdf)
        print(f"   Transformed SDF at {point}: {distance:.6f}")
        
    except Exception as e:
        print(f"   ❌ Failed to build transformation SDF: {e}")
        success = False
    
    if success:
        print("   ✅ All PCC composition tests passed")
    else:
        print("   ❌ Some PCC composition tests failed")
    
    return success


def test_voxel_grid_sampling():
    """Test voxel grid sampling accuracy and efficiency"""
    print("\n🔍 Testing Voxel Grid Sampling...")
    
    success = True
    
    # Create test bounds
    bounds = ChunkBounds(
        min_point=np.array([-1.0, -1.0, -1.0]),
        max_point=np.array([1.0, 1.0, 1.0])
    )
    
    # Create voxel grids of different resolutions
    resolutions = [8, 16, 32]
    
    for resolution in resolutions:
        print(f"   Testing resolution {resolution}^3 = {resolution**3} voxels:")
        
        voxel_grid = VoxelGrid(bounds, resolution)
        
        # Verify grid properties
        expected_voxels = resolution ** 3
        if voxel_grid.total_voxels != expected_voxels:
            print(f"   ❌ Wrong voxel count: expected {expected_voxels}, got {voxel_grid.total_voxels}")
            success = False
        else:
            print(f"   ✅ Correct voxel count: {voxel_grid.total_voxels}")
        
        # Test voxel position generation
        if voxel_grid.voxel_positions.shape != (expected_voxels, 3):
            print(f"   ❌ Wrong position array shape: {voxel_grid.voxel_positions.shape}")
            success = False
        else:
            print(f"   ✅ Correct position array shape: {voxel_grid.voxel_positions.shape}")
        
        # Test that all positions are within bounds
        min_pos = np.min(voxel_grid.voxel_positions, axis=0)
        max_pos = np.max(voxel_grid.voxel_positions, axis=0)
        
        bounds_tolerance = 1e-10
        if (np.any(min_pos < bounds.min_point - bounds_tolerance) or 
            np.any(max_pos > bounds.max_point + bounds_tolerance)):
            print(f"   ❌ Positions outside bounds: {min_pos} to {max_pos}")
            success = False
        else:
            print(f"   ✅ All positions within bounds")
        
        # Test sampling with simple sphere
        sphere = SDFSphere(center=[0, 0, 0], radius=0.5, seed=42)
        
        start_time = time.time()
        scalar_field = sphere.evaluate_batch(voxel_grid.voxel_positions)
        sampling_time = time.time() - start_time
        
        print(f"   Sampling time: {sampling_time*1000:.2f} ms")
        
        # Verify scalar field properties
        if scalar_field.shape != (expected_voxels,):
            print(f"   ❌ Wrong scalar field shape: {scalar_field.shape}")
            success = False
        else:
            print(f"   ✅ Correct scalar field shape: {scalar_field.shape}")
        
        # Check for reasonable distance values
        if not np.all(np.isfinite(scalar_field)):
            print(f"   ❌ Non-finite values in scalar field")
            success = False
        else:
            print(f"   ✅ All finite values in scalar field")
        
        # Test reshape to 3D
        scalar_field_3d = voxel_grid.reshape_scalar_field(scalar_field)
        expected_3d_shape = (resolution, resolution, resolution)
        if scalar_field_3d.shape != expected_3d_shape:
            print(f"   ❌ Wrong 3D shape: {scalar_field_3d.shape}, expected {expected_3d_shape}")
            success = False
        else:
            print(f"   ✅ Correct 3D reshape: {scalar_field_3d.shape}")
    
    if success:
        print("   ✅ All voxel grid sampling tests passed")
    else:
        print("   ❌ Some voxel grid sampling tests failed")
    
    return success


def test_deterministic_chunk_sampling():
    """Test deterministic SDF sampling per chunk"""
    print("\n🔍 Testing Deterministic Chunk Sampling...")
    
    success = True
    
    # Create evaluator
    evaluator = SDFEvaluator(seed=42)
    
    # Define chunk bounds (simulating different chunks)
    chunk_bounds_list = [
        ChunkBounds(np.array([0, 0, 0]), np.array([2, 2, 2])),     # Chunk A
        ChunkBounds(np.array([2, 0, 0]), np.array([4, 2, 2])),     # Chunk B (adjacent)
        ChunkBounds(np.array([0, 2, 0]), np.array([2, 4, 2])),     # Chunk C (adjacent)
    ]
    
    # Create cave system for each chunk
    print("   Testing cave system generation:")
    for i, bounds in enumerate(chunk_bounds_list):
        cave_spec = create_cave_system_sdf(bounds, seed=42)  # Same seed
        cave_sdf = evaluator.build_sdf_from_pcc(cave_spec)
        
        # Sample voxel grid
        voxel_grid = VoxelGrid(bounds, resolution=16)
        scalar_field_a = evaluator.sample_voxel_grid(voxel_grid, cave_sdf)
        
        # Sample again with same parameters
        scalar_field_b = evaluator.sample_voxel_grid(voxel_grid, cave_sdf)
        
        # Check determinism
        if not np.allclose(scalar_field_a, scalar_field_b):
            print(f"   ❌ Non-deterministic sampling for chunk {i}")
            success = False
        else:
            print(f"   ✅ Deterministic sampling for chunk {i}")
        
        # Check field properties
        surface_voxels = np.sum(np.abs(scalar_field_a) < voxel_grid.min_voxel_size)
        inside_voxels = np.sum(scalar_field_a < 0)
        outside_voxels = np.sum(scalar_field_a > 0)
        
        print(f"   Chunk {i}: {surface_voxels} surface, {inside_voxels} inside, {outside_voxels} outside")
    
    # Test seed variation produces different results
    print("   Testing seed variation:")
    bounds = chunk_bounds_list[0]
    voxel_grid = VoxelGrid(bounds, resolution=16)
    
    seeds = [42, 123, 456]
    scalar_fields = []
    
    for seed in seeds:
        cave_spec = create_cave_system_sdf(bounds, seed=seed)
        cave_sdf = evaluator.build_sdf_from_pcc(cave_spec)
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, cave_sdf)
        scalar_fields.append(scalar_field)
        
        surface_voxels = np.sum(np.abs(scalar_field) < voxel_grid.min_voxel_size)
        print(f"   Seed {seed}: {surface_voxels} surface voxels")
    
    # Check that different seeds produce different results
    differences_found = 0
    for i in range(len(scalar_fields)):
        for j in range(i+1, len(scalar_fields)):
            if not np.allclose(scalar_fields[i], scalar_fields[j]):
                differences_found += 1
    
    expected_differences = len(seeds) * (len(seeds) - 1) // 2
    if differences_found != expected_differences:
        print(f"   ⚠️ Expected {expected_differences} differences, found {differences_found}")
        success = False
    else:
        print(f"   ✅ Different seeds produce different results ({differences_found} combinations)")
    
    # Test overhang generation
    print("   Testing overhang generation:")
    for i, bounds in enumerate(chunk_bounds_list[:2]):  # Test first two chunks
        overhang_spec = create_overhang_sdf(bounds, seed=42)
        overhang_sdf = evaluator.build_sdf_from_pcc(overhang_spec)
        
        voxel_grid = VoxelGrid(bounds, resolution=16)
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, overhang_sdf)
        
        inside_voxels = np.sum(scalar_field < 0)
        print(f"   Overhang chunk {i}: {inside_voxels} inside voxels")
        
        if not np.all(np.isfinite(scalar_field)):
            print(f"   ❌ Non-finite values in overhang chunk {i}")
            success = False
    
    if success:
        print("   ✅ All deterministic chunk sampling tests passed")
    else:
        print("   ❌ Some deterministic chunk sampling tests failed")
    
    return success


def benchmark_sdf_performance():
    """Benchmark SDF evaluation performance"""
    print("\n🚀 SDF Performance Benchmarks...")
    
    # Create test scenarios
    scenarios = [
        ("Simple Sphere", SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)),
        ("Complex Gyroid", SDFGyroid(scale=2.0, thickness=0.1, seed=42)),
        ("Noisy Sphere", SDFNoiseDisplace(
            SDFSphere(center=[0, 0, 0], radius=1.0, seed=42),
            displacement_scale=0.1, noise_frequency=2.0, octaves=4, seed=42
        )),
        ("Boolean Union", SDFUnion(
            SDFSphere(center=[-0.5, 0, 0], radius=0.8, seed=42),
            SDFSphere(center=[0.5, 0, 0], radius=0.8, seed=43),
            seed=42
        ))
    ]
    
    # Test different point counts
    point_counts = [1000, 10000, 100000]
    
    for scenario_name, sdf in scenarios:
        print(f"\n   {scenario_name}:")
        
        for point_count in point_counts:
            # Generate random test points
            points = np.random.random((point_count, 3)) * 4.0 - 2.0  # Range [-2, 2]
            
            # Benchmark evaluation
            start_time = time.time()
            distances = sdf.evaluate_batch(points)
            evaluation_time = time.time() - start_time
            
            evaluations_per_second = point_count / evaluation_time
            
            print(f"     {point_count:6d} points: {evaluation_time*1000:6.2f} ms ({evaluations_per_second:8.0f} eval/sec)")
    
    print("\n   ✅ Performance benchmarks completed")


def run_comprehensive_sdf_tests():
    """Run all SDF system tests"""
    print("🚀 T09 SDF System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Primitive Distance Calculations", test_primitive_distances),
        ("Boolean Operations", test_boolean_operations),
        ("Noise Displacement", test_noise_displacement),
        ("PCC Composition", test_pcc_composition),
        ("Voxel Grid Sampling", test_voxel_grid_sampling),
        ("Deterministic Chunk Sampling", test_deterministic_chunk_sampling),
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
    benchmark_sdf_performance()
    
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
        print(f"\n🎉 All SDF system tests passed!")
        return True
    else:
        print(f"\n⚠️ Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    success = run_comprehensive_sdf_tests()
    sys.exit(0 if success else 1)