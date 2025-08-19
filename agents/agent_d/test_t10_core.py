#!/usr/bin/env python3
"""
T10 Core System Test - Simplified
=================================

Core functionality tests for T10 Marching Cubes system.
Tests essential components without full triangle table dependency.
"""

import numpy as np
import os
import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'marching_cubes'))

def test_sdf_gradient_normals():
    """Test SDF gradient normal generation"""
    print("🔍 Testing SDF gradient normals...")
    
    try:
        from sdf_primitives import SDFSphere, sdf_gradient
        
        sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
        
        # Test point on sphere surface  
        point = np.array([1.0, 0.0, 0.0])
        gradient = sdf_gradient(sphere, point, 1e-4)
        
        # Should point outward from center
        expected = np.array([1.0, 0.0, 0.0])
        error = np.linalg.norm(gradient - expected)
        
        if error < 1e-3:
            print("   ✅ SDF gradient calculation working")
            return True
        else:
            print(f"   ❌ SDF gradient error: {error}")
            return False
            
    except Exception as e:
        print(f"   ❌ SDF gradient test failed: {e}")
        return False


def test_voxel_grid_sampling():
    """Test voxel grid SDF sampling"""
    print("🔍 Testing voxel grid sampling...")
    
    try:
        from sdf_evaluator import VoxelGrid, ChunkBounds, SDFEvaluator
        from sdf_primitives import SDFSphere
        
        # Simple test setup
        bounds = ChunkBounds(
            min_point=np.array([-1.5, -1.5, -1.5]),
            max_point=np.array([1.5, 1.5, 1.5])
        )
        
        voxel_grid = VoxelGrid(bounds, resolution=8)
        sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
        
        evaluator = SDFEvaluator()
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, sphere)
        
        # Should have both positive and negative values
        has_negative = np.any(scalar_field < 0)
        has_positive = np.any(scalar_field > 0) 
        
        if has_negative and has_positive:
            print(f"   ✅ Voxel sampling working ({len(scalar_field)} voxels)")
            return True
        else:
            print("   ❌ Voxel sampling: no surface crossing found")
            return False
            
    except Exception as e:
        print(f"   ❌ Voxel grid test failed: {e}")
        return False


def test_cave_manifest_system():
    """Test cave manifest basic functionality"""
    print("🔍 Testing cave manifest system...")
    
    try:
        from cave_manifest import CaveManifestManager
        
        manager = CaveManifestManager()
        
        # Test material colors
        color = manager.get_material_color(1)
        if len(color) == 3:
            print("   ✅ Cave manifest system working")
            return True
        else:
            print("   ❌ Invalid material color format")
            return False
            
    except Exception as e:
        print(f"   ❌ Cave manifest test failed: {e}")
        return False


def test_chunk_cave_generation():
    """Test basic cave generation workflow"""
    print("🔍 Testing cave generation workflow...")
    
    try:
        from sdf_evaluator import create_cave_system_sdf, ChunkBounds, SDFEvaluator
        
        bounds = ChunkBounds(
            min_point=np.array([-2, -2, -2]),
            max_point=np.array([2, 2, 2])
        )
        
        # Create cave SDF specification
        cave_spec = create_cave_system_sdf(bounds, seed=42)
        
        # Build SDF tree
        evaluator = SDFEvaluator()
        cave_sdf = evaluator.build_sdf_from_pcc(cave_spec)
        
        # Test evaluation at a point
        test_point = np.array([0, 0, 0])
        distance = cave_sdf.evaluate(test_point)
        
        if np.isfinite(distance):
            print("   ✅ Cave generation workflow working")
            return True
        else:
            print("   ❌ Invalid distance from cave SDF")
            return False
            
    except Exception as e:
        print(f"   ❌ Cave generation test failed: {e}")
        return False


def run_t10_core_tests():
    """Run core T10 functionality tests"""
    print("🚀 T10 Core System Tests")
    print("=" * 40)
    
    tests = [
        ("SDF Gradient Normals", test_sdf_gradient_normals),
        ("Voxel Grid Sampling", test_voxel_grid_sampling),
        ("Cave Manifest System", test_cave_manifest_system),
        ("Cave Generation Workflow", test_chunk_cave_generation),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 T10 core system functional!")
        return True
    else:
        print("⚠️ Some core tests failed")
        return False


if __name__ == "__main__":
    success = run_t10_core_tests()
    sys.exit(0 if success else 1)