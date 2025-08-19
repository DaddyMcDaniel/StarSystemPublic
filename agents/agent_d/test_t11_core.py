#!/usr/bin/env python3
"""
T11 Core System Test - Surface-SDF Fusion
==========================================

Core functionality tests for T11 surface-SDF fusion system.
Tests terrain-cave blending, border continuity, and seamless rendering.
"""

import numpy as np
import os
import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'marching_cubes'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'fusion'))

def test_heightfield_to_sdf_conversion():
    """Test conversion of heightfield terrain to SDF representation"""
    print("🔍 Testing heightfield to SDF conversion...")
    
    try:
        from fusion.surface_sdf_fusion import TerrainHeightfield, HeightfieldSDF
        from sdf_evaluator import ChunkBounds
        
        # Create simple heightfield
        bounds = ChunkBounds(
            min_point=np.array([-2, -1, -2]),
            max_point=np.array([2, 1, 2])
        )
        
        heights = np.zeros((16, 16), dtype=np.float32)
        for i in range(16):
            for j in range(16):
                heights[j, i] = 0.2 * np.sin(i * 0.5) * np.cos(j * 0.5)
        
        heightfield = TerrainHeightfield(16, 16, heights, bounds)
        
        # Convert to SDF
        heightfield_sdf = HeightfieldSDF(heightfield)
        
        # Test SDF evaluation
        test_points = [
            np.array([0, 0, 0]),      # Should be near surface
            np.array([0, 0.5, 0]),    # Above surface
            np.array([0, -0.5, 0])    # Below surface
        ]
        
        valid_distances = 0
        for point in test_points:
            distance = heightfield_sdf.evaluate(point)
            if np.isfinite(distance):
                valid_distances += 1
        
        if valid_distances == len(test_points):
            print("   ✅ Heightfield to SDF conversion working")
            return True
        else:
            print("   ❌ Invalid SDF distances")
            return False
            
    except Exception as e:
        print(f"   ❌ Heightfield to SDF test failed: {e}")
        return False


def test_terrain_cave_fusion():
    """Test basic terrain-cave fusion functionality"""
    print("🔍 Testing terrain-cave fusion...")
    
    try:
        from fusion.surface_sdf_fusion import TerrainSdfFusion
        from sdf_evaluator import ChunkBounds
        from sdf_primitives import SDFSphere
        
        # Create fusion system
        fusion = TerrainSdfFusion(resolution=16, overlap_voxels=1)
        
        # Create test bounds and chunk
        bounds = ChunkBounds(
            min_point=np.array([-1.5, -1, -1.5]),
            max_point=np.array([1.5, 1, 1.5])
        )
        
        terrain_chunk = {
            "positions": np.array([
                [0, 0, 0], [1, 0, 0], [0, 0, 1],
                [-1, 0, 0], [0, 0, -1]
            ]),
            "chunk_info": {
                "chunk_id": "test_fusion",
                "aabb": {
                    "min": bounds.min_point.tolist(),
                    "max": bounds.max_point.tolist()
                }
            }
        }
        
        # Create simple cave SDF
        cave_sdf = SDFSphere(center=[0, 0, 0], radius=0.8, seed=42)
        
        # Test fusion
        fused_mesh = fusion.fuse_terrain_and_caves(terrain_chunk, cave_sdf, bounds, "test")
        
        if (len(fused_mesh.vertices) > 0 and 
            len(fused_mesh.triangles) > 0 and
            fused_mesh.has_caves):
            print(f"   ✅ Fusion working ({len(fused_mesh.vertices)} vertices, {len(fused_mesh.triangles)} triangles)")
            return True
        else:
            print("   ❌ No fusion mesh generated")
            return False
            
    except Exception as e:
        print(f"   ❌ Terrain-cave fusion test failed: {e}")
        return False


def test_chunk_border_continuity():
    """Test chunk border continuity system"""
    print("🔍 Testing chunk border continuity...")
    
    try:
        from fusion.chunk_border_fusion import ChunkBorderManager, ChunkNeighbors
        from sdf_evaluator import ChunkBounds
        
        # Create border manager
        border_mgr = ChunkBorderManager()
        
        # Create adjacent test chunks
        chunk_bounds_list = [
            ("chunk_a", ChunkBounds(
                np.array([-2, -1, -2]), 
                np.array([0, 1, 0])
            )),
            ("chunk_b", ChunkBounds(
                np.array([0, -1, -2]),
                np.array([2, 1, 0])
            ))
        ]
        
        # Analyze layout
        border_mgr.analyze_chunk_layout(chunk_bounds_list)
        
        # Check neighbor detection
        neighbors_a = border_mgr.chunk_neighbors["chunk_a"]
        neighbors_b = border_mgr.chunk_neighbors["chunk_b"]
        
        if (neighbors_a.right == "chunk_b" and 
            neighbors_b.left == "chunk_a"):
            print("   ✅ Chunk neighbor detection working")
            return True
        else:
            print("   ❌ Chunk neighbor detection failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Chunk border continuity test failed: {e}")
        return False


def test_sdf_debug_overlay():
    """Test SDF debug overlay system"""
    print("🔍 Testing SDF debug overlay...")
    
    try:
        from fusion.sdf_debug_overlay import SdfDebugOverlay
        from fusion.surface_sdf_fusion import FusedMeshData
        from marching_cubes import MarchingCubesVertex
        from sdf_primitives import SDFSphere
        from sdf_evaluator import ChunkBounds
        
        # Create debug overlay
        overlay = SdfDebugOverlay()
        
        # Create test SDF
        test_sdf = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
        
        # Create mock fused mesh
        test_vertices = [
            MarchingCubesVertex(np.array([0, 0, 0]), np.array([0, 1, 0])),
            MarchingCubesVertex(np.array([1, 0, 0]), np.array([1, 0, 0]))
        ]
        
        test_bounds = ChunkBounds(
            np.array([-1.5, -1.5, -1.5]),
            np.array([1.5, 1.5, 1.5])
        )
        
        test_mesh = FusedMeshData(
            vertices=test_vertices,
            triangles=np.array([[0, 1]]),
            chunk_id="test_debug",
            bounds=test_bounds,
            has_caves=True,
            fusion_stats={}
        )
        
        # Generate contours
        contour_set = overlay.generate_surface_contours(test_sdf, test_mesh)
        
        if len(contour_set.contours) > 0:
            print(f"   ✅ Debug overlay working ({len(contour_set.contours)} contours)")
            return True
        else:
            print("   ❌ No debug contours generated")
            return False
            
    except Exception as e:
        print(f"   ❌ SDF debug overlay test failed: {e}")
        return False


def test_seamless_chunk_processing():
    """Test end-to-end seamless chunk processing"""
    print("🔍 Testing seamless chunk processing...")
    
    try:
        from fusion.chunk_border_fusion import SeamlessChunkProcessor
        from sdf_primitives import SDFSphere
        
        # Create processor
        processor = SeamlessChunkProcessor(fusion_resolution=16, overlap_voxels=1)
        
        # Create test chunks
        terrain_chunks = {
            "chunk_0": {
                "positions": np.array([[0, 0, 0], [1, 0, 0]]),
                "chunk_info": {
                    "chunk_id": "chunk_0",
                    "aabb": {
                        "min": [-1, -1, -1],
                        "max": [1, 1, 1]
                    }
                }
            },
            "chunk_1": {
                "positions": np.array([[1, 0, 0], [2, 0, 0]]),
                "chunk_info": {
                    "chunk_id": "chunk_1", 
                    "aabb": {
                        "min": [1, -1, -1],
                        "max": [3, 1, 1]
                    }
                }
            }
        }
        
        # Create cave SDFs
        cave_sdfs = {
            "chunk_0": SDFSphere(center=[0, 0, 0], radius=0.5, seed=42),
            "chunk_1": SDFSphere(center=[2, 0, 0], radius=0.5, seed=43)
        }
        
        # Process chunks
        seamless_meshes = processor.process_terrain_chunks_with_caves(
            terrain_chunks, cave_sdfs
        )
        
        if (len(seamless_meshes) == 2 and
            all(len(mesh.vertices) > 0 for mesh in seamless_meshes.values())):
            print(f"   ✅ Seamless processing working ({len(seamless_meshes)} chunks)")
            return True
        else:
            print("   ❌ Seamless processing failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Seamless chunk processing test failed: {e}")
        return False


def run_t11_core_tests():
    """Run core T11 functionality tests"""
    print("🚀 T11 Surface-SDF Fusion Core Tests")
    print("=" * 50)
    
    tests = [
        ("Heightfield to SDF Conversion", test_heightfield_to_sdf_conversion),
        ("Terrain-Cave Fusion", test_terrain_cave_fusion),
        ("Chunk Border Continuity", test_chunk_border_continuity),
        ("SDF Debug Overlay", test_sdf_debug_overlay),
        ("Seamless Chunk Processing", test_seamless_chunk_processing),
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
        print("🎉 T11 surface-SDF fusion system functional!")
        return True
    else:
        print("⚠️ Some core tests failed")
        return False


if __name__ == "__main__":
    success = run_t11_core_tests()
    sys.exit(0 if success else 1)