#!/usr/bin/env python3
"""
Test script for T03 normals and tangents verification
Creates a scene that uses the T03 cube-sphere with proper shading basis
"""
import json
import tempfile
from pathlib import Path

def create_t03_test_scene():
    """Create a test scene that uses the T03 cube-sphere with normals/tangents"""
    
    # Path to the T03 generated cube-sphere manifest (with verified analytical normals)
    cubesphere_manifest = Path(__file__).parent / "agents/agent_d/mesh/t03_cubesphere_analytical.json"
    
    scene = {
        "objects": [
            {
                "type": "MESH",
                "pos": [0, 0, 0],
                "manifest": str(cubesphere_manifest),
                "material": "terrain"
            },
            {
                "type": "CUBE",
                "pos": [2, 0, 0],
                "size": [0.2, 0.2, 0.2],
                "material": "reference"
            }
        ],
        "terrain": {
            "type": "sphere",
            "center": [0, 0, 0], 
            "radius": 1.0
        },
        "metadata": {
            "test": "T03_normals_tangents",
            "description": "Verification of angle-weighted normals and tangent basis"
        }
    }
    return scene

if __name__ == "__main__":
    # Create test scene
    scene = create_t03_test_scene()
    
    # Write to temporary file
    test_file = Path("/tmp/test_t03_normals_scene.json")
    with open(test_file, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"✅ T03 normals test scene created: {test_file}")
    print("🎮 To test the T03 normals viewer, run:")
    print(f"   python renderer/pcc_game_viewer.py {test_file}")
    print()
    print("🔧 Debug controls for verification:")
    print("   F - Toggle wireframe to see triangle structure")
    print("   B - Toggle AABB bounding box")
    print("   N - Toggle normal visualization (cyan lines)")
    print("   👁️ Verify normals point outward from sphere surface")
    print("   📐 Check that normals are perpendicular to surface")
    print()
    print("🔍 What to look for:")
    print("   - Cyan lines should point outward from sphere")
    print("   - Normal length should be consistent (~0.1 units)")
    print("   - No inward-pointing normals")
    print("   - Smooth normal distribution across surface")