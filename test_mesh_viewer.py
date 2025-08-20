#!/usr/bin/env python3
"""
Test script for the mesh viewer functionality
Creates a simple scene to test the VAO/VBO/EBO mesh rendering
"""
import json
import tempfile
from pathlib import Path

def create_test_scene():
    """Create a simple test scene with spheres to test mesh rendering"""
    scene = {
        "objects": [
            {
                "type": "SPHERE",
                "pos": [0, 1, 0],
                "radius": 1.0,
                "material": "player"
            },
            {
                "type": "SPHERE", 
                "pos": [3, 1, 0],
                "radius": 0.5,
                "material": "collectible"
            },
            {
                "type": "CUBE",
                "pos": [0, 0.5, 3],
                "size": [1, 1, 1],
                "material": "structure"
            }
        ]
    }
    return scene

if __name__ == "__main__":
    # Create test scene
    scene = create_test_scene()
    
    # Write to temporary file
    test_file = Path("/tmp/test_mesh_scene.json")
    with open(test_file, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"✅ Test scene created: {test_file}")
    print("🎮 To test the mesh viewer, run:")
    print(f"   python renderer/pcc_game_viewer.py {test_file}")
    print()
    print("🔧 Debug controls:")
    print("   F - Toggle wireframe mode")
    print("   B - Toggle AABB debug")
    print("   Look for grid meshes instead of solid spheres!")