#!/usr/bin/env python3
"""
SUMMARY: OpenGL Runtime Launcher
================================
Quick launcher for StarSystem OpenGL viewer with proper environment setup.
Supports mini-planet generation and FP navigation for Week 3 alpha proof.

KEY FEATURES:
- FORGE_RUNTIME=gl environment configuration
- Automatic mini-planet seed generation if no scene provided
- First-person navigation with WASD+mouse controls
- Bridge protocol support for Agent B interaction

USAGE:
  python scripts/run_gl.py                      # Generate and run default mini-planet
  python scripts/run_gl.py scene.json           # Run specific scene
  python scripts/run_gl.py --seed 12345         # Generate with specific seed
  python scripts/run_gl.py --bridge-port 8765   # Enable Agent B bridge

RELATED FILES:
- renderer/pcc_simple_viewer.py - Enhanced OpenGL viewer
- forge/modules/miniplanet/ - Mini-planet generation system
- Week 3 requirement for alpha proof demonstration
"""

import os
import sys
import subprocess
import argparse
import json
import random
import math
from pathlib import Path

# Set runtime environment
os.environ.setdefault('FORGE_RUNTIME', 'gl')

def generate_miniplanet_seed(seed=None):
    """Generate a simple mini-planet scene"""
    if seed is None:
        seed = random.randint(1000, 9999)
    
    random.seed(seed)
    
    # Simple mini-planet with terrain and some objects
    scene = {
        "metadata": {
            "scene_type": "miniplanet",
            "seed": seed,
            "generated_at": "2025-01-01T00:00:00Z",
            "layer": "surface"
        },
        "terrain": {
            "type": "sphere",
            "radius": 15.0,
            "center": [0, 0, 0],
            "material": "rock"
        },
        "objects": []
    }
    
    # Add objects positioned on sphere surface
    planet_radius = 15.0
    for i in range(12):
        # Generate random position on sphere surface
        theta = random.uniform(0, 2 * math.pi)  # Azimuth
        phi = random.uniform(0.2, math.pi - 0.2)  # Elevation (avoid poles)
        
        # Convert spherical to cartesian (surface position)
        x = planet_radius * math.sin(phi) * math.cos(theta)
        y = planet_radius * math.cos(phi)
        z = planet_radius * math.sin(phi) * math.sin(theta)
        
        # Offset slightly above surface
        surface_offset = random.uniform(0.5, 2.0)
        normal_x, normal_y, normal_z = x/planet_radius, y/planet_radius, z/planet_radius
        
        final_x = x + normal_x * surface_offset
        final_y = y + normal_y * surface_offset
        final_z = z + normal_z * surface_offset
        
        if random.random() < 0.6:  # 60% terrain blocks
            height = random.uniform(1.0, 3.0)
            scene["objects"].append({
                "type": "CUBE",
                "pos": [final_x, final_y, final_z],
                "size": [1.2, height, 1.2],
                "material": "terrain"
            })
        else:  # 40% resource spheres
            scene["objects"].append({
                "type": "SPHERE", 
                "pos": [final_x + normal_x * 0.5, final_y + normal_y * 0.5, final_z + normal_z * 0.5],
                "radius": random.uniform(0.4, 0.9),
                "material": "resource"
            })
    
    # Add special landmark objects
    # North pole beacon
    scene["objects"].append({
        "type": "SPHERE",
        "pos": [0, planet_radius + 2.0, 0],  # North pole
        "radius": 1.2,
        "material": "beacon"
    })
    
    # Equatorial structure
    equator_x = planet_radius + 1.5
    scene["objects"].append({
        "type": "CUBE",
        "pos": [equator_x, 0, 0],  # On equator
        "size": [2, 4, 2],
        "material": "structure"
    })
    
    # South pole marker
    scene["objects"].append({
        "type": "CUBE",
        "pos": [0, -planet_radius - 1.0, 0],  # South pole
        "size": [1.5, 2, 1.5],
        "material": "structure"
    })
    
    return scene, seed

def main():
    parser = argparse.ArgumentParser(description="Launch StarSystem OpenGL viewer")
    parser.add_argument("scene_file", nargs="?", help="Scene file to load (auto-generates if not provided)")
    parser.add_argument("--seed", type=int, help="Seed for mini-planet generation")
    parser.add_argument("--bridge-port", type=int, default=8765, help="Port for Agent B bridge")
    parser.add_argument("--generate-only", action="store_true", help="Generate scene file and exit")
    
    args = parser.parse_args()
    
    # Determine scene file
    if args.scene_file:
        scene_file = args.scene_file
        if not Path(scene_file).exists():
            print(f"❌ Scene file not found: {scene_file}")
            return 1
    else:
        # Generate mini-planet
        print("🌍 Generating mini-planet...")
        scene_data, used_seed = generate_miniplanet_seed(args.seed)
        
        # Save to temporary file
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        scene_file = runs_dir / f"miniplanet_seed_{used_seed}.json"
        
        with open(scene_file, 'w') as f:
            json.dump(scene_data, f, indent=2)
        
        print(f"✅ Generated mini-planet with seed {used_seed}")
        print(f"📁 Scene saved to: {scene_file}")
        
        if args.generate_only:
            return 0
    
    print(f"🚀 Starting OpenGL viewer...")
    print(f"📋 Runtime: {os.environ.get('FORGE_RUNTIME', 'unknown')}")
    print(f"🎮 Bridge port: {args.bridge_port}")
    print()
    
    # Launch spherical viewer with bridge support
    cmd = [
        sys.executable, 
        "renderer/pcc_spherical_viewer.py", 
        str(scene_file),
        "--bridge-port", str(args.bridge_port)
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n👋 Viewer closed by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to launch viewer: {e}")
        return 1

if __name__ == "__main__":
    exit(main())