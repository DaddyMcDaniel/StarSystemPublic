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
            "radius": 50.0,
            "center": [0, 0, 0],
            "material": "rock"
        },
        "objects": []
    }
    
    # Human feedback: Double planet size for better exploration
    planet_radius = 50.0  # Increased from 25.0 based on human feedback
    
    # Grounded placement utilities
    def _surface_point(theta: float, phi: float):
        sx = planet_radius * math.sin(phi) * math.cos(theta)
        sy = planet_radius * math.cos(phi)
        sz = planet_radius * math.sin(phi) * math.sin(theta)
        nx, ny, nz = sx/planet_radius, sy/planet_radius, sz/planet_radius
        return (sx, sy, sz), (nx, ny, nz)

    def _place_cube(center, normal, size_y):
        nx, ny, nz = normal
        # Seat base on surface by offsetting half height along normal
        return [center[0] + nx * (size_y * 0.5), center[1] + ny * (size_y * 0.5), center[2] + nz * (size_y * 0.5)]

    def _place_sphere(center, normal, radius):
        nx, ny, nz = normal
        return [center[0] + nx * radius, center[1] + ny * radius, center[2] + nz * radius]

    # Human feedback: More objects for larger planet, but all grounded and aligned to surface
    for i in range(40):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0.2, math.pi - 0.2)
        center, normal = _surface_point(theta, phi)

        # Terrain zones (theme) influences material and height variation, not floating
        terrain_zone = random.choice(["flat", "rolling", "mountainous"])
        if terrain_zone == "flat":
            terrain_type = random.choice(["grass", "sand", "metal"])  
            height = random.uniform(0.5, 1.5)
        elif terrain_zone == "rolling":
            terrain_type = random.choice(["rock", "grass", "crystal"]) 
            height = random.uniform(1.5, 3.0)
        else:
            terrain_type = random.choice(["rock", "crystal", "metal"]) 
            height = random.uniform(3.0, 6.0)

        r = random.random()
        if r < 0.45:
            size_x = random.uniform(0.5, 1.2)
            size_y = height
            size_z = random.uniform(0.5, 1.2)
            pos = _place_cube(center, normal, size_y)
            scene["objects"].append({
                "type": "CUBE",
                "pos": pos,
                "size": [size_x, size_y, size_z],
                "material": f"terrain_{terrain_type}",
                "grounded": True,
                "align_to_surface": True,
                "up": [normal[0], normal[1], normal[2]]
            })
        elif r < 0.75:
            resource_type = random.choice(["ore", "crystal", "energy", "rare"])
            radius = random.uniform(0.5, 1.2)
            pos = _place_sphere(center, normal, radius)
            scene["objects"].append({
                "type": "SPHERE",
                "pos": pos,
                "radius": radius,
                "material": f"resource_{resource_type}",
                "grounded": True,
                "up": [normal[0], normal[1], normal[2]]
            })
        else:
            landmark_type = random.choice(["pillar", "arch", "crater", "spire"])
            if landmark_type == "pillar":
                h = random.uniform(3.0, 6.0)
                pos = _place_cube(center, normal, h)
                scene["objects"].append({
                    "type": "CUBE",
                    "pos": pos,
                    "size": [0.8, h, 0.8],
                    "material": "landmark_stone",
                    "grounded": True,
                    "align_to_surface": True,
                    "up": [normal[0], normal[1], normal[2]]
                })
            elif landmark_type == "crater":
                rim_radius = 2.0
                for angle in range(0, 360, 45):
                    ang = math.radians(angle)
                    nx, ny, nz = normal
                    up = (0.0, 1.0, 0.0)
                    if abs(ny) > 0.9:
                        up = (1.0, 0.0, 0.0)
                    ux = up[1]*nz - up[2]*ny
                    uy = up[2]*nx - up[0]*nz
                    uz = up[0]*ny - up[1]*nx
                    ul = max(1e-6, math.sqrt(ux*ux + uy*uy + uz*uz))
                    ux, uy, uz = ux/ul, uy/ul, uz/ul
                    vx = ny*uz - nz*uy
                    vy = nz*ux - nx*uz
                    vz = nx*uy - ny*ux
                    rx = center[0] + rim_radius * (ux * math.cos(ang) + vx * math.sin(ang))
                    ry = center[1] + rim_radius * (uy * math.cos(ang) + vy * math.sin(ang))
                    rz = center[2] + rim_radius * (uz * math.cos(ang) + vz * math.sin(ang))
                    rl = max(1e-6, math.sqrt(rx*rx + ry*ry + rz*rz))
                    sx, sy, sz = (planet_radius * rx/rl, planet_radius * ry/rl, planet_radius * rz/rl)
                    sn = (sx/planet_radius, sy/planet_radius, sz/planet_radius)
                    pos = _place_cube((sx, sy, sz), sn, 0.5)
                    scene["objects"].append({
                        "type": "CUBE",
                        "pos": pos,
                        "size": [1.0, 0.5, 1.0],
                        "material": "landmark_crater",
                        "grounded": True,
                        "align_to_surface": True,
                        "up": [sn[0], sn[1], sn[2]]
                    })
            else:
                h = random.uniform(2.0, 4.0)
                pos = _place_cube(center, normal, h)
                scene["objects"].append({
                    "type": "CUBE",
                    "pos": pos,
                    "size": [1.5, h, 1.5],
                    "material": f"landmark_{landmark_type}",
                    "grounded": True,
                    "align_to_surface": True,
                    "up": [normal[0], normal[1], normal[2]]
                })
    
    # Add special landmark objects for larger planet
    # North pole beacon (enhanced)
    scene["objects"].append({
        "type": "SPHERE",
        "pos": [0, planet_radius + 3.0, 0],  # North pole
        "radius": 1.8,
        "material": "beacon_major",
        "grounded": True,
        "up": [0.0, 1.0, 0.0]
    })
    
    # Equatorial structures (multiple for larger planet)
    for angle in [0, 90, 180, 270]:
        ex = (planet_radius) * math.cos(math.radians(angle))
        ez = (planet_radius) * math.sin(math.radians(angle))
        # Normal at equator ring
        nx, nz = ex/planet_radius, ez/planet_radius
        ny = 0.0
        height = 5.0
        pos = [ex + nx * (height * 0.5 + 2.0), 0 + ny * (height * 0.5 + 2.0), ez + nz * (height * 0.5 + 2.0)]
        scene["objects"].append({
            "type": "CUBE",
            "pos": pos,
            "size": [2.5, height, 2.5],
            "material": "structure_temple",
            "grounded": True,
            "align_to_surface": True,
            "up": [nx, ny, nz]
        })
    
    # South pole marker (enhanced)
    scene["objects"].append({
        "type": "CUBE",
        "pos": [0, -planet_radius - 1.5, 0],  # South pole
        "size": [2.0, 3, 2.0],
        "material": "structure_monument"
    })
    
    # Agent B feedback: Add cave entrances (represented as dark spheres)
    for i in range(4):
        cave_theta = random.uniform(0, 2 * math.pi)
        cave_phi = random.uniform(0.3, math.pi - 0.3)
        cave_x = planet_radius * 0.95 * math.sin(cave_phi) * math.cos(cave_theta)
        cave_y = planet_radius * 0.95 * math.cos(cave_phi)
        cave_z = planet_radius * 0.95 * math.sin(cave_phi) * math.sin(cave_theta)
        
        scene["objects"].append({
            "type": "SPHERE",
            "pos": [cave_x, cave_y, cave_z],
            "radius": 1.5,
            "material": "cave_entrance",
            "grounded": True,
            "up": [cave_x/planet_radius, cave_y/planet_radius, cave_z/planet_radius]
        })
    
    return scene, seed

def main():
    parser = argparse.ArgumentParser(description="Launch StarSystem OpenGL viewer")
    parser.add_argument("scene_file", nargs="?", help="Scene file to load (auto-generates if not provided)")
    parser.add_argument("--seed", type=int, help="Seed for mini-planet generation")
    parser.add_argument("--bridge-port", type=int, default=8765, help="Port for Agent B bridge")
    parser.add_argument("--generate-only", action="store_true", help="Generate scene file and exit")
    parser.add_argument("--from-pcc", help="Generate scene from PCC AST instead of random scene")
    
    args = parser.parse_args()
    
    # Determine scene file
    if args.from_pcc:
        # Convert PCC AST to grounded scene
        pcc_path = Path(args.from_pcc)
        if not pcc_path.exists():
            print(f"❌ PCC file not found: {pcc_path}")
            return 1
        print("🧭 Converting PCC AST to grounded scene...")
        # Import converter with path fix
        import sys as _sys
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in _sys.path:
            _sys.path.insert(0, str(project_root))
        from scripts.ast_to_scene import main as ast_to_scene_main
        # Write to runs/<pcc>_scene.json
        out_scene = Path("runs") / (pcc_path.stem + "_scene.json")
        # Simulate CLI call
        import sys as _sys
        _argv_backup = _sys.argv[:]
        _sys.argv = ["ast_to_scene.py", str(pcc_path), "-o", str(out_scene)]
        ret = ast_to_scene_main()
        _sys.argv = _argv_backup
        if ret != 0:
            return ret
        scene_file = out_scene
    elif args.scene_file:
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