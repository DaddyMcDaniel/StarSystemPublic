#!/usr/bin/env python3
"""
Unified Mini-Planet System
Fixes both flat terrain and floating objects issues by creating a complete pipeline:
Seed → Terrain Parameters → Height Field → Spherical Mesh → Surface-Placed Objects → Direct Viewer

Focus on seed 42 for debugging, then expand to any seed with AAA quality.
"""

import json
import math
import random
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess
import time

class MiniPlanetSystem:
    def __init__(self, debug=False):
        self.debug = debug
        self.seed = None
        self.terrain_params = None
        self.height_field = None
        self.mesh_data = None
        self.objects = None
        
        # Planet parameters
        self.planet_radius = 50.0
        self.mesh_resolution = 64  # Higher for better quality
        self.height_scale = 10.0
        
        if self.debug:
            print("🚀 Unified Mini-Planet System initialized")
    
    def log(self, message: str):
        """Debug logging"""
        if self.debug:
            print(f"  {message}")
    
    def generate_from_seed(self, seed: int) -> str:
        """Complete pipeline: Seed → Terrain → Mesh → Objects → Viewer"""
        self.seed = seed
        self.log(f"🌱 Starting generation with seed {seed}")
        
        # Step 1: Seed → Terrain Parameters
        self.log("Step 1: Generating terrain parameters from seed")
        self.terrain_params = self.seed_to_terrain_params(seed)
        
        # Step 2: Terrain Parameters → Height Field
        self.log("Step 2: Creating height field")
        self.height_field = self.generate_height_field(self.terrain_params)
        
        # Step 3: Height Field → Spherical Mesh Geometry
        self.log("Step 3: Converting to spherical mesh")
        self.mesh_data = self.height_field_to_mesh(self.height_field)
        
        # Step 4: Add Objects ON Surface (not floating)
        self.log("Step 4: Placing objects on terrain surface")
        self.objects = self.place_objects_on_terrain(self.mesh_data, self.terrain_params)
        
        # Step 5: Export to viewer-compatible format
        self.log("Step 5: Exporting scene data")
        output_file = f"unified_planet_seed_{seed}.json"
        self.export_scene(output_file)
        
        self.log(f"✅ Planet generation complete: {output_file}")
        return output_file
    
    def generate_with_debug_steps(self, seed: int, save_steps: bool = False) -> str:
        """Generate with optional step-by-step debugging"""
        output_file = self.generate_from_seed(seed)
        
        if save_steps:
            debug_prefix = f"debug_seed_{seed}"
            self.save_debug_steps(debug_prefix)
        
        return output_file
    
    def seed_to_terrain_params(self, seed: int) -> Dict[str, Any]:
        """Convert seed to deterministic terrain parameters with biome system"""
        rng = random.Random(seed)
        
        # Define biome configurations
        R = self.planet_radius
        biomes = [
            {
                "id": 0,
                "name": "ocean",
                "amplitude": 0.00 * R,
                "frequency": 0.00 / R,
                "octaves": 0,
                "ridged": False,
                "color_rgb": [0.106, 0.239, 0.373]  # #1b3d5f
            },
            {
                "id": 1,
                "name": "plains",
                "amplitude": 0.02 * R,
                "frequency": 0.60 / R,
                "octaves": 4,
                "ridged": False,
                "color_rgb": [0.431, 0.659, 0.306]  # #6ea84e
            },
            {
                "id": 2,
                "name": "desert",
                "amplitude": 0.03 * R,
                "frequency": 0.80 / R,
                "octaves": 5,
                "ridged": False,
                "color_rgb": [0.847, 0.698, 0.416]  # #d8b26a
            },
            {
                "id": 3,
                "name": "mountain",
                "amplitude": 0.10 * R,
                "frequency": 1.40 / R,
                "octaves": 6,
                "ridged": True,
                "color_rgb": [0.557, 0.557, 0.557]  # #8e8e8e
            }
        ]
        
        # Base terrain layers
        params = {
            "seed": seed,
            "radius": self.planet_radius,
            "seeds": {
                "base": seed,
                "mountain": seed + 1000,
                "detail": seed + 2000,
                "warp": seed + 3000,
                "biomeMask": seed + 4000
            },
            "biomes": biomes,
            "domain_warp": {
                "enabled": True,
                "amplitude": 0.1,
                "frequency": 0.5 / R
            },
            "enable_caves": rng.random() < 0.3,  # 30% chance
            "building_density": rng.uniform(0.1, 0.3),  # Buildings per unit area
            "ore_density": rng.uniform(0.05, 0.15)  # Ore deposits per unit area
        }
        
        return params
    
    def generate_height_field(self, params: Dict[str, Any]) -> np.ndarray:
        """Generate height field using biome-driven deformation"""
        resolution = self.mesh_resolution
        height_field = np.zeros((resolution, resolution))
        biome_field = np.zeros((resolution, resolution), dtype=int)
        
        # Generate spherical coordinate grid
        for i in range(resolution):
            for j in range(resolution):
                # Convert grid to spherical coordinates (theta, phi)
                theta = (i / resolution) * math.pi  # 0 to π
                phi = (j / resolution) * 2 * math.pi  # 0 to 2π
                
                # Convert to Cartesian for noise sampling
                x = math.sin(theta) * math.cos(phi)
                y = math.cos(theta)
                z = math.sin(theta) * math.sin(phi)
                
                # Apply domain warping if enabled
                orig_x, orig_y, orig_z = x, y, z
                if params["domain_warp"]["enabled"]:
                    x, y, z = self.domain_warp3d(
                        x, y, z,
                        params["domain_warp"]["amplitude"],
                        params["domain_warp"]["frequency"],
                        params["seeds"]["warp"]
                    )
                
                # Determine biome at this position
                biome_id = self.get_biome_at_position(orig_x, orig_y, orig_z, params)
                biome_field[i, j] = biome_id
                biome = params["biomes"][biome_id]
                
                # Generate height based on biome configuration
                height = 0.0
                
                if biome["octaves"] > 0:
                    if biome["ridged"]:
                        # Use ridged noise for mountains
                        height = self.ridged3d(
                            x, y, z,
                            biome["frequency"],
                            biome["octaves"],
                            params["seeds"]["mountain"]
                        ) * biome["amplitude"]
                    else:
                        # Use FBM for other biomes
                        height = self.fbm3d(
                            x, y, z,
                            biome["frequency"],
                            biome["octaves"],
                            params["seeds"]["base"]
                        ) * biome["amplitude"]
                
                # Add detail noise for all non-ocean biomes
                if biome_id > 0:  # Skip detail for ocean
                    detail_noise = self.fbm3d(
                        x, y, z,
                        biome["frequency"] * 2.0,  # Higher frequency for details
                        max(2, biome["octaves"] - 2),  # Fewer octaves for details
                        params["seeds"]["detail"]
                    )
                    height += detail_noise * (biome["amplitude"] * 0.3)  # 30% detail contribution
                
                height_field[i, j] = height
        
        # Store biome field for later use in mesh generation
        self.biome_field = biome_field
        return height_field
    
    def noise_3d(self, x: float, y: float, z: float, frequency: float, seed: int) -> float:
        """Simple 3D noise function (Perlin-like)"""
        # Scale coordinates by frequency
        x_freq = x * frequency
        y_freq = y * frequency 
        z_freq = z * frequency
        
        # Create deterministic pseudo-random value
        # Use a more complex hash function for better distribution
        hash_val = int(
            (x_freq * 374761393) + 
            (y_freq * 668265263) + 
            (z_freq * 1013904223) + 
            seed
        ) % 2147483647
        
        # Convert to [-1, 1] range
        normalized = (hash_val / 2147483647.0) * 2.0 - 1.0
        
        # Apply smoothing and octaves for more natural noise
        return normalized * 0.8 + math.sin(x_freq * 3.14159) * math.cos(y_freq * 3.14159) * 0.2

    def fbm3d(self, x: float, y: float, z: float, frequency: float, octaves: int, seed: int) -> float:
        """Fractional Brownian Motion (FBM) 3D noise with multiple octaves"""
        value = 0.0
        amplitude = 1.0
        freq = frequency
        max_value = 0.0
        
        for i in range(octaves):
            value += self.noise_3d(x, y, z, freq, seed + i * 100) * amplitude
            max_value += amplitude
            amplitude *= 0.5
            freq *= 2.0
        
        return value / max_value if max_value > 0 else 0.0

    def ridged3d(self, x: float, y: float, z: float, frequency: float, octaves: int, seed: int) -> float:
        """Ridged noise for mountain-like terrain"""
        value = 0.0
        amplitude = 1.0
        freq = frequency
        max_value = 0.0
        
        for i in range(octaves):
            noise_val = abs(self.noise_3d(x, y, z, freq, seed + i * 100))
            noise_val = 1.0 - noise_val  # Invert for ridges
            noise_val = noise_val * noise_val  # Sharpen ridges
            value += noise_val * amplitude
            max_value += amplitude
            amplitude *= 0.5
            freq *= 2.0
        
        return value / max_value if max_value > 0 else 0.0

    def domain_warp3d(self, x: float, y: float, z: float, amplitude: float, frequency: float, seed: int) -> Tuple[float, float, float]:
        """Domain warping for more organic terrain features"""
        offset_x = self.noise_3d(x, y, z, frequency, seed) * amplitude
        offset_y = self.noise_3d(x, y, z, frequency, seed + 100) * amplitude
        offset_z = self.noise_3d(x, y, z, frequency, seed + 200) * amplitude
        
        return x + offset_x, y + offset_y, z + offset_z

    def get_biome_at_position(self, x: float, y: float, z: float, params: Dict[str, Any]) -> int:
        """Determine biome ID at a given position using low-frequency noise"""
        # Use low frequency noise to create biome regions
        biome_noise = self.noise_3d(x, y, z, 0.3, params["seeds"]["biomeMask"])
        
        # Map noise value to biome IDs
        num_biomes = len(params["biomes"])
        biome_index = int((biome_noise + 1) * 0.5 * num_biomes) % num_biomes
        return biome_index
    
    def height_field_to_mesh(self, height_field: np.ndarray) -> Dict[str, Any]:
        """Convert height field to multiple chunks for proper LOD performance"""
        resolution = height_field.shape[0]
        
        # Generate chunks: Start with 9 chunks (3x3 subdivision)
        chunks_per_side = 3
        chunk_resolution = resolution // chunks_per_side
        chunks = []
        
        total_vertices = 0
        total_triangles = 0
        
        self.log(f"Creating {chunks_per_side}x{chunks_per_side} = {chunks_per_side*chunks_per_side} chunks")
        
        for chunk_i in range(chunks_per_side):
            for chunk_j in range(chunks_per_side):
                chunk_id = f"chunk_{chunk_i}_{chunk_j}"
                
                # Calculate chunk bounds in height field
                i_start = chunk_i * chunk_resolution
                i_end = min((chunk_i + 1) * chunk_resolution + 1, resolution)  # +1 for overlap
                j_start = chunk_j * chunk_resolution  
                j_end = min((chunk_j + 1) * chunk_resolution + 1, resolution)  # +1 for overlap
                
                chunk_vertices = []
                chunk_indices = []
                chunk_normals = []
                chunk_colors = []
                biome_ids_in_chunk = set()
                
                # Generate vertices for this chunk
                vertex_map = {}  # Map (i,j) to vertex index in chunk
                vertex_idx = 0
                
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        # Spherical coordinates
                        theta = (i / (resolution - 1)) * math.pi  # 0 to π (latitude)
                        phi = (j / (resolution - 1)) * 2 * math.pi  # 0 to 2π (longitude)
                        
                        # Add small offset to avoid singularities at poles
                        if theta == 0:
                            theta = 0.001
                        elif theta == math.pi:
                            theta = math.pi - 0.001
                        
                        # Height displacement from noise
                        height_offset = height_field[i, j] / self.height_scale
                        radius = self.planet_radius + height_offset
                        
                        # Convert to Cartesian coordinates
                        x = radius * math.sin(theta) * math.cos(phi)
                        z = radius * math.sin(theta) * math.sin(phi)
                        y = radius * math.cos(theta)
                        
                        chunk_vertices.append([float(x), float(y), float(z)])
                        vertex_map[(i, j)] = vertex_idx
                        vertex_idx += 1
                        
                        # Calculate normal (spherical approximation)
                        center_x, center_y, center_z = 0.0, 0.0, 0.0
                        norm_x, norm_y, norm_z = x - center_x, y - center_y, z - center_z
                        length = math.sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z)
                        if length > 0:
                            chunk_normals.append([float(norm_x/length), float(norm_y/length), float(norm_z/length)])
                        else:
                            chunk_normals.append([0.0, 1.0, 0.0])
                        
                        # Get biome color for this vertex
                        if hasattr(self, 'biome_field') and i < self.biome_field.shape[0] and j < self.biome_field.shape[1]:
                            biome_id = int(self.biome_field[i, j])  # Convert numpy int to Python int
                            biome_ids_in_chunk.add(biome_id)
                            biome = self.terrain_params["biomes"][biome_id]
                            chunk_colors.append([float(c) for c in biome["color_rgb"]])
                        else:
                            # Fallback color (plains)
                            chunk_colors.append([0.431, 0.659, 0.306])
                
                # Generate triangle indices for this chunk
                for i in range(i_start, min(i_end - 1, resolution - 1)):
                    for j in range(j_start, min(j_end - 1, resolution - 1)):
                        if (i, j) in vertex_map and (i, j+1) in vertex_map and (i+1, j) in vertex_map and (i+1, j+1) in vertex_map:
                            # Quad made of 2 triangles
                            v0 = vertex_map[(i, j)]
                            v1 = vertex_map[(i, j+1)]
                            v2 = vertex_map[(i+1, j)]
                            v3 = vertex_map[(i+1, j+1)]
                            
                            # First triangle
                            chunk_indices.extend([v0, v1, v2])
                            # Second triangle
                            chunk_indices.extend([v1, v3, v2])
                
                # Calculate chunk bounding box
                if chunk_vertices:
                    vertices_array = np.array(chunk_vertices)
                    min_coords = np.min(vertices_array, axis=0)
                    max_coords = np.max(vertices_array, axis=0)
                    aabb = {
                        "min": min_coords.tolist(),
                        "max": max_coords.tolist()
                    }
                else:
                    aabb = {"min": [0, 0, 0], "max": [0, 0, 0]}
                
                # Determine dominant biome for chunk-level fallback color
                dominant_biome_id = int(max(biome_ids_in_chunk)) if biome_ids_in_chunk else 1
                
                chunk = {
                    "chunk_id": chunk_id,
                    "positions": chunk_vertices,
                    "indices": chunk_indices,
                    "normals": chunk_normals,
                    "vertex_colors": chunk_colors,  # Per-vertex biome colors
                    "biome_id": dominant_biome_id,  # Chunk-level biome for fallback
                    "vertex_count": len(chunk_vertices),
                    "index_count": len(chunk_indices),
                    "triangle_count": len(chunk_indices) // 3,
                    "aabb": aabb,
                    "material": "terrain",
                    "chunk_bounds": {
                        "i_range": [i_start, i_end-1],
                        "j_range": [j_start, j_end-1]
                    }
                }
                
                chunks.append(chunk)
                total_vertices += len(chunk_vertices)
                total_triangles += len(chunk_indices) // 3
                
                if self.debug:
                    self.log(f"  {chunk_id}: {len(chunk_vertices)} vertices, {len(chunk_indices)//3} triangles")
        
        mesh_data = {
            "type": "chunked_spherical_mesh",
            "chunks": chunks,
            "total_vertices": total_vertices,
            "total_triangles": total_triangles,
            "chunks_per_side": chunks_per_side,
            "chunk_resolution": chunk_resolution,
            "radius": self.planet_radius
        }
        
        if self.debug:
            self.log(f"Total: {total_vertices} vertices, {total_triangles} triangles in {len(chunks)} chunks")
        
        return mesh_data
    
    def place_objects_on_terrain(self, mesh_data: Dict[str, Any], terrain_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Place buildings and objects ON the terrain surface (not floating)"""
        objects = []
        
        # Collect all vertices from all chunks
        all_vertices = []
        for chunk in mesh_data["chunks"]:
            all_vertices.extend(chunk["positions"])
        
        vertices = np.array(all_vertices)
        rng = random.Random(self.seed + 5000)
        
        # Calculate surface area for object density
        surface_area = 4 * math.pi * self.planet_radius * self.planet_radius
        
        # Buildings
        num_buildings = max(1, int(terrain_params["building_density"] * surface_area / 100))
        for _ in range(num_buildings):
            # Pick random vertex (on surface)
            vertex_idx = rng.randint(0, len(vertices) - 1)
            surface_pos = vertices[vertex_idx]
            
            # Place building ON surface (not floating)
            building = {
                "type": "building",
                "subtype": "cottage_small",
                "pos": surface_pos.tolist(),
                "material": "wood",
                "on_surface": True,
                "scale": rng.uniform(0.8, 1.2)
            }
            objects.append(building)
        
        # Ore deposits  
        num_ores = max(1, int(terrain_params["ore_density"] * surface_area / 100))
        for _ in range(num_ores):
            vertex_idx = rng.randint(0, len(vertices) - 1)
            surface_pos = vertices[vertex_idx]
            
            ore = {
                "type": "ore",
                "subtype": "crystal",
                "pos": surface_pos.tolist(),
                "material": "crystal",
                "on_surface": True,
                "value": rng.randint(10, 50)
            }
            objects.append(ore)
        
        # Add navigation target
        target_vertex = rng.randint(0, len(vertices) - 1)
        target_pos = vertices[target_vertex]
        target = {
            "type": "target",
            "subtype": "navigation_goal",
            "pos": target_pos.tolist(),
            "material": "reference",
            "on_surface": True
        }
        objects.append(target)
        
        return objects
    
    def save_debug_steps(self, output_prefix: str):
        """Save intermediate data for debugging"""
        if not self.debug:
            return
        
        # Save terrain parameters
        with open(f"{output_prefix}_terrain_params.json", 'w') as f:
            json.dump(self.terrain_params, f, indent=2)
        
        # Save height field as 2D array
        if self.height_field is not None:
            height_data = {
                "height_field": self.height_field.tolist(),
                "resolution": self.height_field.shape[0],
                "min_height": float(np.min(self.height_field)),
                "max_height": float(np.max(self.height_field)),
                "height_scale": self.height_scale
            }
            with open(f"{output_prefix}_height_field.json", 'w') as f:
                json.dump(height_data, f, indent=2)
        
        # Save mesh data
        if self.mesh_data is not None:
            with open(f"{output_prefix}_mesh_data.json", 'w') as f:
                json.dump(self.mesh_data, f, indent=2)
        
        # Save object data
        if self.objects is not None:
            with open(f"{output_prefix}_objects.json", 'w') as f:
                json.dump(self.objects, f, indent=2)
        
        self.log(f"📁 Debug data saved with prefix: {output_prefix}")

    def export_scene(self, output_file: str):
        """Export to viewer-compatible JSON format"""
        # Create chunked planet format for proper mesh rendering
        scene_data = {
            "metadata": {
                "scene_type": "unified_miniplanet",
                "seed": self.seed,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "system": "unified_miniplanet",
                "version": "1.0"
            },
            "terrain_params": self.terrain_params,  # Include biome configuration for reproducibility
            "planet_info": {
                "type": "chunked_quadtree",  # Use chunked format for mesh rendering
                "radius": self.planet_radius,
                "center": [0, 0, 0],
                "vertex_count": self.mesh_data["total_vertices"],
                "triangle_count": self.mesh_data["total_triangles"],
                "chunks_per_side": self.mesh_data["chunks_per_side"]
            },
            "chunks": self.mesh_data["chunks"],
            "objects": self.objects,
            "statistics": {
                "total_vertices": self.mesh_data["total_vertices"],
                "total_triangles": self.mesh_data["total_triangles"],
                "total_chunks": len(self.mesh_data["chunks"]),
                "objects_on_surface": len([obj for obj in self.objects if obj.get("on_surface", False)]),
                "floating_objects": len([obj for obj in self.objects if not obj.get("on_surface", True)])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(scene_data, f, indent=2)
    
    def calculate_aabb(self, vertices: List[List[float]]) -> Dict[str, List[float]]:
        """Calculate axis-aligned bounding box"""
        vertices = np.array(vertices)
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        return {
            "min": min_coords.tolist(),
            "max": max_coords.tolist()
        }
    
    def launch_viewer(self, scene_file: str):
        """Launch the viewer directly"""
        self.log(f"🎮 Launching viewer with {scene_file}")
        try:
            # Use the project's virtual environment python
            venv_python = str(Path.cwd() / ".venv" / "bin" / "python")
            if not Path(venv_python).exists():
                venv_python = sys.executable  # Fallback to current python
            
            subprocess.run([
                venv_python, 
                "renderer/pcc_game_viewer.py", 
                scene_file
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Viewer failed: {e}")
        except FileNotFoundError:
            print(f"❌ Viewer not found: renderer/pcc_game_viewer.py")

def main():
    """Main entry point for unified mini-planet generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Mini-Planet System")
    parser.add_argument("--seed", type=int, default=42, help="Planet generation seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--view", action="store_true", help="Launch viewer after generation")
    parser.add_argument("--save-steps", action="store_true", help="Save intermediate data")
    parser.add_argument("--profile", action="store_true", help="Show performance metrics")
    parser.add_argument("--random-seed", action="store_true", help="Generate random seed for new terrain")
    parser.add_argument("--compare-reference", action="store_true", help="Compare with reference working system")
    
    args = parser.parse_args()
    
    # Generate random seed if requested
    if args.random_seed:
        import random
        args.seed = random.randint(1, 999999)
    
    print("🌍 Unified Mini-Planet System")
    print(f"🌱 Seed: {args.seed}")
    print()
    
    # Create system
    system = MiniPlanetSystem(debug=args.debug)
    
    # Generate planet
    start_time = time.time()
    
    try:
        if args.save_steps:
            output_file = system.generate_with_debug_steps(args.seed, save_steps=True)
        else:
            output_file = system.generate_from_seed(args.seed)
        
        generation_time = time.time() - start_time
        
        # Show results
        print()
        print("🎯 Generation Results:")
        print(f"   ✅ Planet file: {output_file}")
        print(f"   ⏱️ Generation time: {generation_time:.2f}s")
        print(f"   🔺 Triangles: {system.mesh_data['total_triangles']:,}")
        print(f"   🧩 Chunks: {len(system.mesh_data['chunks'])}")
        print(f"   📍 Objects: {len(system.objects)}")
        print(f"   🏠 Buildings: {len([obj for obj in system.objects if obj['type'] == 'building'])}")
        print(f"   💎 Ore deposits: {len([obj for obj in system.objects if obj['type'] == 'ore'])}")
        
        if args.profile:
            print()
            print("📊 Performance Metrics:")
            print(f"   Vertices per second: {system.mesh_data['total_vertices'] / generation_time:.0f}")
            print(f"   Triangles per second: {system.mesh_data['total_triangles'] / generation_time:.0f}")
        
        # Launch viewer if requested
        if args.view:
            print()
            system.launch_viewer(output_file)
    
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())