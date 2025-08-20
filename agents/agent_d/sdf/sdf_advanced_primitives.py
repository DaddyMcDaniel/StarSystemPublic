#!/usr/bin/env python3
"""
Advanced SDF Primitives - T17
=============================

Enhanced SDF implementations for complex cave systems including gyroidal 
surfaces and distributed sphere fields for the hero planet showcase.

Features:
- Gyroidal SDF surfaces with configurable thickness and phase
- Distributed sphere fields with noise-based placement
- Advanced SDF union operations with smooth blending
- Localized masking for cave distribution control
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class GyroidParameters:
    """Parameters for gyroidal SDF generation"""
    frequency: float = 0.01
    thickness: float = 0.15
    amplitude: float = 1.0
    phase_shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    seed: int = 0


@dataclass
class SphereFieldParameters:
    """Parameters for distributed sphere fields"""
    sphere_radius: float = 50.0
    density: float = 0.001
    radius_variation: float = 0.3
    seed: int = 0
    distribution_noise: Dict[str, float] = None


class AdvancedSDFPrimitives:
    """Advanced SDF primitive generators for complex cave systems"""
    
    def __init__(self):
        self.noise_cache = {}
        
    def evaluate_gyroid(self, position: Tuple[float, float, float], 
                       params: GyroidParameters) -> float:
        """
        Evaluate gyroidal SDF at given position
        
        Gyroidal surfaces create complex, interconnected tunnel networks
        similar to biological structures or crystal lattices.
        """
        x, y, z = position
        
        # Apply frequency scaling
        fx = x * params.frequency
        fy = y * params.frequency  
        fz = z * params.frequency
        
        # Apply phase shifts for variation
        fx += params.phase_shift[0]
        fy += params.phase_shift[1]
        fz += params.phase_shift[2]
        
        # Gyroidal surface equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
        gyroid_value = (math.sin(fx) * math.cos(fy) + 
                       math.sin(fy) * math.cos(fz) + 
                       math.sin(fz) * math.cos(fx))
        
        # Scale by amplitude
        gyroid_value *= params.amplitude
        
        # Convert to SDF with thickness
        # Positive inside tunnels, negative outside
        return abs(gyroid_value) - params.thickness
    
    def evaluate_sphere_field(self, position: Tuple[float, float, float],
                             params: SphereFieldParameters) -> float:
        """
        Evaluate distributed sphere field SDF
        
        Creates randomly distributed spherical cavities based on
        noise-driven placement with varying sizes.
        """
        x, y, z = position
        
        # Grid-based sphere placement for performance
        grid_size = 1.0 / math.sqrt(params.density)
        
        # Find grid cell
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)
        grid_z = int(z / grid_size)
        
        min_distance = float('inf')
        
        # Check surrounding grid cells (3x3x3 neighborhood)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cell_x = grid_x + dx
                    cell_y = grid_y + dy
                    cell_z = grid_z + dz
                    
                    # Generate sphere in this cell
                    sphere_distance = self._evaluate_grid_sphere(
                        position, (cell_x, cell_y, cell_z), grid_size, params
                    )
                    
                    min_distance = min(min_distance, sphere_distance)
        
        return min_distance
    
    def _evaluate_grid_sphere(self, position: Tuple[float, float, float],
                             grid_cell: Tuple[int, int, int], grid_size: float,
                             params: SphereFieldParameters) -> float:
        """Evaluate sphere in specific grid cell"""
        x, y, z = position
        gx, gy, gz = grid_cell
        
        # Pseudo-random sphere center within grid cell
        seed_x = self._hash_3d(gx, gy, gz, params.seed)
        seed_y = self._hash_3d(gx, gy, gz, params.seed + 1)
        seed_z = self._hash_3d(gx, gy, gz, params.seed + 2)
        
        # Cell center
        cell_center_x = gx * grid_size + grid_size * 0.5
        cell_center_y = gy * grid_size + grid_size * 0.5
        cell_center_z = gz * grid_size + grid_size * 0.5
        
        # Random offset within cell
        offset_range = grid_size * 0.4
        sphere_x = cell_center_x + (seed_x - 0.5) * offset_range
        sphere_y = cell_center_y + (seed_y - 0.5) * offset_range
        sphere_z = cell_center_z + (seed_z - 0.5) * offset_range
        
        # Check if sphere should exist based on distribution noise
        if params.distribution_noise:
            noise_freq = params.distribution_noise.get('frequency', 0.01)
            noise_threshold = params.distribution_noise.get('threshold', 0.5)
            
            noise_value = self._simple_noise_3d(
                sphere_x * noise_freq, 
                sphere_y * noise_freq, 
                sphere_z * noise_freq,
                params.seed + 100
            )
            
            if noise_value < noise_threshold:
                return float('inf')  # No sphere in this cell
        
        # Random sphere radius with variation
        radius_seed = self._hash_3d(gx, gy, gz, params.seed + 3)
        radius_variation = 1.0 + (radius_seed - 0.5) * params.radius_variation
        sphere_radius = params.sphere_radius * radius_variation
        
        # Distance to sphere center
        dx = x - sphere_x
        dy = y - sphere_y
        dz = z - sphere_z
        distance_to_center = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # SDF: positive inside sphere, negative outside
        return sphere_radius - distance_to_center
    
    def sdf_smooth_union(self, sdf1: float, sdf2: float, blend_radius: float) -> float:
        """Smooth union of two SDFs"""
        if blend_radius <= 0:
            return max(sdf1, sdf2)
        
        h = max(blend_radius - abs(sdf1 - sdf2), 0.0) / blend_radius
        return max(sdf1, sdf2) + h * h * h * blend_radius / 6.0
    
    def sdf_intersection(self, sdf1: float, sdf2: float) -> float:
        """Intersection of two SDFs"""
        return min(sdf1, sdf2)
    
    def sdf_subtraction(self, sdf1: float, sdf2: float) -> float:
        """Subtract sdf2 from sdf1"""
        return min(sdf1, -sdf2)
    
    def apply_localized_mask(self, sdf_value: float, position: Tuple[float, float, float],
                           mask_params: Dict[str, Any]) -> float:
        """Apply localized masking to SDF based on position and noise"""
        x, y, z = position
        
        # Mask strength based on parameters
        mask_strength = mask_params.get('strength', 1.0)
        mask_frequency = mask_params.get('frequency', 0.001)
        mask_threshold = mask_params.get('threshold', 0.5)
        mask_seed = mask_params.get('seed', 0)
        
        # Generate mask noise
        mask_noise = self._simple_noise_3d(
            x * mask_frequency,
            y * mask_frequency, 
            z * mask_frequency,
            mask_seed
        )
        
        # Apply threshold
        if mask_noise < mask_threshold:
            return -abs(sdf_value)  # Force solid
        
        # Smooth falloff near threshold
        falloff = (mask_noise - mask_threshold) / (1.0 - mask_threshold)
        falloff = max(0.0, min(1.0, falloff))
        
        return sdf_value * falloff * mask_strength
    
    def _hash_3d(self, x: int, y: int, z: int, seed: int) -> float:
        """3D hash function for pseudo-random values"""
        n = (x * 1619 + y * 31337 + z * 6971 + seed * 1013) & 0x7fffffff
        n = (n ^ (n >> 13)) * 1274126097
        n = n ^ (n >> 16)
        return (n & 0x7fffffff) / 0x7fffffff
    
    def _simple_noise_3d(self, x: float, y: float, z: float, seed: int) -> float:
        """Simple 3D noise function"""
        # Integer coordinates
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        iz = int(math.floor(z))
        
        # Fractional coordinates
        fx = x - ix
        fy = y - iy
        fz = z - iz
        
        # Smooth interpolation weights
        wx = fx * fx * (3.0 - 2.0 * fx)
        wy = fy * fy * (3.0 - 2.0 * fy)
        wz = fz * fz * (3.0 - 2.0 * fz)
        
        # Trilinear interpolation
        c000 = self._hash_3d(ix, iy, iz, seed)
        c001 = self._hash_3d(ix, iy, iz + 1, seed)
        c010 = self._hash_3d(ix, iy + 1, iz, seed)
        c011 = self._hash_3d(ix, iy + 1, iz + 1, seed)
        c100 = self._hash_3d(ix + 1, iy, iz, seed)
        c101 = self._hash_3d(ix + 1, iy, iz + 1, seed)
        c110 = self._hash_3d(ix + 1, iy + 1, iz, seed)
        c111 = self._hash_3d(ix + 1, iy + 1, iz + 1, seed)
        
        # Interpolate
        c00 = c000 * (1 - wx) + c100 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        
        return c0 * (1 - wz) + c1 * wz


class HeroCaveSystem:
    """Complete cave system implementation for hero planet"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sdf_primitives = AdvancedSDFPrimitives()
        
        # Parse configuration
        self.gyroid_primary = GyroidParameters(**config.get('gyroid_primary', {}))
        self.gyroid_secondary = GyroidParameters(**config.get('gyroid_secondary', {}))
        self.sphere_large = SphereFieldParameters(**config.get('sphere_large', {}))
        self.sphere_small = SphereFieldParameters(**config.get('sphere_small', {}))
        self.cave_mask = config.get('cave_mask', {})
        
    def evaluate_cave_system(self, position: Tuple[float, float, float]) -> float:
        """Evaluate complete cave system SDF at position"""
        
        # Primary gyroidal caves
        gyroid1 = self.sdf_primitives.evaluate_gyroid(position, self.gyroid_primary)
        
        # Secondary gyroidal caves  
        gyroid2 = self.sdf_primitives.evaluate_gyroid(position, self.gyroid_secondary)
        
        # Large sphere cavities
        spheres_large = self.sdf_primitives.evaluate_sphere_field(position, self.sphere_large)
        
        # Small sphere cavities
        spheres_small = self.sdf_primitives.evaluate_sphere_field(position, self.sphere_small)
        
        # Combine gyroidal systems
        combined_gyroids = self.sdf_primitives.sdf_smooth_union(
            gyroid1, gyroid2, 20.0
        )
        
        # Combine sphere systems
        combined_spheres = self.sdf_primitives.sdf_smooth_union(
            spheres_large, spheres_small, 15.0
        )
        
        # Unite all cave systems
        cave_sdf = self.sdf_primitives.sdf_smooth_union(
            combined_gyroids, combined_spheres, 25.0
        )
        
        # Apply localized masking
        if self.cave_mask:
            cave_sdf = self.sdf_primitives.apply_localized_mask(
                cave_sdf, position, self.cave_mask
            )
        
        return cave_sdf
    
    def generate_cave_mesh_chunk(self, chunk_bounds: Tuple[Tuple[float, float, float], 
                                                         Tuple[float, float, float]],
                                resolution: int = 32) -> Dict[str, Any]:
        """Generate cave mesh for specific chunk"""
        min_bounds, max_bounds = chunk_bounds
        
        # Create sample grid
        step_x = (max_bounds[0] - min_bounds[0]) / resolution
        step_y = (max_bounds[1] - min_bounds[1]) / resolution
        step_z = (max_bounds[2] - min_bounds[2]) / resolution
        
        sdf_values = []
        
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                for k in range(resolution + 1):
                    x = min_bounds[0] + i * step_x
                    y = min_bounds[1] + j * step_y
                    z = min_bounds[2] + k * step_z
                    
                    sdf_value = self.evaluate_cave_system((x, y, z))
                    sdf_values.append(sdf_value)
        
        # Would integrate with marching cubes here
        return {
            'sdf_values': sdf_values,
            'resolution': resolution,
            'bounds': chunk_bounds,
            'has_caves': any(v > 0 for v in sdf_values)
        }


if __name__ == "__main__":
    # Test advanced SDF primitives
    print("🚀 T17 Advanced SDF Primitives")
    print("=" * 60)
    
    # Test gyroidal surface
    gyroid_params = GyroidParameters(
        frequency=0.012,
        thickness=0.15,
        phase_shift=(0.0, 0.5, 1.0),
        seed=30000001
    )
    
    sdf = AdvancedSDFPrimitives()
    
    # Sample gyroid at test positions
    test_positions = [
        (0.0, 0.0, 0.0),
        (50.0, 50.0, 50.0),
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 100.0)
    ]
    
    print("📊 Testing gyroidal SDF...")
    for pos in test_positions:
        gyroid_value = sdf.evaluate_gyroid(pos, gyroid_params)
        inside_tunnel = gyroid_value > 0
        print(f"   Position {pos}: SDF={gyroid_value:.3f}, Inside={'Yes' if inside_tunnel else 'No'}")
    
    # Test sphere field
    sphere_params = SphereFieldParameters(
        sphere_radius=80.0,
        density=0.0002,
        radius_variation=0.4,
        seed=30000003,
        distribution_noise={'frequency': 0.005, 'threshold': 0.6}
    )
    
    print(f"\n📊 Testing sphere field SDF...")
    for pos in test_positions:
        sphere_value = sdf.evaluate_sphere_field(pos, sphere_params)
        inside_sphere = sphere_value > 0
        print(f"   Position {pos}: SDF={sphere_value:.3f}, Inside={'Yes' if inside_sphere else 'No'}")
    
    # Test complete cave system
    cave_config = {
        'gyroid_primary': {
            'frequency': 0.012,
            'thickness': 0.15,
            'seed': 30000001
        },
        'gyroid_secondary': {
            'frequency': 0.008,
            'thickness': 0.25,
            'seed': 30000002
        },
        'sphere_large': {
            'sphere_radius': 80.0,
            'density': 0.0002,
            'seed': 30000003
        },
        'sphere_small': {
            'sphere_radius': 25.0,
            'density': 0.001,
            'seed': 30000004
        },
        'cave_mask': {
            'frequency': 0.001,
            'threshold': 0.3,
            'strength': 0.7,
            'seed': 30000005
        }
    }
    
    cave_system = HeroCaveSystem(cave_config)
    
    print(f"\n🔧 Testing complete cave system...")
    for pos in test_positions:
        cave_value = cave_system.evaluate_cave_system(pos)
        inside_cave = cave_value > 0
        print(f"   Position {pos}: SDF={cave_value:.3f}, Inside={'Yes' if inside_cave else 'No'}")
    
    # Test chunk generation
    chunk_bounds = ((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0))
    chunk_data = cave_system.generate_cave_mesh_chunk(chunk_bounds, resolution=16)
    
    print(f"\n📦 Cave chunk generation:")
    print(f"   Resolution: {chunk_data['resolution']}")
    print(f"   Has caves: {chunk_data['has_caves']}")
    print(f"   SDF samples: {len(chunk_data['sdf_values'])}")
    
    cave_samples = [v for v in chunk_data['sdf_values'] if v > 0]
    print(f"   Cave volume samples: {len(cave_samples)}/{len(chunk_data['sdf_values'])}")
    
    print(f"\n✅ Advanced SDF primitives functional")
    print(f"   Gyroidal caves: Complex tunnel networks")
    print(f"   Sphere fields: Distributed cavities")
    print(f"   Smooth unions: Seamless cave connections")
    print(f"   Localized masking: Controlled cave distribution")