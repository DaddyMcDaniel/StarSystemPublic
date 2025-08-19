#!/usr/bin/env python3
"""
SDF Evaluator and Voxelization System - T09
============================================

Implements SDF evaluation with PCC composition and voxel grid sampling for
chunk-local cave and overhang generation.

Features:
- EvaluateSDF(p3) function with PCC-style composition
- VoxelGrid sampler for deterministic chunk-local sampling
- Efficient batch evaluation for voxel grids
- Integration with T08 chunk system

Usage:
    from sdf_evaluator import SDFEvaluator, VoxelGrid
    
    evaluator = SDFEvaluator(sdf_tree, seed=42)
    voxel_grid = VoxelGrid(chunk_bounds, resolution=32)
    scalar_field = evaluator.sample_voxel_grid(voxel_grid)
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
import os

# Import SDF primitives
sys.path.append(os.path.dirname(__file__))
from sdf_primitives import *


@dataclass
class ChunkBounds:
    """3D bounds for a chunk in world space"""
    min_point: np.ndarray  # [x, y, z] minimum corner
    max_point: np.ndarray  # [x, y, z] maximum corner
    
    @property
    def center(self) -> np.ndarray:
        """Get center point of bounds"""
        return (self.min_point + self.max_point) * 0.5
    
    @property
    def size(self) -> np.ndarray:
        """Get size of bounds"""
        return self.max_point - self.min_point
    
    @property
    def diagonal(self) -> float:
        """Get diagonal length of bounds"""
        return np.linalg.norm(self.size)


class VoxelGrid:
    """3D voxel grid for SDF sampling within chunk bounds"""
    
    def __init__(self, chunk_bounds: ChunkBounds, resolution: int = 32):
        """
        Initialize voxel grid
        
        Args:
            chunk_bounds: 3D bounds to sample within
            resolution: Number of voxels per axis (resolution^3 total voxels)
        """
        self.chunk_bounds = chunk_bounds
        self.resolution = resolution
        self.total_voxels = resolution ** 3
        
        # Compute voxel spacing
        self.voxel_size = chunk_bounds.size / resolution
        self.min_voxel_size = np.min(self.voxel_size)
        
        # Pre-compute voxel positions
        self._generate_voxel_positions()
    
    def _generate_voxel_positions(self):
        """Generate 3D positions for all voxels"""
        # Create coordinate arrays
        x_coords = np.linspace(self.chunk_bounds.min_point[0], 
                              self.chunk_bounds.max_point[0], self.resolution)
        y_coords = np.linspace(self.chunk_bounds.min_point[1], 
                              self.chunk_bounds.max_point[1], self.resolution)
        z_coords = np.linspace(self.chunk_bounds.min_point[2], 
                              self.chunk_bounds.max_point[2], self.resolution)
        
        # Create 3D mesh grid
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Flatten to list of points
        self.voxel_positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # Store grid shape for reconstruction
        self.grid_shape = (self.resolution, self.resolution, self.resolution)
    
    def get_voxel_position(self, i: int, j: int, k: int) -> np.ndarray:
        """Get world position of voxel at grid coordinates"""
        index = i * self.resolution * self.resolution + j * self.resolution + k
        return self.voxel_positions[index]
    
    def world_to_voxel(self, world_pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world position to voxel grid coordinates"""
        relative_pos = world_pos - self.chunk_bounds.min_point
        voxel_coords = relative_pos / self.voxel_size
        
        # Clamp to valid range
        i = np.clip(int(voxel_coords[0]), 0, self.resolution - 1)
        j = np.clip(int(voxel_coords[1]), 0, self.resolution - 1)
        k = np.clip(int(voxel_coords[2]), 0, self.resolution - 1)
        
        return i, j, k
    
    def reshape_scalar_field(self, scalar_values: np.ndarray) -> np.ndarray:
        """Reshape flat scalar field to 3D grid"""
        return scalar_values.reshape(self.grid_shape)


class SDFEvaluator:
    """
    SDF evaluation system with PCC-style composition
    """
    
    def __init__(self, seed: int = 42):
        """Initialize SDF evaluator with deterministic seed"""
        self.seed = seed
        self.noise = SeededNoise(seed)
        self.sdf_tree = None
    
    def build_sdf_from_pcc(self, pcc_specification: Dict) -> SDFNode:
        """
        Build SDF tree from PCC-style specification
        
        Args:
            pcc_specification: Dictionary describing SDF composition
            
        Returns:
            Root SDF node of the composed tree
        """
        return self._parse_sdf_node(pcc_specification)
    
    def _parse_sdf_node(self, spec: Dict) -> SDFNode:
        """Recursively parse SDF node specification"""
        node_type = spec.get("type", "sphere")
        seed = spec.get("seed", self.seed)
        
        if node_type == "sphere":
            return SDFSphere(
                center=spec.get("center", [0, 0, 0]),
                radius=spec.get("radius", 1.0),
                seed=seed
            )
        
        elif node_type == "capsule":
            return SDFCapsule(
                point_a=spec.get("point_a", [0, -1, 0]),
                point_b=spec.get("point_b", [0, 1, 0]),
                radius=spec.get("radius", 0.5),
                seed=seed
            )
        
        elif node_type == "box":
            return SDFBox(
                center=spec.get("center", [0, 0, 0]),
                size=spec.get("size", [1, 1, 1]),
                seed=seed
            )
        
        elif node_type == "gyroid":
            return SDFGyroid(
                scale=spec.get("scale", 1.0),
                thickness=spec.get("thickness", 0.1),
                offset=spec.get("offset", 0.0),
                seed=seed
            )
        
        elif node_type == "torus":
            return SDFTorus(
                center=spec.get("center", [0, 0, 0]),
                major_radius=spec.get("major_radius", 2.0),
                minor_radius=spec.get("minor_radius", 0.5),
                seed=seed
            )
        
        elif node_type == "noise_displace":
            base_sdf = self._parse_sdf_node(spec.get("base", {"type": "sphere"}))
            return SDFNoiseDisplace(
                base_sdf=base_sdf,
                displacement_scale=spec.get("displacement_scale", 0.1),
                noise_frequency=spec.get("noise_frequency", 1.0),
                octaves=spec.get("octaves", 4),
                seed=seed
            )
        
        elif node_type == "transform":
            base_sdf = self._parse_sdf_node(spec.get("base", {"type": "sphere"}))
            return SDFTransform(
                base_sdf=base_sdf,
                translation=spec.get("translation", [0, 0, 0]),
                rotation=spec.get("rotation", [0, 0, 0]),
                scale=spec.get("scale", [1, 1, 1]),
                seed=seed
            )
        
        elif node_type == "union":
            sdf_a = self._parse_sdf_node(spec.get("sdf_a", {"type": "sphere"}))
            sdf_b = self._parse_sdf_node(spec.get("sdf_b", {"type": "sphere"}))
            
            if spec.get("smooth", False):
                return SDFSmoothUnion(
                    sdf_a=sdf_a, sdf_b=sdf_b,
                    blend_radius=spec.get("blend_radius", 0.1),
                    seed=seed
                )
            else:
                return SDFUnion(sdf_a=sdf_a, sdf_b=sdf_b, seed=seed)
        
        elif node_type == "subtract":
            sdf_a = self._parse_sdf_node(spec.get("sdf_a", {"type": "sphere"}))
            sdf_b = self._parse_sdf_node(spec.get("sdf_b", {"type": "sphere"}))
            
            if spec.get("smooth", False):
                return SDFSmoothSubtract(
                    sdf_a=sdf_a, sdf_b=sdf_b,
                    blend_radius=spec.get("blend_radius", 0.1),
                    seed=seed
                )
            else:
                return SDFSubtract(sdf_a=sdf_a, sdf_b=sdf_b, seed=seed)
        
        elif node_type == "intersect":
            sdf_a = self._parse_sdf_node(spec.get("sdf_a", {"type": "sphere"}))
            sdf_b = self._parse_sdf_node(spec.get("sdf_b", {"type": "sphere"}))
            return SDFIntersect(sdf_a=sdf_a, sdf_b=sdf_b, seed=seed)
        
        else:
            # Default to sphere for unknown types
            return SDFSphere(seed=seed)
    
    def evaluate_sdf(self, point: np.ndarray, sdf_tree: Optional[SDFNode] = None) -> float:
        """
        Evaluate SDF at given 3D point
        
        Args:
            point: 3D world position [x, y, z]
            sdf_tree: SDF tree to evaluate (uses self.sdf_tree if None)
            
        Returns:
            Signed distance value (negative = inside, positive = outside)
        """
        if sdf_tree is None:
            sdf_tree = self.sdf_tree
        
        if sdf_tree is None:
            return float('inf')  # No SDF defined
        
        return sdf_tree.evaluate(point)
    
    def sample_voxel_grid(self, voxel_grid: VoxelGrid, sdf_tree: Optional[SDFNode] = None) -> np.ndarray:
        """
        Sample SDF over entire voxel grid
        
        Args:
            voxel_grid: VoxelGrid to sample
            sdf_tree: SDF tree to evaluate (uses self.sdf_tree if None)
            
        Returns:
            Flat array of scalar values (length = resolution^3)
        """
        if sdf_tree is None:
            sdf_tree = self.sdf_tree
        
        if sdf_tree is None:
            return np.full(voxel_grid.total_voxels, float('inf'))
        
        # Batch evaluate all voxel positions
        return sdf_tree.evaluate_batch(voxel_grid.voxel_positions)
    
    def sample_voxel_grid_3d(self, voxel_grid: VoxelGrid, sdf_tree: Optional[SDFNode] = None) -> np.ndarray:
        """
        Sample SDF over voxel grid and return as 3D array
        
        Args:
            voxel_grid: VoxelGrid to sample
            sdf_tree: SDF tree to evaluate
            
        Returns:
            3D array of scalar values (shape = resolution x resolution x resolution)
        """
        scalar_field = self.sample_voxel_grid(voxel_grid, sdf_tree)
        return voxel_grid.reshape_scalar_field(scalar_field)
    
    def extract_isosurface_points(self, voxel_grid: VoxelGrid, scalar_field: np.ndarray, 
                                 iso_value: float = 0.0) -> List[np.ndarray]:
        """
        Extract points near the isosurface from voxel grid
        
        Args:
            voxel_grid: VoxelGrid that was sampled
            scalar_field: Scalar field values from sampling
            iso_value: Isosurface value (typically 0.0 for SDF surface)
            
        Returns:
            List of 3D points near the isosurface
        """
        surface_points = []
        threshold = voxel_grid.min_voxel_size  # Use voxel size as threshold
        
        for i, distance in enumerate(scalar_field):
            if abs(distance - iso_value) < threshold:
                surface_points.append(voxel_grid.voxel_positions[i])
        
        return surface_points
    
    def compute_sdf_gradient_field(self, voxel_grid: VoxelGrid, sdf_tree: Optional[SDFNode] = None,
                                  epsilon: float = None) -> np.ndarray:
        """
        Compute gradient field for SDF over voxel grid
        
        Args:
            voxel_grid: VoxelGrid to sample
            sdf_tree: SDF tree to evaluate
            epsilon: Finite difference step size (defaults to voxel size)
            
        Returns:
            Array of gradient vectors (shape = num_voxels x 3)
        """
        if sdf_tree is None:
            sdf_tree = self.sdf_tree
        
        if sdf_tree is None:
            return np.zeros((voxel_grid.total_voxels, 3))
        
        if epsilon is None:
            epsilon = voxel_grid.min_voxel_size * 0.5
        
        gradients = np.zeros((voxel_grid.total_voxels, 3))
        
        for i, position in enumerate(voxel_grid.voxel_positions):
            gradients[i] = sdf_gradient(sdf_tree, position, epsilon)
        
        return gradients


def create_cave_system_sdf(bounds: ChunkBounds, seed: int = 42) -> Dict:
    """
    Create a procedural cave system SDF specification
    
    Args:
        bounds: Chunk bounds for scaling
        seed: Deterministic seed
        
    Returns:
        PCC-style SDF specification for cave system
    """
    center = bounds.center
    size = bounds.size
    max_size = np.max(size)
    
    # Base cave structure using gyroid
    cave_spec = {
        "type": "subtract",
        "seed": seed,
        "sdf_a": {
            "type": "box",
            "center": center.tolist(),
            "size": (size * 2.0).tolist(),  # Large containing volume
            "seed": seed
        },
        "sdf_b": {
            "type": "noise_displace",
            "seed": seed + 1,
            "displacement_scale": max_size * 0.05,
            "noise_frequency": 2.0 / max_size,
            "octaves": 3,
            "base": {
                "type": "gyroid",
                "scale": 4.0 / max_size,
                "thickness": max_size * 0.1,
                "offset": 0.2,
                "seed": seed + 2
            }
        }
    }
    
    return cave_spec


def create_overhang_sdf(bounds: ChunkBounds, seed: int = 42) -> Dict:
    """
    Create an overhang structure SDF specification
    
    Args:
        bounds: Chunk bounds for scaling
        seed: Deterministic seed
        
    Returns:
        PCC-style SDF specification for overhang
    """
    center = bounds.center
    size = bounds.size
    
    # Overhang using intersected and displaced primitives
    overhang_spec = {
        "type": "subtract",
        "seed": seed,
        "sdf_a": {
            "type": "box",
            "center": center.tolist(),
            "size": size.tolist(),
            "seed": seed
        },
        "sdf_b": {
            "type": "union",
            "seed": seed + 1,
            "sdf_a": {
                "type": "noise_displace",
                "seed": seed + 2,
                "displacement_scale": 0.2,
                "noise_frequency": 1.0,
                "base": {
                    "type": "sphere",
                    "center": [center[0], center[1] + size[1] * 0.3, center[2]],
                    "radius": size[0] * 0.6,
                    "seed": seed + 3
                }
            },
            "sdf_b": {
                "type": "capsule",
                "point_a": [center[0], center[1] - size[1] * 0.4, center[2]],
                "point_b": [center[0], center[1] + size[1] * 0.4, center[2]],
                "radius": size[0] * 0.3,
                "seed": seed + 4
            }
        }
    }
    
    return overhang_spec


if __name__ == "__main__":
    # Example usage and testing
    print("🚀 T09 SDF Evaluator and Voxelization System")
    print("=" * 60)
    
    # Create test bounds
    bounds = ChunkBounds(
        min_point=np.array([-2.0, -2.0, -2.0]),
        max_point=np.array([2.0, 2.0, 2.0])
    )
    
    print(f"Test bounds: {bounds.min_point} to {bounds.max_point}")
    print(f"Bounds size: {bounds.size}")
    print(f"Bounds center: {bounds.center}")
    
    # Create voxel grid
    voxel_grid = VoxelGrid(bounds, resolution=16)
    print(f"\nVoxel grid: {voxel_grid.resolution}^3 = {voxel_grid.total_voxels} voxels")
    print(f"Voxel size: {voxel_grid.voxel_size}")
    
    # Test SDF evaluator
    evaluator = SDFEvaluator(seed=42)
    
    # Create cave system
    cave_spec = create_cave_system_sdf(bounds, seed=42)
    cave_sdf = evaluator.build_sdf_from_pcc(cave_spec)
    
    print(f"\nCave SDF: {cave_sdf}")
    
    # Sample voxel grid
    print("\n🔧 Sampling voxel grid...")
    scalar_field = evaluator.sample_voxel_grid(voxel_grid, cave_sdf)
    scalar_field_3d = voxel_grid.reshape_scalar_field(scalar_field)
    
    print(f"Scalar field shape: {scalar_field_3d.shape}")
    print(f"Distance range: {np.min(scalar_field):.3f} to {np.max(scalar_field):.3f}")
    
    # Find surface points
    surface_points = evaluator.extract_isosurface_points(voxel_grid, scalar_field, iso_value=0.0)
    print(f"Surface points found: {len(surface_points)}")
    
    print("\n✅ SDF evaluation and voxelization system functional")