#!/usr/bin/env python3
"""
Marching Cubes Polygonization - T10
===================================

Implements Marching Cubes algorithm to convert SDF voxel grids to triangle meshes
for cave and overhang geometry generation.

Features:
- Complete Marching Cubes lookup tables (256 cases)
- Edge interpolation for smooth surfaces
- Normal generation from SDF gradients
- Per-chunk mesh generation
- Material ID support

Usage:
    from marching_cubes import MarchingCubes
    
    mc = MarchingCubes()
    vertices, triangles, normals = mc.polygonize(voxel_grid, scalar_field)
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import sys
import os

# Import SDF system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))
from sdf_evaluator import VoxelGrid, ChunkBounds, SDFEvaluator
from sdf_primitives import SDFNode, sdf_gradient


@dataclass
class MarchingCubesVertex:
    """Vertex data for marching cubes mesh"""
    position: np.ndarray  # [x, y, z] world position
    normal: np.ndarray    # [x, y, z] surface normal
    material_id: int = 0  # Material ID for rendering


@dataclass
class CaveMeshData:
    """Complete mesh data for a cave chunk"""
    vertices: List[MarchingCubesVertex]
    triangles: np.ndarray  # Triangle indices (Nx3)
    chunk_id: str
    material_id: int
    bounds: ChunkBounds


class MarchingCubes:
    """
    Marching Cubes implementation for SDF polygonization
    """
    
    def __init__(self, iso_value: float = 0.0, use_sdf_normals: bool = True):
        """
        Initialize Marching Cubes with T20 improvements
        
        Args:
            iso_value: Surface isosurface value (0.0 for SDF surface)
            use_sdf_normals: Use SDF gradient for normals (T20 improvement)
        """
        self.iso_value = iso_value
        self.use_sdf_normals = use_sdf_normals
        self._initialize_lookup_tables()
    
    def _initialize_lookup_tables(self):
        """Initialize Marching Cubes lookup tables"""
        from .mc_tables import EDGE_TABLE, COMPLETE_TRIANGLE_TABLE, EDGE_VERTICES, CUBE_VERTICES
        
        # Use complete lookup tables
        self.edge_table = EDGE_TABLE
        self.triangle_table = COMPLETE_TRIANGLE_TABLE
        self.cube_vertices = CUBE_VERTICES
        self.edge_connections = EDGE_VERTICES
    
    
    def polygonize(self, voxel_grid: VoxelGrid, scalar_field: np.ndarray, 
                  sdf_tree: Optional[SDFNode] = None) -> Tuple[List[MarchingCubesVertex], np.ndarray]:
        """
        Convert SDF voxel grid to triangle mesh using Marching Cubes
        
        Args:
            voxel_grid: VoxelGrid defining the sampling space
            scalar_field: Scalar field values (resolution^3 array)
            sdf_tree: SDF tree for gradient computation (optional)
            
        Returns:
            Tuple of (vertices, triangles) where:
            - vertices: List of MarchingCubesVertex objects
            - triangles: Nx3 array of triangle indices
        """
        vertices = []
        triangles = []
        
        # Reshape scalar field to 3D
        scalar_3d = voxel_grid.reshape_scalar_field(scalar_field)
        resolution = voxel_grid.resolution
        
        vertex_cache = {}  # Cache vertices to avoid duplicates
        vertex_index = 0
        
        # Process each cube in the voxel grid
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                for k in range(resolution - 1):
                    # Get scalar values at 8 cube corners
                    cube_values = np.array([
                        scalar_3d[i, j, k],         # 0
                        scalar_3d[i+1, j, k],       # 1
                        scalar_3d[i+1, j+1, k],     # 2
                        scalar_3d[i, j+1, k],       # 3
                        scalar_3d[i, j, k+1],       # 4
                        scalar_3d[i+1, j, k+1],     # 5
                        scalar_3d[i+1, j+1, k+1],   # 6
                        scalar_3d[i, j+1, k+1]      # 7
                    ])
                    
                    # Get world positions of cube corners
                    cube_positions = np.array([
                        voxel_grid.get_voxel_position(i, j, k),
                        voxel_grid.get_voxel_position(i+1, j, k),
                        voxel_grid.get_voxel_position(i+1, j+1, k),
                        voxel_grid.get_voxel_position(i, j+1, k),
                        voxel_grid.get_voxel_position(i, j, k+1),
                        voxel_grid.get_voxel_position(i+1, j, k+1),
                        voxel_grid.get_voxel_position(i+1, j+1, k+1),
                        voxel_grid.get_voxel_position(i, j+1, k+1)
                    ])
                    
                    # Compute cube configuration index
                    cube_index = 0
                    for vertex_idx in range(8):
                        if cube_values[vertex_idx] < self.iso_value:
                            cube_index |= (1 << vertex_idx)
                    
                    # Skip if cube is completely inside or outside
                    if cube_index == 0 or cube_index == 255:
                        continue
                    
                    # Get intersected edges
                    edge_mask = self.edge_table[cube_index]
                    if edge_mask == 0:
                        continue
                    
                    # Find intersection points on edges
                    edge_vertices = {}
                    for edge_idx in range(12):
                        if edge_mask & (1 << edge_idx):
                            # Get edge endpoints
                            v1_idx, v2_idx = self.edge_connections[edge_idx]
                            v1_pos = cube_positions[v1_idx]
                            v2_pos = cube_positions[v2_idx]
                            v1_val = cube_values[v1_idx]
                            v2_val = cube_values[v2_idx]
                            
                            # Interpolate intersection point
                            intersection_pos = self._interpolate_edge(
                                v1_pos, v2_pos, v1_val, v2_val, self.iso_value
                            )
                            
                            # T20: Compute normal from SDF gradient with improved quality
                            normal = self._compute_normal_improved(intersection_pos, sdf_tree, voxel_grid, scalar_3d)
                            
                            # Create vertex
                            vertex = MarchingCubesVertex(
                                position=intersection_pos,
                                normal=normal,
                                material_id=0  # Default material
                            )
                            
                            edge_vertices[edge_idx] = vertex_index
                            vertices.append(vertex)
                            vertex_index += 1
                    
                    # Generate triangles using triangle table
                    triangle_config = self.triangle_table[cube_index]
                    tri_idx = 0
                    while tri_idx < len(triangle_config) and triangle_config[tri_idx] != -1:
                        if tri_idx + 2 < len(triangle_config) and triangle_config[tri_idx + 2] != -1:
                            # Create triangle
                            triangle = np.array([
                                edge_vertices[triangle_config[tri_idx]],
                                edge_vertices[triangle_config[tri_idx + 1]],
                                edge_vertices[triangle_config[tri_idx + 2]]
                            ])
                            triangles.append(triangle)
                        tri_idx += 3
        
        # Convert triangles to numpy array
        triangles_array = np.array(triangles) if triangles else np.empty((0, 3), dtype=np.int32)
        
        # T20: Post-process to remove duplicate vertices in overlap regions
        if len(vertices) > 0:
            vertices, triangles_array = self._remove_duplicate_vertices(vertices, triangles_array)
        
        return vertices, triangles_array
    
    def _interpolate_edge(self, v1_pos: np.ndarray, v2_pos: np.ndarray, 
                         v1_val: float, v2_val: float, iso_value: float) -> np.ndarray:
        """
        Interpolate intersection point on edge using linear interpolation
        
        Args:
            v1_pos, v2_pos: Endpoint positions
            v1_val, v2_val: Scalar values at endpoints
            iso_value: Target isosurface value
            
        Returns:
            Interpolated intersection position
        """
        if abs(iso_value - v1_val) < 1e-8:
            return v1_pos
        if abs(iso_value - v2_val) < 1e-8:
            return v2_pos
        if abs(v1_val - v2_val) < 1e-8:
            return v1_pos
        
        # Linear interpolation factor
        t = (iso_value - v1_val) / (v2_val - v1_val)
        t = np.clip(t, 0.0, 1.0)
        
        return v1_pos + t * (v2_pos - v1_pos)
    
    def _compute_normal(self, position: np.ndarray, sdf_tree: Optional[SDFNode], 
                       voxel_grid: VoxelGrid) -> np.ndarray:
        """
        Compute surface normal from SDF gradient at position
        
        Args:
            position: World position to compute normal at
            sdf_tree: SDF tree for gradient computation
            voxel_grid: Voxel grid for epsilon calculation
            
        Returns:
            Unit normal vector
        """
        if sdf_tree is None:
            # Default upward normal if no SDF available
            return np.array([0.0, 0.0, 1.0])
        
        # Use voxel size for finite difference epsilon
        epsilon = voxel_grid.min_voxel_size * 0.5
        
        # Compute gradient using central differences
        grad = sdf_gradient(sdf_tree, position, epsilon)
        
        # Normalize to unit vector
        length = np.linalg.norm(grad)
        if length < 1e-8:
            return np.array([0.0, 0.0, 1.0])  # Default up normal
        
        return grad / length
    
    def _compute_normal_improved(self, position: np.ndarray, sdf_tree: Optional[SDFNode], 
                               voxel_grid: VoxelGrid, scalar_field: np.ndarray) -> np.ndarray:
        """
        T20: Compute surface normal with improved quality using SDF gradient
        
        Args:
            position: World position to compute normal at
            sdf_tree: SDF tree for gradient computation
            voxel_grid: Voxel grid for epsilon and fallback calculation
            scalar_field: 3D scalar field for fallback estimation
            
        Returns:
            Unit normal vector
        """
        if self.use_sdf_normals and sdf_tree is not None:
            # Use SDF gradient for best quality
            epsilon = voxel_grid.min_voxel_size * 0.25  # Smaller epsilon for better accuracy
            grad = sdf_gradient(sdf_tree, position, epsilon)
            
            # Normalize to unit vector
            length = np.linalg.norm(grad)
            if length > 1e-8:
                return grad / length
        
        # Fallback: estimate from scalar field
        return self._estimate_normal_from_scalar_field(voxel_grid, position, scalar_field)
    
    def _estimate_normal_from_scalar_field(self, voxel_grid: VoxelGrid, 
                                         world_pos: np.ndarray, 
                                         scalar_field: np.ndarray) -> np.ndarray:
        """
        T20: Estimate normal from scalar field using central differences (fallback)
        
        Args:
            voxel_grid: Voxel grid
            world_pos: World position
            scalar_field: 3D scalar field
            
        Returns:
            Estimated normal vector
        """
        epsilon = voxel_grid.min_voxel_size
        
        # Sample scalar field at neighboring positions
        def sample_scalar(pos):
            voxel_pos = (pos - voxel_grid.bounds.min_point) / voxel_grid.min_voxel_size
            x, y, z = np.clip(voxel_pos.astype(int), 0, voxel_grid.resolution - 1)
            return scalar_field[x, y, z]
        
        # Central differences
        grad_x = (sample_scalar(world_pos + np.array([epsilon, 0, 0])) - 
                 sample_scalar(world_pos - np.array([epsilon, 0, 0]))) / (2 * epsilon)
        grad_y = (sample_scalar(world_pos + np.array([0, epsilon, 0])) - 
                 sample_scalar(world_pos - np.array([0, epsilon, 0]))) / (2 * epsilon)
        grad_z = (sample_scalar(world_pos + np.array([0, 0, epsilon])) - 
                 sample_scalar(world_pos - np.array([0, 0, epsilon]))) / (2 * epsilon)
        
        normal = np.array([grad_x, grad_y, grad_z])
        norm = np.linalg.norm(normal)
        
        if norm > 1e-8:
            return normal / norm
        else:
            return np.array([0.0, 0.0, 1.0])  # Default up normal
    
    def _remove_duplicate_vertices(self, vertices: List[MarchingCubesVertex], 
                                 triangles: np.ndarray, tolerance: float = 1e-6) -> Tuple[List[MarchingCubesVertex], np.ndarray]:
        """
        T20: Remove duplicate vertices in overlap regions to prevent seams
        
        Args:
            vertices: List of vertices
            triangles: Triangle index array
            tolerance: Distance tolerance for considering vertices duplicate
            
        Returns:
            Cleaned vertices and updated triangle indices
        """
        if not vertices or len(triangles) == 0:
            return vertices, triangles
        
        # Build spatial hash for efficient duplicate detection
        unique_vertices = []
        vertex_map = {}
        old_to_new_idx = {}
        
        for i, vertex in enumerate(vertices):
            # Create position key for spatial hashing
            pos_key = tuple(np.round(vertex.position / tolerance).astype(int))
            
            # Check if we already have a vertex at this position
            found_duplicate = False
            for existing_idx in vertex_map.get(pos_key, []):
                existing_vertex = unique_vertices[existing_idx]
                distance = np.linalg.norm(vertex.position - existing_vertex.position)
                
                if distance < tolerance:
                    # Use existing vertex
                    old_to_new_idx[i] = existing_idx
                    found_duplicate = True
                    break
            
            if not found_duplicate:
                # Add new unique vertex
                new_idx = len(unique_vertices)
                unique_vertices.append(vertex)
                vertex_map.setdefault(pos_key, []).append(new_idx)
                old_to_new_idx[i] = new_idx
        
        # Update triangle indices
        new_triangles = []
        for triangle in triangles:
            t0 = old_to_new_idx.get(triangle[0], triangle[0])
            t1 = old_to_new_idx.get(triangle[1], triangle[1])
            t2 = old_to_new_idx.get(triangle[2], triangle[2])
            
            # Skip degenerate triangles
            if t0 != t1 and t1 != t2 and t0 != t2:
                new_triangles.append([t0, t1, t2])
        
        return unique_vertices, np.array(new_triangles) if new_triangles else np.empty((0, 3), dtype=np.int32)


class CaveMeshGenerator:
    """
    Generates cave meshes for chunks using Marching Cubes
    """
    
    def __init__(self, material_id: int = 1):
        """
        Initialize cave mesh generator
        
        Args:
            material_id: Default material ID for cave surfaces
        """
        self.marching_cubes = MarchingCubes(iso_value=0.0)
        self.default_material_id = material_id
    
    def generate_cave_mesh(self, chunk_bounds: ChunkBounds, sdf_tree: SDFNode, 
                          resolution: int = 32, chunk_id: str = "unknown") -> Optional[CaveMeshData]:
        """
        Generate cave mesh for a chunk using SDF and Marching Cubes
        
        Args:
            chunk_bounds: Chunk bounds to generate mesh for
            sdf_tree: SDF tree defining cave geometry
            resolution: Voxel grid resolution
            chunk_id: Identifier for the chunk
            
        Returns:
            CaveMeshData containing mesh geometry or None if no surface found
        """
        # Create voxel grid for chunk
        voxel_grid = VoxelGrid(chunk_bounds, resolution)
        
        # Sample SDF over voxel grid
        evaluator = SDFEvaluator()
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, sdf_tree)
        
        # Generate mesh using Marching Cubes
        vertices, triangles = self.marching_cubes.polygonize(voxel_grid, scalar_field, sdf_tree)
        
        if len(vertices) == 0 or len(triangles) == 0:
            return None  # No surface found
        
        # Set material IDs for all vertices
        for vertex in vertices:
            vertex.material_id = self.default_material_id
        
        return CaveMeshData(
            vertices=vertices,
            triangles=triangles,
            chunk_id=chunk_id,
            material_id=self.default_material_id,
            bounds=chunk_bounds
        )
    
    def export_cave_mesh(self, cave_mesh: CaveMeshData, output_path: str):
        """
        Export cave mesh to file format compatible with viewer
        
        Args:
            cave_mesh: CaveMeshData to export
            output_path: Output file path
        """
        if cave_mesh is None:
            return
        
        mesh_data = {
            "chunk_id": cave_mesh.chunk_id,
            "material_id": cave_mesh.material_id,
            "vertex_count": len(cave_mesh.vertices),
            "triangle_count": len(cave_mesh.triangles),
            "bounds": {
                "min": cave_mesh.bounds.min_point.tolist(),
                "max": cave_mesh.bounds.max_point.tolist()
            },
            "vertices": [],
            "triangles": cave_mesh.triangles.tolist()
        }
        
        # Pack vertex data
        for vertex in cave_mesh.vertices:
            mesh_data["vertices"].append({
                "position": vertex.position.tolist(),
                "normal": vertex.normal.tolist(),
                "material_id": vertex.material_id
            })
        
        # Save to JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(mesh_data, f, indent=2)


if __name__ == "__main__":
    # Example usage and testing
    print("🚀 T10 Marching Cubes Polygonization System")
    print("=" * 60)
    
    # Test with simple sphere SDF
    from sdf_primitives import SDFSphere
    
    sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    bounds = ChunkBounds(
        min_point=np.array([-1.5, -1.5, -1.5]),
        max_point=np.array([1.5, 1.5, 1.5])
    )
    
    print(f"Test bounds: {bounds.min_point} to {bounds.max_point}")
    
    # Generate cave mesh
    cave_generator = CaveMeshGenerator(material_id=1)
    cave_mesh = cave_generator.generate_cave_mesh(bounds, sphere, resolution=16, chunk_id="test_sphere")
    
    if cave_mesh:
        print(f"\n✅ Cave mesh generated:")
        print(f"   Vertices: {len(cave_mesh.vertices)}")
        print(f"   Triangles: {len(cave_mesh.triangles)}")
        print(f"   Material ID: {cave_mesh.material_id}")
        
        # Test vertex data
        if len(cave_mesh.vertices) > 0:
            first_vertex = cave_mesh.vertices[0]
            print(f"   First vertex position: {first_vertex.position}")
            print(f"   First vertex normal: {first_vertex.normal}")
    else:
        print("❌ No cave mesh generated")
    
    print("\n✅ Marching Cubes system initialized")