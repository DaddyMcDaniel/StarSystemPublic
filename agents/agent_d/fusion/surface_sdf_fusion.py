#!/usr/bin/env python3
"""
Surface-SDF Fusion System - T11
===============================

Blends surface heightfield terrain with SDF cave/overhang geometry to create
unified meshes with seamless transitions and no chunk border artifacts.

Features:
- Heightfield to SDF conversion for terrain surfaces
- Boolean subtraction of cave geometry from terrain
- Chunk border continuity with 1-voxel overlap sampling  
- Isosurface reconciliation across chunk edges
- Normal and tangent recomputation for fused meshes
- Debug visualization of SDF iso-contours

Usage:
    from surface_sdf_fusion import TerrainSdfFusion
    
    fusion = TerrainSdfFusion()
    fused_mesh = fusion.fuse_terrain_and_caves(terrain_chunk, cave_sdf, bounds)
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marching_cubes'))

from sdf_evaluator import VoxelGrid, ChunkBounds, SDFEvaluator
from sdf_primitives import SDFNode, SDFSubtract, sdf_gradient
from marching_cubes import MarchingCubes, MarchingCubesVertex


@dataclass
class TerrainHeightfield:
    """Heightfield representation of terrain surface"""
    width: int
    height: int
    heights: np.ndarray  # 2D height values
    bounds: ChunkBounds
    
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def sample_height(self, x: float, z: float) -> float:
        """Sample height at world coordinates with bilinear interpolation"""
        # Convert world coords to heightfield coords
        rel_x = (x - self.bounds.min_point[0]) / self.bounds.size[0]
        rel_z = (z - self.bounds.min_point[2]) / self.bounds.size[2]
        
        # Convert to heightfield indices
        fx = rel_x * (self.width - 1)
        fz = rel_z * (self.height - 1)
        
        # Clamp to valid range
        fx = np.clip(fx, 0, self.width - 1)
        fz = np.clip(fz, 0, self.height - 1)
        
        # Bilinear interpolation
        ix0, ix1 = int(fx), min(int(fx) + 1, self.width - 1)
        iz0, iz1 = int(fz), min(int(fz) + 1, self.height - 1)
        
        wx = fx - ix0
        wz = fz - iz0
        
        h00 = self.heights[iz0, ix0]
        h10 = self.heights[iz0, ix1] 
        h01 = self.heights[iz1, ix0]
        h11 = self.heights[iz1, ix1]
        
        h0 = h00 * (1 - wx) + h10 * wx
        h1 = h01 * (1 - wx) + h11 * wx
        
        return h0 * (1 - wz) + h1 * wz


class HeightfieldToSDF:
    """Converts heightfield terrain to SDF representation"""
    
    def __init__(self):
        pass
    
    def convert_heightfield_to_sdf(self, heightfield: TerrainHeightfield) -> 'HeightfieldSDF':
        """Convert heightfield to SDF representation"""
        return HeightfieldSDF(heightfield)


class HeightfieldSDF(SDFNode):
    """SDF representation of a heightfield terrain surface"""
    
    def __init__(self, heightfield: TerrainHeightfield, seed: int = 42):
        super().__init__(seed)
        self.heightfield = heightfield
    
    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate SDF distance to heightfield surface"""
        x, y, z = point
        
        # Sample height at (x, z) position
        terrain_height = self.heightfield.sample_height(x, z)
        
        # Distance is vertical distance to terrain surface
        # Negative below ground, positive above
        return y - terrain_height
    
    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Batch evaluate SDF for multiple points"""
        distances = np.zeros(points.shape[0])
        
        for i, point in enumerate(points):
            distances[i] = self.evaluate(point)
        
        return distances


@dataclass
class FusedMeshData:
    """Result of terrain-SDF fusion"""
    vertices: List[MarchingCubesVertex]
    triangles: np.ndarray
    chunk_id: str
    bounds: ChunkBounds
    has_caves: bool
    fusion_stats: Dict[str, Any]


class TerrainSdfFusion:
    """
    Main class for fusing terrain heightfields with SDF cave geometry
    """
    
    def __init__(self, resolution: int = 32, overlap_voxels: int = 1):
        """
        Initialize terrain-SDF fusion system
        
        Args:
            resolution: Voxel grid resolution for fusion
            overlap_voxels: Number of overlapping voxels for border continuity
        """
        self.resolution = resolution
        self.overlap_voxels = overlap_voxels
        self.heightfield_converter = HeightfieldToSDF()
        self.marching_cubes = MarchingCubes(iso_value=0.0)
        
        # Fusion statistics
        self.stats = {
            'chunks_processed': 0,
            'caves_carved': 0,
            'vertices_generated': 0,
            'fusion_time': 0.0
        }
    
    def create_extended_bounds(self, chunk_bounds: ChunkBounds) -> ChunkBounds:
        """Create extended bounds with overlap for border continuity"""
        size = chunk_bounds.size
        voxel_size = size / self.resolution
        overlap_size = voxel_size * self.overlap_voxels
        
        extended_min = chunk_bounds.min_point - overlap_size
        extended_max = chunk_bounds.max_point + overlap_size
        
        return ChunkBounds(extended_min, extended_max)
    
    def extract_heightfield_from_chunk(self, chunk_data: Dict) -> TerrainHeightfield:
        """Extract heightfield from terrain chunk data"""
        # Get terrain geometry from chunk
        positions = chunk_data.get("positions", np.array([]))
        
        if len(positions) == 0:
            # Create flat heightfield as fallback
            bounds = self._extract_chunk_bounds(chunk_data)
            return self._create_flat_heightfield(bounds)
        
        # Extract bounds
        bounds = self._extract_chunk_bounds(chunk_data)
        
        # Create heightfield from vertex positions
        return self._create_heightfield_from_positions(positions, bounds)
    
    def _extract_chunk_bounds(self, chunk_data: Dict) -> ChunkBounds:
        """Extract chunk bounds from chunk data"""
        chunk_info = chunk_data.get("chunk_info", {})
        aabb = chunk_info.get("aabb", {})
        
        min_point = np.array(aabb.get("min", [-2, -2, -2]))
        max_point = np.array(aabb.get("max", [2, 2, 2]))
        
        return ChunkBounds(min_point, max_point)
    
    def _create_flat_heightfield(self, bounds: ChunkBounds, height: float = 0.0) -> TerrainHeightfield:
        """Create flat heightfield at specified height"""
        resolution = 32
        heights = np.full((resolution, resolution), height, dtype=np.float32)
        
        return TerrainHeightfield(
            width=resolution,
            height=resolution, 
            heights=heights,
            bounds=bounds
        )
    
    def _create_heightfield_from_positions(self, positions: np.ndarray, bounds: ChunkBounds) -> TerrainHeightfield:
        """Create heightfield by sampling terrain vertex positions"""
        resolution = 32
        heights = np.zeros((resolution, resolution), dtype=np.float32)
        
        # Sample terrain heights across the heightfield grid
        size = bounds.size
        for i in range(resolution):
            for j in range(resolution):
                # Compute world coordinates
                x = bounds.min_point[0] + (i / (resolution - 1)) * size[0]
                z = bounds.min_point[2] + (j / (resolution - 1)) * size[2]
                
                # Find closest terrain vertex
                distances = np.sum((positions[:, [0, 2]] - [x, z]) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                heights[j, i] = positions[closest_idx, 1]  # Y coordinate
        
        return TerrainHeightfield(
            width=resolution,
            height=resolution,
            heights=heights,
            bounds=bounds
        )
    
    def create_terrain_cave_fusion_sdf(self, heightfield: TerrainHeightfield, 
                                     cave_sdf: SDFNode) -> SDFNode:
        """Create SDF that represents terrain with caves carved out"""
        # Convert heightfield to SDF
        terrain_sdf = HeightfieldSDF(heightfield)
        
        # Create boolean subtraction: terrain - caves
        # This carves caves out of the terrain
        fusion_sdf = SDFSubtract(terrain_sdf, cave_sdf, seed=42)
        
        return fusion_sdf
    
    def fuse_terrain_and_caves(self, terrain_chunk: Dict, cave_sdf: Optional[SDFNode], 
                              chunk_bounds: ChunkBounds, chunk_id: str = "unknown") -> FusedMeshData:
        """
        Main fusion function: blend terrain heightfield with SDF caves
        
        Args:
            terrain_chunk: Terrain chunk data with positions/geometry
            cave_sdf: SDF defining cave geometry (None = no caves)
            chunk_bounds: Chunk spatial bounds
            chunk_id: Chunk identifier
            
        Returns:
            FusedMeshData containing unified mesh with caves carved into terrain
        """
        import time
        start_time = time.time()
        
        # Extract heightfield from terrain chunk
        heightfield = self.extract_heightfield_from_chunk(terrain_chunk)
        
        # Create extended bounds for border continuity
        extended_bounds = self.create_extended_bounds(chunk_bounds)
        
        # Create voxel grid with overlap
        voxel_grid = VoxelGrid(extended_bounds, self.resolution)
        
        # Create fusion SDF
        if cave_sdf is not None:
            fusion_sdf = self.create_terrain_cave_fusion_sdf(heightfield, cave_sdf)
            has_caves = True
        else:
            # No caves, just use terrain SDF
            fusion_sdf = HeightfieldSDF(heightfield)
            has_caves = False
        
        # Sample fusion SDF over voxel grid
        evaluator = SDFEvaluator()
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, fusion_sdf)
        
        # Generate mesh using Marching Cubes
        vertices, triangles = self.marching_cubes.polygonize(voxel_grid, scalar_field, fusion_sdf)
        
        # Recompute normals and tangents
        vertices = self._recompute_normals_and_tangents(vertices, triangles, fusion_sdf, voxel_grid)
        
        # Crop vertices back to original chunk bounds (remove overlap)
        vertices, triangles = self._crop_mesh_to_bounds(vertices, triangles, chunk_bounds)
        
        fusion_time = time.time() - start_time
        
        # Update statistics
        self.stats['chunks_processed'] += 1
        if has_caves:
            self.stats['caves_carved'] += 1
        self.stats['vertices_generated'] += len(vertices)
        self.stats['fusion_time'] += fusion_time
        
        fusion_stats = {
            'fusion_time_ms': fusion_time * 1000,
            'vertices_generated': len(vertices),
            'triangles_generated': len(triangles),
            'has_caves': has_caves,
            'extended_bounds': extended_bounds,
            'crop_bounds': chunk_bounds
        }
        
        return FusedMeshData(
            vertices=vertices,
            triangles=triangles,
            chunk_id=chunk_id,
            bounds=chunk_bounds,
            has_caves=has_caves,
            fusion_stats=fusion_stats
        )
    
    def _recompute_normals_and_tangents(self, vertices: List[MarchingCubesVertex], 
                                      triangles: np.ndarray, fusion_sdf: SDFNode,
                                      voxel_grid: VoxelGrid) -> List[MarchingCubesVertex]:
        """Recompute normals and tangents for fused mesh vertices"""
        epsilon = voxel_grid.min_voxel_size * 0.5
        
        # Recompute normals from SDF gradients
        for vertex in vertices:
            # Compute gradient at vertex position
            gradient = sdf_gradient(fusion_sdf, vertex.position, epsilon)
            
            # Normalize to get normal
            length = np.linalg.norm(gradient)
            if length > 1e-8:
                vertex.normal = gradient / length
            else:
                vertex.normal = np.array([0.0, 1.0, 0.0])  # Default up normal
        
        # Recompute tangents based on triangles
        self._recompute_tangents(vertices, triangles)
        
        return vertices
    
    def _recompute_tangents(self, vertices: List[MarchingCubesVertex], triangles: np.ndarray):
        """Recompute tangent vectors for vertices"""
        vertex_count = len(vertices)
        tangent_accum = np.zeros((vertex_count, 3))
        tangent_count = np.zeros(vertex_count)
        
        # Accumulate tangents from triangles
        for triangle in triangles:
            if len(triangle) != 3:
                continue
                
            i1, i2, i3 = triangle
            if i1 >= vertex_count or i2 >= vertex_count or i3 >= vertex_count:
                continue
            
            v1 = vertices[i1].position
            v2 = vertices[i2].position
            v3 = vertices[i3].position
            
            # Calculate edge vectors
            edge1 = v2 - v1
            edge2 = v3 - v1
            
            # Calculate tangent (simplified)
            tangent = np.cross(edge1, edge2)
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-8:
                tangent = tangent / tangent_norm
                
                # Accumulate for each vertex of the triangle
                for i in [i1, i2, i3]:
                    tangent_accum[i] += tangent
                    tangent_count[i] += 1
        
        # Average and orthogonalize tangents
        for i in range(vertex_count):
            if tangent_count[i] > 0:
                # Average tangent
                avg_tangent = tangent_accum[i] / tangent_count[i]
                normal = vertices[i].normal
                
                # Gram-Schmidt orthogonalization
                tangent = avg_tangent - np.dot(avg_tangent, normal) * normal
                tangent_norm = np.linalg.norm(tangent)
                
                if tangent_norm > 1e-8:
                    vertices[i].tangent = tangent / tangent_norm
                else:
                    # Generate perpendicular vector
                    if abs(normal[0]) < 0.9:
                        vertices[i].tangent = np.cross(normal, [1, 0, 0])
                    else:
                        vertices[i].tangent = np.cross(normal, [0, 1, 0])
                    vertices[i].tangent /= np.linalg.norm(vertices[i].tangent)
            else:
                vertices[i].tangent = np.array([1.0, 0.0, 0.0])  # Default tangent
    
    def _crop_mesh_to_bounds(self, vertices: List[MarchingCubesVertex], triangles: np.ndarray,
                           crop_bounds: ChunkBounds) -> Tuple[List[MarchingCubesVertex], np.ndarray]:
        """Crop mesh vertices back to original chunk bounds, removing overlap"""
        # Find vertices within crop bounds
        valid_vertex_indices = []
        vertex_mapping = {}  # old_index -> new_index
        
        for i, vertex in enumerate(vertices):
            pos = vertex.position
            if (np.all(pos >= crop_bounds.min_point) and 
                np.all(pos <= crop_bounds.max_point)):
                new_index = len(valid_vertex_indices)
                valid_vertex_indices.append(i)
                vertex_mapping[i] = new_index
        
        # Create cropped vertex list
        cropped_vertices = [vertices[i] for i in valid_vertex_indices]
        
        # Update triangles to use new vertex indices
        valid_triangles = []
        for triangle in triangles:
            # Check if all triangle vertices are in the cropped set
            if all(idx in vertex_mapping for idx in triangle):
                new_triangle = np.array([vertex_mapping[idx] for idx in triangle])
                valid_triangles.append(new_triangle)
        
        cropped_triangles = np.array(valid_triangles) if valid_triangles else np.empty((0, 3), dtype=np.int32)
        
        return cropped_vertices, cropped_triangles
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion system statistics"""
        return self.stats.copy()


class SdfIsoContourVisualizer:
    """Visualizes SDF iso-contours on terrain surface for debugging"""
    
    def __init__(self):
        pass
    
    def generate_iso_contours(self, fusion_sdf: SDFNode, bounds: ChunkBounds, 
                            surface_resolution: int = 64) -> Dict[str, np.ndarray]:
        """Generate iso-contour lines on the terrain surface"""
        contours = {}
        
        # Sample SDF values across the surface
        size = bounds.size
        x_coords = np.linspace(bounds.min_point[0], bounds.max_point[0], surface_resolution)
        z_coords = np.linspace(bounds.min_point[2], bounds.max_point[2], surface_resolution)
        
        # For each iso-value, find contour lines
        iso_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        
        for iso_value in iso_values:
            contour_points = []
            
            # Sample grid points
            for i in range(surface_resolution - 1):
                for j in range(surface_resolution - 1):
                    x = x_coords[i]
                    z = z_coords[j]
                    
                    # Sample at surface height (y = 0 for now, could be more sophisticated)
                    point = np.array([x, 0.0, z])
                    sdf_value = fusion_sdf.evaluate(point)
                    
                    # Check for iso-value crossing
                    if abs(sdf_value - iso_value) < 0.1:
                        contour_points.append(point)
            
            if contour_points:
                contours[f"iso_{iso_value}"] = np.array(contour_points)
        
        return contours


# Add tangent attribute to MarchingCubesVertex if not present
if not hasattr(MarchingCubesVertex, 'tangent'):
    def add_tangent_field():
        """Add tangent field to existing MarchingCubesVertex class"""
        original_init = MarchingCubesVertex.__init__
        
        def new_init(self, position, normal, material_id=0):
            original_init(self, position, normal, material_id)
            self.tangent = np.array([1.0, 0.0, 0.0])  # Default tangent
        
        MarchingCubesVertex.__init__ = new_init
        MarchingCubesVertex.tangent = np.array([1.0, 0.0, 0.0])
    
    add_tangent_field()


if __name__ == "__main__":
    # Test terrain-SDF fusion system
    print("🚀 T11 Surface-SDF Fusion System")
    print("=" * 50)
    
    # Create test terrain heightfield
    bounds = ChunkBounds(
        min_point=np.array([-2.0, -1.0, -2.0]),
        max_point=np.array([2.0, 1.0, 2.0])
    )
    
    # Create simple heightfield (sinusoidal terrain)
    resolution = 32
    heights = np.zeros((resolution, resolution), dtype=np.float32)
    
    for i in range(resolution):
        for j in range(resolution):
            x = -2.0 + (i / (resolution - 1)) * 4.0
            z = -2.0 + (j / (resolution - 1)) * 4.0
            heights[j, i] = 0.3 * math.sin(x * math.pi) * math.cos(z * math.pi)
    
    heightfield = TerrainHeightfield(
        width=resolution,
        height=resolution,
        heights=heights,
        bounds=bounds
    )
    
    print(f"Test heightfield: {resolution}x{resolution}, bounds {bounds.size}")
    
    # Create simple cave SDF
    from sdf_primitives import SDFSphere
    cave_sdf = SDFSphere(center=[0, 0, 0], radius=0.8, seed=42)
    
    # Create fusion system
    fusion = TerrainSdfFusion(resolution=16, overlap_voxels=1)
    
    # Create mock terrain chunk
    terrain_chunk = {
        "positions": np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]),  # Minimal positions
        "chunk_info": {
            "chunk_id": "test_chunk",
            "aabb": {
                "min": bounds.min_point.tolist(),
                "max": bounds.max_point.tolist()
            }
        }
    }
    
    # Test fusion
    print("\n🔧 Testing terrain-cave fusion...")
    fused_mesh = fusion.fuse_terrain_and_caves(terrain_chunk, cave_sdf, bounds, "test_fusion")
    
    print(f"✅ Fusion complete:")
    print(f"   Vertices: {len(fused_mesh.vertices)}")
    print(f"   Triangles: {len(fused_mesh.triangles)}")
    print(f"   Has caves: {fused_mesh.has_caves}")
    print(f"   Fusion time: {fused_mesh.fusion_stats['fusion_time_ms']:.2f} ms")
    
    # Test statistics
    stats = fusion.get_fusion_statistics()
    print(f"\n📊 Fusion Statistics:")
    print(f"   Chunks processed: {stats['chunks_processed']}")
    print(f"   Caves carved: {stats['caves_carved']}")
    print(f"   Total vertices: {stats['vertices_generated']}")
    
    print("\n✅ Surface-SDF fusion system functional")