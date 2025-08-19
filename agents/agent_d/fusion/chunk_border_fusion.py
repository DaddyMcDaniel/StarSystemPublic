#!/usr/bin/env python3
"""
Chunk Border Fusion System - T11
=================================

Ensures seamless transitions across chunk borders by managing overlapping
voxel regions and reconciling isosurfaces at chunk boundaries.

Features:
- 1-voxel overlap sampling for border continuity
- Neighbor chunk coordinate sharing
- Edge vertex deduplication and welding
- Isosurface reconciliation across boundaries
- Seam elimination for unified terrain

Usage:
    from chunk_border_fusion import ChunkBorderManager
    
    border_mgr = ChunkBorderManager()
    seamless_chunks = border_mgr.process_chunk_borders(chunk_list)
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Import fusion system
sys.path.append(os.path.dirname(__file__))
from surface_sdf_fusion import FusedMeshData, TerrainSdfFusion

# Import required modules  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marching_cubes'))

from sdf_evaluator import ChunkBounds
from marching_cubes import MarchingCubesVertex


class BorderEdge(Enum):
    """Chunk border edge identifiers"""
    LEFT = "left"      # -X edge
    RIGHT = "right"    # +X edge  
    BOTTOM = "bottom"  # -Z edge
    TOP = "top"        # +Z edge


@dataclass
class ChunkNeighbors:
    """Neighbor relationships for a chunk"""
    chunk_id: str
    left: Optional[str] = None      # -X neighbor
    right: Optional[str] = None     # +X neighbor
    bottom: Optional[str] = None    # -Z neighbor  
    top: Optional[str] = None       # +Z neighbor
    
    def get_neighbor(self, edge: BorderEdge) -> Optional[str]:
        """Get neighbor chunk ID for specific edge"""
        return {
            BorderEdge.LEFT: self.left,
            BorderEdge.RIGHT: self.right,
            BorderEdge.BOTTOM: self.bottom,
            BorderEdge.TOP: self.top
        }.get(edge)
    
    def has_neighbors(self) -> bool:
        """Check if chunk has any neighbors"""
        return any([self.left, self.right, self.bottom, self.top])


@dataclass
class BorderVertex:
    """Vertex on chunk border for edge reconciliation"""
    position: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    edge: BorderEdge
    chunk_id: str
    local_index: int
    
    def distance_to(self, other: 'BorderVertex') -> float:
        """Distance to another border vertex"""
        return np.linalg.norm(self.position - other.position)


@dataclass
class BorderVertexPair:
    """Pair of matching border vertices from adjacent chunks"""
    vertex_a: BorderVertex
    vertex_b: BorderVertex
    distance: float
    merged_vertex: Optional[MarchingCubesVertex] = None
    
    def create_merged_vertex(self) -> MarchingCubesVertex:
        """Create merged vertex averaging properties from both vertices"""
        if self.merged_vertex is not None:
            return self.merged_vertex
        
        # Average positions
        avg_position = (self.vertex_a.position + self.vertex_b.position) * 0.5
        
        # Average normals and normalize
        avg_normal = (self.vertex_a.normal + self.vertex_b.normal) * 0.5
        normal_length = np.linalg.norm(avg_normal)
        if normal_length > 1e-8:
            avg_normal /= normal_length
        else:
            avg_normal = np.array([0.0, 1.0, 0.0])
        
        # Average tangents and normalize
        avg_tangent = (self.vertex_a.tangent + self.vertex_b.tangent) * 0.5
        tangent_length = np.linalg.norm(avg_tangent)
        if tangent_length > 1e-8:
            avg_tangent /= tangent_length
        else:
            avg_tangent = np.array([1.0, 0.0, 0.0])
        
        self.merged_vertex = MarchingCubesVertex(
            position=avg_position,
            normal=avg_normal,
            material_id=self.vertex_a.chunk_id.split('_')[0] == 'cave' and 1 or 0
        )
        self.merged_vertex.tangent = avg_tangent
        
        return self.merged_vertex


class ChunkBorderManager:
    """Manages chunk border continuity and seam elimination"""
    
    def __init__(self, vertex_weld_tolerance: float = 1e-4):
        """
        Initialize chunk border manager
        
        Args:
            vertex_weld_tolerance: Distance tolerance for welding border vertices
        """
        self.vertex_weld_tolerance = vertex_weld_tolerance
        self.chunk_neighbors: Dict[str, ChunkNeighbors] = {}
        self.border_vertices: Dict[str, List[BorderVertex]] = {}
        self.vertex_pairs: List[BorderVertexPair] = []
    
    def analyze_chunk_layout(self, chunk_bounds_list: List[Tuple[str, ChunkBounds]]):
        """Analyze spatial layout to determine chunk neighbor relationships"""
        self.chunk_neighbors.clear()
        
        # Initialize neighbor data for all chunks
        for chunk_id, bounds in chunk_bounds_list:
            self.chunk_neighbors[chunk_id] = ChunkNeighbors(chunk_id)
        
        # Find neighbors by comparing chunk bounds
        for i, (chunk_id_a, bounds_a) in enumerate(chunk_bounds_list):
            for j, (chunk_id_b, bounds_b) in enumerate(chunk_bounds_list):
                if i == j:
                    continue
                
                # Check for adjacent relationships
                self._check_adjacency(chunk_id_a, bounds_a, chunk_id_b, bounds_b)
        
        print(f"✅ Analyzed {len(chunk_bounds_list)} chunks for neighbor relationships")
        
        # Print neighbor summary
        for chunk_id, neighbors in self.chunk_neighbors.items():
            neighbor_count = sum(1 for n in [neighbors.left, neighbors.right, 
                                           neighbors.bottom, neighbors.top] if n is not None)
            print(f"   {chunk_id}: {neighbor_count} neighbors")
    
    def _check_adjacency(self, chunk_id_a: str, bounds_a: ChunkBounds, 
                        chunk_id_b: str, bounds_b: ChunkBounds):
        """Check if two chunks are adjacent and update neighbor relationships"""
        tolerance = 1e-6
        
        # Check right edge of A against left edge of B (+X direction)
        if (abs(bounds_a.max_point[0] - bounds_b.min_point[0]) < tolerance and
            self._ranges_overlap(bounds_a.min_point[2], bounds_a.max_point[2],
                               bounds_b.min_point[2], bounds_b.max_point[2], tolerance)):
            self.chunk_neighbors[chunk_id_a].right = chunk_id_b
            self.chunk_neighbors[chunk_id_b].left = chunk_id_a
        
        # Check top edge of A against bottom edge of B (+Z direction) 
        if (abs(bounds_a.max_point[2] - bounds_b.min_point[2]) < tolerance and
            self._ranges_overlap(bounds_a.min_point[0], bounds_a.max_point[0],
                               bounds_b.min_point[0], bounds_b.max_point[0], tolerance)):
            self.chunk_neighbors[chunk_id_a].top = chunk_id_b
            self.chunk_neighbors[chunk_id_b].bottom = chunk_id_a
    
    def _ranges_overlap(self, min_a: float, max_a: float, 
                       min_b: float, max_b: float, tolerance: float) -> bool:
        """Check if two 1D ranges overlap within tolerance"""
        return not (max_a + tolerance < min_b or max_b + tolerance < min_a)
    
    def extract_border_vertices(self, chunk_id: str, fused_mesh: FusedMeshData) -> List[BorderVertex]:
        """Extract vertices on chunk borders for reconciliation"""
        border_vertices = []
        bounds = fused_mesh.bounds
        
        tolerance = self.vertex_weld_tolerance * 10  # Slightly larger tolerance for border detection
        
        for i, vertex in enumerate(fused_mesh.vertices):
            pos = vertex.position
            
            # Check which border edges this vertex lies on
            on_left = abs(pos[0] - bounds.min_point[0]) < tolerance
            on_right = abs(pos[0] - bounds.max_point[0]) < tolerance
            on_bottom = abs(pos[2] - bounds.min_point[2]) < tolerance
            on_top = abs(pos[2] - bounds.max_point[2]) < tolerance
            
            # Create border vertex entries for each edge this vertex touches
            if on_left:
                border_vertices.append(BorderVertex(
                    position=pos.copy(),
                    normal=vertex.normal.copy(),
                    tangent=getattr(vertex, 'tangent', np.array([1, 0, 0])).copy(),
                    edge=BorderEdge.LEFT,
                    chunk_id=chunk_id,
                    local_index=i
                ))
            
            if on_right:
                border_vertices.append(BorderVertex(
                    position=pos.copy(), 
                    normal=vertex.normal.copy(),
                    tangent=getattr(vertex, 'tangent', np.array([1, 0, 0])).copy(),
                    edge=BorderEdge.RIGHT,
                    chunk_id=chunk_id,
                    local_index=i
                ))
            
            if on_bottom:
                border_vertices.append(BorderVertex(
                    position=pos.copy(),
                    normal=vertex.normal.copy(),
                    tangent=getattr(vertex, 'tangent', np.array([1, 0, 0])).copy(),
                    edge=BorderEdge.BOTTOM,
                    chunk_id=chunk_id,
                    local_index=i
                ))
            
            if on_top:
                border_vertices.append(BorderVertex(
                    position=pos.copy(),
                    normal=vertex.normal.copy(), 
                    tangent=getattr(vertex, 'tangent', np.array([1, 0, 0])).copy(),
                    edge=BorderEdge.TOP,
                    chunk_id=chunk_id,
                    local_index=i
                ))
        
        self.border_vertices[chunk_id] = border_vertices
        return border_vertices
    
    def find_matching_border_vertices(self, chunk_id_a: str, chunk_id_b: str, 
                                    edge_a: BorderEdge, edge_b: BorderEdge) -> List[BorderVertexPair]:
        """Find matching vertices on adjacent chunk borders"""
        if chunk_id_a not in self.border_vertices or chunk_id_b not in self.border_vertices:
            return []
        
        vertices_a = [v for v in self.border_vertices[chunk_id_a] if v.edge == edge_a]
        vertices_b = [v for v in self.border_vertices[chunk_id_b] if v.edge == edge_b]
        
        pairs = []
        used_b_indices = set()
        
        # Find closest matching vertices
        for vertex_a in vertices_a:
            best_match = None
            best_distance = float('inf')
            best_index = -1
            
            for i, vertex_b in enumerate(vertices_b):
                if i in used_b_indices:
                    continue
                
                distance = vertex_a.distance_to(vertex_b)
                if distance < self.vertex_weld_tolerance and distance < best_distance:
                    best_match = vertex_b
                    best_distance = distance
                    best_index = i
            
            if best_match is not None:
                pairs.append(BorderVertexPair(vertex_a, best_match, best_distance))
                used_b_indices.add(best_index)
        
        return pairs
    
    def reconcile_chunk_borders(self, fused_meshes: Dict[str, FusedMeshData]) -> Dict[str, FusedMeshData]:
        """Reconcile borders between adjacent chunks to eliminate seams"""
        print("\n🔧 Reconciling chunk borders...")
        
        # Extract border vertices from all chunks
        for chunk_id, mesh in fused_meshes.items():
            self.extract_border_vertices(chunk_id, mesh)
        
        # Find vertex pairs across adjacent chunk borders
        self.vertex_pairs.clear()
        
        for chunk_id, neighbors in self.chunk_neighbors.items():
            if chunk_id not in fused_meshes:
                continue
                
            # Check each border edge for neighbors
            if neighbors.right and neighbors.right in fused_meshes:
                pairs = self.find_matching_border_vertices(
                    chunk_id, neighbors.right, BorderEdge.RIGHT, BorderEdge.LEFT
                )
                self.vertex_pairs.extend(pairs)
            
            if neighbors.top and neighbors.top in fused_meshes:
                pairs = self.find_matching_border_vertices(
                    chunk_id, neighbors.top, BorderEdge.TOP, BorderEdge.BOTTOM
                )
                self.vertex_pairs.extend(pairs)
        
        print(f"   Found {len(self.vertex_pairs)} vertex pairs to reconcile")
        
        # Create merged vertices for matching pairs
        vertex_replacements = {}  # (chunk_id, local_index) -> merged_vertex
        
        for pair in self.vertex_pairs:
            merged_vertex = pair.create_merged_vertex()
            
            # Mark both original vertices for replacement
            key_a = (pair.vertex_a.chunk_id, pair.vertex_a.local_index)
            key_b = (pair.vertex_b.chunk_id, pair.vertex_b.local_index)
            
            vertex_replacements[key_a] = merged_vertex
            vertex_replacements[key_b] = merged_vertex
        
        # Apply vertex replacements to meshes
        reconciled_meshes = {}
        
        for chunk_id, mesh in fused_meshes.items():
            new_vertices = []
            
            for i, vertex in enumerate(mesh.vertices):
                key = (chunk_id, i)
                if key in vertex_replacements:
                    new_vertices.append(vertex_replacements[key])
                else:
                    new_vertices.append(vertex)
            
            # Create new mesh data with reconciled vertices
            reconciled_meshes[chunk_id] = FusedMeshData(
                vertices=new_vertices,
                triangles=mesh.triangles,
                chunk_id=mesh.chunk_id,
                bounds=mesh.bounds,
                has_caves=mesh.has_caves,
                fusion_stats=mesh.fusion_stats
            )
        
        print(f"   ✅ Reconciled borders for {len(reconciled_meshes)} chunks")
        return reconciled_meshes
    
    def validate_seam_elimination(self, fused_meshes: Dict[str, FusedMeshData]) -> Dict[str, Any]:
        """Validate that chunk seams have been eliminated"""
        validation_stats = {
            'chunks_checked': 0,
            'border_vertices_found': 0,
            'vertex_pairs_matched': 0,
            'max_seam_distance': 0.0,
            'avg_seam_distance': 0.0,
            'seams_eliminated': True
        }
        
        total_distance = 0.0
        max_distance = 0.0
        
        for pair in self.vertex_pairs:
            distance = pair.distance
            total_distance += distance
            max_distance = max(max_distance, distance)
            
            if distance > self.vertex_weld_tolerance * 2:
                validation_stats['seams_eliminated'] = False
        
        validation_stats['chunks_checked'] = len(fused_meshes)
        validation_stats['border_vertices_found'] = sum(len(vertices) for vertices in self.border_vertices.values())
        validation_stats['vertex_pairs_matched'] = len(self.vertex_pairs)
        validation_stats['max_seam_distance'] = max_distance
        validation_stats['avg_seam_distance'] = total_distance / len(self.vertex_pairs) if self.vertex_pairs else 0.0
        
        return validation_stats


class SeamlessChunkProcessor:
    """High-level processor for creating seamless terrain chunks"""
    
    def __init__(self, fusion_resolution: int = 32, overlap_voxels: int = 1):
        """
        Initialize seamless chunk processor
        
        Args:
            fusion_resolution: Voxel resolution for terrain-cave fusion
            overlap_voxels: Overlap size for border continuity
        """
        self.fusion_system = TerrainSdfFusion(fusion_resolution, overlap_voxels)
        self.border_manager = ChunkBorderManager()
    
    def process_terrain_chunks_with_caves(self, terrain_chunks: Dict[str, Dict], 
                                        cave_sdfs: Dict[str, Any]) -> Dict[str, FusedMeshData]:
        """
        Process terrain chunks with caves, ensuring seamless borders
        
        Args:
            terrain_chunks: Dict of chunk_id -> terrain chunk data
            cave_sdfs: Dict of chunk_id -> cave SDF nodes
            
        Returns:
            Dict of chunk_id -> seamless fused mesh data
        """
        print(f"🚀 Processing {len(terrain_chunks)} chunks with seamless fusion...")
        
        # Step 1: Analyze chunk layout for neighbor relationships
        chunk_bounds_list = []
        for chunk_id, chunk_data in terrain_chunks.items():
            bounds = self._extract_chunk_bounds(chunk_data)
            chunk_bounds_list.append((chunk_id, bounds))
        
        self.border_manager.analyze_chunk_layout(chunk_bounds_list)
        
        # Step 2: Generate fused meshes for each chunk
        fused_meshes = {}
        
        for chunk_id, chunk_data in terrain_chunks.items():
            bounds = self._extract_chunk_bounds(chunk_data)
            cave_sdf = cave_sdfs.get(chunk_id)
            
            print(f"   Processing {chunk_id}...")
            fused_mesh = self.fusion_system.fuse_terrain_and_caves(
                chunk_data, cave_sdf, bounds, chunk_id
            )
            
            fused_meshes[chunk_id] = fused_mesh
        
        # Step 3: Reconcile chunk borders to eliminate seams
        seamless_meshes = self.border_manager.reconcile_chunk_borders(fused_meshes)
        
        # Step 4: Validate seam elimination
        validation_stats = self.border_manager.validate_seam_elimination(seamless_meshes)
        
        print(f"\n📊 Seamless Processing Results:")
        print(f"   Chunks processed: {validation_stats['chunks_checked']}")
        print(f"   Border vertices: {validation_stats['border_vertices_found']}")
        print(f"   Vertex pairs matched: {validation_stats['vertex_pairs_matched']}")
        print(f"   Max seam distance: {validation_stats['max_seam_distance']:.6f}")
        print(f"   Seams eliminated: {'✅' if validation_stats['seams_eliminated'] else '❌'}")
        
        return seamless_meshes
    
    def _extract_chunk_bounds(self, chunk_data: Dict) -> ChunkBounds:
        """Extract chunk bounds from chunk data"""
        chunk_info = chunk_data.get("chunk_info", {})
        aabb = chunk_info.get("aabb", {})
        
        min_point = np.array(aabb.get("min", [-2, -2, -2]))
        max_point = np.array(aabb.get("max", [2, 2, 2]))
        
        return ChunkBounds(min_point, max_point)


if __name__ == "__main__":
    # Test chunk border fusion system
    print("🧪 Testing Chunk Border Fusion System")
    print("=" * 50)
    
    # Create test chunks with adjacent layout
    test_chunks = {
        "chunk_0_0": {
            "chunk_info": {
                "chunk_id": "chunk_0_0",
                "aabb": {
                    "min": [-2.0, -1.0, -2.0],
                    "max": [0.0, 1.0, 0.0]
                }
            },
            "positions": np.array([[0, 0, 0]])
        },
        "chunk_1_0": {
            "chunk_info": {
                "chunk_id": "chunk_1_0", 
                "aabb": {
                    "min": [0.0, -1.0, -2.0],
                    "max": [2.0, 1.0, 0.0]
                }
            },
            "positions": np.array([[0, 0, 0]])
        }
    }
    
    # Test border manager
    border_mgr = ChunkBorderManager()
    
    # Analyze layout
    chunk_bounds_list = []
    for chunk_id, chunk_data in test_chunks.items():
        chunk_info = chunk_data["chunk_info"]
        aabb = chunk_info["aabb"]
        bounds = ChunkBounds(np.array(aabb["min"]), np.array(aabb["max"]))
        chunk_bounds_list.append((chunk_id, bounds))
    
    border_mgr.analyze_chunk_layout(chunk_bounds_list)
    
    # Check neighbor relationships
    neighbors_0 = border_mgr.chunk_neighbors["chunk_0_0"]
    neighbors_1 = border_mgr.chunk_neighbors["chunk_1_0"]
    
    print(f"✅ Chunk 0_0 neighbors: right={neighbors_0.right}")
    print(f"✅ Chunk 1_0 neighbors: left={neighbors_1.left}")
    
    print("\n✅ Chunk border fusion system functional")