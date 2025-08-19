#!/usr/bin/env python3
"""
Quadtree Chunking System for Agent D - T06
==========================================

Subdivides cube faces into quadtree chunks for static LOD scaffolding.
Each chunk is baked as its own mesh with separate manifest+buffers.

Features:
- Quadtree node structure with level and UV bounds
- Static subdivision to target chunk resolution
- Per-chunk mesh generation with heightfield displacement
- Manifest export per chunk with shared format
- AABB computation for debug visualization

Usage:
    python quadtree_chunking.py --max_depth 3 --chunk_res 16 --output chunks/
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import sys
import os

# Import dependencies
sys.path.append(os.path.dirname(__file__))
from cubesphere import CubeSphereGenerator, export_manifest
from shading_basis import compute_angle_weighted_normals, compute_tangent_basis, validate_shading_basis

# Import terrain system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'terrain'))
try:
    from heightfield import create_heightfield_from_pcc, HeightField
    TERRAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Terrain system not available: {e}")
    TERRAIN_AVAILABLE = False


@dataclass
class QuadtreeNode:
    """
    Quadtree node structure for cube face subdivision
    
    Attributes:
        level: Subdivision level (0 = root, increases with depth)
        face_id: Which cube face this node belongs to (0-5)
        uv_min: Minimum UV coordinates in face space [0,1]
        uv_max: Maximum UV coordinates in face space [0,1]
        is_leaf: Whether this is a leaf node (will generate mesh)
        children: Child nodes (4 for internal nodes, None for leaves)
        chunk_id: Unique identifier for this chunk
    """
    level: int
    face_id: int
    uv_min: Tuple[float, float]
    uv_max: Tuple[float, float]
    is_leaf: bool = False
    children: Optional[List['QuadtreeNode']] = None
    chunk_id: str = ""
    
    def __post_init__(self):
        """Generate chunk ID after initialization"""
        if not self.chunk_id:
            u_mid = (self.uv_min[0] + self.uv_max[0]) * 0.5
            v_mid = (self.uv_min[1] + self.uv_max[1]) * 0.5
            self.chunk_id = f"face{self.face_id}_L{self.level}_{u_mid:.3f}_{v_mid:.3f}"
    
    @property
    def uv_center(self) -> Tuple[float, float]:
        """Get UV center of this node"""
        return (
            (self.uv_min[0] + self.uv_max[0]) * 0.5,
            (self.uv_min[1] + self.uv_max[1]) * 0.5
        )
    
    @property
    def uv_size(self) -> Tuple[float, float]:
        """Get UV size of this node"""
        return (
            self.uv_max[0] - self.uv_min[0],
            self.uv_max[1] - self.uv_min[1]
        )
    
    def subdivide(self) -> List['QuadtreeNode']:
        """
        Subdivide this node into 4 children
        
        Returns:
            List of 4 child nodes
        """
        if self.is_leaf or self.children is not None:
            raise ValueError("Cannot subdivide leaf node or already subdivided node")
        
        u_min, v_min = self.uv_min
        u_max, v_max = self.uv_max
        u_mid = (u_min + u_max) * 0.5
        v_mid = (v_min + v_max) * 0.5
        
        # Create 4 children in quadrant order: SW, SE, NW, NE
        children = [
            QuadtreeNode(self.level + 1, self.face_id, (u_min, v_min), (u_mid, v_mid)),  # SW
            QuadtreeNode(self.level + 1, self.face_id, (u_mid, v_min), (u_max, v_mid)),  # SE
            QuadtreeNode(self.level + 1, self.face_id, (u_min, v_mid), (u_mid, v_max)),  # NW
            QuadtreeNode(self.level + 1, self.face_id, (u_mid, v_mid), (u_max, v_max)),  # NE
        ]
        
        self.children = children
        return children


class QuadtreeChunker:
    """
    Generates quadtree chunks for cube faces with heightfield displacement
    """
    
    def __init__(self, max_depth: int = 3, chunk_res: int = 16, 
                 base_radius: float = 1.0, heightfield: Optional[HeightField] = None,
                 displacement_scale: float = 0.0):
        """
        Initialize quadtree chunker
        
        Args:
            max_depth: Maximum quadtree depth
            chunk_res: Grid resolution per chunk (N×N)
            base_radius: Base sphere radius
            heightfield: Optional heightfield for displacement
            displacement_scale: Displacement scaling factor
        """
        self.max_depth = max_depth
        self.chunk_res = chunk_res
        self.base_radius = base_radius
        self.heightfield = heightfield
        self.displacement_scale = displacement_scale
        self.face_configs = self._get_face_configurations()
        
    def _get_face_configurations(self) -> List[Tuple[Tuple[float, float, float], 
                                                   Tuple[float, float, float], 
                                                   Tuple[float, float, float]]]:
        """Get cube face configurations (normal, up, right vectors)"""
        return [
            # +X face (right)
            (( 1,  0,  0), ( 0,  1,  0), ( 0,  0, -1)),
            # -X face (left) 
            ((-1,  0,  0), ( 0,  1,  0), ( 0,  0,  1)),
            # +Y face (top)
            (( 0,  1,  0), ( 0,  0,  1), ( 1,  0,  0)),
            # -Y face (bottom)
            (( 0, -1,  0), ( 0,  0, -1), ( 1,  0,  0)),
            # +Z face (front)
            (( 0,  0,  1), ( 0,  1,  0), ( 1,  0,  0)),
            # -Z face (back)
            (( 0,  0, -1), ( 0,  1,  0), (-1,  0,  0)),
        ]
    
    def generate_quadtree(self) -> List[QuadtreeNode]:
        """
        Generate complete quadtree for all 6 cube faces
        
        Returns:
            List of all leaf nodes ready for mesh generation
        """
        print(f"🌳 Generating quadtree: max_depth={self.max_depth}, chunk_res={self.chunk_res}")
        
        all_leaves = []
        
        # Process each cube face
        for face_id in range(6):
            print(f"   Processing face {face_id}...")
            
            # Create root node for this face
            root = QuadtreeNode(
                level=0,
                face_id=face_id,
                uv_min=(0.0, 0.0),
                uv_max=(1.0, 1.0)
            )
            
            # Recursively subdivide to target depth
            leaves = self._subdivide_recursive(root, self.max_depth)
            all_leaves.extend(leaves)
        
        print(f"✅ Generated {len(all_leaves)} leaf chunks ({len(all_leaves)//6} per face)")
        return all_leaves
    
    def _subdivide_recursive(self, node: QuadtreeNode, remaining_depth: int) -> List[QuadtreeNode]:
        """
        Recursively subdivide node to target depth
        
        Args:
            node: Current node to subdivide
            remaining_depth: Remaining subdivision depth
            
        Returns:
            List of leaf nodes from this subtree
        """
        if remaining_depth <= 0:
            # Mark as leaf and return
            node.is_leaf = True
            return [node]
        
        # Subdivide into 4 children
        children = node.subdivide()
        
        # Recursively subdivide children
        leaves = []
        for child in children:
            leaves.extend(self._subdivide_recursive(child, remaining_depth - 1))
        
        return leaves
    
    def generate_chunk_mesh(self, node: QuadtreeNode) -> Dict[str, Any]:
        """
        Generate mesh data for a single chunk
        
        Args:
            node: Quadtree leaf node to generate mesh for
            
        Returns:
            Dictionary containing mesh data (positions, normals, uvs, indices)
        """
        if not node.is_leaf:
            raise ValueError("Can only generate mesh for leaf nodes")
        
        face_id = node.face_id
        normal, up, right = self.face_configs[face_id]
        
        vertices = []
        uvs = []
        indices = []
        
        # Generate grid within UV bounds
        res = self.chunk_res
        u_min, v_min = node.uv_min
        u_max, v_max = node.uv_max
        
        # Create vertices
        for j in range(res):
            for i in range(res):
                # Local grid coordinates
                local_u = i / (res - 1)
                local_v = j / (res - 1)
                
                # Map to chunk UV bounds
                u = u_min + local_u * (u_max - u_min)
                v = v_min + local_v * (v_max - v_min)
                
                # Convert to [-1,1] cube coordinates
                cube_u = u * 2 - 1
                cube_v = v * 2 - 1
                
                # Calculate position on cube face
                pos = np.array(normal) + cube_u * np.array(right) + cube_v * np.array(up)
                
                # Project to unit sphere
                sphere_pos = pos / np.linalg.norm(pos)
                
                # Apply displacement if heightfield available
                if self.heightfield is not None and self.displacement_scale > 0:
                    height_offset = self.heightfield.sample(sphere_pos[0], sphere_pos[1], sphere_pos[2])
                    displaced_radius = self.base_radius + height_offset * self.displacement_scale
                    final_pos = sphere_pos * displaced_radius
                else:
                    final_pos = sphere_pos * self.base_radius
                
                vertices.append(final_pos)
                
                # Texture coordinates (face-relative)
                uvs.append((u, v))
        
        # Generate indices
        for j in range(res - 1):
            for i in range(res - 1):
                # Quad corners
                bl = j * res + i           # bottom-left
                br = j * res + (i + 1)     # bottom-right
                tl = (j + 1) * res + i     # top-left
                tr = (j + 1) * res + (i + 1)  # top-right
                
                # Two triangles per quad
                indices.extend([bl, br, tl])
                indices.extend([br, tr, tl])
        
        # Convert to numpy arrays
        positions = np.array(vertices, dtype=np.float32)
        uvs = np.array(uvs, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Compute normals and tangents
        normals = compute_angle_weighted_normals(positions, indices)
        tangents, bitangents = compute_tangent_basis(positions, normals, uvs, indices)
        
        # Validate shading basis
        validate_shading_basis(normals, tangents, bitangents)
        
        return {
            "positions": positions,
            "normals": normals,
            "tangents": tangents,
            "bitangents": bitangents,
            "uvs": uvs,
            "indices": indices,
            "node": node
        }
    
    def compute_chunk_aabb(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute axis-aligned bounding box for chunk
        
        Args:
            mesh_data: Mesh data containing positions
            
        Returns:
            AABB dictionary with min, max, center, size
        """
        positions = mesh_data["positions"]
        
        aabb_min = np.min(positions, axis=0)
        aabb_max = np.max(positions, axis=0)
        aabb_center = (aabb_min + aabb_max) * 0.5
        aabb_size = aabb_max - aabb_min
        
        return {
            "min": aabb_min.tolist(),
            "max": aabb_max.tolist(), 
            "center": aabb_center.tolist(),
            "size": aabb_size.tolist()
        }
    
    def export_chunk_manifest(self, mesh_data: Dict[str, Any], output_dir: Path) -> Path:
        """
        Export single chunk as manifest + binary buffers
        
        Args:
            mesh_data: Generated mesh data
            output_dir: Output directory for files
            
        Returns:
            Path to generated manifest file
        """
        node = mesh_data["node"]
        chunk_id = node.chunk_id
        
        # Create output paths
        manifest_path = output_dir / f"{chunk_id}.json"
        
        # Compute AABB for this chunk
        aabb = self.compute_chunk_aabb(mesh_data)
        
        # Export using standard format but add chunk metadata
        manifest = export_manifest(mesh_data, manifest_path, output_dir)
        
        # Add chunk-specific metadata
        manifest["chunk"] = {
            "chunk_id": chunk_id,
            "face_id": node.face_id,
            "level": node.level,
            "uv_bounds": {
                "min": list(node.uv_min),
                "max": list(node.uv_max),
                "center": list(node.uv_center),
                "size": list(node.uv_size)
            },
            "aabb": aabb,
            "resolution": self.chunk_res
        }
        
        # Update metadata
        if "metadata" not in manifest:
            manifest["metadata"] = {}
        
        manifest["metadata"]["chunk_type"] = "quadtree_leaf"
        manifest["metadata"]["displacement_scale"] = self.displacement_scale
        
        # Rewrite manifest with chunk data
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path


def generate_chunked_planet(max_depth: int = 3, chunk_res: int = 16, 
                          output_dir: Path = Path("chunks"),
                          heightfield: Optional[HeightField] = None,
                          displacement_scale: float = 0.0,
                          base_radius: float = 1.0) -> Dict[str, Any]:
    """
    Generate complete chunked planet with quadtree subdivision
    
    Args:
        max_depth: Maximum quadtree depth
        chunk_res: Resolution per chunk
        output_dir: Output directory for chunk files
        heightfield: Optional heightfield for terrain
        displacement_scale: Displacement scaling
        base_radius: Base sphere radius
        
    Returns:
        Summary dictionary with generation stats
    """
    print(f"🪐 Generating chunked planet: depth={max_depth}, chunk_res={chunk_res}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize chunker
    chunker = QuadtreeChunker(
        max_depth=max_depth,
        chunk_res=chunk_res,
        base_radius=base_radius,
        heightfield=heightfield,
        displacement_scale=displacement_scale
    )
    
    # Generate quadtree
    leaf_nodes = chunker.generate_quadtree()
    
    # Generate and export each chunk
    chunk_manifests = []
    total_vertices = 0
    total_triangles = 0
    
    print(f"🔧 Generating {len(leaf_nodes)} chunk meshes...")
    
    for i, node in enumerate(leaf_nodes):
        print(f"   Chunk {i+1}/{len(leaf_nodes)}: {node.chunk_id}")
        
        # Generate mesh for this chunk
        mesh_data = chunker.generate_chunk_mesh(node)
        
        # Export chunk manifest
        manifest_path = chunker.export_chunk_manifest(mesh_data, output_dir)
        chunk_manifests.append(str(manifest_path.name))
        
        # Accumulate stats
        total_vertices += len(mesh_data["positions"])
        total_triangles += len(mesh_data["indices"]) // 3
    
    # Create master manifest listing all chunks
    master_manifest = {
        "planet": {
            "type": "chunked_quadtree",
            "max_depth": max_depth,
            "chunk_resolution": chunk_res,
            "total_chunks": len(chunk_manifests),
            "chunks_per_face": len(chunk_manifests) // 6,
            "base_radius": base_radius,
            "displacement_scale": displacement_scale,
            "has_terrain": heightfield is not None
        },
        "chunks": chunk_manifests,
        "statistics": {
            "total_vertices": total_vertices,
            "total_triangles": total_triangles,
            "vertices_per_chunk": total_vertices // len(chunk_manifests),
            "triangles_per_chunk": total_triangles // len(chunk_manifests)
        },
        "metadata": {
            "generator": "quadtree_chunking.py",
            "version": "T06",
            "format": "quadtree_chunks"
        }
    }
    
    # Write master manifest
    master_path = output_dir / "planet_chunks.json"
    with open(master_path, 'w') as f:
        json.dump(master_manifest, f, indent=2)
    
    print(f"✅ Chunked planet generation complete!")
    print(f"📊 Generated {len(chunk_manifests)} chunks")
    print(f"📊 Total vertices: {total_vertices}")
    print(f"📊 Total triangles: {total_triangles}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Master manifest: {master_path}")
    
    return master_manifest


def main():
    """Command-line interface for quadtree chunking"""
    parser = argparse.ArgumentParser(description="Generate quadtree chunks for Agent D")
    parser.add_argument("--max_depth", type=int, default=3,
                       help="Maximum quadtree depth")
    parser.add_argument("--chunk_res", type=int, default=16,
                       help="Grid resolution per chunk")
    parser.add_argument("--output", type=str, default="chunks",
                       help="Output directory for chunk files")
    parser.add_argument("--base_radius", type=float, default=1.0,
                       help="Base sphere radius")
    parser.add_argument("--displacement_scale", type=float, default=0.0,
                       help="Heightfield displacement scale")
    parser.add_argument("--terrain_seed", type=int, default=42,
                       help="Terrain generation seed")
    parser.add_argument("--terrain_spec", type=str, default=None,
                       help="Path to terrain specification JSON")
    
    args = parser.parse_args()
    
    # Create heightfield if displacement requested
    heightfield = None
    if args.displacement_scale > 0.0 or args.terrain_spec:
        if not TERRAIN_AVAILABLE:
            print("❌ Terrain system not available for heightfield displacement")
            return
        
        if args.terrain_spec and Path(args.terrain_spec).exists():
            print(f"🏔️ Loading terrain: {args.terrain_spec}")
            heightfield = create_heightfield_from_pcc(args.terrain_spec, args.terrain_seed)
        else:
            print(f"🏔️ Creating default terrain with seed {args.terrain_seed}")
            # Simple terrain for testing
            terrain_spec = {
                "terrain": {
                    "heightfield": {"base_height": 0.0, "height_scale": 0.25},
                    "nodes": {
                        "type": "RidgedMF",
                        "seed": 1000,
                        "octaves": 4,
                        "frequency": 2.0,
                        "lacunarity": 2.1,
                        "persistence": 0.6,
                        "ridge_offset": 1.0
                    }
                }
            }
            heightfield = create_heightfield_from_pcc(terrain_spec, args.terrain_seed)
    
    # Generate chunked planet
    output_dir = Path(args.output)
    master_manifest = generate_chunked_planet(
        max_depth=args.max_depth,
        chunk_res=args.chunk_res,
        output_dir=output_dir,
        heightfield=heightfield,
        displacement_scale=args.displacement_scale,
        base_radius=args.base_radius
    )


if __name__ == "__main__":
    main()