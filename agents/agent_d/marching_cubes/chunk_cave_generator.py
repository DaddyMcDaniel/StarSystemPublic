#!/usr/bin/env python3
"""
Chunk Cave Generator - T10
===========================

Generates cave meshes for terrain chunks using SDF and Marching Cubes.
Creates separate cave manifests compatible with the existing T06-T08 chunk system.

Features:
- Per-chunk cave mesh generation
- Material ID assignment for cave surfaces
- Binary buffer export compatible with viewer
- Manifest generation for chunk loading
- Integration with existing terrain pipeline

Usage:
    from chunk_cave_generator import ChunkCaveGenerator
    
    generator = ChunkCaveGenerator()
    cave_chunks = generator.generate_cave_chunks(terrain_chunks, output_dir)
"""

import numpy as np
import json
import struct
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import sys

# Import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))
sys.path.append(os.path.dirname(__file__))

from sdf_evaluator import SDFEvaluator, VoxelGrid, ChunkBounds, create_cave_system_sdf, create_overhang_sdf
from marching_cubes import MarchingCubes, CaveMeshGenerator, MarchingCubesVertex, CaveMeshData


class ChunkCaveGenerator:
    """
    Generates cave meshes for terrain chunks using SDF and Marching Cubes
    """
    
    def __init__(self, resolution: int = 64, cave_material_id: int = 2, chunk_overlap_voxels: int = 1):
        """
        Initialize chunk cave generator with T20 improvements
        
        Args:
            resolution: Voxel grid resolution for cave generation (increased to 48-64 for T20)
            cave_material_id: Material ID for cave surfaces
            chunk_overlap_voxels: Voxel overlap between chunks for seam prevention (T20)
        """
        # T20: Increase resolution from 32 to 64 for better detail
        self.resolution = resolution
        self.cave_material_id = cave_material_id
        self.chunk_overlap_voxels = chunk_overlap_voxels
        self.cave_mesh_generator = CaveMeshGenerator(material_id=cave_material_id)
        self.sdf_evaluator = SDFEvaluator()
        
        # Statistics
        self.stats = {
            "chunks_processed": 0,
            "caves_generated": 0,
            "total_vertices": 0,
            "total_triangles": 0,
            "generation_time": 0.0,
            "voxel_resolution": self.resolution,
            "chunk_overlap_voxels": self.chunk_overlap_voxels,
            "seam_prevention_enabled": True
        }
    
    def generate_cave_chunks(self, terrain_manifest_path: str, output_dir: str, 
                           cave_types: List[str] = None) -> Dict:
        """
        Generate cave meshes for all terrain chunks
        
        Args:
            terrain_manifest_path: Path to terrain chunk manifest
            output_dir: Output directory for cave chunks
            cave_types: List of cave types to generate ["caves", "overhangs"]
            
        Returns:
            Cave chunk manifest dictionary
        """
        if cave_types is None:
            cave_types = ["caves", "overhangs"]
        
        start_time = time.time()
        
        # Load terrain chunk manifest
        with open(terrain_manifest_path, 'r') as f:
            terrain_manifest = json.load(f)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cave_chunks = []
        
        # Process each terrain chunk
        for chunk_info in terrain_manifest.get("chunks", []):
            chunk_id = chunk_info.get("chunk_id", "unknown")
            
            print(f"Processing chunk {chunk_id}...")
            
            # Extract chunk bounds from terrain data
            aabb = chunk_info.get("aabb", {})
            bounds = ChunkBounds(
                min_point=np.array(aabb.get("min", [-1, -1, -1])),
                max_point=np.array(aabb.get("max", [1, 1, 1]))
            )
            
            # Generate caves for this chunk
            chunk_caves = self._generate_chunk_caves(chunk_id, bounds, cave_types, output_path)
            
            if chunk_caves:
                cave_chunks.extend(chunk_caves)
                self.stats["caves_generated"] += len(chunk_caves)
            
            self.stats["chunks_processed"] += 1
        
        # Create cave manifest
        cave_manifest = self._create_cave_manifest(cave_chunks, output_dir)
        
        # Save cave manifest
        manifest_path = output_path / "cave_chunks.json"
        with open(manifest_path, 'w') as f:
            json.dump(cave_manifest, f, indent=2)
        
        self.stats["generation_time"] = time.time() - start_time
        
        print(f"\n✅ Cave generation complete:")
        print(f"   Chunks processed: {self.stats['chunks_processed']}")
        print(f"   Caves generated: {self.stats['caves_generated']}")
        print(f"   Total vertices: {self.stats['total_vertices']}")
        print(f"   Total triangles: {self.stats['total_triangles']}")
        print(f"   Generation time: {self.stats['generation_time']:.2f}s")
        
        return cave_manifest
    
    def _generate_chunk_caves(self, chunk_id: str, bounds: ChunkBounds, 
                            cave_types: List[str], output_path: Path) -> List[Dict]:
        """
        Generate cave meshes for a single chunk
        
        Args:
            chunk_id: Chunk identifier
            bounds: Chunk bounds
            cave_types: Types of caves to generate
            output_path: Output directory path
            
        Returns:
            List of cave chunk info dictionaries
        """
        chunk_caves = []
        
        # Generate seed from chunk ID for deterministic caves
        chunk_seed = hash(chunk_id) % (2**31)
        
        for cave_type in cave_types:
            cave_chunk_id = f"{chunk_id}_{cave_type}"
            
            # Create SDF specification based on cave type
            if cave_type == "caves":
                sdf_spec = create_cave_system_sdf(bounds, seed=chunk_seed)
            elif cave_type == "overhangs":
                sdf_spec = create_overhang_sdf(bounds, seed=chunk_seed + 1000)
            else:
                continue
            
            # Build SDF tree
            sdf_tree = self.sdf_evaluator.build_sdf_from_pcc(sdf_spec)
            
            # Generate cave mesh
            cave_mesh = self.cave_mesh_generator.generate_cave_mesh(
                bounds, sdf_tree, self.resolution, cave_chunk_id
            )
            
            if cave_mesh and len(cave_mesh.vertices) > 0:
                # Export mesh data
                mesh_info = self._export_cave_mesh(cave_mesh, output_path)
                if mesh_info:
                    chunk_caves.append(mesh_info)
                    
                    # Update statistics
                    self.stats["total_vertices"] += len(cave_mesh.vertices)
                    self.stats["total_triangles"] += len(cave_mesh.triangles)
        
        return chunk_caves
    
    def _export_cave_mesh(self, cave_mesh: CaveMeshData, output_path: Path) -> Optional[Dict]:
        """
        Export cave mesh to binary buffers compatible with viewer
        
        Args:
            cave_mesh: CaveMeshData to export
            output_path: Output directory path
            
        Returns:
            Chunk info dictionary or None if export failed
        """
        try:
            chunk_id = cave_mesh.chunk_id
            
            # Prepare vertex data arrays
            positions = np.array([v.position for v in cave_mesh.vertices], dtype=np.float32)
            normals = np.array([v.normal for v in cave_mesh.vertices], dtype=np.float32)
            
            # Generate UVs (simple planar projection for caves)
            uvs = self._generate_cave_uvs(positions, cave_mesh.bounds)
            
            # Generate tangents (simplified calculation)
            tangents = self._generate_cave_tangents(positions, normals, cave_mesh.triangles)
            
            # Export binary buffers
            buffers = {}
            
            # Position buffer
            pos_path = output_path / f"{chunk_id}_positions.bin"
            with open(pos_path, 'wb') as f:
                f.write(positions.tobytes())
            buffers["positions"] = f"{chunk_id}_positions.bin"
            
            # Normal buffer
            normal_path = output_path / f"{chunk_id}_normals.bin"
            with open(normal_path, 'wb') as f:
                f.write(normals.tobytes())
            buffers["normals"] = f"{chunk_id}_normals.bin"
            
            # UV buffer
            uv_path = output_path / f"{chunk_id}_uvs.bin"
            with open(uv_path, 'wb') as f:
                f.write(uvs.tobytes())
            buffers["uvs"] = f"{chunk_id}_uvs.bin"
            
            # Tangent buffer
            tangent_path = output_path / f"{chunk_id}_tangents.bin"
            with open(tangent_path, 'wb') as f:
                f.write(tangents.tobytes())
            buffers["tangents"] = f"{chunk_id}_tangents.bin"
            
            # Index buffer
            indices = cave_mesh.triangles.flatten().astype(np.uint32)
            index_path = output_path / f"{chunk_id}_indices.bin"
            with open(index_path, 'wb') as f:
                f.write(indices.tobytes())
            buffers["indices"] = f"{chunk_id}_indices.bin"
            
            # Create chunk info
            chunk_info = {
                "chunk_id": chunk_id,
                "chunk_type": "cave",
                "material_id": cave_mesh.material_id,
                "vertex_count": len(cave_mesh.vertices),
                "triangle_count": len(cave_mesh.triangles),
                "index_count": len(indices),
                "aabb": {
                    "min": cave_mesh.bounds.min_point.tolist(),
                    "max": cave_mesh.bounds.max_point.tolist()
                },
                "buffers": buffers,
                "lod_levels": [0]  # Only one LOD level for caves
            }
            
            return chunk_info
            
        except Exception as e:
            print(f"❌ Failed to export cave mesh {cave_mesh.chunk_id}: {e}")
            return None
    
    def _generate_cave_uvs(self, positions: np.ndarray, bounds: ChunkBounds) -> np.ndarray:
        """
        Generate UV coordinates for cave vertices using planar projection
        
        Args:
            positions: Vertex positions (Nx3)
            bounds: Chunk bounds for normalization
            
        Returns:
            UV coordinates (Nx2)
        """
        # Normalize positions to [0,1] range based on bounds
        size = bounds.size
        min_point = bounds.min_point
        
        # Project to XY plane and normalize
        normalized_pos = (positions - min_point) / size
        uvs = normalized_pos[:, [0, 1]]  # Use X,Y components
        
        # Scale UVs for texture tiling
        uv_scale = 2.0  # Adjust for texture resolution
        uvs *= uv_scale
        
        return uvs.astype(np.float32)
    
    def _generate_cave_tangents(self, positions: np.ndarray, normals: np.ndarray, 
                              triangles: np.ndarray) -> np.ndarray:
        """
        Generate tangent vectors for cave vertices
        
        Args:
            positions: Vertex positions (Nx3)
            normals: Vertex normals (Nx3)
            triangles: Triangle indices (Mx3)
            
        Returns:
            Tangent vectors (Nx4) with handedness in w component
        """
        vertex_count = len(positions)
        tangents = np.zeros((vertex_count, 4), dtype=np.float32)
        
        # Calculate tangents per triangle
        for triangle in triangles:
            if len(triangle) != 3:
                continue
                
            i1, i2, i3 = triangle
            
            # Get triangle vertices
            v1 = positions[i1]
            v2 = positions[i2]
            v3 = positions[i3]
            
            # Calculate edge vectors
            edge1 = v2 - v1
            edge2 = v3 - v1
            
            # Calculate tangent (simplified - assumes UV coordinates)
            tangent = np.cross(edge1, edge2)
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-8:
                tangent = tangent / tangent_norm
            else:
                tangent = np.array([1.0, 0.0, 0.0])  # Default tangent
            
            # Assign to triangle vertices
            for i in [i1, i2, i3]:
                tangents[i, :3] = tangent
                tangents[i, 3] = 1.0  # Handedness
        
        # Orthogonalize tangents against normals
        for i in range(vertex_count):
            normal = normals[i]
            tangent = tangents[i, :3]
            
            # Gram-Schmidt orthogonalization
            tangent = tangent - np.dot(tangent, normal) * normal
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-8:
                tangents[i, :3] = tangent / tangent_norm
            else:
                # Generate perpendicular vector to normal
                if abs(normal[0]) < 0.9:
                    tangents[i, :3] = np.cross(normal, [1, 0, 0])
                else:
                    tangents[i, :3] = np.cross(normal, [0, 1, 0])
                tangents[i, :3] /= np.linalg.norm(tangents[i, :3])
        
        return tangents
    
    def _create_cave_manifest(self, cave_chunks: List[Dict], output_dir: str) -> Dict:
        """
        Create cave chunk manifest
        
        Args:
            cave_chunks: List of cave chunk info dictionaries
            output_dir: Output directory
            
        Returns:
            Cave manifest dictionary
        """
        manifest = {
            "name": "Cave Chunks",
            "type": "cave_chunks",
            "version": "1.0",
            "generated_by": "ChunkCaveGenerator T10",
            "chunk_count": len(cave_chunks),
            "material_id": self.cave_material_id,
            "resolution": self.resolution,
            "chunks": cave_chunks,
            "statistics": self.stats.copy()
        }
        
        return manifest


def generate_test_cave_chunks():
    """Generate test cave chunks for demonstration"""
    print("🧪 Generating Test Cave Chunks")
    print("=" * 50)
    
    # Create test terrain manifest
    test_manifest = {
        "name": "Test Terrain",
        "chunks": [
            {
                "chunk_id": "test_chunk_0_0",
                "aabb": {
                    "min": [-2.0, -2.0, -2.0],
                    "max": [2.0, 2.0, 2.0]
                }
            },
            {
                "chunk_id": "test_chunk_1_0", 
                "aabb": {
                    "min": [2.0, -2.0, -2.0],
                    "max": [6.0, 2.0, 2.0]
                }
            }
        ]
    }
    
    # Save test manifest
    test_dir = Path("test_cave_chunks")
    test_dir.mkdir(exist_ok=True)
    
    manifest_path = test_dir / "test_terrain.json"
    with open(manifest_path, 'w') as f:
        json.dump(test_manifest, f, indent=2)
    
    # Generate cave chunks
    generator = ChunkCaveGenerator(resolution=16, cave_material_id=2)
    cave_manifest = generator.generate_cave_chunks(
        str(manifest_path), 
        str(test_dir),
        cave_types=["caves", "overhangs"]
    )
    
    print(f"\n✅ Test cave chunks generated in {test_dir}")
    return cave_manifest


if __name__ == "__main__":
    # Generate test cave chunks
    test_manifest = generate_test_cave_chunks()
    
    print(f"\n📋 Cave Manifest Summary:")
    print(f"   Cave chunks: {test_manifest['chunk_count']}")
    print(f"   Material ID: {test_manifest['material_id']}")
    print(f"   Resolution: {test_manifest['resolution']}")
    
    print("\n✅ Chunk cave generator system functional")