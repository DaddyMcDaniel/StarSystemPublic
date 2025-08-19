#!/usr/bin/env python3
"""
Cube-Sphere Primitive Generator for Agent D - T02
=================================================

Generates a uniform cube-sphere with shared vertices and seam-aware UVs.
Implements the terrain pipeline requirements from HOP_terrain specification.

Features:
- 6 face grids at configurable resolution (N×N)
- Shared edge vertices for seamless geometry
- Seam-safe UV mapping per face (no polar pinch)
- Triangle indices (two triangles per quad)
- Export to manifest format with binary buffers

Usage:
    python cubesphere.py --face_res 32 --output sphere_manifest.json
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
import os

# Import shading basis computation
from shading_basis import compute_angle_weighted_normals, compute_tangent_basis, validate_shading_basis

# Import terrain heightfield system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'terrain'))
try:
    from heightfield import create_heightfield_from_pcc, create_example_terrain_stack, HeightField
    from noise_nodes import NoiseFBM, RidgedMF, DomainWarp
    TERRAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Terrain system not available: {e}")
    TERRAIN_AVAILABLE = False


class CubeSphereGenerator:
    """
    Generates cube-sphere primitives with shared vertices and seam-aware UVs
    Supports heightfield displacement for terrain generation
    """
    
    def __init__(self, face_res: int = 32, base_radius: float = 1.0, 
                 heightfield: Optional[HeightField] = None, displacement_scale: float = 0.1):
        """
        Initialize cube-sphere generator
        
        Args:
            face_res: Resolution per face (N×N grid)
            base_radius: Base radius of the sphere before displacement
            heightfield: Optional heightfield for terrain displacement
            displacement_scale: Scale factor for heightfield displacement
        """
        self.face_res = face_res
        self.base_radius = base_radius
        self.heightfield = heightfield
        self.displacement_scale = displacement_scale
        self.vertices = []
        self.uvs = []
        self.indices = []
        self.vertex_map = {}  # For shared vertex detection
        
    def generate(self) -> Dict[str, Any]:
        """
        Generate complete cube-sphere mesh with optional heightfield displacement
        
        Returns:
            Dictionary containing positions, normals, uvs, indices arrays
        """
        print(f"🔮 Generating cube-sphere with face resolution {self.face_res}×{self.face_res}")
        
        # Generate 6 faces with shared vertices
        self._generate_all_faces()
        
        # Convert to numpy arrays
        positions = np.array(self.vertices, dtype=np.float32)
        uvs = np.array(self.uvs, dtype=np.float32)
        indices = np.array(self.indices, dtype=np.uint32)
        
        print(f"✅ Generated {len(positions)} vertices, {len(indices)//3} triangles")
        print(f"📊 Shared vertices: {(self.face_res * self.face_res * 6) - len(positions)} vertices saved")
        
        # Apply heightfield displacement if available
        if self.heightfield is not None:
            print(f"🏔️ Applying heightfield displacement (scale: {self.displacement_scale})...")
            displaced_positions = np.zeros_like(positions)
            
            for i, pos in enumerate(positions):
                # Normalize position to unit sphere (should already be normalized)
                unit_pos = pos / np.linalg.norm(pos)
                
                # Sample heightfield at unit sphere position
                height_offset = self.heightfield.sample(unit_pos[0], unit_pos[1], unit_pos[2])
                
                # Apply displacement: p' = normalize(p) * (radius + h * scale)
                displaced_radius = self.base_radius + height_offset * self.displacement_scale
                displaced_positions[i] = unit_pos * displaced_radius
            
            positions = displaced_positions
            print(f"✅ Displacement applied to {len(positions)} vertices")
        else:
            # Scale to base radius without displacement
            positions = positions * self.base_radius
            print(f"📏 Scaled vertices to radius {self.base_radius}")
        
        # Recompute normals after displacement using angle-weighted method
        print("🔢 Computing angle-weighted normals...")
        normals = compute_angle_weighted_normals(positions, indices)
        
        # Compute tangent basis for normal mapping
        print("🔢 Computing tangent basis...")
        tangents, bitangents = compute_tangent_basis(positions, normals, uvs, indices)
        
        # Validate shading basis
        validate_shading_basis(normals, tangents, bitangents)
        
        print("✅ Shading basis computation complete")
        
        return {
            "positions": positions,
            "normals": normals,
            "tangents": tangents,
            "bitangents": bitangents,
            "uvs": uvs,
            "indices": indices
        }
    
    def _generate_all_faces(self):
        """Generate all 6 cube faces with proper orientation"""
        
        # Define cube face configurations (normal, up, right vectors)
        faces = [
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
        
        for face_id, (normal, up, right) in enumerate(faces):
            self._generate_face(face_id, normal, up, right)
    
    def _generate_face(self, face_id: int, normal: Tuple[float, float, float], 
                      up: Tuple[float, float, float], right: Tuple[float, float, float]):
        """
        Generate a single face grid with shared edge vertices
        
        Args:
            face_id: Face identifier (0-5)
            normal: Face normal vector
            up: Face up vector  
            right: Face right vector
        """
        n = self.face_res
        face_vertices = []
        
        # Generate grid vertices for this face
        for j in range(n):
            for i in range(n):
                # UV coordinates in [0,1] range for this face
                u = i / (n - 1)
                v = j / (n - 1)
                
                # Convert to [-1,1] cube coordinates
                cube_u = u * 2 - 1
                cube_v = v * 2 - 1
                
                # Calculate position on cube face
                pos = np.array(normal) + cube_u * np.array(right) + cube_v * np.array(up)
                
                # Project to unit sphere
                sphere_pos = pos / np.linalg.norm(pos)
                
                # Check if this vertex is shared with another face
                vertex_key = self._get_vertex_key(sphere_pos)
                
                if vertex_key in self.vertex_map:
                    # Reuse existing vertex
                    vertex_idx = self.vertex_map[vertex_key]
                    face_vertices.append(vertex_idx)
                else:
                    # Create new vertex
                    vertex_idx = len(self.vertices)
                    self.vertices.append(sphere_pos)
                    
                    # Compute seam-safe UV for this face
                    face_uv = self._compute_face_uv(face_id, u, v)
                    self.uvs.append(face_uv)
                    
                    self.vertex_map[vertex_key] = vertex_idx
                    face_vertices.append(vertex_idx)
        
        # Generate triangle indices for this face
        self._generate_face_indices(face_vertices, n)
    
    def _get_vertex_key(self, pos: np.ndarray, epsilon: float = 1e-6) -> Tuple[int, int, int]:
        """
        Get a quantized key for vertex sharing detection
        
        Args:
            pos: Vertex position
            epsilon: Quantization epsilon
            
        Returns:
            Quantized position tuple for dictionary key
        """
        scale = 1.0 / epsilon
        return (
            int(pos[0] * scale),
            int(pos[1] * scale), 
            int(pos[2] * scale)
        )
    
    def _compute_face_uv(self, face_id: int, u: float, v: float) -> Tuple[float, float]:
        """
        Compute seam-safe UV coordinates for a face
        
        Args:
            face_id: Face identifier (0-5)
            u, v: Local face coordinates in [0,1]
            
        Returns:
            UV coordinates for texture mapping
        """
        # Simple face-based UV mapping to avoid seams
        # Each face gets a region in UV space
        face_u_offset = (face_id % 3) / 3.0
        face_v_offset = (face_id // 3) / 2.0
        
        final_u = face_u_offset + u / 3.0
        final_v = face_v_offset + v / 2.0
        
        return (final_u, final_v)
    
    def _generate_face_indices(self, face_vertices: List[int], n: int):
        """
        Generate triangle indices for a face grid
        
        Args:
            face_vertices: List of vertex indices for this face
            n: Grid resolution
        """
        for j in range(n - 1):
            for i in range(n - 1):
                # Quad corners
                bl = face_vertices[j * n + i]           # bottom-left
                br = face_vertices[j * n + (i + 1)]     # bottom-right  
                tl = face_vertices[(j + 1) * n + i]     # top-left
                tr = face_vertices[(j + 1) * n + (i + 1)]  # top-right
                
                # Two triangles per quad
                # Triangle 1: bl -> br -> tl
                self.indices.extend([bl, br, tl])
                # Triangle 2: br -> tr -> tl  
                self.indices.extend([br, tr, tl])


def export_manifest(mesh_data: Dict[str, Any], output_path: Path, buffer_dir: Path = None) -> Dict[str, Any]:
    """
    Export mesh data to manifest format with binary buffers
    
    Args:
        mesh_data: Generated mesh data
        output_path: Output manifest file path
        buffer_dir: Directory for binary buffers (defaults to same as manifest)
        
    Returns:
        Manifest dictionary
    """
    if buffer_dir is None:
        buffer_dir = output_path.parent
    
    buffer_dir.mkdir(parents=True, exist_ok=True)
    
    # Write binary buffers
    positions_file = buffer_dir / f"{output_path.stem}_positions.bin"
    normals_file = buffer_dir / f"{output_path.stem}_normals.bin"
    tangents_file = buffer_dir / f"{output_path.stem}_tangents.bin"
    uvs_file = buffer_dir / f"{output_path.stem}_uvs.bin"
    indices_file = buffer_dir / f"{output_path.stem}_indices.bin"
    
    mesh_data["positions"].tofile(positions_file)
    mesh_data["normals"].tofile(normals_file)
    mesh_data["tangents"].tofile(tangents_file)
    mesh_data["uvs"].tofile(uvs_file)
    mesh_data["indices"].tofile(indices_file)
    
    # Calculate bounds
    positions = mesh_data["positions"].reshape(-1, 3)
    center = np.mean(positions, axis=0)
    max_dist = np.max(np.linalg.norm(positions - center, axis=1))
    
    # Create manifest
    manifest = {
        "mesh": {
            "primitive_topology": "triangles",
            "positions": f"buffer://{positions_file.name}",
            "normals": f"buffer://{normals_file.name}",
            "tangents": f"buffer://{tangents_file.name}",
            "uv0": f"buffer://{uvs_file.name}",
            "indices": f"buffer://{indices_file.name}",
            "bounds": {
                "center": center.tolist(),
                "radius": float(max_dist)
            }
        },
        "metadata": {
            "generator": "cubesphere.py",
            "face_resolution": len(mesh_data["positions"]) // 6,
            "vertex_count": len(mesh_data["positions"]),
            "triangle_count": len(mesh_data["indices"]) // 3,
            "shading_basis": "angle_weighted_normals_mikktspace_tangents",
            "displacement": "heightfield_terrain" if "displacement_scale" in str(mesh_data) else "none"
        }
    }
    
    # Write manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📁 Exported manifest: {output_path}")
    print(f"📁 Binary buffers in: {buffer_dir}")
    
    return manifest


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Generate cube-sphere primitive for Agent D")
    parser.add_argument("--face_res", type=int, default=32, 
                       help="Resolution per face (N×N grid)")
    parser.add_argument("--output", type=str, default="cubesphere_manifest.json",
                       help="Output manifest file path")
    parser.add_argument("--buffer_dir", type=str, default=None,
                       help="Directory for binary buffers")
    
    # Terrain parameters for T05 displacement
    parser.add_argument("--radius", type=float, default=1.0,
                       help="Base sphere radius")
    parser.add_argument("--displacement_scale", type=float, default=0.0,
                       help="Heightfield displacement scale (0.0 = no displacement)")
    parser.add_argument("--terrain_seed", type=int, default=42,
                       help="Terrain generation seed")
    parser.add_argument("--terrain_frequency", type=float, default=2.0,
                       help="Base terrain frequency")
    parser.add_argument("--terrain_amplitude", type=float, default=0.25,
                       help="Terrain amplitude scaling")
    parser.add_argument("--terrain_spec", type=str, default=None,
                       help="Path to terrain specification JSON file")
    
    args = parser.parse_args()
    
    # Create heightfield if displacement is requested
    heightfield = None
    if args.displacement_scale > 0.0 or args.terrain_spec:
        if not TERRAIN_AVAILABLE:
            print("❌ Terrain system not available for heightfield displacement")
            return
        
        if args.terrain_spec and Path(args.terrain_spec).exists():
            print(f"🏔️ Loading terrain from specification: {args.terrain_spec}")
            heightfield = create_heightfield_from_pcc(args.terrain_spec, args.terrain_seed)
        else:
            print(f"🏔️ Creating example terrain with seed {args.terrain_seed}")
            # Create simple terrain specification
            terrain_spec = {
                "terrain": {
                    "heightfield": {
                        "base_height": 0.0,
                        "height_scale": args.terrain_amplitude
                    },
                    "nodes": {
                        "type": "Composite",
                        "operation": "add", 
                        "weights": [1.0, 0.3],
                        "nodes": [
                            {
                                "type": "RidgedMF",
                                "seed": 1000,
                                "octaves": 4,
                                "frequency": args.terrain_frequency,
                                "lacunarity": 2.1,
                                "persistence": 0.6,
                                "ridge_offset": 1.0
                            },
                            {
                                "type": "NoiseFBM",
                                "seed": 2000,
                                "octaves": 3,
                                "frequency": args.terrain_frequency * 4.0,
                                "lacunarity": 2.0,
                                "persistence": 0.4
                            }
                        ]
                    }
                }
            }
            heightfield = create_heightfield_from_pcc(terrain_spec, args.terrain_seed)
    
    # Generate cube-sphere with optional displacement
    generator = CubeSphereGenerator(
        face_res=args.face_res,
        base_radius=args.radius,
        heightfield=heightfield,
        displacement_scale=args.displacement_scale
    )
    mesh_data = generator.generate()
    
    # Export to manifest format
    output_path = Path(args.output)
    buffer_dir = Path(args.buffer_dir) if args.buffer_dir else None
    
    manifest = export_manifest(mesh_data, output_path, buffer_dir)
    
    print(f"✅ Cube-sphere generation complete!")
    print(f"📋 Face resolution: {args.face_res}×{args.face_res}")
    print(f"📋 Base radius: {args.radius}")
    print(f"📋 Displacement scale: {args.displacement_scale}")
    print(f"📋 Total vertices: {len(mesh_data['positions'])}")
    print(f"📋 Total triangles: {len(mesh_data['indices']) // 3}")
    
    if heightfield:
        print(f"🏔️ Terrain applied with seed {args.terrain_seed}")
        print(f"🏔️ Terrain frequency: {args.terrain_frequency}")
        print(f"🏔️ Terrain amplitude: {args.terrain_amplitude}")


if __name__ == "__main__":
    main()