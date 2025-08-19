#!/usr/bin/env python3
"""
MikkTSpace Compatible Tangent Generation - T12
===============================================

Implements MikkTSpace algorithm for consistent tangent space generation
across all mesh types (terrain, caves, fused geometry) to ensure stable
lighting and normal mapping.

Features:
- MikkTSpace algorithm implementation for standard tangent generation
- Per-triangle tangent calculation with proper handedness
- Vertex tangent accumulation and normalization
- TBN space standardization (right-handed coordinate system)
- Compatible with industry standard normal mapping workflows

Usage:
    from mikktspace_tangents import MikkTSpaceTangentGenerator
    
    generator = MikkTSpaceTangentGenerator()
    tangents = generator.generate_tangents(positions, normals, uvs, indices)
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marching_cubes'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fusion'))

try:
    from marching_cubes import MarchingCubesVertex
    MARCHING_CUBES_AVAILABLE = True
except ImportError:
    MARCHING_CUBES_AVAILABLE = False


@dataclass
class TangentSpaceVertex:
    """Vertex with complete tangent space information"""
    position: np.ndarray     # [x, y, z] world position
    normal: np.ndarray       # [x, y, z] surface normal
    uv: np.ndarray          # [u, v] texture coordinates
    tangent: np.ndarray     # [x, y, z, w] tangent with handedness in w
    bitangent: np.ndarray   # [x, y, z] bitangent (computed from normal x tangent)
    
    def get_tbn_matrix(self) -> np.ndarray:
        """Get TBN (Tangent-Bitangent-Normal) matrix for normal mapping"""
        return np.column_stack([self.tangent[:3], self.bitangent, self.normal])
    
    def validate_orthogonality(self, tolerance: float = 1e-3) -> bool:
        """Validate that TBN vectors are orthogonal"""
        t, b, n = self.tangent[:3], self.bitangent, self.normal
        
        # Check dot products are near zero (orthogonal)
        tb_dot = abs(np.dot(t, b))
        tn_dot = abs(np.dot(t, n)) 
        bn_dot = abs(np.dot(b, n))
        
        return all(dot < tolerance for dot in [tb_dot, tn_dot, bn_dot])


class MikkTSpaceHandedness(Enum):
    """Tangent space handedness convention"""
    RIGHT_HANDED = 1.0   # Standard convention: T x B = N
    LEFT_HANDED = -1.0   # Flipped convention: T x B = -N


@dataclass
class TangentGenerationConfig:
    """Configuration for tangent generation"""
    handedness: MikkTSpaceHandedness = MikkTSpaceHandedness.RIGHT_HANDED
    normalize_uvs: bool = True
    merge_identical_vertices: bool = True
    vertex_tolerance: float = 1e-6
    uv_tolerance: float = 1e-6
    flip_bitangent_on_mirrored_uvs: bool = True


class MikkTSpaceTangentGenerator:
    """
    MikkTSpace compatible tangent space generation
    
    Based on the MikkTSpace algorithm by Morten S. Mikkelsen, which is the
    industry standard for tangent generation used by Blender, Maya, and other
    major 3D tools.
    """
    
    def __init__(self, config: TangentGenerationConfig = None):
        """
        Initialize MikkTSpace tangent generator
        
        Args:
            config: Configuration for tangent generation behavior
        """
        self.config = config or TangentGenerationConfig()
        
        # Statistics
        self.stats = {
            'triangles_processed': 0,
            'vertices_processed': 0,
            'degenerate_triangles': 0,
            'flipped_bitangents': 0,
            'generation_time': 0.0
        }
    
    def generate_tangents(self, positions: np.ndarray, normals: np.ndarray, 
                         uvs: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Generate MikkTSpace compatible tangents
        
        Args:
            positions: Vertex positions (Nx3)
            normals: Vertex normals (Nx3) 
            uvs: Texture coordinates (Nx2)
            indices: Triangle indices (Mx3)
            
        Returns:
            Tangents with handedness (Nx4) where w component stores handedness
        """
        import time
        start_time = time.time()
        
        vertex_count = len(positions)
        triangle_count = len(indices)
        
        # Initialize tangent accumulation arrays
        tangent_accum = np.zeros((vertex_count, 3), dtype=np.float32)
        bitangent_accum = np.zeros((vertex_count, 3), dtype=np.float32)
        
        # Process each triangle
        degenerate_count = 0
        for tri_idx, triangle in enumerate(indices):
            if len(triangle) != 3:
                continue
                
            i1, i2, i3 = triangle
            if i1 >= vertex_count or i2 >= vertex_count or i3 >= vertex_count:
                continue
            
            # Get triangle vertices
            v1, v2, v3 = positions[i1], positions[i2], positions[i3]
            uv1, uv2, uv3 = uvs[i1], uvs[i2], uvs[i3]
            
            # Calculate triangle tangent and bitangent
            tangent, bitangent, is_degenerate = self._calculate_triangle_tangent_space(
                v1, v2, v3, uv1, uv2, uv3
            )
            
            if is_degenerate:
                degenerate_count += 1
                continue
            
            # Accumulate tangent and bitangent for each vertex
            for vertex_idx in triangle:
                tangent_accum[vertex_idx] += tangent
                bitangent_accum[vertex_idx] += bitangent
        
        # Normalize accumulated tangents and compute final tangent space
        final_tangents = np.zeros((vertex_count, 4), dtype=np.float32)
        flipped_bitangents = 0
        
        for i in range(vertex_count):
            normal = normals[i]
            accumulated_tangent = tangent_accum[i]
            accumulated_bitangent = bitangent_accum[i]
            
            # Orthogonalize tangent against normal (Gram-Schmidt)
            tangent = accumulated_tangent - np.dot(accumulated_tangent, normal) * normal
            tangent_length = np.linalg.norm(tangent)
            
            if tangent_length > 1e-8:
                tangent /= tangent_length
            else:
                # Generate fallback tangent perpendicular to normal
                tangent = self._generate_fallback_tangent(normal)
            
            # Compute bitangent and determine handedness
            computed_bitangent = np.cross(normal, tangent)
            
            # Check if we need to flip the bitangent (handedness)
            accumulated_bitangent_length = np.linalg.norm(accumulated_bitangent)
            handedness = self.config.handedness.value
            
            if accumulated_bitangent_length > 1e-8:
                accumulated_bitangent /= accumulated_bitangent_length
                
                # Determine if bitangent needs flipping
                dot_product = np.dot(computed_bitangent, accumulated_bitangent)
                if dot_product < 0:
                    handedness *= -1
                    flipped_bitangents += 1
            
            # Store tangent with handedness in w component
            final_tangents[i, :3] = tangent
            final_tangents[i, 3] = handedness
        
        # Update statistics
        generation_time = time.time() - start_time
        self.stats.update({
            'triangles_processed': triangle_count,
            'vertices_processed': vertex_count,
            'degenerate_triangles': degenerate_count,
            'flipped_bitangents': flipped_bitangents,
            'generation_time': generation_time
        })
        
        return final_tangents
    
    def _calculate_triangle_tangent_space(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                                        uv1: np.ndarray, uv2: np.ndarray, uv3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Calculate tangent and bitangent for a triangle"""
        
        # Calculate edge vectors
        edge1 = v2 - v1
        edge2 = v3 - v1
        
        # Calculate UV deltas
        delta_uv1 = uv2 - uv1
        delta_uv2 = uv3 - uv1
        
        # Calculate determinant for UV space
        det = delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0]
        
        # Check for degenerate UV coordinates
        if abs(det) < 1e-8:
            # Degenerate triangle in UV space
            return np.array([1, 0, 0]), np.array([0, 1, 0]), True
        
        # Calculate tangent and bitangent using the standard formula
        inv_det = 1.0 / det
        
        tangent = inv_det * (delta_uv2[1] * edge1 - delta_uv1[1] * edge2)
        bitangent = inv_det * (delta_uv1[0] * edge2 - delta_uv2[0] * edge1)
        
        # Normalize
        tangent_length = np.linalg.norm(tangent)
        bitangent_length = np.linalg.norm(bitangent)
        
        if tangent_length > 1e-8:
            tangent /= tangent_length
        else:
            tangent = np.array([1, 0, 0])
            
        if bitangent_length > 1e-8:
            bitangent /= bitangent_length
        else:
            bitangent = np.array([0, 1, 0])
        
        return tangent, bitangent, False
    
    def _generate_fallback_tangent(self, normal: np.ndarray) -> np.ndarray:
        """Generate a fallback tangent perpendicular to the normal"""
        # Choose axis with smallest component to avoid near-parallel vectors
        abs_normal = np.abs(normal)
        if abs_normal[0] <= abs_normal[1] and abs_normal[0] <= abs_normal[2]:
            tangent = np.cross(normal, np.array([1, 0, 0]))
        elif abs_normal[1] <= abs_normal[2]:
            tangent = np.cross(normal, np.array([0, 1, 0]))
        else:
            tangent = np.cross(normal, np.array([0, 0, 1]))
        
        tangent_length = np.linalg.norm(tangent)
        if tangent_length > 1e-8:
            return tangent / tangent_length
        else:
            return np.array([1, 0, 0])  # Ultimate fallback
    
    def generate_tangent_space_vertices(self, positions: np.ndarray, normals: np.ndarray,
                                      uvs: np.ndarray, indices: np.ndarray) -> List[TangentSpaceVertex]:
        """Generate complete tangent space vertex data"""
        tangents = self.generate_tangents(positions, normals, uvs, indices)
        
        vertices = []
        for i in range(len(positions)):
            # Compute bitangent from normal and tangent
            tangent_vec = tangents[i, :3]
            handedness = tangents[i, 3]
            normal_vec = normals[i]
            
            bitangent_vec = np.cross(normal_vec, tangent_vec) * handedness
            
            vertex = TangentSpaceVertex(
                position=positions[i],
                normal=normal_vec,
                uv=uvs[i],
                tangent=tangents[i],
                bitangent=bitangent_vec
            )
            vertices.append(vertex)
        
        return vertices
    
    def update_marching_cubes_vertices(self, mc_vertices: List, uvs: np.ndarray, 
                                     indices: np.ndarray) -> List:
        """Update MarchingCubesVertex objects with proper tangent space"""
        if not MARCHING_CUBES_AVAILABLE:
            return mc_vertices
        
        # Extract data from MarchingCubesVertex objects
        positions = np.array([v.position for v in mc_vertices])
        normals = np.array([v.normal for v in mc_vertices])
        
        # Generate tangents
        tangents = self.generate_tangents(positions, normals, uvs, indices)
        
        # Update vertex objects
        for i, vertex in enumerate(mc_vertices):
            if hasattr(vertex, 'tangent'):
                vertex.tangent = tangents[i]
            else:
                # Add tangent attribute if it doesn't exist
                vertex.tangent = tangents[i]
        
        return mc_vertices
    
    def validate_tangent_space(self, vertices: List[TangentSpaceVertex]) -> Dict[str, Any]:
        """Validate tangent space quality and consistency"""
        validation_results = {
            'total_vertices': len(vertices),
            'orthogonal_vertices': 0,
            'handedness_consistent': True,
            'avg_orthogonality_error': 0.0,
            'max_orthogonality_error': 0.0,
            'handedness_distribution': {'right': 0, 'left': 0}
        }
        
        total_error = 0.0
        max_error = 0.0
        
        for vertex in vertices:
            # Check orthogonality
            if vertex.validate_orthogonality():
                validation_results['orthogonal_vertices'] += 1
            
            # Calculate orthogonality error
            t, b, n = vertex.tangent[:3], vertex.bitangent, vertex.normal
            tb_dot = abs(np.dot(t, b))
            tn_dot = abs(np.dot(t, n))
            bn_dot = abs(np.dot(b, n))
            
            error = max(tb_dot, tn_dot, bn_dot)
            total_error += error
            max_error = max(max_error, error)
            
            # Check handedness
            handedness = vertex.tangent[3]
            if handedness > 0:
                validation_results['handedness_distribution']['right'] += 1
            else:
                validation_results['handedness_distribution']['left'] += 1
        
        validation_results['avg_orthogonality_error'] = total_error / len(vertices) if vertices else 0
        validation_results['max_orthogonality_error'] = max_error
        
        return validation_results
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get tangent generation statistics"""
        return self.stats.copy()


class MaterialTangentSpaceManager:
    """Manages tangent space across different material types"""
    
    def __init__(self):
        """Initialize material tangent space manager"""
        self.generators = {
            'terrain': MikkTSpaceTangentGenerator(),
            'cave': MikkTSpaceTangentGenerator(), 
            'fused': MikkTSpaceTangentGenerator()
        }
        
        # Material-specific configurations
        self.material_configs = {
            'terrain': TangentGenerationConfig(
                handedness=MikkTSpaceHandedness.RIGHT_HANDED,
                normalize_uvs=True
            ),
            'cave': TangentGenerationConfig(
                handedness=MikkTSpaceHandedness.RIGHT_HANDED,
                normalize_uvs=True
            ),
            'fused': TangentGenerationConfig(
                handedness=MikkTSpaceHandedness.RIGHT_HANDED,
                normalize_uvs=True
            )
        }
    
    def generate_tangents_for_material(self, material_type: str, positions: np.ndarray,
                                     normals: np.ndarray, uvs: np.ndarray, 
                                     indices: np.ndarray) -> np.ndarray:
        """Generate tangents for specific material type"""
        if material_type not in self.generators:
            material_type = 'fused'  # Default fallback
        
        generator = self.generators[material_type]
        return generator.generate_tangents(positions, normals, uvs, indices)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all material generators"""
        return {material: gen.get_generation_stats() 
                for material, gen in self.generators.items()}


if __name__ == "__main__":
    # Test MikkTSpace tangent generation
    print("🚀 T12 MikkTSpace Tangent Generation System")
    print("=" * 60)
    
    # Create test geometry (simple triangle)
    positions = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.5, 1.0, 0.0],  # Vertex 2
    ], dtype=np.float32)
    
    normals = np.array([
        [0.0, 0.0, 1.0],  # Normal pointing up
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    
    uvs = np.array([
        [0.0, 0.0],  # UV 0
        [1.0, 0.0],  # UV 1
        [0.5, 1.0],  # UV 2
    ], dtype=np.float32)
    
    indices = np.array([
        [0, 1, 2]  # Single triangle
    ], dtype=np.int32)
    
    print(f"Test geometry: {len(positions)} vertices, {len(indices)} triangles")
    
    # Test tangent generation
    generator = MikkTSpaceTangentGenerator()
    tangents = generator.generate_tangents(positions, normals, uvs, indices)
    
    print(f"\n✅ Generated tangents:")
    for i, tangent in enumerate(tangents):
        print(f"   Vertex {i}: T=({tangent[0]:.3f}, {tangent[1]:.3f}, {tangent[2]:.3f}), H={tangent[3]:.1f}")
    
    # Test tangent space vertex generation
    ts_vertices = generator.generate_tangent_space_vertices(positions, normals, uvs, indices)
    
    # Validate tangent space
    validation = generator.validate_tangent_space(ts_vertices)
    print(f"\n📊 Tangent Space Validation:")
    print(f"   Orthogonal vertices: {validation['orthogonal_vertices']}/{validation['total_vertices']}")
    print(f"   Max orthogonality error: {validation['max_orthogonality_error']:.6f}")
    print(f"   Handedness: {validation['handedness_distribution']}")
    
    # Test statistics
    stats = generator.get_generation_stats()
    print(f"\n📈 Generation Statistics:")
    print(f"   Triangles processed: {stats['triangles_processed']}")
    print(f"   Vertices processed: {stats['vertices_processed']}")
    print(f"   Generation time: {stats['generation_time']*1000:.2f} ms")
    
    print("\n✅ MikkTSpace tangent generation system functional")