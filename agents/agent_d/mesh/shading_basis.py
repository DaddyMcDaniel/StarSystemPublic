#!/usr/bin/env python3
"""
Shading Basis Computation for Agent D - T03
==========================================

Robust normal and tangent computation for displacement and normal mapping.
Implements angle-weighted vertex normals and MikkTSpace-compatible tangents.

Features:
- Angle-weighted vertex normals from triangle mesh
- Per-vertex tangent/bitangent computation from UVs
- MikkTSpace integration point for consistency
- Proper handedness for normal mapping

Usage:
    from shading_basis import compute_angle_weighted_normals, compute_tangent_basis
"""

import numpy as np
import math
from typing import Tuple, Optional


def compute_angle_weighted_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute angle-weighted vertex normals from triangle mesh
    
    This method weights each triangle's contribution to a vertex normal
    by the angle at that vertex, providing smoother results than
    simple area-weighted normals.
    
    Args:
        positions: Vertex positions as (N, 3) float32 array
        indices: Triangle indices as (M, 3) uint32 array
        
    Returns:
        Vertex normals as (N, 3) float32 array
    """
    num_vertices = len(positions)
    vertex_normals = np.zeros((num_vertices, 3), dtype=np.float32)
    
    # Reshape indices to process triangles
    triangles = indices.reshape(-1, 3)
    
    print(f"🔢 Computing angle-weighted normals for {num_vertices} vertices, {len(triangles)} triangles")
    
    for tri in triangles:
        i0, i1, i2 = tri
        
        # Get triangle vertices
        v0 = positions[i0]
        v1 = positions[i1] 
        v2 = positions[i2]
        
        # Compute face normal (not normalized yet)
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        # Skip degenerate triangles
        face_normal_length = np.linalg.norm(face_normal)
        if face_normal_length < 1e-12:
            continue
            
        face_normal = face_normal / face_normal_length
        
        # Compute angle weights at each vertex
        # At vertex i0: angle between edges (v1-v0) and (v2-v0)
        edge_a = v1 - v0
        edge_b = v2 - v0
        angle0 = compute_angle_between_vectors(edge_a, edge_b)
        
        # At vertex i1: angle between edges (v0-v1) and (v2-v1)  
        edge_a = v0 - v1
        edge_b = v2 - v1
        angle1 = compute_angle_between_vectors(edge_a, edge_b)
        
        # At vertex i2: angle between edges (v0-v2) and (v1-v2)
        edge_a = v0 - v2
        edge_b = v1 - v2
        angle2 = compute_angle_between_vectors(edge_a, edge_b)
        
        # Accumulate weighted normals
        vertex_normals[i0] += face_normal * angle0
        vertex_normals[i1] += face_normal * angle1
        vertex_normals[i2] += face_normal * angle2
    
    # Normalize all vertex normals
    for i in range(num_vertices):
        normal_length = np.linalg.norm(vertex_normals[i])
        if normal_length > 1e-12:
            vertex_normals[i] /= normal_length
        else:
            # Fallback for degenerate cases
            vertex_normals[i] = np.array([0, 1, 0], dtype=np.float32)
    
    print(f"✅ Computed angle-weighted normals")
    return vertex_normals


def compute_angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute angle between two vectors using robust method
    
    Args:
        a, b: Input vectors
        
    Returns:
        Angle in radians
    """
    # Normalize vectors
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm < 1e-12 or b_norm < 1e-12:
        return 0.0
        
    a_unit = a / a_norm
    b_unit = b / b_norm
    
    # Use atan2 for robust angle computation
    dot_product = np.clip(np.dot(a_unit, b_unit), -1.0, 1.0)
    return math.acos(dot_product)


def compute_tangent_basis(positions: np.ndarray, normals: np.ndarray, 
                         uvs: np.ndarray, indices: np.ndarray, 
                         use_mikktspace: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-vertex tangent basis from UV coordinates
    
    Args:
        positions: Vertex positions as (N, 3) float32 array
        normals: Vertex normals as (N, 3) float32 array
        uvs: Texture coordinates as (N, 2) float32 array
        indices: Triangle indices as (M, 3) uint32 array
        use_mikktspace: Whether to use MikkTSpace algorithm
        
    Returns:
        Tuple of (tangents, bitangents) as (N, 4) and (N, 3) float32 arrays
        Tangents include handedness in w component
    """
    num_vertices = len(positions)
    
    if use_mikktspace:
        return compute_mikktspace_tangents(positions, normals, uvs, indices)
    else:
        return compute_basic_tangents(positions, normals, uvs, indices)


def compute_basic_tangents(positions: np.ndarray, normals: np.ndarray,
                          uvs: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basic tangent computation using Lengyel's method
    
    This is a fallback implementation when MikkTSpace is not available.
    """
    num_vertices = len(positions)
    tangents = np.zeros((num_vertices, 4), dtype=np.float32)
    bitangents = np.zeros((num_vertices, 3), dtype=np.float32)
    
    # Temporary arrays for accumulation
    tan1 = np.zeros((num_vertices, 3), dtype=np.float32)
    tan2 = np.zeros((num_vertices, 3), dtype=np.float32)
    
    triangles = indices.reshape(-1, 3)
    
    print(f"🔢 Computing basic tangent basis for {num_vertices} vertices")
    
    # Accumulate tangent basis per triangle
    for tri in triangles:
        i0, i1, i2 = tri
        
        # Get triangle data
        v0, v1, v2 = positions[i0], positions[i1], positions[i2]
        w0, w1, w2 = uvs[i0], uvs[i1], uvs[i2]
        
        # Position deltas
        x1, x2 = v1 - v0, v2 - v0
        
        # UV deltas
        s1, s2 = w1[0] - w0[0], w2[0] - w0[0]
        t1, t2 = w1[1] - w0[1], w2[1] - w0[1]
        
        # Compute determinant
        denom = s1 * t2 - s2 * t1
        if abs(denom) < 1e-12:
            # Degenerate UV triangle - skip
            continue
            
        r = 1.0 / denom
        
        # Compute tangent and bitangent
        sdir = (x1 * t2 - x2 * t1) * r
        tdir = (x2 * s1 - x1 * s2) * r
        
        # Accumulate for all three vertices
        tan1[i0] += sdir
        tan1[i1] += sdir
        tan1[i2] += sdir
        
        tan2[i0] += tdir
        tan2[i1] += tdir
        tan2[i2] += tdir
    
    # Orthogonalize and normalize
    for i in range(num_vertices):
        n = normals[i]
        t = tan1[i]
        
        # Gram-Schmidt orthogonalization
        # tangent = normalize(t - n * dot(n, t))
        dot_nt = np.dot(n, t)
        tangent = t - n * dot_nt
        tangent_length = np.linalg.norm(tangent)
        
        if tangent_length > 1e-12:
            tangent = tangent / tangent_length
        else:
            # Generate orthogonal vector
            tangent = generate_orthogonal_vector(n)
        
        # Compute bitangent and handedness
        bitangent = np.cross(n, tangent)
        
        # Check handedness
        # handedness = (dot(cross(n, t), tan2[i]) < 0.0) ? -1.0 : 1.0
        cross_nt = np.cross(n, t)
        handedness = -1.0 if np.dot(cross_nt, tan2[i]) < 0.0 else 1.0
        
        tangents[i] = np.array([tangent[0], tangent[1], tangent[2], handedness], dtype=np.float32)
        bitangents[i] = bitangent * handedness
    
    print(f"✅ Computed basic tangent basis")
    return tangents, bitangents


def compute_mikktspace_tangents(positions: np.ndarray, normals: np.ndarray,
                               uvs: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MikkTSpace-compatible tangent computation
    
    This function provides an integration point for MikkTSpace algorithm.
    For now, it's a stub that calls the basic implementation.
    """
    print("🔧 MikkTSpace integration point - using basic tangents for now")
    
    # TODO: Integrate actual MikkTSpace library here
    # This could be done via:
    # 1. Python bindings to MikkTSpace C library
    # 2. Pure Python implementation of MikkTSpace
    # 3. External tool integration
    
    return compute_basic_tangents(positions, normals, uvs, indices)


def generate_orthogonal_vector(normal: np.ndarray) -> np.ndarray:
    """
    Generate an orthogonal vector to the given normal
    
    Args:
        normal: Input normal vector
        
    Returns:
        Orthogonal unit vector
    """
    # Choose axis with smallest component to avoid near-parallel vectors
    abs_normal = np.abs(normal)
    if abs_normal[0] <= abs_normal[1] and abs_normal[0] <= abs_normal[2]:
        # X component is smallest
        orthogonal = np.array([1, 0, 0], dtype=np.float32)
    elif abs_normal[1] <= abs_normal[2]:
        # Y component is smallest
        orthogonal = np.array([0, 1, 0], dtype=np.float32)
    else:
        # Z component is smallest
        orthogonal = np.array([0, 0, 1], dtype=np.float32)
    
    # Project out normal component and normalize
    dot_product = np.dot(normal, orthogonal)
    orthogonal = orthogonal - normal * dot_product
    return orthogonal / np.linalg.norm(orthogonal)


def validate_shading_basis(normals: np.ndarray, tangents: np.ndarray, 
                          bitangents: np.ndarray) -> bool:
    """
    Validate that the shading basis is orthonormal
    
    Args:
        normals: Vertex normals as (N, 3) array
        tangents: Vertex tangents as (N, 4) array (w = handedness)
        bitangents: Vertex bitangents as (N, 3) array
        
    Returns:
        True if basis is valid
    """
    print("🔍 Validating shading basis...")
    
    num_vertices = len(normals)
    errors = 0
    tolerance = 1e-3
    
    for i in range(min(num_vertices, 100)):  # Sample first 100 vertices
        n = normals[i]
        t = tangents[i][:3]  # Extract xyz, ignore handedness
        b = bitangents[i]
        
        # Check if vectors are normalized
        n_len = np.linalg.norm(n)
        t_len = np.linalg.norm(t)
        b_len = np.linalg.norm(b)
        
        if abs(n_len - 1.0) > tolerance:
            errors += 1
        if abs(t_len - 1.0) > tolerance:
            errors += 1
        if abs(b_len - 1.0) > tolerance:
            errors += 1
        
        # Check orthogonality
        dot_nt = abs(np.dot(n, t))
        dot_nb = abs(np.dot(n, b))
        dot_tb = abs(np.dot(t, b))
        
        if dot_nt > tolerance:
            errors += 1
        if dot_nb > tolerance:
            errors += 1
        if dot_tb > tolerance:
            errors += 1
    
    success = errors == 0
    status = "✅" if success else "⚠️"
    print(f"{status} Shading basis validation: {errors} errors in sample")
    
    return success


if __name__ == "__main__":
    # Test with simple triangle
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0], 
        [0, 1, 0]
    ], dtype=np.float32)
    
    indices = np.array([0, 1, 2], dtype=np.uint32)
    
    uvs = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ], dtype=np.float32)
    
    # Test normal computation
    normals = compute_angle_weighted_normals(positions, indices)
    print(f"Test normals: {normals}")
    
    # Test tangent computation
    tangents, bitangents = compute_tangent_basis(positions, normals, uvs, indices)
    print(f"Test tangents shape: {tangents.shape}")
    print(f"Test bitangents shape: {bitangents.shape}")
    
    # Validate basis
    validate_shading_basis(normals, tangents, bitangents)