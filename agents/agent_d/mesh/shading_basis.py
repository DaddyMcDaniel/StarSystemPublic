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


def compute_angle_weighted_normals(positions: np.ndarray, indices: np.ndarray, 
                                   post_displacement: bool = True) -> np.ndarray:
    """
    T22: Compute angle-weighted vertex normals from triangle mesh after displacement
    
    This method weights each triangle's contribution to a vertex normal
    by the angle at that vertex, providing smoother results than
    simple area-weighted normals. Enhanced for post-displacement quality.
    
    Args:
        positions: Vertex positions as (N, 3) float32 array
        indices: Triangle indices as (M, 3) uint32 array
        post_displacement: Whether this is after heightfield displacement (enables optimizations)
        
    Returns:
        Vertex normals as (N, 3) float32 array
    """
    num_vertices = len(positions)
    vertex_normals = np.zeros((num_vertices, 3), dtype=np.float32)
    
    # Reshape indices to process triangles
    triangles = indices.reshape(-1, 3)
    
    if post_displacement:
        print(f"🔢 T22: Computing post-displacement angle-weighted normals for {num_vertices} vertices, {len(triangles)} triangles")
    else:
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
    
    # T22: Additional quality pass for post-displacement normals
    if post_displacement:
        vertex_normals = _smooth_displacement_normals(vertex_normals, positions, indices)
        print(f"✅ T22: Computed enhanced post-displacement angle-weighted normals")
    else:
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
                         use_mikktspace: bool = True, post_displacement: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    T22: Compute per-vertex tangent basis from UV coordinates with enhanced quality
    
    Args:
        positions: Vertex positions as (N, 3) float32 array
        normals: Vertex normals as (N, 3) float32 array
        uvs: Texture coordinates as (N, 2) float32 array
        indices: Triangle indices as (M, 3) uint32 array
        use_mikktspace: Whether to use MikkTSpace algorithm
        post_displacement: Whether this is after displacement (enables quality improvements)
        
    Returns:
        Tuple of (tangents, bitangents) as (N, 4) and (N, 3) float32 arrays
        Tangents include handedness in w component
    """
    num_vertices = len(positions)
    
    if use_mikktspace:
        return compute_mikktspace_tangents(positions, normals, uvs, indices, post_displacement)
    else:
        return compute_basic_tangents(positions, normals, uvs, indices, post_displacement)


def compute_basic_tangents(positions: np.ndarray, normals: np.ndarray,
                          uvs: np.ndarray, indices: np.ndarray, post_displacement: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    T22: Enhanced tangent computation using improved Lengyel's method
    
    This is a fallback implementation when MikkTSpace is not available.
    Enhanced for post-displacement quality and seam consistency.
    """
    num_vertices = len(positions)
    tangents = np.zeros((num_vertices, 4), dtype=np.float32)
    bitangents = np.zeros((num_vertices, 3), dtype=np.float32)
    
    # Temporary arrays for accumulation
    tan1 = np.zeros((num_vertices, 3), dtype=np.float32)
    tan2 = np.zeros((num_vertices, 3), dtype=np.float32)
    
    triangles = indices.reshape(-1, 3)
    
    if post_displacement:
        print(f"🔢 T22: Computing enhanced post-displacement tangent basis for {num_vertices} vertices")
    else:
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
    
    # T22: Post-process tangents for improved quality and seam consistency
    if post_displacement:
        tangents, bitangents = _improve_tangent_quality(tangents, bitangents, normals, positions, indices)
        print(f"✅ T22: Computed enhanced post-displacement tangent basis")
    else:
        print(f"✅ Computed basic tangent basis")
    
    return tangents, bitangents


def compute_mikktspace_tangents(positions: np.ndarray, normals: np.ndarray,
                               uvs: np.ndarray, indices: np.ndarray, post_displacement: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    T22: MikkTSpace-compatible tangent computation with displacement enhancements
    
    This function provides an integration point for MikkTSpace algorithm.
    Enhanced for post-displacement quality and normal map compatibility.
    """
    if post_displacement:
        print("🔧 T22: MikkTSpace integration point with displacement enhancements - using enhanced basic tangents")
    else:
        print("🔧 MikkTSpace integration point - using basic tangents for now")
    
    # TODO: Integrate actual MikkTSpace library here
    # This could be done via:
    # 1. Python bindings to MikkTSpace C library
    # 2. Pure Python implementation of MikkTSpace
    # 3. External tool integration
    
    return compute_basic_tangents(positions, normals, uvs, indices, post_displacement)


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


def _smooth_displacement_normals(normals: np.ndarray, positions: np.ndarray, 
                                indices: np.ndarray, smoothing_factor: float = 0.3) -> np.ndarray:
    """
    T22: Smooth normals after heightfield displacement to reduce artifacts
    
    Args:
        normals: Input vertex normals
        positions: Vertex positions after displacement
        indices: Triangle indices
        smoothing_factor: Amount of smoothing to apply (0.0 = none, 1.0 = full)
        
    Returns:
        Smoothed vertex normals
    """
    if smoothing_factor <= 0.0:
        return normals
    
    num_vertices = len(normals)
    smoothed_normals = np.copy(normals)
    
    # Build vertex adjacency for smoothing
    adjacency = [[] for _ in range(num_vertices)]
    triangles = indices.reshape(-1, 3)
    
    for tri in triangles:
        i0, i1, i2 = tri
        adjacency[i0].extend([i1, i2])
        adjacency[i1].extend([i0, i2])
        adjacency[i2].extend([i0, i1])
    
    # Remove duplicates and convert to sets
    for i in range(num_vertices):
        adjacency[i] = list(set(adjacency[i]))
    
    # Apply smoothing based on adjacent vertices
    for i in range(num_vertices):
        if len(adjacency[i]) == 0:
            continue
            
        # Accumulate normals from adjacent vertices
        accumulated_normal = np.zeros(3, dtype=np.float32)
        total_weight = 0.0
        
        for adj_idx in adjacency[i]:
            # Weight by inverse distance to preserve shape
            distance = np.linalg.norm(positions[adj_idx] - positions[i])
            weight = 1.0 / max(distance, 1e-6)
            
            accumulated_normal += normals[adj_idx] * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_normal = accumulated_normal / total_weight
            
            # Normalize the averaged normal
            avg_length = np.linalg.norm(avg_normal)
            if avg_length > 1e-8:
                avg_normal = avg_normal / avg_length
                
                # Blend with original normal
                blended_normal = (1.0 - smoothing_factor) * normals[i] + smoothing_factor * avg_normal
                
                # Re-normalize
                blend_length = np.linalg.norm(blended_normal)
                if blend_length > 1e-8:
                    smoothed_normals[i] = blended_normal / blend_length
    
    return smoothed_normals


def _improve_tangent_quality(tangents: np.ndarray, bitangents: np.ndarray, normals: np.ndarray,
                           positions: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    T22: Improve tangent quality after displacement for better normal mapping
    
    Args:
        tangents: Input tangent vectors with handedness
        bitangents: Input bitangent vectors
        normals: Vertex normals
        positions: Vertex positions
        indices: Triangle indices
        
    Returns:
        Improved (tangents, bitangents) tuple
    """
    num_vertices = len(tangents)
    improved_tangents = np.copy(tangents)
    improved_bitangents = np.copy(bitangents)
    
    # Build vertex adjacency for consistency checking
    adjacency = [[] for _ in range(num_vertices)]
    triangles = indices.reshape(-1, 3)
    
    for tri in triangles:
        i0, i1, i2 = tri
        adjacency[i0].extend([i1, i2])
        adjacency[i1].extend([i0, i2])
        adjacency[i2].extend([i0, i1])
    
    # Remove duplicates
    for i in range(num_vertices):
        adjacency[i] = list(set(adjacency[i]))
    
    # Improve tangent consistency across seams
    for i in range(num_vertices):
        if len(adjacency[i]) == 0:
            continue
        
        current_tangent = tangents[i][:3]  # Extract XYZ
        current_normal = normals[i]
        
        # Check for tangent flipping across edges
        flip_count = 0
        for adj_idx in adjacency[i]:
            adj_tangent = tangents[adj_idx][:3]
            
            # Check if tangents are roughly opposite
            dot_product = np.dot(current_tangent, adj_tangent)
            if dot_product < -0.5:  # Roughly opposite
                flip_count += 1
        
        # If majority of adjacent tangents are flipped, consider flipping this one
        if flip_count > len(adjacency[i]) // 2:
            improved_tangents[i][:3] = -current_tangent
            improved_tangents[i][3] = -tangents[i][3]  # Flip handedness too
            improved_bitangents[i] = -bitangents[i]
        
        # Re-orthogonalize tangent with respect to normal
        t = improved_tangents[i][:3]
        n = current_normal
        
        # Gram-Schmidt: t' = normalize(t - n * dot(n, t))
        dot_nt = np.dot(n, t)
        orthogonal_tangent = t - n * dot_nt
        tangent_length = np.linalg.norm(orthogonal_tangent)
        
        if tangent_length > 1e-8:
            orthogonal_tangent = orthogonal_tangent / tangent_length
            improved_tangents[i][:3] = orthogonal_tangent
            
            # Recompute bitangent to ensure orthogonality
            improved_bitangents[i] = np.cross(n, orthogonal_tangent) * improved_tangents[i][3]
    
    return improved_tangents, improved_bitangents


def validate_seam_consistency(normals: np.ndarray, tangents: np.ndarray, 
                             positions: np.ndarray, indices: np.ndarray, 
                             uv_coords: np.ndarray, tolerance: float = 0.1) -> bool:
    """
    T22: Validate that normals and tangents are consistent across UV seams
    
    Args:
        normals: Vertex normals
        tangents: Vertex tangents with handedness
        positions: Vertex positions
        indices: Triangle indices
        uv_coords: UV coordinates for seam detection
        tolerance: Angular tolerance for consistency (radians)
        
    Returns:
        True if seams are consistent
    """
    print("🔍 T22: Validating seam consistency...")
    
    # Find potential UV seam vertices (vertices with same position but different UVs)
    seam_vertices = []
    position_tolerance = 1e-6
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos_diff = np.linalg.norm(positions[i] - positions[j])
            uv_diff = np.linalg.norm(uv_coords[i] - uv_coords[j])
            
            if pos_diff < position_tolerance and uv_diff > 0.01:
                seam_vertices.append((i, j))
    
    # Check consistency across seams
    inconsistent_seams = 0
    
    for i, j in seam_vertices:
        normal_i, normal_j = normals[i], normals[j]
        tangent_i, tangent_j = tangents[i][:3], tangents[j][:3]
        
        # Check normal consistency
        normal_angle = math.acos(np.clip(np.dot(normal_i, normal_j), -1.0, 1.0))
        if normal_angle > tolerance:
            inconsistent_seams += 1
            continue
        
        # Check tangent consistency (considering possible flip)
        tangent_angle = math.acos(np.clip(abs(np.dot(tangent_i, tangent_j)), 0.0, 1.0))
        if tangent_angle > tolerance:
            inconsistent_seams += 1
    
    success = inconsistent_seams == 0
    status = "✅" if success else "⚠️"
    total_seams = len(seam_vertices)
    
    print(f"{status} T22: Seam consistency check: {inconsistent_seams}/{total_seams} seams inconsistent")
    
    return success


def create_normal_map_test_pattern(width: int = 512, height: int = 512) -> np.ndarray:
    """
    T22: Create test normal map pattern for tangent space validation
    
    Args:
        width: Normal map width
        height: Normal map height
        
    Returns:
        RGB normal map for testing tangent space
    """
    normal_map = np.zeros((height, width, 3), dtype=np.float32)
    
    # Create checkerboard normal pattern
    for y in range(height):
        for x in range(width):
            u = x / width
            v = y / height
            
            # Create bumps in checkerboard pattern
            check_u = int(u * 8) % 2
            check_v = int(v * 8) % 2
            is_raised = (check_u + check_v) % 2 == 0
            
            if is_raised:
                # Raised areas - slight normal variation
                nx = math.sin(u * 16 * math.pi) * 0.3
                ny = math.sin(v * 16 * math.pi) * 0.3
                nz = math.sqrt(max(0.0, 1.0 - nx*nx - ny*ny))
            else:
                # Flat areas
                nx, ny, nz = 0.0, 0.0, 1.0
            
            # Convert from [-1,1] to [0,1] for storage
            normal_map[y, x] = [(nx + 1.0) * 0.5, (ny + 1.0) * 0.5, (nz + 1.0) * 0.5]
    
    return normal_map


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