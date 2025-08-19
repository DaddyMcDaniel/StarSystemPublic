#!/usr/bin/env python3
"""
T06 Chunked Planet Verification
===============================

Tests the complete quadtree chunking pipeline without requiring OpenGL.
Validates chunk loading, AABB computation, and chunked planet structure.
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh'))

def verify_chunk_manifest(chunk_path: Path):
    """Verify a single chunk manifest file"""
    try:
        with open(chunk_path, 'r') as f:
            manifest = json.load(f)
        
        # Check required sections
        required_sections = ["mesh", "metadata", "chunk"]
        for section in required_sections:
            if section not in manifest:
                return False, f"Missing section: {section}"
        
        # Check chunk metadata
        chunk_info = manifest["chunk"]
        required_chunk_fields = ["chunk_id", "face_id", "level", "uv_bounds", "aabb", "resolution"]
        for field in required_chunk_fields:
            if field not in chunk_info:
                return False, f"Missing chunk field: {field}"
        
        # Check AABB structure
        aabb = chunk_info["aabb"]
        required_aabb_fields = ["min", "max", "center", "size"]
        for field in required_aabb_fields:
            if field not in aabb:
                return False, f"Missing AABB field: {field}"
            if len(aabb[field]) != 3:
                return False, f"AABB {field} should have 3 components"
        
        # Check UV bounds structure
        uv_bounds = chunk_info["uv_bounds"]
        required_uv_fields = ["min", "max", "center", "size"]
        for field in required_uv_fields:
            if field not in uv_bounds:
                return False, f"Missing UV bounds field: {field}"
            if len(uv_bounds[field]) != 2:
                return False, f"UV bounds {field} should have 2 components"
        
        return True, "Valid chunk manifest"
        
    except Exception as e:
        return False, f"Error reading manifest: {e}"


def verify_chunk_buffers(chunk_path: Path):
    """Verify that chunk binary buffers exist and have reasonable sizes"""
    chunk_dir = chunk_path.parent
    chunk_name = chunk_path.stem
    
    expected_buffers = [
        f"{chunk_name}_positions.bin",
        f"{chunk_name}_normals.bin", 
        f"{chunk_name}_tangents.bin",
        f"{chunk_name}_uvs.bin",
        f"{chunk_name}_indices.bin"
    ]
    
    missing_buffers = []
    buffer_stats = {}
    
    for buffer_name in expected_buffers:
        buffer_path = chunk_dir / buffer_name
        if not buffer_path.exists():
            missing_buffers.append(buffer_name)
        else:
            size = buffer_path.stat().st_size
            buffer_stats[buffer_name] = size
    
    if missing_buffers:
        return False, f"Missing buffers: {missing_buffers}", {}
    
    # Check reasonable buffer sizes
    positions_size = buffer_stats.get(f"{chunk_name}_positions.bin", 0)
    expected_vertex_count = positions_size // (3 * 4)  # 3 floats * 4 bytes per float
    
    indices_size = buffer_stats.get(f"{chunk_name}_indices.bin", 0)
    expected_triangle_count = (indices_size // 4) // 3  # 4 bytes per uint32, 3 indices per triangle
    
    if expected_vertex_count < 9:  # At least 3x3 vertices for minimum chunk
        return False, f"Too few vertices: {expected_vertex_count}", buffer_stats
    
    if expected_triangle_count < 8:  # At least 8 triangles for 3x3 grid
        return False, f"Too few triangles: {expected_triangle_count}", buffer_stats
    
    return True, f"Valid buffers (vertices: {expected_vertex_count}, triangles: {expected_triangle_count})", buffer_stats


def verify_planet_manifest(planet_path: Path):
    """Verify the master planet manifest"""
    try:
        with open(planet_path, 'r') as f:
            manifest = json.load(f)
        
        # Check required sections
        required_sections = ["planet", "chunks", "statistics", "metadata"]
        for section in required_sections:
            if section not in manifest:
                return False, f"Missing section: {section}"
        
        # Check planet info
        planet_info = manifest["planet"]
        if planet_info.get("type") != "chunked_quadtree":
            return False, f"Invalid planet type: {planet_info.get('type')}"
        
        required_planet_fields = ["max_depth", "chunk_resolution", "total_chunks", "chunks_per_face"]
        for field in required_planet_fields:
            if field not in planet_info:
                return False, f"Missing planet field: {field}"
        
        # Verify chunk count consistency
        expected_chunks_per_face = 4 ** planet_info["max_depth"]
        if planet_info["chunks_per_face"] != expected_chunks_per_face:
            return False, f"Chunks per face mismatch: expected {expected_chunks_per_face}, got {planet_info['chunks_per_face']}"
        
        expected_total_chunks = expected_chunks_per_face * 6
        if planet_info["total_chunks"] != expected_total_chunks:
            return False, f"Total chunks mismatch: expected {expected_total_chunks}, got {planet_info['total_chunks']}"
        
        if len(manifest["chunks"]) != expected_total_chunks:
            return False, f"Chunk list length mismatch: expected {expected_total_chunks}, got {len(manifest['chunks'])}"
        
        return True, f"Valid planet manifest (depth: {planet_info['max_depth']}, chunks: {planet_info['total_chunks']})"
        
    except Exception as e:
        return False, f"Error reading planet manifest: {e}"


def analyze_chunk_distribution(planet_path: Path):
    """Analyze the distribution of chunks across faces and levels"""
    try:
        planet_dir = planet_path.parent
        
        with open(planet_path, 'r') as f:
            planet_manifest = json.load(f)
        
        chunk_files = planet_manifest.get("chunks", [])
        
        face_counts = {}
        level_counts = {}
        uv_bounds_analysis = []
        
        for chunk_file in chunk_files:
            chunk_path = planet_dir / chunk_file
            
            with open(chunk_path, 'r') as f:
                chunk_manifest = json.load(f)
            
            chunk_info = chunk_manifest.get("chunk", {})
            face_id = chunk_info.get("face_id", -1)
            level = chunk_info.get("level", -1)
            uv_bounds = chunk_info.get("uv_bounds", {})
            
            # Count by face
            face_counts[face_id] = face_counts.get(face_id, 0) + 1
            
            # Count by level
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # Analyze UV bounds
            uv_center = uv_bounds.get("center", [0, 0])
            uv_size = uv_bounds.get("size", [0, 0])
            uv_bounds_analysis.append({
                "face_id": face_id,
                "level": level,
                "center": uv_center,
                "size": uv_size
            })
        
        return {
            "face_distribution": face_counts,
            "level_distribution": level_counts,
            "total_chunks": len(chunk_files),
            "uv_bounds_sample": uv_bounds_analysis[:5]  # First 5 for inspection
        }
        
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main verification function"""
    print("🚀 T06 Chunked Planet Verification")
    print("=" * 60)
    
    # Find test chunks directory
    chunks_dir = Path(__file__).parent / "mesh" / "test_chunks"
    planet_manifest_path = chunks_dir / "planet_chunks.json"
    
    if not planet_manifest_path.exists():
        print(f"❌ Planet manifest not found: {planet_manifest_path}")
        print("   Run: python quadtree_chunking.py --max_depth 2 --chunk_res 12 --output test_chunks")
        return False
    
    print(f"📁 Verifying planet manifest: {planet_manifest_path}")
    
    # Verify planet manifest
    valid, message = verify_planet_manifest(planet_manifest_path)
    if not valid:
        print(f"❌ Planet manifest validation failed: {message}")
        return False
    
    print(f"✅ Planet manifest valid: {message}")
    
    # Load and analyze planet
    with open(planet_manifest_path, 'r') as f:
        planet_manifest = json.load(f)
    
    planet_info = planet_manifest["planet"]
    chunk_files = planet_manifest["chunks"]
    
    print(f"\n📊 Planet Configuration:")
    print(f"   Max depth: {planet_info['max_depth']}")
    print(f"   Chunk resolution: {planet_info['chunk_resolution']}")
    print(f"   Total chunks: {planet_info['total_chunks']}")
    print(f"   Chunks per face: {planet_info['chunks_per_face']}")
    print(f"   Has terrain: {planet_info.get('has_terrain', False)}")
    print(f"   Displacement scale: {planet_info.get('displacement_scale', 0.0)}")
    
    # Verify individual chunks
    print(f"\n🔍 Verifying individual chunks...")
    
    chunk_validation_results = []
    total_vertices = 0
    total_triangles = 0
    
    for i, chunk_file in enumerate(chunk_files[:10]):  # Test first 10 chunks
        chunk_path = chunks_dir / chunk_file
        print(f"   Chunk {i+1}/10: {chunk_file}")
        
        # Verify manifest
        valid, message = verify_chunk_manifest(chunk_path)
        if not valid:
            print(f"      ❌ Manifest: {message}")
            chunk_validation_results.append(False)
            continue
        
        # Verify buffers
        valid, message, buffer_stats = verify_chunk_buffers(chunk_path)
        if not valid:
            print(f"      ❌ Buffers: {message}")
            chunk_validation_results.append(False)
            continue
        
        print(f"      ✅ Valid chunk: {message}")
        chunk_validation_results.append(True)
        
        # Extract vertex/triangle counts from buffer stats
        positions_size = buffer_stats.get(f"{chunk_path.stem}_positions.bin", 0)
        indices_size = buffer_stats.get(f"{chunk_path.stem}_indices.bin", 0)
        
        vertices = positions_size // (3 * 4)
        triangles = (indices_size // 4) // 3
        
        total_vertices += vertices
        total_triangles += triangles
    
    successful_chunks = sum(chunk_validation_results)
    tested_chunks = len(chunk_validation_results)
    
    print(f"\n📈 Chunk Validation Results:")
    print(f"   Tested chunks: {tested_chunks}")
    print(f"   Valid chunks: {successful_chunks}")
    print(f"   Success rate: {successful_chunks/tested_chunks*100:.1f}%")
    print(f"   Total vertices (sample): {total_vertices}")
    print(f"   Total triangles (sample): {total_triangles}")
    
    # Analyze chunk distribution
    print(f"\n🗺️ Chunk Distribution Analysis:")
    distribution = analyze_chunk_distribution(planet_manifest_path)
    
    if "error" in distribution:
        print(f"   ❌ Distribution analysis failed: {distribution['error']}")
    else:
        print(f"   Faces: {distribution['face_distribution']}")
        print(f"   Levels: {distribution['level_distribution']}")
        print(f"   Total chunks: {distribution['total_chunks']}")
        
        # Check if distribution is balanced
        face_counts = list(distribution['face_distribution'].values())
        if len(set(face_counts)) == 1:
            print(f"   ✅ Balanced face distribution ({face_counts[0]} chunks per face)")
        else:
            print(f"   ⚠️ Unbalanced face distribution: {face_counts}")
    
    # Overall assessment
    print(f"\n✅ T06 Chunked Planet Verification Complete")
    
    if successful_chunks == tested_chunks and tested_chunks > 0:
        print(f"🎉 All tested chunks are valid!")
        print(f"🪐 Chunked planet successfully generated and verified")
        return True
    else:
        print(f"⚠️ Some chunks failed validation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)