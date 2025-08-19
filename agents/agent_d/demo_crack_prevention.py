#!/usr/bin/env python3
"""
T07 Crack Prevention Demonstration
==================================

Demonstrates crack prevention on mixed LOD chunks and shows before/after comparison.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mesh'))

from mesh.crack_prevention import LODCrackPrevention, validate_crack_prevention
from test_crack_prevention import create_mixed_lod_test_chunks, load_chunk_for_testing


def demonstrate_crack_prevention():
    """Run crack prevention demonstration"""
    print("🚀 T07 Crack Prevention Demonstration")
    print("=" * 60)
    
    # Create mixed LOD test chunks
    print("🔧 Creating mixed LOD test chunks...")
    test_chunks = create_mixed_lod_test_chunks()
    
    if not test_chunks:
        print("❌ No test chunks available")
        return False
    
    print(f"✅ Created {len(test_chunks)} test chunks")
    
    # Analyze original chunks
    print("\n📊 Original Chunks Analysis:")
    print("-" * 40)
    
    original_validation = validate_crack_prevention(test_chunks)
    
    # Detect potential cracks
    crack_preventer = LODCrackPrevention()
    original_cracks = crack_preventer.detect_cracks(test_chunks)
    
    print(f"🔍 Potential cracks detected: {len(original_cracks)}")
    
    if original_cracks:
        transition_types = {}
        for crack in original_cracks:
            transition = crack.get("transition_type", "unknown")
            transition_types[transition] = transition_types.get(transition, 0) + 1
        
        for transition, count in transition_types.items():
            print(f"   {transition} transitions: {count}")
    
    # Apply crack prevention with stitching
    print("\n🔧 Applying Index Stitching...")
    stitching_preventer = LODCrackPrevention(enable_stitching=True, enable_skirts=False)
    stitched_chunks = stitching_preventer.apply_crack_prevention(test_chunks)
    
    # Validate stitched chunks
    print("\n📊 Stitched Chunks Analysis:")
    print("-" * 40)
    stitched_validation = validate_crack_prevention(stitched_chunks)
    
    # Apply crack prevention with skirts
    print("\n🔧 Applying Skirt Generation...")
    skirt_preventer = LODCrackPrevention(enable_stitching=False, enable_skirts=True, skirt_depth=0.02)
    skirt_chunks = skirt_preventer.apply_crack_prevention(test_chunks)
    
    # Validate skirt chunks
    print("\n📊 Skirt Chunks Analysis:")
    print("-" * 40)
    skirt_validation = validate_crack_prevention(skirt_chunks)
    
    # Compare results
    print("\n📈 Comparison Results:")
    print("=" * 40)
    
    print(f"{'Method':<15} {'Chunks':<8} {'NaN Verts':<10} {'Invalid Idx':<12} {'Errors':<8}")
    print("-" * 60)
    print(f"{'Original':<15} {original_validation['total_chunks']:<8} {original_validation['nan_vertices']:<10} {original_validation['invalid_indices']:<12} {len(original_validation['validation_errors']):<8}")
    print(f"{'Stitching':<15} {stitched_validation['total_chunks']:<8} {stitched_validation['nan_vertices']:<10} {stitched_validation['invalid_indices']:<12} {len(stitched_validation['validation_errors']):<8}")
    print(f"{'Skirts':<15} {skirt_validation['total_chunks']:<8} {skirt_validation['nan_vertices']:<10} {skirt_validation['invalid_indices']:<12} {len(skirt_validation['validation_errors']):<8}")
    
    # Check for improvements
    print(f"\n✅ Results Summary:")
    
    if (stitched_validation['nan_vertices'] == 0 and 
        stitched_validation['invalid_indices'] == 0):
        print("   ✅ Index stitching: No geometry errors")
    else:
        print("   ❌ Index stitching: Has geometry errors")
    
    if (skirt_validation['nan_vertices'] == 0 and 
        skirt_validation['invalid_indices'] == 0):
        print("   ✅ Skirt generation: No geometry errors")
    else:
        print("   ❌ Skirt generation: Has geometry errors")
    
    # Vertex count analysis
    print(f"\n📊 Vertex Count Analysis:")
    original_verts = sum(len(chunk.get("positions", [])) // 3 for chunk in test_chunks)
    stitched_verts = sum(len(chunk.get("positions", [])) // 3 for chunk in stitched_chunks)
    skirt_verts = sum(len(chunk.get("positions", [])) // 3 for chunk in skirt_chunks)
    
    print(f"   Original vertices: {original_verts}")
    print(f"   Stitched vertices: {stitched_verts} ({stitched_verts - original_verts:+d})")
    print(f"   Skirt vertices: {skirt_verts} ({skirt_verts - original_verts:+d})")
    
    if skirt_verts > original_verts:
        print("   ✅ Skirts added geometry as expected")
    
    print(f"\n🎉 Crack Prevention Demonstration Complete!")
    
    return True


def analyze_crack_distribution():
    """Analyze crack distribution patterns"""
    print("\n🔍 Crack Distribution Analysis")
    print("=" * 40)
    
    # Load existing chunked planets for analysis
    test_dirs = [
        Path("test_chunks_depth1"),
        Path("test_chunks_depth2"),
        Path("mesh/test_chunks"),
        Path("mesh/small_chunks")
    ]
    
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
        
        planet_manifest = test_dir / "planet_chunks.json"
        if not planet_manifest.exists():
            continue
        
        print(f"\n📁 Analyzing: {test_dir}")
        
        # Load chunks from this directory
        chunks = []
        with open(planet_manifest, 'r') as f:
            import json
            planet_data = json.load(f)
        
        chunk_files = planet_data.get("chunks", [])[:10]  # Sample first 10
        
        for chunk_file in chunk_files:
            chunk_path = test_dir / chunk_file
            chunk_data = load_chunk_for_testing(chunk_path)
            if chunk_data:
                chunks.append(chunk_data)
        
        if not chunks:
            continue
        
        # Analyze this set
        crack_preventer = LODCrackPrevention()
        cracks = crack_preventer.detect_cracks(chunks)
        
        print(f"   Chunks analyzed: {len(chunks)}")
        print(f"   Potential cracks: {len(cracks)}")
        
        if cracks:
            face_distribution = {}
            level_distribution = {}
            
            for crack in cracks:
                # Extract face and level info if available in chunk IDs
                chunk_id = crack.get("chunk_id", "")
                if "face" in chunk_id and "_L" in chunk_id:
                    try:
                        face_part = chunk_id.split("face")[1].split("_")[0]
                        level_part = chunk_id.split("_L")[1].split("_")[0]
                        
                        face_id = int(face_part)
                        level = int(level_part)
                        
                        face_distribution[face_id] = face_distribution.get(face_id, 0) + 1
                        level_distribution[level] = level_distribution.get(level, 0) + 1
                    except (ValueError, IndexError):
                        pass
            
            if face_distribution:
                print(f"   Face distribution: {face_distribution}")
            if level_distribution:
                print(f"   Level distribution: {level_distribution}")


if __name__ == "__main__":
    success = demonstrate_crack_prevention()
    analyze_crack_distribution()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 T07 Crack Prevention Demonstration Successful!")
    else:
        print("❌ T07 Crack Prevention Demonstration Failed!")
    
    sys.exit(0 if success else 1)