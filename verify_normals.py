#!/usr/bin/env python3
"""
Verify that cube-sphere normals are outward-facing
"""
import numpy as np
import json
from pathlib import Path

def verify_outward_normals(manifest_path):
    """Verify that normals point outward from sphere center"""
    
    manifest_path_obj = Path(manifest_path)
    manifest_dir = manifest_path_obj.parent
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    mesh_data = manifest.get("mesh", {})
    
    # Load positions and normals from binary files
    pos_file = manifest_dir / mesh_data["positions"].replace("buffer://", "")
    norm_file = manifest_dir / mesh_data["normals"].replace("buffer://", "")
    
    positions = np.fromfile(str(pos_file), dtype=np.float32).reshape(-1, 3)
    normals = np.fromfile(str(norm_file), dtype=np.float32).reshape(-1, 3)
    
    print(f"🔍 Verifying {len(positions)} vertex normals...")
    
    # For a sphere centered at origin, the outward normal at any point
    # should be the normalized position vector
    errors = 0
    max_error = 0
    
    for i in range(len(positions)):
        pos = positions[i]
        normal = normals[i]
        
        # Expected normal (outward from center)
        expected_normal = pos / np.linalg.norm(pos)
        
        # Check if computed normal matches expected
        error = np.linalg.norm(normal - expected_normal)
        max_error = max(max_error, error)
        
        if error > 0.01:  # Tolerance for numerical differences
            errors += 1
    
    print(f"📊 Verification results:")
    print(f"   Errors: {errors}/{len(positions)} vertices")
    print(f"   Max error: {max_error:.6f}")
    print(f"   Status: {'✅ PASS' if errors == 0 else '❌ FAIL'}")
    
    # Check a few sample dot products to verify outward direction
    center = np.array([0, 0, 0])
    sample_indices = [0, len(positions)//4, len(positions)//2, 3*len(positions)//4, len(positions)-1]
    
    print(f"🔍 Sample outward direction check:")
    for i in sample_indices:
        pos = positions[i]
        normal = normals[i]
        
        # Vector from center to vertex
        outward_dir = pos - center
        outward_dir = outward_dir / np.linalg.norm(outward_dir)
        
        # Dot product should be close to 1 for outward normals
        dot_product = np.dot(normal, outward_dir)
        print(f"   Vertex {i}: dot = {dot_product:.6f} {'✅' if dot_product > 0.99 else '❌'}")

if __name__ == "__main__":
    # Verify the T03 cube-sphere
    manifest_path = "agents/agent_d/mesh/t03_cubesphere_analytical.json"
    verify_outward_normals(manifest_path)