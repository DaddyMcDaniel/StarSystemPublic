#!/usr/bin/env python3
"""
Example Terrain Stack Logger - T04
==================================

Demonstrates and logs an example terrain composition:
- Ridged base layer for mountain-like features
- FBM detail layer for natural variation
- Small domain warp for organic flow

Logs the complete stack configuration and sample outputs.
"""

import json
import numpy as np
from pathlib import Path

try:
    from .heightfield import create_example_terrain_stack, log_terrain_composition, create_heightfield_from_pcc
    from .noise_nodes import NoiseFBM, RidgedMF, DomainWarp
except ImportError:
    from heightfield import create_example_terrain_stack, log_terrain_composition, create_heightfield_from_pcc
    from noise_nodes import NoiseFBM, RidgedMF, DomainWarp


def create_detailed_terrain_stack():
    """Create a detailed terrain stack and return both the specification and heightfield"""
    
    # Define the complete terrain specification
    terrain_spec = {
        "terrain": {
            "name": "Example Mountainous Terrain",
            "description": "Ridged mountain base with FBM detail and domain warping for organic flow",
            "heightfield": {
                "base_height": 0.0,
                "height_scale": 0.25,
                "evaluation_space": "unit_sphere"
            },
            "nodes": {
                "type": "Composite",
                "operation": "add",
                "weights": [1.0, 0.4, 0.15],
                "description": "Layered composition: ridged base + detail + micro variation",
                "nodes": [
                    {
                        "name": "Mountain Base Layer",
                        "type": "RidgedMF",
                        "seed": 1000,
                        "octaves": 4,
                        "frequency": 1.8,
                        "amplitude": 1.0,
                        "lacunarity": 2.2,
                        "persistence": 0.65,
                        "ridge_offset": 0.95,
                        "description": "Sharp mountain ridges and valleys"
                    },
                    {
                        "name": "Warped Detail Layer",
                        "type": "DomainWarp",
                        "seed": 2000,
                        "strength": 0.08,
                        "frequency": 1.2,
                        "description": "FBM detail with gentle domain distortion for flow",
                        "source": {
                            "name": "Detail Noise",
                            "type": "NoiseFBM",
                            "seed": 3000,
                            "octaves": 5,
                            "frequency": 3.5,
                            "amplitude": 1.0,
                            "lacunarity": 2.0,
                            "persistence": 0.45,
                            "description": "Medium-frequency detail variation"
                        }
                    },
                    {
                        "name": "Micro Variation",
                        "type": "NoiseFBM",
                        "seed": 4000,
                        "octaves": 3,
                        "frequency": 8.0,
                        "amplitude": 1.0,
                        "lacunarity": 2.1,
                        "persistence": 0.35,
                        "description": "High-frequency surface detail"
                    }
                ]
            }
        }
    }
    
    return terrain_spec


def log_example_terrain_stack():
    """Create and log the example terrain stack"""
    print("🏔️ T04 Example Terrain Stack - Detailed Logging")
    print("=" * 60)
    
    # Create terrain specification
    terrain_spec = create_detailed_terrain_stack()
    
    # Create heightfield with deterministic seed
    seed = 424242
    print(f"🌱 Global Seed: {seed}")
    heightfield = create_heightfield_from_pcc(terrain_spec, global_seed=seed)
    
    print("\n📋 Terrain Specification:")
    print("-" * 30)
    terrain_info = terrain_spec["terrain"]
    print(f"Name: {terrain_info['name']}")
    print(f"Description: {terrain_info['description']}")
    print(f"Base Height: {terrain_info['heightfield']['base_height']}")
    print(f"Height Scale: {terrain_info['heightfield']['height_scale']}")
    print(f"Evaluation Space: {terrain_info['heightfield']['evaluation_space']}")
    
    print("\n🔍 Layer Composition:")
    print("-" * 30)
    log_terrain_composition(heightfield)
    
    print("\n📊 Sample Evaluations:")
    print("-" * 30)
    
    # Define test points in unit-sphere space with geographic meaning
    test_points = [
        {"name": "North Pole", "pos": (0.0, 0.0, 1.0)},
        {"name": "Equator East", "pos": (1.0, 0.0, 0.0)},
        {"name": "Equator North", "pos": (0.0, 1.0, 0.0)},
        {"name": "South Pole", "pos": (0.0, 0.0, -1.0)},
        {"name": "45° NE", "pos": (0.707, 0.707, 0.0)},
        {"name": "30° Zenith", "pos": (0.866, 0.0, 0.5)},
        {"name": "Random Point A", "pos": (0.5, 0.3, 0.8)},
        {"name": "Random Point B", "pos": (-0.6, 0.4, 0.7)},
        {"name": "Random Point C", "pos": (0.2, -0.8, 0.6)}
    ]
    
    for point in test_points:
        height = heightfield.sample(*point["pos"])
        pos_str = f"({point['pos'][0]:6.3f}, {point['pos'][1]:6.3f}, {point['pos'][2]:6.3f})"
        print(f"   {point['name']:15s}: {pos_str} -> {height:8.6f}")
    
    # Analyze statistical properties
    print("\n📈 Statistical Analysis:")
    print("-" * 30)
    
    # Generate larger sample for statistics
    num_samples = 1000
    np.random.seed(seed)  # Deterministic sample points
    
    # Generate random points on unit sphere
    samples = []
    for _ in range(num_samples):
        # Generate random point on sphere using normal distribution
        vec = np.random.normal(0, 1, 3)
        vec = vec / np.linalg.norm(vec)
        height = heightfield.sample(vec[0], vec[1], vec[2])
        samples.append(height)
    
    samples = np.array(samples)
    
    print(f"   Sample Count: {len(samples)}")
    print(f"   Mean Height: {np.mean(samples):8.6f}")
    print(f"   Std Deviation: {np.std(samples):8.6f}")
    print(f"   Min Height: {np.min(samples):8.6f}")
    print(f"   Max Height: {np.max(samples):8.6f}")
    print(f"   Range: {np.max(samples) - np.min(samples):8.6f}")
    
    # Percentiles
    print(f"   5th Percentile: {np.percentile(samples, 5):8.6f}")
    print(f"   25th Percentile: {np.percentile(samples, 25):8.6f}")
    print(f"   Median: {np.percentile(samples, 50):8.6f}")
    print(f"   75th Percentile: {np.percentile(samples, 75):8.6f}")
    print(f"   95th Percentile: {np.percentile(samples, 95):8.6f}")
    
    print("\n🔬 Layer Analysis:")
    print("-" * 30)
    
    # Analyze individual layers at a test point
    test_pos = (0.5, 0.3, 0.8)
    print(f"Analysis at position: ({test_pos[0]}, {test_pos[1]}, {test_pos[2]})")
    
    # Create individual layer heightfields for analysis
    ridged_spec = {
        "terrain": {
            "heightfield": {"base_height": 0.0, "height_scale": 1.0},
            "nodes": terrain_spec["terrain"]["nodes"]["nodes"][0]
        }
    }
    
    ridged_hf = create_heightfield_from_pcc(ridged_spec, global_seed=seed)
    ridged_val = ridged_hf.sample(*test_pos)
    
    detail_spec = {
        "terrain": {
            "heightfield": {"base_height": 0.0, "height_scale": 1.0},
            "nodes": terrain_spec["terrain"]["nodes"]["nodes"][1]
        }
    }
    
    detail_hf = create_heightfield_from_pcc(detail_spec, global_seed=seed)
    detail_val = detail_hf.sample(*test_pos)
    
    micro_spec = {
        "terrain": {
            "heightfield": {"base_height": 0.0, "height_scale": 1.0},
            "nodes": terrain_spec["terrain"]["nodes"]["nodes"][2]
        }
    }
    
    micro_hf = create_heightfield_from_pcc(micro_spec, global_seed=seed)
    micro_val = micro_hf.sample(*test_pos)
    
    final_val = heightfield.sample(*test_pos)
    
    weights = terrain_spec["terrain"]["nodes"]["weights"]
    expected = (ridged_val * weights[0] + detail_val * weights[1] + micro_val * weights[2]) * heightfield.height_scale
    
    print(f"   Ridged Base (weight {weights[0]}): {ridged_val:8.6f}")
    print(f"   Warped Detail (weight {weights[1]}): {detail_val:8.6f}")
    print(f"   Micro Variation (weight {weights[2]}): {micro_val:8.6f}")
    print(f"   Expected Combined: {expected:8.6f}")
    print(f"   Actual Combined: {final_val:8.6f}")
    print(f"   Difference: {abs(expected - final_val):8.6f}")
    
    print("\n🔖 Determinism Verification:")
    print("-" * 30)
    
    # Test determinism by recreating the heightfield
    heightfield2 = create_heightfield_from_pcc(terrain_spec, global_seed=seed)
    
    determinism_errors = 0
    for point in test_points[:5]:  # Test first 5 points
        val1 = heightfield.sample(*point["pos"])
        val2 = heightfield2.sample(*point["pos"])
        diff = abs(val1 - val2)
        if diff > 1e-12:
            determinism_errors += 1
        print(f"   {point['name']:15s}: {diff:2.2e} {'✅' if diff < 1e-12 else '❌'}")
    
    print(f"   Determinism Status: {'✅ PASS' if determinism_errors == 0 else '❌ FAIL'}")
    
    print("\n💾 Exporting Terrain Specification:")
    print("-" * 30)
    
    # Export the complete specification
    output_file = Path(__file__).parent / "example_terrain_spec.json"
    with open(output_file, 'w') as f:
        json.dump(terrain_spec, f, indent=2)
    
    print(f"   Exported to: {output_file}")
    print(f"   File size: {output_file.stat().st_size} bytes")
    
    print("\n✅ T04 Example Terrain Stack Logging Complete")
    print("=" * 60)
    
    return heightfield, terrain_spec


if __name__ == "__main__":
    # Log the example terrain stack
    heightfield, spec = log_example_terrain_stack()