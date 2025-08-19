#!/usr/bin/env python3
"""
Noise Nodes for Heightfield Generation - T04
=============================================

Deterministic noise functions for PCC terrain specification.
All functions are seeded for reproducible results.

Features:
- NoiseFBM: Fractional Brownian Motion
- RidgedMF: Ridged Multifractal  
- DomainWarp: Procedural domain distortion
- Deterministic seeding using PCG64 generator
"""

import math
import numpy as np
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod


class NoiseNode(ABC):
    """Base class for all noise nodes"""
    
    def __init__(self, seed: int = 42):
        """Initialize with deterministic seed"""
        self.seed = seed
        self.rng = np.random.PCG64(seed)
        
    @abstractmethod
    def sample(self, x: float, y: float, z: float) -> float:
        """Sample noise at 3D position"""
        pass
    
    def sample_vec3(self, pos: np.ndarray) -> float:
        """Sample noise at 3D position vector"""
        return self.sample(pos[0], pos[1], pos[2])


def hash_position(x: float, y: float, z: float, seed: int) -> int:
    """Deterministic 3D position hash for noise"""
    # Simple but effective hash combining position and seed
    x_int = int(x * 1000000) if abs(x) < 1e6 else int(x)
    y_int = int(y * 1000000) if abs(y) < 1e6 else int(y)
    z_int = int(z * 1000000) if abs(z) < 1e6 else int(z)
    
    # FNV-1a hash variant
    hash_val = 2166136261
    for val in [x_int, y_int, z_int, seed]:
        hash_val ^= val & 0xFFFFFFFF
        hash_val *= 16777619
        hash_val &= 0xFFFFFFFF
    
    return hash_val


def interpolate_quintic(t: float) -> float:
    """Quintic interpolation (6t^5 - 15t^4 + 10t^3)"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def gradient_3d(hash_val: int, x: float, y: float, z: float) -> float:
    """3D gradient function for Perlin noise"""
    h = hash_val & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def perlin_noise_3d(x: float, y: float, z: float, seed: int) -> float:
    """3D Perlin noise implementation"""
    # Find unit cube containing point
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255  
    zi = int(math.floor(z)) & 255
    
    # Find relative position in cube
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)
    
    # Compute fade curves
    u = interpolate_quintic(x)
    v = interpolate_quintic(y)
    w = interpolate_quintic(z)
    
    # Hash coordinates of cube corners
    a = hash_position(xi, yi, zi, seed)
    aa = hash_position(xi, yi, zi + 1, seed)
    ab = hash_position(xi, yi + 1, zi, seed)
    abb = hash_position(xi, yi + 1, zi + 1, seed)
    b = hash_position(xi + 1, yi, zi, seed)
    ba = hash_position(xi + 1, yi, zi + 1, seed)
    bb = hash_position(xi + 1, yi + 1, zi, seed)
    bba = hash_position(xi + 1, yi + 1, zi + 1, seed)
    
    # Blend the results from 8 corners
    x1 = lerp(gradient_3d(a, x, y, z), gradient_3d(b, x - 1, y, z), u)
    x2 = lerp(gradient_3d(ab, x, y - 1, z), gradient_3d(bb, x - 1, y - 1, z), u)
    y1 = lerp(x1, x2, v)
    
    x1 = lerp(gradient_3d(aa, x, y, z - 1), gradient_3d(ba, x - 1, y, z - 1), u)
    x2 = lerp(gradient_3d(abb, x, y - 1, z - 1), gradient_3d(bba, x - 1, y - 1, z - 1), u)
    y2 = lerp(x1, x2, v)
    
    return lerp(y1, y2, w)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation"""
    return a + t * (b - a)


class NoiseFBM(NoiseNode):
    """
    Fractional Brownian Motion noise
    
    Combines multiple octaves of Perlin noise with decreasing amplitude
    and increasing frequency to create natural-looking terrain features.
    """
    
    def __init__(self, seed: int = 42, octaves: int = 4, frequency: float = 1.0, 
                 amplitude: float = 1.0, lacunarity: float = 2.0, persistence: float = 0.5):
        """
        Initialize FBM noise
        
        Args:
            seed: Random seed for deterministic results
            octaves: Number of noise octaves to combine
            frequency: Base frequency of noise
            amplitude: Base amplitude of noise
            lacunarity: Frequency multiplier between octaves
            persistence: Amplitude multiplier between octaves
        """
        super().__init__(seed)
        self.octaves = octaves
        self.frequency = frequency
        self.amplitude = amplitude
        self.lacunarity = lacunarity
        self.persistence = persistence
        
    def sample(self, x: float, y: float, z: float) -> float:
        """Sample FBM noise at 3D position"""
        value = 0.0
        freq = self.frequency
        amp = self.amplitude
        
        for i in range(self.octaves):
            # Use different seed per octave for variety
            octave_seed = self.seed + i * 1000
            value += perlin_noise_3d(x * freq, y * freq, z * freq, octave_seed) * amp
            freq *= self.lacunarity
            amp *= self.persistence
            
        return value


class RidgedMF(NoiseNode):
    """
    Ridged Multifractal noise
    
    Creates sharp ridges and valleys by inverting and emphasizing 
    the peaks of FBM noise. Excellent for mountain ranges and canyons.
    """
    
    def __init__(self, seed: int = 42, octaves: int = 4, frequency: float = 1.0,
                 amplitude: float = 1.0, lacunarity: float = 2.0, persistence: float = 0.5,
                 ridge_offset: float = 1.0):
        """
        Initialize Ridged Multifractal noise
        
        Args:
            ridge_offset: Controls ridge sharpness (typically 0.9-1.1)
        """
        super().__init__(seed)
        self.octaves = octaves
        self.frequency = frequency
        self.amplitude = amplitude
        self.lacunarity = lacunarity
        self.persistence = persistence
        self.ridge_offset = ridge_offset
        
    def sample(self, x: float, y: float, z: float) -> float:
        """Sample ridged multifractal noise at 3D position"""
        value = 0.0
        freq = self.frequency
        amp = self.amplitude
        weight = 1.0
        
        for i in range(self.octaves):
            octave_seed = self.seed + i * 1000
            noise = perlin_noise_3d(x * freq, y * freq, z * freq, octave_seed)
            
            # Create ridged effect
            noise = abs(noise)
            noise = self.ridge_offset - noise
            noise = noise * noise
            noise *= weight
            
            # Weight successive octaves by previous results
            weight = noise * 2.0
            weight = max(0.0, min(1.0, weight))  # Clamp to [0,1]
            
            value += noise * amp
            freq *= self.lacunarity
            amp *= self.persistence
            
        return value


class DomainWarp(NoiseNode):
    """
    Domain Warp node
    
    Applies procedural distortion to the sampling domain before
    evaluating another noise function. Creates flowing, organic shapes.
    """
    
    def __init__(self, seed: int = 42, strength: float = 0.1, frequency: float = 1.0,
                 source_node: Optional[NoiseNode] = None):
        """
        Initialize Domain Warp
        
        Args:
            seed: Random seed
            strength: Distortion strength 
            frequency: Frequency of distortion pattern
            source_node: Source noise to warp (if None, creates FBM)
        """
        super().__init__(seed)
        self.strength = strength
        self.frequency = frequency
        
        # Create source noise if not provided
        if source_node is None:
            self.source_node = NoiseFBM(seed=seed + 5000)
        else:
            self.source_node = source_node
            
        # Create orthogonal warp patterns
        self.warp_x = NoiseFBM(seed=seed + 1000, frequency=frequency)
        self.warp_y = NoiseFBM(seed=seed + 2000, frequency=frequency)
        self.warp_z = NoiseFBM(seed=seed + 3000, frequency=frequency)
        
    def sample(self, x: float, y: float, z: float) -> float:
        """Sample domain-warped noise at 3D position"""
        # Calculate warp offsets
        warp_offset_x = self.warp_x.sample(x, y, z) * self.strength
        warp_offset_y = self.warp_y.sample(x, y, z) * self.strength
        warp_offset_z = self.warp_z.sample(x, y, z) * self.strength
        
        # Apply warp to sampling position
        warped_x = x + warp_offset_x
        warped_y = y + warp_offset_y
        warped_z = z + warp_offset_z
        
        # Sample source noise at warped position
        return self.source_node.sample(warped_x, warped_y, warped_z)


def create_noise_node(node_spec: dict) -> NoiseNode:
    """
    Create a noise node from PCC specification
    
    Args:
        node_spec: Dictionary containing node type and parameters
        
    Returns:
        Configured noise node
    """
    node_type = node_spec.get("type", "")
    seed = node_spec.get("seed", 42)
    
    if node_type == "NoiseFBM":
        return NoiseFBM(
            seed=seed,
            octaves=node_spec.get("octaves", 4),
            frequency=node_spec.get("frequency", 1.0),
            amplitude=node_spec.get("amplitude", 1.0),
            lacunarity=node_spec.get("lacunarity", 2.0),
            persistence=node_spec.get("persistence", 0.5)
        )
    elif node_type == "RidgedMF":
        return RidgedMF(
            seed=seed,
            octaves=node_spec.get("octaves", 4),
            frequency=node_spec.get("frequency", 1.0),
            amplitude=node_spec.get("amplitude", 1.0),
            lacunarity=node_spec.get("lacunarity", 2.0),
            persistence=node_spec.get("persistence", 0.5),
            ridge_offset=node_spec.get("ridge_offset", 1.0)
        )
    elif node_type == "DomainWarp":
        # Handle nested source node
        source_spec = node_spec.get("source", None)
        source_node = create_noise_node(source_spec) if source_spec else None
        
        return DomainWarp(
            seed=seed,
            strength=node_spec.get("strength", 0.1),
            frequency=node_spec.get("frequency", 1.0),
            source_node=source_node
        )
    else:
        raise ValueError(f"Unknown noise node type: {node_type}")


if __name__ == "__main__":
    # Test deterministic behavior
    print("🔧 Testing noise node determinism...")
    
    # Test position
    test_pos = (0.5, 0.3, 0.7)
    
    # Test FBM
    fbm1 = NoiseFBM(seed=12345)
    fbm2 = NoiseFBM(seed=12345)
    val1 = fbm1.sample(*test_pos)
    val2 = fbm2.sample(*test_pos)
    print(f"FBM determinism: {val1:.6f} == {val2:.6f} -> {'✅' if abs(val1 - val2) < 1e-10 else '❌'}")
    
    # Test RidgedMF
    ridged1 = RidgedMF(seed=54321)
    ridged2 = RidgedMF(seed=54321)
    val1 = ridged1.sample(*test_pos)
    val2 = ridged2.sample(*test_pos)
    print(f"RidgedMF determinism: {val1:.6f} == {val2:.6f} -> {'✅' if abs(val1 - val2) < 1e-10 else '❌'}")
    
    # Test DomainWarp
    warp1 = DomainWarp(seed=99999)
    warp2 = DomainWarp(seed=99999)
    val1 = warp1.sample(*test_pos)
    val2 = warp2.sample(*test_pos)
    print(f"DomainWarp determinism: {val1:.6f} == {val2:.6f} -> {'✅' if abs(val1 - val2) < 1e-10 else '❌'}")
    
    print("✅ Noise node testing complete")