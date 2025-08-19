#!/usr/bin/env python3
"""
SDF Primitives Module - T09
===========================

Implements Signed Distance Function (SDF) primitive nodes for cave and overhang generation.
All primitives are deterministic and seeded for reproducible terrain generation.

Features:
- Core geometric primitives: Sphere, Capsule, Gyroid
- Noise displacement for organic shapes
- Boolean operations: Union, Subtract, Intersect
- Deterministic seeded evaluation
- Efficient distance computations

Usage:
    from sdf_primitives import SDFSphere, SDFCapsule, SDFGyroid, SDFUnion
    
    sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    distance = sphere.evaluate(point)
"""

import numpy as np
import math
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import sys
import os

# Seeded noise for deterministic generation
class SeededNoise:
    """Deterministic noise generator using linear congruential generator"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.state = seed
    
    def _lcg(self, x: int) -> int:
        """Linear congruential generator for pseudo-randomness"""
        return (x * 1664525 + 1013904223) % (2**32)
    
    def noise1d(self, x: float) -> float:
        """1D noise function"""
        ix = int(x * 1000) % (2**31)
        self.state = self._lcg(ix + self.seed)
        return (self.state / (2**32)) * 2.0 - 1.0
    
    def noise3d(self, x: float, y: float, z: float) -> float:
        """3D noise function using hash of coordinates"""
        ix = int(x * 1000) % (2**16)
        iy = int(y * 1000) % (2**16) 
        iz = int(z * 1000) % (2**16)
        
        hash_val = ix + iy * 65536 + iz * 65536 * 65536
        self.state = self._lcg(hash_val + self.seed)
        return (self.state / (2**32)) * 2.0 - 1.0
    
    def fractal_noise3d(self, x: float, y: float, z: float, 
                       octaves: int = 4, frequency: float = 1.0, 
                       amplitude: float = 1.0, lacunarity: float = 2.0, 
                       persistence: float = 0.5) -> float:
        """Multi-octave fractal noise"""
        total = 0.0
        max_value = 0.0
        
        for i in range(octaves):
            total += self.noise3d(x * frequency, y * frequency, z * frequency) * amplitude
            max_value += amplitude
            
            frequency *= lacunarity
            amplitude *= persistence
        
        return total / max_value


class SDFNode(ABC):
    """Abstract base class for all SDF nodes"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.noise = SeededNoise(seed)
    
    @abstractmethod
    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate SDF at given 3D point, returning signed distance"""
        pass
    
    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate SDF for batch of points (Nx3 array)"""
        if points.ndim == 1:
            return self.evaluate(points)
        
        distances = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            distances[i] = self.evaluate(point)
        return distances


class SDFSphere(SDFNode):
    """SDF for sphere primitive"""
    
    def __init__(self, center: List[float] = None, radius: float = 1.0, seed: int = 42):
        super().__init__(seed)
        self.center = np.array(center or [0.0, 0.0, 0.0])
        self.radius = radius
    
    def evaluate(self, point: np.ndarray) -> float:
        """Sphere SDF: distance from surface"""
        return np.linalg.norm(point - self.center) - self.radius
    
    def __repr__(self):
        return f"SDFSphere(center={self.center}, radius={self.radius}, seed={self.seed})"


class SDFCapsule(SDFNode):
    """SDF for capsule primitive (cylinder with hemispherical caps)"""
    
    def __init__(self, point_a: List[float] = None, point_b: List[float] = None, 
                 radius: float = 0.5, seed: int = 42):
        super().__init__(seed)
        self.point_a = np.array(point_a or [0.0, -1.0, 0.0])
        self.point_b = np.array(point_b or [0.0, 1.0, 0.0])
        self.radius = radius
    
    def evaluate(self, point: np.ndarray) -> float:
        """Capsule SDF: distance from capsule surface"""
        pa = point - self.point_a
        ba = self.point_b - self.point_a
        
        # Project point onto line segment
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
        
        # Distance from point to closest point on line segment
        closest_point = self.point_a + h * ba
        return np.linalg.norm(point - closest_point) - self.radius
    
    def __repr__(self):
        return f"SDFCapsule(a={self.point_a}, b={self.point_b}, radius={self.radius}, seed={self.seed})"


class SDFBox(SDFNode):
    """SDF for axis-aligned box primitive"""
    
    def __init__(self, center: List[float] = None, size: List[float] = None, seed: int = 42):
        super().__init__(seed)
        self.center = np.array(center or [0.0, 0.0, 0.0])
        self.size = np.array(size or [1.0, 1.0, 1.0])
        self.half_size = self.size * 0.5
    
    def evaluate(self, point: np.ndarray) -> float:
        """Box SDF: distance from box surface"""
        # Transform to box-local coordinates
        local_point = np.abs(point - self.center) - self.half_size
        
        # Distance calculation
        outside_distance = np.linalg.norm(np.maximum(local_point, 0.0))
        inside_distance = np.max(local_point)
        
        return outside_distance + min(inside_distance, 0.0)
    
    def __repr__(self):
        return f"SDFBox(center={self.center}, size={self.size}, seed={self.seed})"


class SDFGyroid(SDFNode):
    """SDF for gyroid triply periodic minimal surface"""
    
    def __init__(self, scale: float = 1.0, thickness: float = 0.1, 
                 offset: float = 0.0, seed: int = 42):
        super().__init__(seed)
        self.scale = scale
        self.thickness = thickness
        self.offset = offset
    
    def evaluate(self, point: np.ndarray) -> float:
        """Gyroid SDF: distance from gyroid surface"""
        x, y, z = point * self.scale
        
        # Gyroid equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
        gyroid_value = (math.sin(x) * math.cos(y) + 
                       math.sin(y) * math.cos(z) + 
                       math.sin(z) * math.cos(x))
        
        # Add offset and thickness
        return abs(gyroid_value + self.offset) - self.thickness
    
    def __repr__(self):
        return f"SDFGyroid(scale={self.scale}, thickness={self.thickness}, offset={self.offset}, seed={self.seed})"


class SDFTorus(SDFNode):
    """SDF for torus primitive"""
    
    def __init__(self, center: List[float] = None, major_radius: float = 2.0, 
                 minor_radius: float = 0.5, seed: int = 42):
        super().__init__(seed)
        self.center = np.array(center or [0.0, 0.0, 0.0])
        self.major_radius = major_radius
        self.minor_radius = minor_radius
    
    def evaluate(self, point: np.ndarray) -> float:
        """Torus SDF: distance from torus surface"""
        local_point = point - self.center
        
        # Distance to torus in XZ plane, then to minor circle
        xz_distance = math.sqrt(local_point[0]**2 + local_point[2]**2)
        torus_point = np.array([xz_distance - self.major_radius, local_point[1]])
        
        return np.linalg.norm(torus_point) - self.minor_radius
    
    def __repr__(self):
        return f"SDFTorus(center={self.center}, major_r={self.major_radius}, minor_r={self.minor_radius}, seed={self.seed})"


class SDFNoiseDisplace(SDFNode):
    """SDF modifier that displaces another SDF using noise"""
    
    def __init__(self, base_sdf: SDFNode, displacement_scale: float = 0.1, 
                 noise_frequency: float = 1.0, octaves: int = 4, seed: int = 42):
        super().__init__(seed)
        self.base_sdf = base_sdf
        self.displacement_scale = displacement_scale
        self.noise_frequency = noise_frequency
        self.octaves = octaves
    
    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate base SDF with noise displacement"""
        # Sample noise at point
        noise_value = self.noise.fractal_noise3d(
            point[0] * self.noise_frequency,
            point[1] * self.noise_frequency, 
            point[2] * self.noise_frequency,
            octaves=self.octaves
        )
        
        # Displace the distance
        base_distance = self.base_sdf.evaluate(point)
        return base_distance + noise_value * self.displacement_scale
    
    def __repr__(self):
        return f"SDFNoiseDisplace(base={self.base_sdf}, scale={self.displacement_scale}, freq={self.noise_frequency}, seed={self.seed})"


class SDFTransform(SDFNode):
    """SDF modifier that applies transformation to another SDF"""
    
    def __init__(self, base_sdf: SDFNode, translation: List[float] = None, 
                 rotation: List[float] = None, scale: List[float] = None, seed: int = 42):
        super().__init__(seed)
        self.base_sdf = base_sdf
        self.translation = np.array(translation or [0.0, 0.0, 0.0])
        self.rotation = np.array(rotation or [0.0, 0.0, 0.0])  # Euler angles in radians
        self.scale = np.array(scale or [1.0, 1.0, 1.0])
        
        # Precompute transformation matrix
        self._compute_transform_matrix()
    
    def _compute_transform_matrix(self):
        """Compute inverse transformation matrix for point transformation"""
        # For SDF, we need to transform the query point inversely
        # Scale
        scale_matrix = np.diag(1.0 / self.scale)
        
        # Rotation (inverse of rotation matrix)
        rx, ry, rz = self.rotation
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        cos_z, sin_z = math.cos(rz), math.sin(rz)
        
        # Rotation matrices
        rot_x = np.array([[1, 0, 0],
                         [0, cos_x, sin_x],
                         [0, -sin_x, cos_x]])
        
        rot_y = np.array([[cos_y, 0, -sin_y],
                         [0, 1, 0],
                         [sin_y, 0, cos_y]])
        
        rot_z = np.array([[cos_z, sin_z, 0],
                         [-sin_z, cos_z, 0],
                         [0, 0, 1]])
        
        # Combined rotation (inverse order)
        rotation_matrix = rot_x @ rot_y @ rot_z
        
        # Combined transformation
        self.transform_matrix = scale_matrix @ rotation_matrix
    
    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate SDF with transformation applied"""
        # Transform point to local space
        translated_point = point - self.translation
        transformed_point = self.transform_matrix @ translated_point
        
        # Evaluate base SDF and scale distance appropriately
        base_distance = self.base_sdf.evaluate(transformed_point)
        
        # Scale the distance by minimum scale factor
        scale_factor = np.min(self.scale)
        return base_distance * scale_factor
    
    def __repr__(self):
        return f"SDFTransform(base={self.base_sdf}, trans={self.translation}, rot={self.rotation}, scale={self.scale}, seed={self.seed})"


# Boolean Operations

class SDFUnion(SDFNode):
    """SDF boolean union operation (OR)"""
    
    def __init__(self, sdf_a: SDFNode, sdf_b: SDFNode, seed: int = 42):
        super().__init__(seed)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b
    
    def evaluate(self, point: np.ndarray) -> float:
        """Union: minimum distance from either SDF"""
        dist_a = self.sdf_a.evaluate(point)
        dist_b = self.sdf_b.evaluate(point)
        return min(dist_a, dist_b)
    
    def __repr__(self):
        return f"SDFUnion({self.sdf_a}, {self.sdf_b}, seed={self.seed})"


class SDFSubtract(SDFNode):
    """SDF boolean subtraction operation (A - B)"""
    
    def __init__(self, sdf_a: SDFNode, sdf_b: SDFNode, seed: int = 42):
        super().__init__(seed)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b
    
    def evaluate(self, point: np.ndarray) -> float:
        """Subtraction: A minus B"""
        dist_a = self.sdf_a.evaluate(point)
        dist_b = self.sdf_b.evaluate(point)
        return max(dist_a, -dist_b)
    
    def __repr__(self):
        return f"SDFSubtract({self.sdf_a}, {self.sdf_b}, seed={self.seed})"


class SDFIntersect(SDFNode):
    """SDF boolean intersection operation (AND)"""
    
    def __init__(self, sdf_a: SDFNode, sdf_b: SDFNode, seed: int = 42):
        super().__init__(seed)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b
    
    def evaluate(self, point: np.ndarray) -> float:
        """Intersection: maximum distance from both SDFs"""
        dist_a = self.sdf_a.evaluate(point)
        dist_b = self.sdf_b.evaluate(point)
        return max(dist_a, dist_b)
    
    def __repr__(self):
        return f"SDFIntersect({self.sdf_a}, {self.sdf_b}, seed={self.seed})"


class SDFSmoothUnion(SDFNode):
    """SDF smooth union operation with blending"""
    
    def __init__(self, sdf_a: SDFNode, sdf_b: SDFNode, blend_radius: float = 0.1, seed: int = 42):
        super().__init__(seed)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b
        self.blend_radius = blend_radius
    
    def evaluate(self, point: np.ndarray) -> float:
        """Smooth union with polynomial blending"""
        dist_a = self.sdf_a.evaluate(point)
        dist_b = self.sdf_b.evaluate(point)
        
        h = max(self.blend_radius - abs(dist_a - dist_b), 0.0) / self.blend_radius
        return min(dist_a, dist_b) - h * h * self.blend_radius * 0.25
    
    def __repr__(self):
        return f"SDFSmoothUnion({self.sdf_a}, {self.sdf_b}, blend={self.blend_radius}, seed={self.seed})"


class SDFSmoothSubtract(SDFNode):
    """SDF smooth subtraction operation with blending"""
    
    def __init__(self, sdf_a: SDFNode, sdf_b: SDFNode, blend_radius: float = 0.1, seed: int = 42):
        super().__init__(seed)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b
        self.blend_radius = blend_radius
    
    def evaluate(self, point: np.ndarray) -> float:
        """Smooth subtraction with polynomial blending"""
        dist_a = self.sdf_a.evaluate(point)
        dist_b = self.sdf_b.evaluate(point)
        
        h = max(self.blend_radius - abs(-dist_b - dist_a), 0.0) / self.blend_radius
        return max(dist_a, -dist_b) + h * h * self.blend_radius * 0.25
    
    def __repr__(self):
        return f"SDFSmoothSubtract({self.sdf_a}, {self.sdf_b}, blend={self.blend_radius}, seed={self.seed})"


# Utility functions for SDF operations

def sdf_gradient(sdf: SDFNode, point: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Compute gradient of SDF at point using finite differences"""
    grad = np.zeros(3)
    
    for i in range(3):
        point_plus = point.copy()
        point_minus = point.copy()
        
        point_plus[i] += epsilon
        point_minus[i] -= epsilon
        
        grad[i] = (sdf.evaluate(point_plus) - sdf.evaluate(point_minus)) / (2.0 * epsilon)
    
    return grad


def sdf_normal(sdf: SDFNode, point: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Compute unit normal vector of SDF surface at point"""
    grad = sdf_gradient(sdf, point, epsilon)
    length = np.linalg.norm(grad)
    
    if length < 1e-8:
        return np.array([0.0, 1.0, 0.0])  # Default up vector
    
    return grad / length


def sdf_raymarch(sdf: SDFNode, ray_origin: np.ndarray, ray_direction: np.ndarray, 
                max_distance: float = 100.0, epsilon: float = 1e-4, max_steps: int = 100) -> Optional[float]:
    """Raymarch to find intersection with SDF surface"""
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    total_distance = 0.0
    
    for _ in range(max_steps):
        current_point = ray_origin + ray_direction * total_distance
        distance = sdf.evaluate(current_point)
        
        if distance < epsilon:
            return total_distance  # Hit surface
        
        if total_distance > max_distance:
            break  # Ray escaped
        
        total_distance += distance
    
    return None  # No intersection found


if __name__ == "__main__":
    # Example usage and basic testing
    print("🚀 T09 SDF Primitives System")
    print("=" * 50)
    
    # Test basic primitives
    sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    capsule = SDFCapsule(point_a=[0, -1, 0], point_b=[0, 1, 0], radius=0.5, seed=42)
    gyroid = SDFGyroid(scale=2.0, thickness=0.1, seed=42)
    
    test_point = np.array([0.5, 0.5, 0.5])
    
    print(f"Test point: {test_point}")
    print(f"Sphere distance: {sphere.evaluate(test_point):.3f}")
    print(f"Capsule distance: {capsule.evaluate(test_point):.3f}")
    print(f"Gyroid distance: {gyroid.evaluate(test_point):.3f}")
    
    # Test boolean operations
    union = SDFUnion(sphere, capsule, seed=42)
    subtract = SDFSubtract(sphere, capsule, seed=42)
    intersect = SDFIntersect(sphere, capsule, seed=42)
    
    print(f"\nBoolean operations at {test_point}:")
    print(f"Union distance: {union.evaluate(test_point):.3f}")
    print(f"Subtract distance: {subtract.evaluate(test_point):.3f}")
    print(f"Intersect distance: {intersect.evaluate(test_point):.3f}")
    
    # Test noise displacement
    noisy_sphere = SDFNoiseDisplace(sphere, displacement_scale=0.1, noise_frequency=2.0, seed=42)
    print(f"Noisy sphere distance: {noisy_sphere.evaluate(test_point):.3f}")
    
    print("\n✅ SDF primitives system initialized")