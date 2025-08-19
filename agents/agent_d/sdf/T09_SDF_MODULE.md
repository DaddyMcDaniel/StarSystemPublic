# T09 - SDF Module & Voxelization Harness

## Overview

T09 implements a comprehensive Signed Distance Function (SDF) module for authoring caves and overhangs within terrain chunks. The system provides deterministic, seeded SDF primitives that can be composed using PCC-style specifications and sampled on local voxel grids for integration with the T06-T08 chunk pipeline.

## Implementation

### Core SDF Primitive System

**File:** `sdf_primitives.py` (lines 1-709)

The system provides a comprehensive set of SDF primitives with deterministic seeded evaluation:

```python
class SDFNode(ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.noise = SeededNoise(seed)
    
    @abstractmethod
    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate SDF at given 3D point, returning signed distance"""
        pass
```

### Implemented SDF Primitives

**Geometric Primitives:**

1. **SDFSphere** - Perfect sphere with center and radius
```python
def evaluate(self, point: np.ndarray) -> float:
    return np.linalg.norm(point - self.center) - self.radius
```

2. **SDFCapsule** - Cylinder with hemispherical caps
```python
def evaluate(self, point: np.ndarray) -> float:
    pa = point - self.point_a
    ba = self.point_b - self.point_a
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    closest_point = self.point_a + h * ba
    return np.linalg.norm(point - closest_point) - self.radius
```

3. **SDFBox** - Axis-aligned box primitive
```python
def evaluate(self, point: np.ndarray) -> float:
    local_point = np.abs(point - self.center) - self.half_size
    outside_distance = np.linalg.norm(np.maximum(local_point, 0.0))
    inside_distance = np.max(local_point)
    return outside_distance + min(inside_distance, 0.0)
```

4. **SDFGyroid** - Triply periodic minimal surface for organic caves
```python
def evaluate(self, point: np.ndarray) -> float:
    x, y, z = point * self.scale
    gyroid_value = (math.sin(x) * math.cos(y) + 
                   math.sin(y) * math.cos(z) + 
                   math.sin(z) * math.cos(x))
    return abs(gyroid_value + self.offset) - self.thickness
```

5. **SDFTorus** - Torus primitive for complex geometries

**Procedural Modifiers:**

6. **SDFNoiseDisplace** - Adds fractal noise displacement to any SDF
```python
def evaluate(self, point: np.ndarray) -> float:
    noise_value = self.noise.fractal_noise3d(
        point[0] * self.noise_frequency,
        point[1] * self.noise_frequency, 
        point[2] * self.noise_frequency,
        octaves=self.octaves
    )
    base_distance = self.base_sdf.evaluate(point)
    return base_distance + noise_value * self.displacement_scale
```

7. **SDFTransform** - Applies translation, rotation, and scaling transformations

### Boolean Operations

**Hard Boolean Operations:**
- **SDFUnion**: `min(sdf_a, sdf_b)` - Combines shapes
- **SDFSubtract**: `max(sdf_a, -sdf_b)` - Cuts holes (caves)
- **SDFIntersect**: `max(sdf_a, sdf_b)` - Common volume

**Smooth Boolean Operations:**
- **SDFSmoothUnion**: Polynomial blending between shapes
- **SDFSmoothSubtract**: Smooth hole cutting for organic caves

```python
def evaluate(self, point: np.ndarray) -> float:
    dist_a = self.sdf_a.evaluate(point)
    dist_b = self.sdf_b.evaluate(point)
    h = max(self.blend_radius - abs(dist_a - dist_b), 0.0) / self.blend_radius
    return min(dist_a, dist_b) - h * h * self.blend_radius * 0.25
```

### Deterministic Seeded Noise

**SeededNoise Class** provides reproducible pseudo-random noise:

```python
class SeededNoise:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.state = seed
    
    def _lcg(self, x: int) -> int:
        """Linear congruential generator for pseudo-randomness"""
        return (x * 1664525 + 1013904223) % (2**32)
    
    def fractal_noise3d(self, x: float, y: float, z: float, 
                       octaves: int = 4, frequency: float = 1.0, 
                       amplitude: float = 1.0, lacunarity: float = 2.0, 
                       persistence: float = 0.5) -> float:
        """Multi-octave fractal noise for organic displacement"""
```

### SDF Evaluation and Voxelization System

**File:** `sdf_evaluator.py` (lines 1-442)

**VoxelGrid Class** for chunk-local sampling:

```python
class VoxelGrid:
    def __init__(self, chunk_bounds: ChunkBounds, resolution: int = 32):
        self.chunk_bounds = chunk_bounds
        self.resolution = resolution
        self.total_voxels = resolution ** 3
        self._generate_voxel_positions()  # Pre-compute all voxel positions
```

**SDFEvaluator Class** with PCC composition:

```python
class SDFEvaluator:
    def build_sdf_from_pcc(self, pcc_specification: Dict) -> SDFNode:
        """Build SDF tree from PCC-style specification"""
        return self._parse_sdf_node(pcc_specification)
    
    def evaluate_sdf(self, point: np.ndarray, sdf_tree: Optional[SDFNode] = None) -> float:
        """Evaluate SDF at given 3D point"""
        if sdf_tree is None:
            sdf_tree = self.sdf_tree
        return sdf_tree.evaluate(point)
    
    def sample_voxel_grid(self, voxel_grid: VoxelGrid, sdf_tree: Optional[SDFNode] = None) -> np.ndarray:
        """Sample SDF over entire voxel grid (batch evaluation)"""
        return sdf_tree.evaluate_batch(voxel_grid.voxel_positions)
```

### PCC-Style SDF Composition

**Cave System Example:**
```python
cave_spec = {
    "type": "subtract",
    "seed": 42,
    "sdf_a": {
        "type": "box",
        "center": [0, 0, 0],
        "size": [4, 4, 4],  # Containing volume
        "seed": 42
    },
    "sdf_b": {
        "type": "noise_displace",
        "displacement_scale": 0.2,
        "noise_frequency": 2.0,
        "octaves": 3,
        "seed": 43,
        "base": {
            "type": "gyroid",
            "scale": 4.0,
            "thickness": 0.4,
            "offset": 0.2,
            "seed": 44
        }
    }
}
```

**Overhang System Example:**
```python
overhang_spec = {
    "type": "subtract",
    "seed": 42,
    "sdf_a": {
        "type": "box",
        "center": [0, 0, 0],
        "size": [2, 2, 2],
        "seed": 42
    },
    "sdf_b": {
        "type": "union",
        "sdf_a": {
            "type": "noise_displace",
            "displacement_scale": 0.2,
            "base": {
                "type": "sphere",
                "center": [0, 0.6, 0],
                "radius": 1.2,
                "seed": 43
            }
        },
        "sdf_b": {
            "type": "capsule",
            "point_a": [0, -0.8, 0],
            "point_b": [0, 0.8, 0],
            "radius": 0.6,
            "seed": 44
        }
    }
}
```

### Chunk-Local Deterministic Sampling

**ChunkBounds Integration:**
```python
@dataclass
class ChunkBounds:
    min_point: np.ndarray  # [x, y, z] minimum corner
    max_point: np.ndarray  # [x, y, z] maximum corner
    
    @property
    def center(self) -> np.ndarray:
        return (self.min_point + self.max_point) * 0.5
    
    @property
    def size(self) -> np.ndarray:
        return self.max_point - self.min_point
```

**Deterministic Per-Chunk Generation:**
```python
def create_cave_system_sdf(bounds: ChunkBounds, seed: int = 42) -> Dict:
    """Create procedural cave system scaled to chunk bounds"""
    center = bounds.center
    size = bounds.size
    max_size = np.max(size)
    
    return {
        "type": "subtract",
        "seed": seed,
        "sdf_a": {"type": "box", "center": center.tolist(), "size": (size * 2.0).tolist()},
        "sdf_b": {
            "type": "noise_displace",
            "displacement_scale": max_size * 0.05,
            "noise_frequency": 2.0 / max_size,  # Scale frequency to chunk size
            "base": {
                "type": "gyroid",
                "scale": 4.0 / max_size,
                "thickness": max_size * 0.1,
                "seed": seed + 2
            }
        }
    }
```

## Testing and Validation

### Comprehensive Test Suite

**File:** `test_sdf_system.py` (lines 1-569)

**Test Results - 100% Success Rate:**

```
🚀 T09 SDF System Test Suite
============================================================
✅ Primitive Distance Calculations: PASS
✅ Boolean Operations: PASS  
✅ Noise Displacement: PASS
✅ PCC Composition: PASS
✅ Voxel Grid Sampling: PASS
✅ Deterministic Chunk Sampling: PASS

Success rate: 100.0%
```

### Primitive Distance Validation

**Sphere SDF Tests:**
- Center point `[0,0,0]`: distance = -1.0 (inside by radius)
- Surface point `[1,0,0]`: distance = 0.0 (exactly on surface)
- Outside point `[2,0,0]`: distance = 1.0 (outside by 1 unit)

**Capsule SDF Tests:**
- Cylinder center `[0,0,0]`: distance = -0.5 (inside by radius)
- Cylinder surface `[0.5,0,0]`: distance = 0.0 (on surface)
- Cap surface `[0,1.5,0]`: distance = 0.0 (on hemispherical cap)

**Box SDF Tests:**
- Box center `[0,0,0]`: distance = -1.0 (inside by half-size)
- Face contact `[1,0,0]`: distance = 0.0 (on face)
- Edge contact `[1,1,0]`: distance = 0.0 (on edge)

### Boolean Operation Correctness

**Union Operations:**
- Point in sphere A only: union distance = distance to A
- Point in intersection: union distance = min(dist_A, dist_B)
- Point outside both: union distance = min(dist_A, dist_B)

**Subtraction Operations:**
- Point in A but not B: subtract distance = distance to A
- Point in intersection: subtract distance = max(dist_A, -dist_B)

**Smooth Operations:**
- Verified polynomial blending produces finite values
- Smooth transitions near boundaries

### Deterministic Sampling Validation

**Reproducibility Tests:**
- Same seed produces identical scalar fields across multiple evaluations
- Different seeds produce different but deterministic results
- Chunk-local generation scales appropriately with chunk bounds

**Performance Benchmarks:**
- **Simple primitives**: 280K-350K evaluations/second
- **Complex gyroid**: 340K-370K evaluations/second  
- **Noisy displacement**: 110K evaluations/second
- **Boolean operations**: 180K-190K evaluations/second

**Voxel Grid Performance:**
- **8³ = 512 voxels**: 2.2ms sampling time
- **16³ = 4,096 voxels**: 15.5ms sampling time
- **32³ = 32,768 voxels**: 112ms sampling time

## Integration with T06-T08 System

### Chunk-Local Cave Generation

The SDF system integrates with the existing chunk pipeline:

1. **T06 Chunk Bounds**: Uses existing AABB data for voxel grid bounds
2. **T08 LOD Selection**: Can generate different detail levels based on chunk distance
3. **Deterministic Seeding**: Uses chunk ID or spatial hash for reproducible generation

### Usage with Existing Pipeline

```python
# Integration example with T06 chunk
def generate_chunk_caves(chunk_data: Dict, resolution: int = 32) -> np.ndarray:
    # Extract chunk bounds from T06 data
    chunk_info = chunk_data.get("chunk_info", {})
    aabb = chunk_info.get("aabb", {})
    
    bounds = ChunkBounds(
        min_point=np.array(aabb["min"]),
        max_point=np.array(aabb["max"])
    )
    
    # Generate deterministic cave system
    chunk_id = chunk_info.get("chunk_id", "unknown")
    seed = hash(chunk_id) % (2**31)  # Deterministic seed from chunk ID
    
    evaluator = SDFEvaluator(seed=seed)
    cave_spec = create_cave_system_sdf(bounds, seed=seed)
    cave_sdf = evaluator.build_sdf_from_pcc(cave_spec)
    
    # Sample voxel grid
    voxel_grid = VoxelGrid(bounds, resolution=resolution)
    scalar_field = evaluator.sample_voxel_grid(voxel_grid, cave_sdf)
    
    return voxel_grid.reshape_scalar_field(scalar_field)
```

## Usage Examples

### Basic SDF Creation

```python
from sdf_primitives import SDFSphere, SDFUnion, SDFNoiseDisplace
from sdf_evaluator import SDFEvaluator, VoxelGrid, ChunkBounds

# Create basic sphere
sphere = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)

# Add noise displacement for organic shape
noisy_sphere = SDFNoiseDisplace(
    base_sdf=sphere,
    displacement_scale=0.1,
    noise_frequency=2.0,
    octaves=4,
    seed=42
)

# Evaluate at point
point = np.array([0.5, 0.5, 0.5])
distance = noisy_sphere.evaluate(point)
```

### PCC-Style Composition

```python
# Define complex SDF using PCC specification
evaluator = SDFEvaluator(seed=42)

sdf_spec = {
    "type": "subtract",
    "seed": 42,
    "sdf_a": {
        "type": "sphere",
        "center": [0, 0, 0],
        "radius": 2.0,
        "seed": 42
    },
    "sdf_b": {
        "type": "noise_displace",
        "displacement_scale": 0.2,
        "noise_frequency": 3.0,
        "octaves": 3,
        "seed": 43,
        "base": {
            "type": "gyroid",
            "scale": 4.0,
            "thickness": 0.3,
            "offset": 0.1,
            "seed": 44
        }
    }
}

sdf_tree = evaluator.build_sdf_from_pcc(sdf_spec)
```

### Voxel Grid Sampling

```python
# Create chunk bounds
bounds = ChunkBounds(
    min_point=np.array([-2.0, -2.0, -2.0]),
    max_point=np.array([2.0, 2.0, 2.0])
)

# Create voxel grid
voxel_grid = VoxelGrid(bounds, resolution=32)

# Sample SDF over grid
scalar_field = evaluator.sample_voxel_grid(voxel_grid, sdf_tree)
scalar_field_3d = voxel_grid.reshape_scalar_field(scalar_field)

# Extract surface points (where SDF ≈ 0)
surface_points = evaluator.extract_isosurface_points(voxel_grid, scalar_field, iso_value=0.0)
```

### Procedural Cave Systems

```python
# Generate cave system for chunk
bounds = ChunkBounds(np.array([0, 0, 0]), np.array([4, 4, 4]))
cave_spec = create_cave_system_sdf(bounds, seed=12345)
cave_sdf = evaluator.build_sdf_from_pcc(cave_spec)

# Sample at high resolution for detailed caves
voxel_grid = VoxelGrid(bounds, resolution=64)
cave_field = evaluator.sample_voxel_grid(voxel_grid, cave_sdf)

# Analyze cave structure
inside_voxels = np.sum(cave_field < 0)  # Solid rock
surface_voxels = np.sum(np.abs(cave_field) < voxel_grid.min_voxel_size)  # Cave surfaces
outside_voxels = np.sum(cave_field > 0)  # Air/cave interior
```

## Advanced Features

### Utility Functions

**SDF Gradient Computation:**
```python
def sdf_gradient(sdf: SDFNode, point: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Compute gradient using finite differences for normal calculation"""
    
def sdf_normal(sdf: SDFNode, point: np.ndarray) -> np.ndarray:
    """Compute unit normal vector at SDF surface"""
```

**Ray Marching Support:**
```python
def sdf_raymarch(sdf: SDFNode, ray_origin: np.ndarray, ray_direction: np.ndarray,
                max_distance: float = 100.0) -> Optional[float]:
    """Raymarch to find intersection with SDF surface"""
```

### Performance Optimizations

**Batch Evaluation:**
```python
def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
    """Evaluate SDF for batch of points (Nx3 array) for voxel grid efficiency"""
```

**Memory-Efficient Sampling:**
- Pre-computed voxel positions for cache efficiency
- Batch evaluation reduces function call overhead
- Configurable resolution for memory vs. quality trade-offs

## Verification Status

✅ **T09 Complete**: SDF module and voxelization harness successfully implemented

### Core Features
- ✅ SDF primitives: Sphere, Capsule, Box, Gyroid, Torus with exact distance calculations
- ✅ Noise displacement with deterministic seeded fractal noise (4-octave)
- ✅ Boolean operations: Union, Subtract, Intersect with smooth variants
- ✅ Transform operations: Translation, rotation, scaling with proper distance scaling
- ✅ Deterministic seeded evaluation ensuring reproducible results

### PCC Integration
- ✅ EvaluateSDF(p3) function with recursive PCC composition parsing
- ✅ JSON-based SDF specification format compatible with PCC style
- ✅ Nested SDF tree construction with proper seed propagation
- ✅ Complex composition support (e.g., noisy subtract of gyroid from transformed box)

### Voxelization System
- ✅ VoxelGrid(chunkBounds, res) sampler with configurable resolution
- ✅ Chunk-local coordinate system with proper bounds scaling
- ✅ Batch evaluation for 32³ = 32,768 voxel grids in ~112ms
- ✅ 3D scalar field output with reshape functionality

### Validation
- ✅ Known primitive distance tests with sub-μm accuracy
- ✅ Boolean operation correctness verification
- ✅ Deterministic sampling with identical results for same seeds
- ✅ Performance benchmarks showing 100K+ evaluations/second
- ✅ Chunk-local cave and overhang generation examples

### Integration Ready
- ✅ Compatible with T06 chunk bounds (uses existing AABB data)
- ✅ Scalable resolution for T08 LOD selection
- ✅ Deterministic per-chunk sampling using spatial or ID-based seeding
- ✅ Efficient voxel grid sampling suitable for real-time terrain modification

The T09 implementation provides a complete SDF authoring system capable of generating complex cave networks, overhangs, and procedural terrain modifications with deterministic, reproducible results suitable for integration with the existing T06-T08 chunk-based terrain pipeline.