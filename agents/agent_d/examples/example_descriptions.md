# PCC Example Files - T15

## Overview

This directory contains example PCC files demonstrating the T15 hardened schema. These examples showcase different complexity levels and use cases for the PCC terrain pipeline.

## Example Files

### 1. Minimal Sphere (`minimal_sphere.pcc`)

**Purpose**: Simplest possible valid PCC terrain file  
**Complexity**: Minimal (2 nodes, 1 connection)  
**Use Case**: Learning PCC basics, schema validation testing

**Features**:
- Single `CubeSphere` primitive generator
- Direct output to `MarchingCubes` mesh generation
- No stochastic nodes (no seed requirements)
- Basic parameter usage with safe defaults

**Pipeline Flow**:
```
CubeSphere → MarchingCubes → Mesh Output
```

**Key Learnings**:
- Minimal valid PCC structure  
- Required fields and basic parameter types
- Simple node connection pattern
- Basic mesh generation from primitives

**Generated Terrain**: Simple smooth sphere mesh at 100m radius

---

### 2. Hero Planet (`hero_planet.pcc`)

**Purpose**: Complex planetary terrain showcasing full PCC capabilities  
**Complexity**: Advanced (14 nodes, 13 connections)  
**Use Case**: Production terrain, feature demonstration, performance testing

**Features**:
- Multi-scale noise generation (continental, mountain, detail)
- Stochastic nodes with explicit seeds and units
- Domain warping for natural terrain variation  
- SDF cave system with boolean operations
- Displacement layers for height variation
- Level-of-detail management with QuadtreeLOD
- Complete parameter range utilization

**Pipeline Flow**:
```
CubeSphere → Continental Noise Displacement → Domain Warp
    ↓
Mountain Ridge Displacement → Detail Noise Displacement
    ↓
Cave System (NoiseFBM + RidgedMF → Union → Smooth)
    ↓
Terrain-Cave Subtraction → MarchingCubes → QuadtreeLOD
```

**Stochastic Nodes**:
- `continental_noise`: Seed 12345, units "km", large-scale landmasses
- `mountain_ridges`: Seed 23456, units "km", ridged mountain ranges  
- `detail_noise`: Seed 34567, units "m", fine surface detail
- `cave_system_base`: Seed 45678, units "m", cave network base
- `cave_detail`: Seed 56789, units "m", cave detail and variation

**Key Learnings**:
- Complex multi-layer terrain generation
- Proper seed management for deterministic results
- Scale-appropriate units usage (km for continents, m for details)
- SDF operations for cave generation
- Domain warping for natural variation
- LOD system integration for performance

**Generated Terrain**: 
- 1000m radius planet with realistic terrain variation
- Continental-scale features (200m amplitude) 
- Mountain ridges up to 400m height
- Fine detail variation (±25m)
- Internal cave systems with smooth SDF operations
- Optimized for viewing from 2km distance

## Parameter Usage Examples

### Stochastic Node Seed Management

The hero planet example demonstrates proper seed derivation:

```json
{
  "continental_noise": {"seed": 12345},  // Base seed
  "mountain_ridges": {"seed": 23456},    // Base + 11111  
  "detail_noise": {"seed": 34567},       // Base + 22222
  "cave_system_base": {"seed": 45678},   // Base + 33333
  "cave_detail": {"seed": 56789}        // Base + 44444
}
```

### Units Usage by Scale

```json
{
  "continental_features": {"units": "km", "frequency": 0.002},  // 500km wavelength
  "mountain_features": {"units": "km", "frequency": 0.008},     // 125km wavelength  
  "surface_detail": {"units": "m", "frequency": 0.05}          // 20m wavelength
}
```

### Amplitude Scaling

```json
{
  "continental_noise": {"amplitude": 200.0},    // Major landmass variation
  "mountain_ridges": {"amplitude": 400.0},      // Dramatic mountain peaks
  "detail_noise": {"amplitude": 25.0}           // Surface texture detail
}
```

## Validation Testing

Both examples are designed to pass T15 schema validation:

### Minimal Sphere Validation
```bash
python validation/pcc_validator.py examples/minimal_sphere.pcc
# Expected: ✅ Valid with 0 errors
```

### Hero Planet Validation  
```bash
python validation/pcc_validator.py examples/hero_planet.pcc
# Expected: ✅ Valid with 0 errors, complex graph validation
```

## Performance Characteristics

### Minimal Sphere
- **Generation Time**: ~1 second
- **Memory Usage**: ~10MB
- **Mesh Complexity**: ~16K triangles
- **Use Case**: Real-time generation, testing

### Hero Planet
- **Generation Time**: ~30-60 seconds (depending on hardware)
- **Memory Usage**: ~500MB-1GB
- **Mesh Complexity**: ~500K-2M triangles (with LOD)
- **Use Case**: Offline generation, cinematic quality

## Extension Examples

### Adding Vegetation Layer
```json
{
  "vegetation_noise": {
    "type": "NoiseFBM",
    "parameters": {
      "seed": 67890,      // Next in sequence
      "units": "m",
      "frequency": 0.1,
      "amplitude": 1.0,
      "octaves": 4
    }
  }
}
```

### Multiple Cave Systems
```json
{
  "large_caves": {"seed": 70000, "frequency": 0.01},
  "small_caves": {"seed": 71000, "frequency": 0.05},
  "cave_intersection": {"type": "SDF.Intersect"}
}
```

### Atmospheric Noise
```json
{
  "cloud_layer": {
    "seed": 80000,
    "units": "km", 
    "frequency": 0.001,
    "amplitude": 100.0
  }
}
```

## Best Practices Demonstrated

1. **Hierarchical Seed Management**: Deterministic but varied seeds
2. **Appropriate Unit Usage**: km for large features, m for details
3. **Amplitude Scaling**: Realistic terrain proportions
4. **Parameter Ranges**: Within documented safe limits
5. **Graph Structure**: No cycles, proper data flow
6. **Performance Considerations**: LOD for complex terrain
7. **Schema Compliance**: Full validation with helpful errors

These examples serve as both learning materials and validation references for the T15 PCC schema hardening system.