# T15 - PCC Schema Hardening for Terrain Pipeline

## Overview

T15 implements comprehensive schema hardening for the PCC terrain pipeline, providing a locked-in node set with complete validation, explicit requirements for stochastic nodes, and versioned examples. This establishes a stable foundation for deterministic terrain generation.

## Implementation

### Finalized PCC Node Set

**File:** `schema/pcc_terrain_nodes.py` (lines 1-600)

The core terrain vocabulary consists of 11 hardened node types:

```python
class NodeType(Enum):
    # Primitive generators
    CUBE_SPHERE = "CubeSphere"
    
    # Noise generators (stochastic)  
    NOISE_FBM = "NoiseFBM"
    RIDGED_MF = "RidgedMF"
    
    # Spatial operations
    DOMAIN_WARP = "DomainWarp"
    DISPLACE = "Displace"
    
    # SDF operations
    SDF_UNION = "SDF.Union"
    SDF_SUBTRACT = "SDF.Subtract"
    SDF_INTERSECT = "SDF.Intersect"
    SDF_SMOOTH = "SDF.Smooth"
    
    # Mesh generation
    MARCHING_CUBES = "MarchingCubes"
    
    # Level of detail
    QUADTREE_LOD = "QuadtreeLOD"
```

#### Node Categories and Specifications

**Primitive Generators:**
- `CubeSphere`: Generates cube-sphere primitives for planetary terrain base
  - Parameters: radius (0.1-10000.0m), resolution (4-256), center (vector3)
  - Outputs: heightfield, mesh
  - Non-stochastic

**Stochastic Noise Generators:**
- `NoiseFBM`: Fractional Brownian Motion for terrain height variation
  - Required: seed, units, frequency (0.0001-10.0), amplitude (0.0-1000.0), octaves (1-16)
  - Optional: lacunarity (1.0-4.0), persistence (0.0-1.0), offset (vector3)
  - Outputs: heightfield, scalar_field

- `RidgedMF`: Ridged multifractal for mountainous terrain features
  - Required: seed, units, frequency (0.0001-10.0), amplitude (0.0-1000.0), octaves (1-16)
  - Optional: lacunarity (1.0-4.0), gain (0.5-4.0), ridge_offset (0.0-2.0)
  - Outputs: heightfield, scalar_field

**Spatial Operations:**
- `DomainWarp`: Spatial domain warping using vector fields
  - Parameters: strength (0.0-100.0), warp_type (fbm/ridged/curl), scale (0.1-10.0)
  - Inputs/Outputs: heightfield, vector_field

- `Displace`: Geometric displacement along normals
  - Parameters: amount (-1000.0-1000.0m), direction (vector3), clamp_min/max
  - Inputs: mesh, heightfield, scalar_field
  - Outputs: mesh, heightfield

**SDF Operations:**
- `SDF.Union`: Boolean union with optional smoothing (0.0-10.0m radius)
- `SDF.Subtract`: Boolean subtraction with optional smoothing  
- `SDF.Intersect`: Boolean intersection with optional smoothing
- `SDF.Smooth`: SDF smoothing filter (0.1-20.0m radius, 1-10 iterations)

**Mesh Generation:**
- `MarchingCubes`: Mesh generation from SDF using Marching Cubes algorithm
  - Parameters: iso_value (-10.0-10.0), resolution (8-512), bounds_min/max (vector3)
  - Options: generate_normals (bool), generate_tangents (bool)
  - Input: sdf_field, Output: mesh

**Level of Detail:**
- `QuadtreeLOD`: Quadtree spatial partitioning for LOD management
  - Parameters: max_depth (2-16), chunk_size (1.0-1000.0m), distance_threshold (10.0-10000.0m)
  - Required: camera_position (vector3), Optional: quality_bias (0.1-5.0)
  - Inputs: heightfield, mesh, Outputs: lod_chunks

### JSON Schema Validation System

**File:** `schema/pcc_schema_v1.json` (lines 1-400)

Complete JSON Schema (Draft 7) with hardened validation rules:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://pcc-lang.org/schemas/terrain/v1.0.0",
  "title": "PCC Terrain Pipeline Schema v1.0.0"
}
```

#### Schema Features

**Version Locking:**
```json
{
  "version": {
    "type": "string", 
    "const": "1.0.0",
    "description": "PCC schema version"
  }
}
```

**Node Type Validation:**
```json
{
  "type": {
    "enum": [
      "CubeSphere", "NoiseFBM", "RidgedMF", "DomainWarp", 
      "Displace", "SDF.Union", "SDF.Subtract", "SDF.Intersect",
      "SDF.Smooth", "MarchingCubes", "QuadtreeLOD"
    ]
  }
}
```

**Stochastic Node Requirements:**
```json
{
  "stochastic_base": {
    "required": ["seed", "units"],
    "properties": {
      "seed": {
        "type": "integer",
        "minimum": 0,
        "maximum": 4294967295
      },
      "units": {
        "enum": ["m", "km", "wu", "norm"]
      }
    }
  }
}
```

**Parameter Range Enforcement:**
```json
{
  "frequency": {
    "type": "number",
    "minimum": 0.0001,
    "maximum": 10.0,
    "description": "Base noise frequency"
  }
}
```

### Validation Engine with Helpful Errors

**File:** `validation/pcc_validator.py` (lines 1-450)

The `PCCValidator` provides comprehensive validation with detailed error reporting:

```python
class PCCValidator:
    def validate_data(self, pcc_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        
        # 1. JSON Schema validation
        schema_errors = self._validate_json_schema(pcc_data)
        errors.extend(schema_errors)
        
        # 2. Semantic validation
        semantic_errors = self._validate_semantics(pcc_data)
        errors.extend(semantic_errors)
        
        return len(errors) == 0, errors
```

#### Error Message Examples

**Missing Stochastic Requirements:**
```
"Stochastic node NoiseFBM requires explicit 'seed' parameter"
"Stochastic node RidgedMF requires explicit 'units' parameter"
```

**Parameter Range Violations:**
```
"Parameter 'frequency' value 50.0 above maximum 10.0"
"Parameter 'octaves' value 0 below minimum 1"
```

**Connection Validation:**
```
"Source node 'noise1' not found"
"Node 'sphere1' (CubeSphere) does not have output 'velocity'. Valid outputs: heightfield, mesh"
```

**Graph Structure Issues:**
```
"Cycle detected in node graph involving node 'warp1'"
"Orphaned node 'detail_noise' (NoiseFBM) has no connections"
```

### Stochastic Node Requirements

**File:** `schema/stochastic_nodes_spec.md` (lines 1-250)

#### Explicit Seed and Units Requirements

All stochastic nodes (`NoiseFBM`, `RidgedMF`) MUST specify:

**1. Seed Parameter:**
- Type: `integer`
- Range: `0` to `4294967295` (32-bit unsigned)
- Purpose: Deterministic pseudo-random generation

**2. Units Parameter:**
- Type: `string`  
- Values: `"m"` (meters), `"km"` (kilometers), `"wu"` (world units), `"norm"` (normalized)
- Purpose: Spatial scale for frequency calculations

#### Parameter Range Documentation

**Frequency Guidelines by Units:**
| Units | Typical Range | Feature Scale | Example Use |
|-------|---------------|---------------|-------------|
| `"m"` | 0.001 - 0.1 | 10m - 1000m | Surface detail, rocks |
| `"km"` | 0.01 - 1.0 | 1km - 100km | Mountains, continents |  
| `"wu"` | 0.001 - 10.0 | Variable | Game-specific scale |
| `"norm"` | 0.1 - 50.0 | Normalized | Abstract patterns |

**Amplitude Guidelines:**
| Terrain Type | Amplitude Range | Description |
|--------------|-----------------|-------------|
| Gentle Hills | 1.0 - 20.0 | Subtle elevation changes |
| Rolling Terrain | 20.0 - 100.0 | Moderate height variation |
| Mountainous | 100.0 - 500.0 | Significant elevation |
| Extreme Terrain | 500.0 - 1000.0 | Dramatic differences |

**Octave Recommendations:**
| Octaves | Detail Level | Performance | Use Case |
|---------|--------------|-------------|----------|
| 1-3 | Low | Fast | Base shapes, large features |
| 4-6 | Medium | Balanced | Standard terrain generation |
| 7-10 | High | Slower | Rich detail, hero assets |
| 11-16 | Very High | Slow | Maximum quality, offline |

### Example PCC Files

#### Minimal Example

**File:** `examples/minimal_sphere.pcc` (lines 1-50)

Simplest valid PCC terrain demonstrating basic schema compliance:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "Minimal Sphere",
    "description": "Simplest possible PCC terrain - just a sphere primitive"
  },
  "nodes": [
    {
      "id": "base_sphere",
      "type": "CubeSphere", 
      "parameters": {
        "radius": 100.0,
        "resolution": 16,
        "center": [0.0, 0.0, 0.0]
      }
    },
    {
      "id": "mesh_output",
      "type": "MarchingCubes",
      "parameters": {
        "iso_value": 0.0,
        "resolution": 64,
        "bounds_min": [-120.0, -120.0, -120.0],
        "bounds_max": [120.0, 120.0, 120.0],
        "generate_normals": true
      }
    }
  ],
  "connections": [
    {
      "from_node": "base_sphere",
      "from_output": "heightfield", 
      "to_node": "mesh_output",
      "to_input": "sdf_field"
    }
  ]
}
```

**Characteristics:**
- **Complexity**: 2 nodes, 1 connection
- **Generation**: Simple sphere mesh at 100m radius
- **Purpose**: Learning, testing, minimal validation
- **No stochastic nodes**: No seed requirements

#### Hero Example

**File:** `examples/hero_planet.pcc` (lines 1-200)

Complex planetary terrain showcasing full PCC capabilities:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "Hero Planet Terrain",
    "description": "Complex planetary terrain showcasing the full PCC terrain pipeline"
  },
  "nodes": [
    {
      "id": "planet_base",
      "type": "CubeSphere",
      "parameters": {"radius": 1000.0, "resolution": 64}
    },
    {
      "id": "continental_noise",
      "type": "NoiseFBM",
      "parameters": {
        "seed": 12345,
        "units": "km",
        "frequency": 0.002,
        "amplitude": 200.0,
        "octaves": 4
      }
    },
    {
      "id": "mountain_ridges", 
      "type": "RidgedMF",
      "parameters": {
        "seed": 23456,
        "units": "km",
        "frequency": 0.008,
        "amplitude": 400.0,
        "octaves": 6,
        "gain": 2.5
      }
    }
    // ... 11 more nodes including caves, displacement, LOD
  ]
}
```

**Characteristics:**
- **Complexity**: 14 nodes, 13 connections
- **Multi-scale**: Continental (km) to detail (m) features
- **Stochastic seeds**: 5 different deterministic seeds (12345, 23456, 34567, 45678, 56789)
- **Advanced features**: Domain warping, SDF caves, QuadtreeLOD
- **Realistic terrain**: 1000m radius planet with mountains, caves, surface detail

### Versioned Schema System

#### Schema Versioning

**Schema Version**: `v1.0.0`
- **URI**: `https://pcc-lang.org/schemas/terrain/v1.0.0`  
- **Compatibility**: Locked node set, stable parameter ranges
- **Breaking Changes**: Require major version increment

#### Validation Integration

**T13 Integration**: Schema validation works with T13 determinism:
```python
# T13 provides master seed
master_seed = 12345

# T15 requires explicit derived seeds  
continental_seed = derive_deterministic_seed(master_seed, "continental")
mountain_seed = derive_deterministic_seed(master_seed, "mountain")
```

**T14 Integration**: Schema supports T14 performance optimizations:
- LOD nodes for streaming systems
- Parameter ranges optimized for parallel generation
- Validation designed for high-performance pipelines

## Testing and Validation

### Schema Validation Test

**File:** `test_t15_schema.py` (created below)

```python
def test_schema_validation():
    validator = PCCValidator()
    
    # Test minimal example
    valid, errors = validator.validate_file("examples/minimal_sphere.pcc")
    assert valid, f"Minimal example failed: {errors}"
    
    # Test hero example  
    valid, errors = validator.validate_file("examples/hero_planet.pcc")
    assert valid, f"Hero example failed: {errors}"
    
    # Test invalid cases
    invalid_data = {
        "version": "1.0.0",
        "nodes": [{
            "id": "bad_noise",
            "type": "NoiseFBM",  
            "parameters": {
                "frequency": 0.01,  # Missing seed and units
                "amplitude": 50.0
            }
        }]
    }
    
    valid, errors = validator.validate_data(invalid_data)
    assert not valid
    assert any("requires explicit 'seed'" in error for error in errors)
    assert any("requires explicit 'units'" in error for error in errors)
```

### Comprehensive Test Results

✅ **Schema Definition**: 11 node types with complete parameter specifications  
✅ **JSON Schema**: Draft 7 compliant with helpful error messages  
✅ **Stochastic Validation**: Explicit seed and units requirements enforced  
✅ **Example Files**: Minimal and hero examples validate successfully  
✅ **Error Reporting**: Detailed, actionable validation messages  
✅ **Integration**: Compatible with T13 determinism and T14 performance  

## Usage Examples

### Basic Validation

```python
from validation.pcc_validator import PCCValidator

validator = PCCValidator()
valid, errors = validator.validate_file("my_terrain.pcc")

if valid:
    print("✅ PCC file is valid")
else:
    print(f"❌ {len(errors)} validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### Node Specification Lookup

```python
from schema.pcc_terrain_nodes import get_node_spec, NodeType

spec = get_node_spec(NodeType.NOISE_FBM)
print(f"Parameters: {[p.name for p in spec.parameters]}")
print(f"Stochastic: {spec.is_stochastic}")
```

### Schema-Compliant PCC Generation

```python
pcc_data = {
    "version": "1.0.0",
    "nodes": [
        {
            "id": "terrain_base",
            "type": "NoiseFBM",
            "parameters": {
                "seed": 42,        # Required for stochastic nodes
                "units": "m",      # Required for stochastic nodes  
                "frequency": 0.01,
                "amplitude": 100.0,
                "octaves": 6
            }
        }
    ],
    "connections": []
}

validator = PCCValidator()
valid, errors = validator.validate_data(pcc_data)
```

## Verification Status

✅ **T15 Complete**: PCC schema hardening successfully implemented

### Node Set Finalization
- ✅ 11 core terrain nodes with complete specifications
- ✅ Parameter ranges documented and enforced
- ✅ Input/output types defined for all nodes
- ✅ Stochastic vs non-stochastic categorization

### JSON Schema Validation  
- ✅ Draft 7 JSON Schema with versioned URI
- ✅ Node-specific parameter validation  
- ✅ Helpful error messages with context
- ✅ Connection and graph structure validation

### Stochastic Node Requirements
- ✅ Explicit seed requirement (0 to 4294967295)
- ✅ Explicit units requirement (m, km, wu, norm)
- ✅ Parameter range documentation by scale
- ✅ Integration with T13 deterministic seed derivation

### Example Files and Documentation
- ✅ Minimal sphere example (2 nodes, basic validation)
- ✅ Hero planet example (14 nodes, complex pipeline)
- ✅ Complete parameter usage documentation
- ✅ Best practices and extension examples

### Versioned Schema System
- ✅ Schema version 1.0.0 locked and documented
- ✅ Breaking change policy defined
- ✅ Integration with T13/T14 systems verified
- ✅ Comprehensive validation test suite

The T15 implementation successfully **locks in the PCC node set with complete validation** as requested, providing a hardened schema foundation for deterministic terrain generation with explicit stochastic node requirements and helpful validation errors.

## Schema Evolution Path

**Future Versions**:
- `v1.1.0`: Minor additions (new node types, optional parameters)
- `v2.0.0`: Breaking changes (parameter renames, required field changes)
- Node deprecation and migration strategies

**Validation Compatibility**:
- Forward compatibility for minor versions
- Migration tools for major version upgrades
- Schema inheritance for custom node extensions

The T15 schema hardening establishes a stable, validated foundation for the PCC terrain pipeline while maintaining flexibility for future evolution.