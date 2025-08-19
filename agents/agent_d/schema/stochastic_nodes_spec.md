# PCC Stochastic Nodes Specification - T15

## Overview

Stochastic nodes in the PCC terrain pipeline are nodes that involve randomness or pseudo-randomness in their computation. To ensure deterministic and reproducible terrain generation, **all stochastic nodes MUST specify explicit seed and units parameters**.

## Required Parameters for All Stochastic Nodes

### 1. Seed Parameter
- **Type**: `integer` 
- **Range**: `0` to `4294967295` (32-bit unsigned integer)
- **Required**: `true`
- **Description**: Random seed for reproducible pseudo-random generation

**Example**:
```json
{
  "seed": 12345
}
```

### 2. Units Parameter
- **Type**: `string`
- **Values**: `"m"`, `"km"`, `"wu"`, `"norm"`
- **Required**: `true`
- **Description**: Spatial units for noise sampling and frequency calculations

**Units Definitions**:
- `"m"` - Meters (real-world metric units)
- `"km"` - Kilometers (large-scale terrain features)
- `"wu"` - World Units (game/simulation specific units)
- `"norm"` - Normalized units (0.0 to 1.0 range)

**Example**:
```json
{
  "units": "m"
}
```

## Stochastic Node Types

### NoiseFBM (Fractional Brownian Motion)

**Category**: Stochastic noise generator  
**Determinism**: Requires seed + units for reproducible output

**Required Parameters**:
```json
{
  "seed": 12345,                    // Range: 0 to 4294967295
  "units": "m",                     // Values: "m", "km", "wu", "norm"
  "frequency": 0.01,                // Range: 0.0001 to 10.0
  "amplitude": 50.0,                // Range: 0.0 to 1000.0  
  "octaves": 6                      // Range: 1 to 16
}
```

**Optional Parameters**:
```json
{
  "lacunarity": 2.0,                // Range: 1.0 to 4.0, default: 2.0
  "persistence": 0.5,               // Range: 0.0 to 1.0, default: 0.5
  "offset": [0.0, 0.0, 0.0]         // 3D vector, default: [0, 0, 0]
}
```

**Frequency Guidelines by Units**:
- `"m"`: 0.001 to 0.1 (features from 10m to 1000m)
- `"km"`: 0.01 to 1.0 (features from 1km to 100km)
- `"wu"`: 0.001 to 10.0 (depends on world unit scale)
- `"norm"`: 0.1 to 50.0 (normalized frequency space)

### RidgedMF (Ridged Multifractal)

**Category**: Stochastic noise generator  
**Determinism**: Requires seed + units for reproducible ridge formations

**Required Parameters**:
```json
{
  "seed": 54321,                    // Range: 0 to 4294967295
  "units": "m",                     // Values: "m", "km", "wu", "norm"
  "frequency": 0.005,               // Range: 0.0001 to 10.0
  "amplitude": 100.0,               // Range: 0.0 to 1000.0
  "octaves": 8                      // Range: 1 to 16
}
```

**Optional Parameters**:
```json
{
  "lacunarity": 2.0,                // Range: 1.0 to 4.0, default: 2.0
  "gain": 2.0,                      // Range: 0.5 to 4.0, default: 2.0
  "ridge_offset": 1.0               // Range: 0.0 to 2.0, default: 1.0
}
```

**Ridge Formation Guidelines**:
- **Low gain (0.5-1.0)**: Smooth, rolling ridges
- **Medium gain (1.0-2.0)**: Moderate ridge sharpness  
- **High gain (2.0-4.0)**: Sharp, dramatic ridges
- **Ridge offset**: Controls ridge prominence (1.0 = standard ridges)

## Parameter Range Documentation

### Frequency Ranges by Terrain Scale

| Units | Min Freq | Max Freq | Typical Range | Feature Scale |
|-------|----------|----------|---------------|---------------|
| `"m"` | 0.0001 | 10.0 | 0.001 - 0.1 | 10m - 1000m |
| `"km"` | 0.0001 | 10.0 | 0.01 - 1.0 | 1km - 100km |
| `"wu"` | 0.0001 | 10.0 | 0.001 - 10.0 | Varies by scale |
| `"norm"` | 0.0001 | 10.0 | 0.1 - 50.0 | Normalized space |

### Amplitude Guidelines

| Terrain Type | Amplitude Range | Description |
|--------------|-----------------|-------------|
| Gentle Hills | 1.0 - 20.0 | Subtle elevation changes |
| Rolling Terrain | 20.0 - 100.0 | Moderate height variation |
| Mountainous | 100.0 - 500.0 | Significant elevation changes |
| Extreme Terrain | 500.0 - 1000.0 | Dramatic height differences |

### Octave Recommendations

| Octaves | Detail Level | Use Case |
|---------|--------------|----------|
| 1-3 | Low | Base terrain shape, large features |
| 4-6 | Medium | Balanced detail/performance |
| 7-10 | High | Rich detail, small features |
| 11-16 | Very High | Maximum detail (performance cost) |

## Seed Management Best Practices

### Deterministic Seed Derivation
When generating multiple terrain features, derive seeds deterministically:

```json
{
  "base_terrain": {
    "type": "NoiseFBM",
    "parameters": {
      "seed": 12345,              // Master seed
      "units": "m"
    }
  },
  "detail_noise": {
    "type": "NoiseFBM", 
    "parameters": {
      "seed": 12346,              // Master seed + 1
      "units": "m"
    }
  },
  "ridge_features": {
    "type": "RidgedMF",
    "parameters": {
      "seed": 12400,              // Master seed + 55
      "units": "m"
    }
  }
}
```

### Seed Ranges for Different Purposes

| Purpose | Seed Range | Example |
|---------|------------|---------|
| Base terrain | 10000-19999 | 12345 |
| Detail noise | 20000-29999 | 25678 |
| Cave systems | 30000-39999 | 34567 |
| Vegetation | 40000-49999 | 43210 |
| Water features | 50000-59999 | 52468 |

## Validation Rules

### Schema Validation
All stochastic nodes MUST pass these validation checks:

1. **Seed Present**: `seed` parameter must be specified
2. **Seed Range**: Must be integer in range [0, 4294967295]
3. **Units Present**: `units` parameter must be specified  
4. **Units Valid**: Must be one of ["m", "km", "wu", "norm"]
5. **Parameter Ranges**: All parameters must be within documented ranges

### Semantic Validation
Additional semantic checks for stochastic nodes:

1. **Frequency vs Units**: Frequency should be reasonable for specified units
2. **Octave Limits**: Higher octaves should have reasonable frequency bounds
3. **Amplitude Scaling**: Amplitude should match terrain scale expectations

## Example Validation Errors

### Missing Seed
```json
{
  "type": "NoiseFBM",
  "parameters": {
    "units": "m",
    "frequency": 0.01,
    "amplitude": 50.0,
    "octaves": 6
  }
}
```
**Error**: "Stochastic node NoiseFBM requires explicit 'seed' parameter"

### Missing Units
```json
{
  "type": "RidgedMF",
  "parameters": {
    "seed": 12345,
    "frequency": 0.005,
    "amplitude": 100.0,
    "octaves": 8
  }
}
```
**Error**: "Stochastic node RidgedMF requires explicit 'units' parameter"

### Invalid Seed Range
```json
{
  "seed": -100
}
```
**Error**: "Parameter 'seed' must be a non-negative integer seed"

### Invalid Units
```json
{
  "units": "inches"
}
```
**Error**: "Invalid value at parameters.units: 'inches', must be one of: 'm', 'km', 'wu', 'norm'"

## Integration with T13 Determinism

The explicit seed requirements in T15 integrate with the T13 determinism system:

1. **Master Seed**: T13 provides master seed for entire terrain generation
2. **Derived Seeds**: T15 stochastic nodes use deterministically derived seeds
3. **Reproducibility**: Same master seed + same PCC file = identical terrain
4. **Hierarchical Control**: Different terrain layers can have independent but deterministic seeds

This ensures that PCC terrain generation is fully deterministic while providing rich stochastic variety in the generated terrain features.