#!/usr/bin/env python3
"""
PCC Terrain Node Definitions - T15
==================================

Finalized node set for the terrain pipeline with complete type definitions,
parameter ranges, and semantic validation. These nodes form the core vocabulary
for procedural terrain generation in PCC.

Node Categories:
- Primitive Generators: CubeSphere
- Noise Generators: NoiseFBM, RidgedMF  
- Spatial Operations: DomainWarp, Displace
- SDF Operations: SDF.Union, SDF.Subtract, SDF.Intersect, SDF.Smooth
- Mesh Generation: MarchingCubes
- Level of Detail: QuadtreeLOD

All stochastic nodes require explicit seed and units.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class NodeType(Enum):
    """Core PCC node types for terrain pipeline"""
    
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


class Units(Enum):
    """Spatial units for PCC operations"""
    METERS = "m"
    KILOMETERS = "km"
    WORLD_UNITS = "wu"
    NORMALIZED = "norm"


@dataclass
class ParameterSpec:
    """Specification for a node parameter"""
    name: str
    type: str  # "float", "int", "string", "bool", "vector3", "seed"
    required: bool = True
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    units: Optional[Units] = None
    
    def validate_value(self, value: Any) -> Tuple[bool, str]:
        """Validate parameter value against specification"""
        if value is None and self.required:
            return False, f"Required parameter '{self.name}' is missing"
        
        if value is None and not self.required:
            return True, ""
        
        # Type validation
        if self.type == "float":
            if not isinstance(value, (int, float)):
                return False, f"Parameter '{self.name}' must be a number, got {type(value).__name__}"
            value = float(value)
        elif self.type == "int":
            if not isinstance(value, int):
                return False, f"Parameter '{self.name}' must be an integer, got {type(value).__name__}"
        elif self.type == "string":
            if not isinstance(value, str):
                return False, f"Parameter '{self.name}' must be a string, got {type(value).__name__}"
        elif self.type == "bool":
            if not isinstance(value, bool):
                return False, f"Parameter '{self.name}' must be a boolean, got {type(value).__name__}"
        elif self.type == "vector3":
            if not isinstance(value, list) or len(value) != 3:
                return False, f"Parameter '{self.name}' must be a 3-element array"
            if not all(isinstance(v, (int, float)) for v in value):
                return False, f"Parameter '{self.name}' vector elements must be numbers"
        elif self.type == "seed":
            if not isinstance(value, int) or value < 0:
                return False, f"Parameter '{self.name}' must be a non-negative integer seed"
        
        # Range validation
        if self.type in ["float", "int"] and isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' value {value} below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' value {value} above maximum {self.max_value}"
        
        return True, ""


@dataclass 
class NodeSpec:
    """Complete specification for a PCC node type"""
    node_type: NodeType
    description: str
    parameters: List[ParameterSpec] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)  # Input node types accepted
    outputs: List[str] = field(default_factory=list)  # Output data types
    is_stochastic: bool = False
    category: str = ""
    
    def validate_node(self, node_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a node instance against this specification"""
        errors = []
        
        # Check required node structure
        if "type" not in node_data:
            errors.append("Node missing required 'type' field")
            return False, errors
        
        if node_data["type"] != self.node_type.value:
            errors.append(f"Node type mismatch: expected {self.node_type.value}, got {node_data['type']}")
        
        # Validate parameters
        params = node_data.get("parameters", {})
        
        # Check stochastic nodes have required seed and units
        if self.is_stochastic:
            if "seed" not in params:
                errors.append(f"Stochastic node {self.node_type.value} requires explicit 'seed' parameter")
            if "units" not in params:
                errors.append(f"Stochastic node {self.node_type.value} requires explicit 'units' parameter")
        
        # Validate each parameter
        for param_spec in self.parameters:
            param_value = params.get(param_spec.name)
            valid, error = param_spec.validate_value(param_value)
            if not valid:
                errors.append(error)
        
        # Check for unknown parameters
        known_params = {p.name for p in self.parameters}
        for param_name in params:
            if param_name not in known_params:
                errors.append(f"Unknown parameter '{param_name}' for node type {self.node_type.value}")
        
        return len(errors) == 0, errors


def create_terrain_node_specs() -> Dict[NodeType, NodeSpec]:
    """Create complete specifications for all terrain nodes"""
    specs = {}
    
    # CubeSphere - Primitive sphere generator
    specs[NodeType.CUBE_SPHERE] = NodeSpec(
        node_type=NodeType.CUBE_SPHERE,
        description="Generates a cube-sphere primitive for planetary terrain base",
        category="primitive_generator",
        parameters=[
            ParameterSpec("radius", "float", True, min_value=0.1, max_value=10000.0,
                         description="Sphere radius", units=Units.METERS),
            ParameterSpec("resolution", "int", True, default=8, min_value=4, max_value=256,
                         description="Tessellation resolution per face"),
            ParameterSpec("center", "vector3", False, default=[0.0, 0.0, 0.0],
                         description="Sphere center position", units=Units.METERS)
        ],
        inputs=[],
        outputs=["heightfield", "mesh"],
        is_stochastic=False
    )
    
    # NoiseFBM - Fractional Brownian Motion noise
    specs[NodeType.NOISE_FBM] = NodeSpec(
        node_type=NodeType.NOISE_FBM,
        description="Fractional Brownian Motion noise for terrain height variation",
        category="noise_generator",
        parameters=[
            ParameterSpec("seed", "seed", True, 
                         description="Random seed for reproducible noise"),
            ParameterSpec("units", "string", True,
                         description="Spatial units (m, km, wu, norm)"),
            ParameterSpec("frequency", "float", True, default=0.01, min_value=0.0001, max_value=10.0,
                         description="Base noise frequency"),
            ParameterSpec("amplitude", "float", True, default=1.0, min_value=0.0, max_value=1000.0,
                         description="Maximum noise amplitude"),
            ParameterSpec("octaves", "int", True, default=6, min_value=1, max_value=16,
                         description="Number of noise octaves"),
            ParameterSpec("lacunarity", "float", False, default=2.0, min_value=1.0, max_value=4.0,
                         description="Frequency multiplier between octaves"),
            ParameterSpec("persistence", "float", False, default=0.5, min_value=0.0, max_value=1.0,
                         description="Amplitude multiplier between octaves"),
            ParameterSpec("offset", "vector3", False, default=[0.0, 0.0, 0.0],
                         description="Noise sampling offset")
        ],
        inputs=["heightfield", "vector_field"],
        outputs=["heightfield", "scalar_field"],
        is_stochastic=True
    )
    
    # RidgedMF - Ridged multifractal noise
    specs[NodeType.RIDGED_MF] = NodeSpec(
        node_type=NodeType.RIDGED_MF, 
        description="Ridged multifractal noise for mountainous terrain features",
        category="noise_generator",
        parameters=[
            ParameterSpec("seed", "seed", True,
                         description="Random seed for reproducible noise"),
            ParameterSpec("units", "string", True,
                         description="Spatial units (m, km, wu, norm)"),
            ParameterSpec("frequency", "float", True, default=0.01, min_value=0.0001, max_value=10.0,
                         description="Base noise frequency"),
            ParameterSpec("amplitude", "float", True, default=1.0, min_value=0.0, max_value=1000.0,
                         description="Maximum noise amplitude"),
            ParameterSpec("octaves", "int", True, default=6, min_value=1, max_value=16,
                         description="Number of noise octaves"),
            ParameterSpec("lacunarity", "float", False, default=2.0, min_value=1.0, max_value=4.0,
                         description="Frequency multiplier between octaves"),
            ParameterSpec("gain", "float", False, default=2.0, min_value=0.5, max_value=4.0,
                         description="Ridge sharpness factor"),
            ParameterSpec("ridge_offset", "float", False, default=1.0, min_value=0.0, max_value=2.0,
                         description="Ridge formation offset")
        ],
        inputs=["heightfield", "vector_field"],
        outputs=["heightfield", "scalar_field"],
        is_stochastic=True
    )
    
    # DomainWarp - Spatial domain warping
    specs[NodeType.DOMAIN_WARP] = NodeSpec(
        node_type=NodeType.DOMAIN_WARP,
        description="Warps spatial domain using vector field for terrain distortion",
        category="spatial_operation",
        parameters=[
            ParameterSpec("strength", "float", True, default=1.0, min_value=0.0, max_value=100.0,
                         description="Warp strength multiplier"),
            ParameterSpec("warp_type", "string", False, default="fbm",
                         description="Warp pattern type (fbm, ridged, curl)"),
            ParameterSpec("scale", "float", False, default=1.0, min_value=0.1, max_value=10.0,
                         description="Warp pattern scale")
        ],
        inputs=["heightfield", "vector_field"],
        outputs=["heightfield", "vector_field"],
        is_stochastic=False
    )
    
    # Displace - Geometric displacement
    specs[NodeType.DISPLACE] = NodeSpec(
        node_type=NodeType.DISPLACE,
        description="Displaces geometry along normals using scalar field",
        category="spatial_operation", 
        parameters=[
            ParameterSpec("amount", "float", True, default=1.0, min_value=-1000.0, max_value=1000.0,
                         description="Displacement amount", units=Units.METERS),
            ParameterSpec("direction", "vector3", False, default=[0.0, 1.0, 0.0],
                         description="Displacement direction (normalized)"),
            ParameterSpec("clamp_min", "float", False, min_value=-1000.0, max_value=1000.0,
                         description="Minimum displacement limit"),
            ParameterSpec("clamp_max", "float", False, min_value=-1000.0, max_value=1000.0,
                         description="Maximum displacement limit")
        ],
        inputs=["mesh", "heightfield", "scalar_field"],
        outputs=["mesh", "heightfield"],
        is_stochastic=False
    )
    
    # SDF.Union - Boolean union of SDF fields
    specs[NodeType.SDF_UNION] = NodeSpec(
        node_type=NodeType.SDF_UNION,
        description="Boolean union operation on signed distance fields",
        category="sdf_operation",
        parameters=[
            ParameterSpec("smooth_radius", "float", False, default=0.0, min_value=0.0, max_value=10.0,
                         description="Smoothing radius for smooth union", units=Units.METERS)
        ],
        inputs=["sdf_field", "sdf_field"],
        outputs=["sdf_field"],
        is_stochastic=False
    )
    
    # SDF.Subtract - Boolean subtraction of SDF fields
    specs[NodeType.SDF_SUBTRACT] = NodeSpec(
        node_type=NodeType.SDF_SUBTRACT,
        description="Boolean subtraction operation on signed distance fields",
        category="sdf_operation",
        parameters=[
            ParameterSpec("smooth_radius", "float", False, default=0.0, min_value=0.0, max_value=10.0,
                         description="Smoothing radius for smooth subtraction", units=Units.METERS)
        ],
        inputs=["sdf_field", "sdf_field"],
        outputs=["sdf_field"],
        is_stochastic=False
    )
    
    # SDF.Intersect - Boolean intersection of SDF fields  
    specs[NodeType.SDF_INTERSECT] = NodeSpec(
        node_type=NodeType.SDF_INTERSECT,
        description="Boolean intersection operation on signed distance fields",
        category="sdf_operation",
        parameters=[
            ParameterSpec("smooth_radius", "float", False, default=0.0, min_value=0.0, max_value=10.0,
                         description="Smoothing radius for smooth intersection", units=Units.METERS)
        ],
        inputs=["sdf_field", "sdf_field"],
        outputs=["sdf_field"],
        is_stochastic=False
    )
    
    # SDF.Smooth - SDF smoothing filter
    specs[NodeType.SDF_SMOOTH] = NodeSpec(
        node_type=NodeType.SDF_SMOOTH,
        description="Applies smoothing filter to signed distance field",
        category="sdf_operation",
        parameters=[
            ParameterSpec("radius", "float", True, default=1.0, min_value=0.1, max_value=20.0,
                         description="Smoothing radius", units=Units.METERS),
            ParameterSpec("iterations", "int", False, default=1, min_value=1, max_value=10,
                         description="Number of smoothing iterations")
        ],
        inputs=["sdf_field"],
        outputs=["sdf_field"],
        is_stochastic=False
    )
    
    # MarchingCubes - Mesh generation from SDF
    specs[NodeType.MARCHING_CUBES] = NodeSpec(
        node_type=NodeType.MARCHING_CUBES,
        description="Generates mesh from signed distance field using Marching Cubes",
        category="mesh_generator",
        parameters=[
            ParameterSpec("iso_value", "float", False, default=0.0, min_value=-10.0, max_value=10.0,
                         description="ISO surface value for mesh extraction"),
            ParameterSpec("resolution", "int", True, default=64, min_value=8, max_value=512,
                         description="Voxel grid resolution per axis"),
            ParameterSpec("bounds_min", "vector3", True, default=[-10.0, -10.0, -10.0],
                         description="Minimum sampling bounds", units=Units.METERS),
            ParameterSpec("bounds_max", "vector3", True, default=[10.0, 10.0, 10.0],
                         description="Maximum sampling bounds", units=Units.METERS),
            ParameterSpec("generate_normals", "bool", False, default=True,
                         description="Generate vertex normals"),
            ParameterSpec("generate_tangents", "bool", False, default=False,
                         description="Generate tangent space vectors")
        ],
        inputs=["sdf_field"],
        outputs=["mesh"],
        is_stochastic=False
    )
    
    # QuadtreeLOD - Level of detail management
    specs[NodeType.QUADTREE_LOD] = NodeSpec(
        node_type=NodeType.QUADTREE_LOD,
        description="Manages level of detail using quadtree spatial partitioning",
        category="lod_system",
        parameters=[
            ParameterSpec("max_depth", "int", True, default=8, min_value=2, max_value=16,
                         description="Maximum quadtree depth"),
            ParameterSpec("chunk_size", "float", True, default=64.0, min_value=1.0, max_value=1000.0,
                         description="Base chunk size", units=Units.METERS),
            ParameterSpec("distance_threshold", "float", True, default=100.0, min_value=10.0, max_value=10000.0,
                         description="Distance threshold for LOD transitions", units=Units.METERS),
            ParameterSpec("camera_position", "vector3", True, default=[0.0, 100.0, 0.0],
                         description="Camera position for distance calculations", units=Units.METERS),
            ParameterSpec("quality_bias", "float", False, default=1.0, min_value=0.1, max_value=5.0,
                         description="Quality bias factor (higher = more detail)")
        ],
        inputs=["heightfield", "mesh"],
        outputs=["lod_chunks"],
        is_stochastic=False
    )
    
    return specs


def get_node_spec(node_type: Union[NodeType, str]) -> Optional[NodeSpec]:
    """Get specification for a specific node type"""
    if isinstance(node_type, str):
        try:
            node_type = NodeType(node_type)
        except ValueError:
            return None
    
    specs = create_terrain_node_specs()
    return specs.get(node_type)


def validate_node_instance(node_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a node instance against its specification"""
    if "type" not in node_data:
        return False, ["Node missing required 'type' field"]
    
    spec = get_node_spec(node_data["type"])
    if spec is None:
        return False, [f"Unknown node type: {node_data['type']}"]
    
    return spec.validate_node(node_data)


if __name__ == "__main__":
    # Test node specifications
    print("🚀 PCC Terrain Node Specifications - T15")
    print("=" * 60)
    
    specs = create_terrain_node_specs()
    print(f"📊 Defined {len(specs)} node types:")
    
    for node_type, spec in specs.items():
        stochastic_marker = " (stochastic)" if spec.is_stochastic else ""
        print(f"   ✅ {node_type.value}: {spec.description[:50]}...{stochastic_marker}")
        print(f"      Parameters: {len(spec.parameters)}, Inputs: {len(spec.inputs)}, Outputs: {len(spec.outputs)}")
    
    # Test validation
    print(f"\n🔍 Testing node validation...")
    
    # Valid node
    test_node = {
        "type": "NoiseFBM",
        "parameters": {
            "seed": 12345,
            "units": "m",
            "frequency": 0.01,
            "amplitude": 50.0,
            "octaves": 6
        }
    }
    
    valid, errors = validate_node_instance(test_node)
    print(f"   ✅ Valid NoiseFBM node: {valid}")
    
    # Invalid node (missing seed)
    invalid_node = {
        "type": "NoiseFBM", 
        "parameters": {
            "frequency": 0.01,
            "amplitude": 50.0
        }
    }
    
    valid, errors = validate_node_instance(invalid_node)
    print(f"   ❌ Invalid NoiseFBM node: {valid}")
    for error in errors:
        print(f"      - {error}")
    
    print(f"\n✅ PCC terrain node specifications complete")