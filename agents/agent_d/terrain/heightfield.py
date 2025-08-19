#!/usr/bin/env python3
"""
HeightField Module for PCC Terrain - T04
========================================

Composes PCC terrain nodes into a callable heightfield function.
Operates in unit-sphere space for cube-sphere terrain generation.

Features:
- Parse PCC JSON terrain specifications
- Compose noise nodes into layered heightfields
- Support for arithmetic operations (add, multiply, etc.)
- Deterministic evaluation with consistent seeding
- Unit-sphere space sampling for cube-sphere integration
"""

import json
import math
import numpy as np
from typing import Dict, List, Any, Callable, Union, Optional
from pathlib import Path

try:
    from .noise_nodes import NoiseNode, NoiseFBM, RidgedMF, DomainWarp, create_noise_node
except ImportError:
    from noise_nodes import NoiseNode, NoiseFBM, RidgedMF, DomainWarp, create_noise_node


class HeightField:
    """
    Callable heightfield function composed from PCC terrain nodes
    
    Evaluates height at 3D positions in unit-sphere space.
    """
    
    def __init__(self, root_node: NoiseNode, base_height: float = 0.0, 
                 height_scale: float = 1.0, seed: int = 42):
        """
        Initialize heightfield
        
        Args:
            root_node: Root noise node for evaluation
            base_height: Base height offset
            height_scale: Height scaling factor
            seed: Global seed for determinism
        """
        self.root_node = root_node
        self.base_height = base_height
        self.height_scale = height_scale
        self.seed = seed
        
    def __call__(self, x: float, y: float, z: float) -> float:
        """
        Evaluate heightfield at 3D position
        
        Args:
            x, y, z: Position in unit-sphere space
            
        Returns:
            Height value
        """
        return self.sample(x, y, z)
        
    def sample(self, x: float, y: float, z: float) -> float:
        """Sample heightfield at 3D position"""
        noise_value = self.root_node.sample(x, y, z)
        return self.base_height + noise_value * self.height_scale
    
    def sample_vec3(self, pos: np.ndarray) -> float:
        """Sample heightfield at 3D position vector"""
        return self.sample(pos[0], pos[1], pos[2])
    
    def sample_grid(self, positions: np.ndarray) -> np.ndarray:
        """
        Sample heightfield at multiple positions
        
        Args:
            positions: (N, 3) array of positions
            
        Returns:
            (N,) array of height values
        """
        heights = np.zeros(len(positions))
        for i, pos in enumerate(positions):
            heights[i] = self.sample(pos[0], pos[1], pos[2])
        return heights


class CompositeNode(NoiseNode):
    """Node that combines multiple noise sources with arithmetic operations"""
    
    def __init__(self, operation: str, nodes: List[NoiseNode], weights: Optional[List[float]] = None):
        """
        Initialize composite node
        
        Args:
            operation: 'add', 'multiply', 'max', 'min'
            nodes: List of noise nodes to combine
            weights: Optional weights for each node
        """
        super().__init__(seed=0)  # Composite nodes don't need their own seed
        self.operation = operation
        self.nodes = nodes
        self.weights = weights or [1.0] * len(nodes)
        
        if len(self.weights) != len(self.nodes):
            raise ValueError("Weights list must match nodes list length")
    
    def sample(self, x: float, y: float, z: float) -> float:
        """Sample composite noise at 3D position"""
        if not self.nodes:
            return 0.0
            
        values = []
        for node, weight in zip(self.nodes, self.weights):
            values.append(node.sample(x, y, z) * weight)
        
        if self.operation == "add":
            return sum(values)
        elif self.operation == "multiply":
            result = 1.0
            for val in values:
                result *= val
            return result
        elif self.operation == "max":
            return max(values)
        elif self.operation == "min":
            return min(values)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


def create_heightfield_from_pcc(pcc_spec: Union[str, dict], global_seed: int = 42) -> HeightField:
    """
    Create heightfield from PCC terrain specification
    
    Args:
        pcc_spec: PCC JSON string or dictionary
        global_seed: Global seed for deterministic results
        
    Returns:
        Configured HeightField instance
    """
    if isinstance(pcc_spec, str):
        # Load from file or parse JSON string
        if Path(pcc_spec).exists():
            with open(pcc_spec, 'r') as f:
                spec = json.load(f)
        else:
            spec = json.loads(pcc_spec)
    else:
        spec = pcc_spec
    
    print(f"🔧 Creating heightfield from PCC spec with global seed {global_seed}")
    
    # Parse terrain specification
    terrain_spec = spec.get("terrain", {})
    if not terrain_spec:
        raise ValueError("No terrain specification found in PCC")
    
    # Get heightfield configuration
    heightfield_config = terrain_spec.get("heightfield", {})
    base_height = heightfield_config.get("base_height", 0.0)
    height_scale = heightfield_config.get("height_scale", 1.0)
    
    # Parse node tree
    root_node_spec = terrain_spec.get("nodes", {})
    if not root_node_spec:
        # Create default FBM if no nodes specified
        print("⚠️ No nodes specified, creating default FBM")
        root_node = NoiseFBM(seed=global_seed)
    else:
        root_node = parse_node_tree(root_node_spec, global_seed)
    
    print(f"✅ Created heightfield: base={base_height}, scale={height_scale}")
    return HeightField(root_node, base_height, height_scale, global_seed)


def parse_node_tree(node_spec: dict, global_seed: int) -> NoiseNode:
    """
    Parse a node tree from PCC specification
    
    Args:
        node_spec: Node specification dictionary
        global_seed: Global seed for deterministic results
        
    Returns:
        Configured noise node
    """
    node_type = node_spec.get("type", "")
    
    # Apply global seed offset to node seed
    node_seed = node_spec.get("seed", 0) + global_seed
    node_spec_with_seed = {**node_spec, "seed": node_seed}
    
    if node_type in ["NoiseFBM", "RidgedMF", "DomainWarp"]:
        # Handle DomainWarp with nested source
        if node_type == "DomainWarp" and "source" in node_spec:
            source_spec = node_spec["source"]
            source_node = parse_node_tree(source_spec, global_seed)
            
            # Extract DomainWarp parameters, excluding metadata fields
            excluded_keys = ["type", "source", "name", "description"]
            warp_params = {k: v for k, v in node_spec_with_seed.items() 
                          if k not in excluded_keys}
            return DomainWarp(**warp_params, source_node=source_node)
        else:
            return create_noise_node(node_spec_with_seed)
    
    elif node_type == "Composite":
        operation = node_spec.get("operation", "add")
        weights = node_spec.get("weights", None)
        
        # Parse child nodes (filter out metadata like description, name)
        child_specs = node_spec.get("nodes", [])
        child_nodes = []
        for child_spec in child_specs:
            child_nodes.append(parse_node_tree(child_spec, global_seed))
        
        return CompositeNode(operation, child_nodes, weights)
    
    else:
        raise ValueError(f"Unknown node type: {node_type}")


def create_example_terrain_stack(seed: int = 42) -> HeightField:
    """
    Create example terrain: ridged base + fbm detail + small warp
    
    Args:
        seed: Random seed for deterministic results
        
    Returns:
        Configured HeightField with example terrain stack
    """
    print(f"🏔️ Creating example terrain stack with seed {seed}")
    
    # Create terrain specification
    terrain_spec = {
        "terrain": {
            "heightfield": {
                "base_height": 0.0,
                "height_scale": 0.3
            },
            "nodes": {
                "type": "Composite",
                "operation": "add",
                "weights": [1.0, 0.5],
                "nodes": [
                    {
                        "type": "RidgedMF",
                        "seed": 1000,
                        "octaves": 3,
                        "frequency": 2.0,
                        "amplitude": 1.0,
                        "lacunarity": 2.1,
                        "persistence": 0.6,
                        "ridge_offset": 1.0
                    },
                    {
                        "type": "DomainWarp", 
                        "seed": 2000,
                        "strength": 0.1,
                        "frequency": 1.5,
                        "source": {
                            "type": "NoiseFBM",
                            "seed": 3000,
                            "octaves": 4,
                            "frequency": 4.0,
                            "amplitude": 1.0,
                            "lacunarity": 2.0,
                            "persistence": 0.5
                        }
                    }
                ]
            }
        }
    }
    
    heightfield = create_heightfield_from_pcc(terrain_spec, seed)
    
    # Log some sample values
    print("📊 Example terrain samples:")
    test_positions = [
        (0.0, 0.0, 1.0),   # North pole
        (1.0, 0.0, 0.0),   # Equator
        (0.0, 1.0, 0.0),   # Equator
        (0.707, 0.707, 0.0), # Diagonal
        (0.0, 0.0, -1.0)   # South pole
    ]
    
    for i, pos in enumerate(test_positions):
        height = heightfield.sample(*pos)
        print(f"   Position {i}: {pos} -> height {height:.6f}")
    
    print("✅ Example terrain stack created")
    return heightfield


def log_terrain_composition(heightfield: HeightField):
    """Log the composition of a terrain heightfield"""
    print("🔍 Terrain Composition Analysis:")
    print(f"   Base height: {heightfield.base_height}")
    print(f"   Height scale: {heightfield.height_scale}")
    print(f"   Global seed: {heightfield.seed}")
    print(f"   Root node type: {type(heightfield.root_node).__name__}")
    
    # Recursively analyze node tree
    def analyze_node(node: NoiseNode, depth: int = 0):
        indent = "   " + "  " * depth
        node_type = type(node).__name__
        
        if hasattr(node, 'seed'):
            print(f"{indent}- {node_type} (seed: {node.seed})")
        else:
            print(f"{indent}- {node_type}")
        
        # Handle composite nodes
        if isinstance(node, CompositeNode):
            print(f"{indent}  Operation: {node.operation}")
            print(f"{indent}  Weights: {node.weights}")
            for i, child in enumerate(node.nodes):
                print(f"{indent}  Child {i}:")
                analyze_node(child, depth + 2)
        
        # Handle domain warp
        elif isinstance(node, DomainWarp):
            print(f"{indent}  Strength: {node.strength}")
            print(f"{indent}  Frequency: {node.frequency}")
            print(f"{indent}  Source:")
            analyze_node(node.source_node, depth + 1)
    
    analyze_node(heightfield.root_node)


if __name__ == "__main__":
    # Test heightfield creation and example terrain
    print("🔧 Testing HeightField module...")
    
    # Create and test example terrain
    example_terrain = create_example_terrain_stack(seed=12345)
    log_terrain_composition(example_terrain)
    
    # Test deterministic behavior
    terrain1 = create_example_terrain_stack(seed=99999)
    terrain2 = create_example_terrain_stack(seed=99999)
    
    test_pos = (0.5, 0.3, 0.8)
    h1 = terrain1.sample(*test_pos)
    h2 = terrain2.sample(*test_pos)
    
    print(f"🔍 Determinism test: {h1:.6f} == {h2:.6f} -> {'✅' if abs(h1 - h2) < 1e-10 else '❌'}")
    print("✅ HeightField testing complete")