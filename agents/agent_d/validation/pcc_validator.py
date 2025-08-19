#!/usr/bin/env python3
"""
PCC Schema Validator - T15
==========================

JSON schema validation with helpful error messages for PCC terrain files.
Validates against the hardened schema and provides semantic validation 
for node connections and parameter ranges.

Features:
- JSON schema validation with detailed error reporting
- Semantic validation of node connections and data flow
- Parameter range checking with helpful error messages  
- Stochastic node validation (seed + units requirements)
- Connection graph validation and cycle detection
"""

import json
import jsonschema
from typing import Dict, List, Any, Tuple, Set, Optional
from pathlib import Path
import sys
import os

# Import node specifications
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from schema.pcc_terrain_nodes import validate_node_instance, get_node_spec


class PCCValidationError(Exception):
    """Exception for PCC validation errors"""
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []


class PCCValidator:
    """Comprehensive PCC terrain file validator"""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize validator with schema"""
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "schema" / "pcc_schema_v1.json"
        
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.validator = jsonschema.Draft7Validator(self.schema)
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema from file"""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise PCCValidationError(f"Schema file not found: {self.schema_path}")
        except json.JSONDecodeError as e:
            raise PCCValidationError(f"Invalid JSON in schema file: {e}")
    
    def validate_file(self, pcc_file_path: Path) -> Tuple[bool, List[str]]:
        """Validate a PCC file against the schema"""
        try:
            with open(pcc_file_path, 'r') as f:
                pcc_data = json.load(f)
        except FileNotFoundError:
            return False, [f"PCC file not found: {pcc_file_path}"]
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON in PCC file: {e}"]
        
        return self.validate_data(pcc_data)
    
    def validate_data(self, pcc_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate PCC data against schema and semantic rules"""
        errors = []
        
        # 1. JSON Schema validation
        schema_errors = self._validate_json_schema(pcc_data)
        errors.extend(schema_errors)
        
        # 2. Semantic validation (even if schema fails, try to provide helpful info)
        if "nodes" in pcc_data:
            semantic_errors = self._validate_semantics(pcc_data)
            errors.extend(semantic_errors)
        
        return len(errors) == 0, errors
    
    def _validate_json_schema(self, pcc_data: Dict[str, Any]) -> List[str]:
        """Validate against JSON schema with helpful error formatting"""
        errors = []
        
        for error in self.validator.iter_errors(pcc_data):
            formatted_error = self._format_schema_error(error)
            errors.append(formatted_error)
        
        return errors
    
    def _format_schema_error(self, error: jsonschema.ValidationError) -> str:
        """Format JSON schema error with helpful context"""
        path = " -> ".join(str(p) for p in error.absolute_path)
        
        if error.validator == "required":
            missing_prop = error.message.split("'")[1]
            return f"Missing required property '{missing_prop}' at {path or 'root'}"
        
        elif error.validator == "type":
            expected_type = error.schema.get("type", "unknown")
            return f"Type error at {path}: expected {expected_type}, got {type(error.instance).__name__}"
        
        elif error.validator == "enum":
            valid_values = ", ".join(f"'{v}'" for v in error.schema["enum"])
            return f"Invalid value at {path}: '{error.instance}', must be one of: {valid_values}"
        
        elif error.validator in ["minimum", "maximum"]:
            limit = error.schema[error.validator]
            return f"Value error at {path}: {error.instance} {error.validator} limit is {limit}"
        
        elif error.validator == "const":
            expected = error.schema["const"]
            return f"Constant error at {path}: expected '{expected}', got '{error.instance}'"
        
        elif error.validator == "additionalProperties":
            extra_props = set(error.instance.keys()) - set(error.schema.get("properties", {}).keys())
            return f"Unknown properties at {path}: {', '.join(sorted(extra_props))}"
        
        else:
            # Generic error with context
            context = error.schema.get("description", "")
            context_str = f" ({context})" if context else ""
            return f"Validation error at {path}: {error.message}{context_str}"
    
    def _validate_semantics(self, pcc_data: Dict[str, Any]) -> List[str]:
        """Validate semantic rules beyond JSON schema"""
        errors = []
        
        nodes = pcc_data.get("nodes", [])
        connections = pcc_data.get("connections", [])
        
        # 1. Node-specific validation
        node_errors = self._validate_nodes(nodes)
        errors.extend(node_errors)
        
        # 2. Connection validation
        connection_errors = self._validate_connections(nodes, connections)
        errors.extend(connection_errors)
        
        # 3. Graph structure validation
        graph_errors = self._validate_graph_structure(nodes, connections)
        errors.extend(graph_errors)
        
        return errors
    
    def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate individual nodes using node specifications"""
        errors = []
        node_ids = set()
        
        for i, node in enumerate(nodes):
            node_path = f"nodes[{i}]"
            
            # Check for duplicate IDs
            node_id = node.get("id")
            if node_id in node_ids:
                errors.append(f"Duplicate node ID '{node_id}' at {node_path}")
            else:
                node_ids.add(node_id)
            
            # Validate node against specification
            valid, node_errors = validate_node_instance(node)
            for error in node_errors:
                errors.append(f"{node_path}: {error}")
        
        return errors
    
    def _validate_connections(self, nodes: List[Dict[str, Any]], 
                            connections: List[Dict[str, Any]]) -> List[str]:
        """Validate node connections and data flow"""
        errors = []
        
        # Build node lookup
        node_lookup = {node.get("id"): node for node in nodes if node.get("id")}
        
        for i, conn in enumerate(connections):
            conn_path = f"connections[{i}]"
            
            from_node_id = conn.get("from_node")
            to_node_id = conn.get("to_node")
            from_output = conn.get("from_output")
            to_input = conn.get("to_input")
            
            # Check nodes exist
            if from_node_id not in node_lookup:
                errors.append(f"{conn_path}: Source node '{from_node_id}' not found")
                continue
            
            if to_node_id not in node_lookup:
                errors.append(f"{conn_path}: Target node '{to_node_id}' not found")
                continue
            
            # Validate output/input compatibility
            from_node = node_lookup[from_node_id]
            to_node = node_lookup[to_node_id]
            
            from_spec = get_node_spec(from_node.get("type"))
            to_spec = get_node_spec(to_node.get("type"))
            
            if from_spec and to_spec:
                # Check if output exists
                if from_output not in from_spec.outputs:
                    valid_outputs = ", ".join(from_spec.outputs)
                    errors.append(f"{conn_path}: Node '{from_node_id}' ({from_spec.node_type.value}) "
                                f"does not have output '{from_output}'. Valid outputs: {valid_outputs}")
                
                # Check if input exists (simplified - would need input specs)
                # For now, just check common patterns
                valid_inputs = ["heightfield", "sdf_field", "vector_field", "scalar_field", "mesh"]
                if to_input not in valid_inputs:
                    errors.append(f"{conn_path}: Potentially invalid input '{to_input}' for node '{to_node_id}'")
        
        return errors
    
    def _validate_graph_structure(self, nodes: List[Dict[str, Any]], 
                                 connections: List[Dict[str, Any]]) -> List[str]:
        """Validate graph structure (cycles, orphans, etc.)"""
        errors = []
        
        # Build adjacency list
        graph = {}
        all_nodes = {node.get("id") for node in nodes if node.get("id")}
        
        for node_id in all_nodes:
            graph[node_id] = []
        
        for conn in connections:
            from_node = conn.get("from_node")
            to_node = conn.get("to_node")
            if from_node in graph and to_node in graph:
                graph[from_node].append(to_node)
        
        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor, visited, rec_stack):
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for node_id in all_nodes:
            if node_id not in visited:
                if has_cycle(node_id, visited, set()):
                    errors.append(f"Cycle detected in node graph involving node '{node_id}'")
                    break
        
        # Check for orphaned nodes (no inputs or outputs)
        nodes_with_inputs = {conn.get("to_node") for conn in connections}
        nodes_with_outputs = {conn.get("from_node") for conn in connections}
        
        for node in nodes:
            node_id = node.get("id")
            node_type = node.get("type")
            
            # Skip certain node types that are expected to be sources/sinks
            if node_type in ["CubeSphere"]:  # Sources
                continue
            if node_type in ["MarchingCubes", "QuadtreeLOD"]:  # Sinks
                if node_id not in nodes_with_inputs:
                    errors.append(f"Sink node '{node_id}' ({node_type}) has no inputs")
                continue
            
            # Other nodes should have both inputs and outputs
            if node_id not in nodes_with_inputs and node_id not in nodes_with_outputs:
                errors.append(f"Orphaned node '{node_id}' ({node_type}) has no connections")
        
        return errors
    
    def generate_validation_report(self, pcc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        valid, errors = self.validate_data(pcc_data)
        
        # Count error types
        error_types = {"schema": 0, "semantic": 0, "connection": 0, "graph": 0}
        for error in errors:
            if "Type error" in error or "Missing required" in error:
                error_types["schema"] += 1
            elif "connection" in error.lower():
                error_types["connection"] += 1  
            elif "cycle" in error.lower() or "orphan" in error.lower():
                error_types["graph"] += 1
            else:
                error_types["semantic"] += 1
        
        # Analyze nodes
        nodes = pcc_data.get("nodes", [])
        node_types = {}
        stochastic_nodes = 0
        
        for node in nodes:
            node_type = node.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Check if stochastic
            spec = get_node_spec(node_type)
            if spec and spec.is_stochastic:
                stochastic_nodes += 1
        
        return {
            "valid": valid,
            "total_errors": len(errors),
            "errors": errors,
            "error_breakdown": error_types,
            "statistics": {
                "total_nodes": len(nodes),
                "total_connections": len(pcc_data.get("connections", [])),
                "node_types": node_types,
                "stochastic_nodes": stochastic_nodes
            },
            "schema_version": self.schema.get("$id", "unknown")
        }


def validate_pcc_file(file_path: Union[str, Path]) -> None:
    """Command-line validation utility"""
    validator = PCCValidator()
    
    valid, errors = validator.validate_file(Path(file_path))
    
    if valid:
        print(f"✅ PCC file '{file_path}' is valid")
    else:
        print(f"❌ PCC file '{file_path}' has {len(errors)} validation errors:")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")


if __name__ == "__main__":
    # Test validator
    print("🚀 PCC Schema Validator - T15")
    print("=" * 60)
    
    # Create test validator
    validator = PCCValidator()
    print(f"✅ Schema loaded: {validator.schema.get('title', 'Unknown')}")
    
    # Test with minimal valid data
    test_data = {
        "version": "1.0.0",
        "nodes": [
            {
                "id": "sphere1",
                "type": "CubeSphere",
                "parameters": {
                    "radius": 100.0,
                    "resolution": 32
                }
            },
            {
                "id": "noise1", 
                "type": "NoiseFBM",
                "parameters": {
                    "seed": 12345,
                    "units": "m",
                    "frequency": 0.01,
                    "amplitude": 10.0,
                    "octaves": 6
                }
            }
        ],
        "connections": [
            {
                "from_node": "sphere1",
                "from_output": "heightfield",
                "to_node": "noise1",
                "to_input": "heightfield"
            }
        ]
    }
    
    print(f"\n🔍 Testing validation with sample data...")
    valid, errors = validator.validate_data(test_data)
    
    if valid:
        print(f"   ✅ Sample data is valid")
    else:
        print(f"   ❌ Sample data has {len(errors)} errors:")
        for error in errors:
            print(f"      - {error}")
    
    # Generate validation report
    report = validator.generate_validation_report(test_data)
    print(f"\n📊 Validation Report:")
    print(f"   Total nodes: {report['statistics']['total_nodes']}")
    print(f"   Total connections: {report['statistics']['total_connections']}")
    print(f"   Stochastic nodes: {report['statistics']['stochastic_nodes']}")
    print(f"   Node types: {list(report['statistics']['node_types'].keys())}")
    
    print(f"\n✅ PCC validator functional")