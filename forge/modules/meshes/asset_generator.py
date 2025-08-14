"""Complex multi-mesh asset generation - houses built from component meshes (2x4s → house)."""
import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import os

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

class AssetGenerator:
    """Generates complex multi-mesh assets from component parts."""
    
    def __init__(self, assets_dir: str = "runs/assets"):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
        # Component mesh library (procedurally generated)
        self.component_library = self._initialize_component_library()
        
        # Asset templates (house = multiple components)
        self.asset_templates = self._initialize_asset_templates()
    
    def _initialize_component_library(self) -> Dict[str, Dict]:
        """Initialize library of basic component meshes (2x4s, planks, etc.)."""
        return {
            # Structural components
            "beam_2x4": {
                "dimensions": [0.089, 0.038, 2.44],  # 2x4 lumber in meters
                "material_type": "wood_frame",
                "connection_points": ["end_a", "end_b"],
                "mesh_type": "box"
            },
            "beam_2x6": {
                "dimensions": [0.089, 0.057, 2.44],  # 2x6 lumber
                "material_type": "wood_frame", 
                "connection_points": ["end_a", "end_b"],
                "mesh_type": "box"
            },
            "plank_1x6": {
                "dimensions": [0.019, 0.140, 2.44],  # 1x6 board
                "material_type": "wood_siding",
                "connection_points": ["edge_long_a", "edge_long_b"],
                "mesh_type": "box"
            },
            
            # Foundation components
            "concrete_block": {
                "dimensions": [0.39, 0.19, 0.19],  # Standard CMU block
                "material_type": "concrete",
                "connection_points": ["face_north", "face_south", "face_east", "face_west"],
                "mesh_type": "box"
            },
            
            # Roofing components  
            "rafter_2x8": {
                "dimensions": [0.089, 0.184, 3.66],  # 2x8 rafter
                "material_type": "wood_frame",
                "connection_points": ["peak", "wall_plate"],
                "mesh_type": "box"
            },
            "shingle_strip": {
                "dimensions": [0.91, 0.30, 0.003],  # Asphalt shingle strip
                "material_type": "roofing_asphalt",
                "connection_points": ["overlap_top", "overlap_bottom"],
                "mesh_type": "box"
            }
        }
    
    def _initialize_asset_templates(self) -> Dict[str, Dict]:
        """Initialize templates for complex assets built from components."""
        return {
            "cottage_small": {
                "description": "Small cottage built from lumber components",
                "footprint": [6.0, 4.0, 3.5],  # width, depth, height
                "components": [
                    # Foundation
                    {"type": "foundation_wall", "component": "concrete_block", "count": 32},
                    
                    # Floor frame
                    {"type": "floor_joist", "component": "beam_2x6", "count": 8},
                    {"type": "rim_board", "component": "beam_2x6", "count": 4},
                    
                    # Walls - frame
                    {"type": "wall_stud", "component": "beam_2x4", "count": 24},
                    {"type": "top_plate", "component": "beam_2x4", "count": 8},
                    {"type": "bottom_plate", "component": "beam_2x4", "count": 4},
                    
                    # Wall sheathing/siding
                    {"type": "siding", "component": "plank_1x6", "count": 48},
                    
                    # Roof frame
                    {"type": "rafter", "component": "rafter_2x8", "count": 12},
                    {"type": "ridge_beam", "component": "rafter_2x8", "count": 1},
                    
                    # Roofing
                    {"type": "shingles", "component": "shingle_strip", "count": 36}
                ],
                "assembly_rules": [
                    "foundation_wall forms rectangle footprint",
                    "floor_joists span foundation_wall width", 
                    "wall_studs vertical on top_plate/bottom_plate",
                    "siding covers wall_studs horizontally",
                    "rafters span from wall to ridge_beam",
                    "shingles layer from bottom to top with overlap"
                ]
            },
            
            "barn_rustic": {
                "description": "Rustic barn from heavy timber components", 
                "footprint": [8.0, 12.0, 6.0],
                "components": [
                    {"type": "post", "component": "beam_2x6", "count": 12},
                    {"type": "beam", "component": "beam_2x6", "count": 16}, 
                    {"type": "siding", "component": "plank_1x6", "count": 80},
                    {"type": "roof_beam", "component": "rafter_2x8", "count": 20}
                ],
                "assembly_rules": [
                    "posts form rectangular grid",
                    "beams connect posts horizontally",
                    "siding covers frame vertically", 
                    "roof_beams form gabled roof structure"
                ]
            },
            
            "watchtower": {
                "description": "Tall defensive watchtower",
                "footprint": [3.0, 3.0, 8.0],
                "components": [
                    {"type": "foundation", "component": "concrete_block", "count": 16},
                    {"type": "corner_post", "component": "beam_2x6", "count": 4},
                    {"type": "wall_board", "component": "plank_1x6", "count": 64},
                    {"type": "platform_beam", "component": "beam_2x6", "count": 8}
                ],
                "assembly_rules": [
                    "foundation forms square base",
                    "corner_posts extend full height",
                    "wall_board creates enclosed walls",
                    "platform_beam creates upper level"
                ]
            }
        }
    
    def generate_complex_asset(self, asset_type: str, position: Tuple[float, float, float], 
                             up_vector: Tuple[float, float, float], 
                             seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a complex asset built from multiple component meshes."""
        if seed is not None:
            random.seed(seed)
        
        if asset_type not in self.asset_templates:
            raise ValueError(f"Unknown asset type: {asset_type}")
        
        template = self.asset_templates[asset_type]
        asset_id = f"{asset_type}_{random.randint(1000, 9999)}"
        
        # Generate individual component meshes
        components = []
        for comp_spec in template["components"]:
            component_meshes = self._generate_component_instances(
                comp_spec, position, up_vector, asset_id
            )
            components.extend(component_meshes)
        
        # Create asset manifest
        asset_manifest = {
            "asset_id": asset_id,
            "asset_type": asset_type, 
            "description": template["description"],
            "position": list(position),
            "up_vector": list(up_vector),
            "footprint": template["footprint"],
            "component_count": len(components),
            "components": components,
            "assembly_rules": template["assembly_rules"],
            "materials_used": list(set(comp["material"] for comp in components)),
            "bounding_box": self._calculate_bounding_box(components),
            "generation_seed": seed
        }
        
        # Save asset manifest
        manifest_file = self.assets_dir / f"{asset_id}_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(asset_manifest, f, indent=2)
        
        return asset_manifest
    
    def _generate_component_instances(self, comp_spec: Dict, base_pos: Tuple[float, float, float],
                                    up_vector: Tuple[float, float, float], asset_id: str) -> List[Dict]:
        """Generate instances of a component type according to assembly rules."""
        component_type = comp_spec["type"] 
        base_component = comp_spec["component"]
        count = comp_spec["count"]
        
        if base_component not in self.component_library:
            raise ValueError(f"Unknown component: {base_component}")
        
        component_def = self.component_library[base_component]
        components = []
        
        # Generate component instances with proper positioning
        for i in range(count):
            # Calculate position based on component type and index
            comp_pos = self._calculate_component_position(
                component_type, i, count, base_pos, up_vector, component_def
            )
            
            # Create component instance
            component_instance = {
                "component_id": f"{asset_id}_{component_type}_{i:03d}",
                "component_type": component_type,
                "base_component": base_component,
                "position": comp_pos,
                "dimensions": component_def["dimensions"],
                "material": component_def["material_type"],
                "mesh_type": component_def["mesh_type"],
                "connection_points": component_def["connection_points"],
                "assembly_index": i
            }
            
            components.append(component_instance)
        
        return components
    
    def _calculate_component_position(self, comp_type: str, index: int, total_count: int,
                                    base_pos: Tuple[float, float, float], 
                                    up_vector: Tuple[float, float, float],
                                    component_def: Dict) -> List[float]:
        """Calculate position for a component instance based on assembly rules."""
        base_x, base_y, base_z = base_pos
        up_x, up_y, up_z = up_vector
        dims = component_def["dimensions"]
        
        # Simplified positioning logic (would be more complex in real implementation)
        if comp_type in ["foundation_wall", "foundation"]:
            # Arrange in rectangle around perimeter
            perimeter_pos = index / total_count * 2 * math.pi
            offset_x = 2.0 * math.cos(perimeter_pos)
            offset_z = 2.0 * math.sin(perimeter_pos)
            return [base_x + offset_x, base_y, base_z + offset_z]
            
        elif comp_type in ["wall_stud", "post", "corner_post"]:
            # Vertical components
            spacing = 0.61  # 24" on center
            row = index // 4
            col = index % 4
            return [base_x + col * spacing, base_y + 1.0, base_z + row * spacing]
            
        elif comp_type in ["floor_joist", "rafter", "roof_beam"]:
            # Horizontal spanning elements
            spacing = 0.41  # 16" on center
            offset = (index - total_count/2) * spacing
            return [base_x + offset, base_y + 0.5, base_z]
            
        elif comp_type in ["siding", "wall_board"]:
            # Wall covering - arranged in courses
            course_height = dims[1]  # Height of each board
            row = index // 8  # 8 boards per row
            col = index % 8
            return [base_x + col * dims[0], base_y + row * course_height, base_z]
            
        else:
            # Default grid arrangement
            grid_size = math.ceil(math.sqrt(total_count))
            row = index // grid_size
            col = index % grid_size
            return [base_x + col * 0.5, base_y, base_z + row * 0.5]
    
    def _calculate_bounding_box(self, components: List[Dict]) -> Dict[str, List[float]]:
        """Calculate bounding box for all components."""
        if not components:
            return {"min": [0, 0, 0], "max": [0, 0, 0]}
        
        positions = [comp["position"] for comp in components]
        dimensions = [comp["dimensions"] for comp in components]
        
        # Calculate min/max considering component dimensions
        min_x = min(pos[0] - dims[0]/2 for pos, dims in zip(positions, dimensions))
        max_x = max(pos[0] + dims[0]/2 for pos, dims in zip(positions, dimensions))
        min_y = min(pos[1] - dims[1]/2 for pos, dims in zip(positions, dimensions))
        max_y = max(pos[1] + dims[1]/2 for pos, dims in zip(positions, dimensions))
        min_z = min(pos[2] - dims[2]/2 for pos, dims in zip(positions, dimensions))
        max_z = max(pos[2] + dims[2]/2 for pos, dims in zip(positions, dimensions))
        
        return {
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z]
        }
    
    def generate_mesh_files(self, asset_manifest: Dict[str, Any]) -> List[str]:
        """Generate actual GLB mesh files for components (if trimesh available)."""
        if not TRIMESH_AVAILABLE:
            print("⚠️ Trimesh not available - skipping mesh file generation")
            return []
        
        generated_files = []
        asset_id = asset_manifest["asset_id"]
        
        for component in asset_manifest["components"]:
            mesh_file = self._generate_component_mesh_file(component, asset_id)
            if mesh_file:
                generated_files.append(mesh_file)
        
        return generated_files
    
    def _generate_component_mesh_file(self, component: Dict, asset_id: str) -> Optional[str]:
        """Generate GLB mesh file for a single component."""
        if not TRIMESH_AVAILABLE:
            return None
        
        try:
            # Create mesh based on component type
            dims = component["dimensions"]
            mesh_type = component["mesh_type"]
            
            if mesh_type == "box":
                mesh = trimesh.creation.box(extents=dims)
            else:
                # Fallback to box
                mesh = trimesh.creation.box(extents=dims)
            
            # Set up materials and colors based on material type
            material_colors = {
                "wood_frame": [0.6, 0.4, 0.2],      # Brown wood
                "wood_siding": [0.5, 0.3, 0.1],     # Darker wood  
                "concrete": [0.7, 0.7, 0.7],        # Gray concrete
                "roofing_asphalt": [0.2, 0.2, 0.3], # Dark shingles
            }
            
            material = component["material"]
            if material in material_colors:
                mesh.visual.face_colors = material_colors[material] + [1.0]  # Add alpha
            
            # Save mesh as GLB
            component_id = component["component_id"]
            mesh_filename = f"{component_id}.glb"
            mesh_path = self.assets_dir / mesh_filename
            
            mesh.export(str(mesh_path))
            
            return str(mesh_path)
            
        except Exception as e:
            print(f"⚠️ Failed to generate mesh for {component['component_id']}: {e}")
            return None

# Asset generation functions for integration
def generate_building_asset(building_type: str, position: Tuple[float, float, float],
                          up_vector: Tuple[float, float, float], 
                          seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a complex building asset."""
    generator = AssetGenerator()
    return generator.generate_complex_asset(building_type, position, up_vector, seed)

def get_available_building_types() -> List[str]:
    """Get list of available building types."""
    generator = AssetGenerator()
    return list(generator.asset_templates.keys())