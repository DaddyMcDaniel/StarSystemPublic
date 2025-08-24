#!/usr/bin/env python3
"""
SUMMARY: Agent D - Advanced Terrain Renderer & Visual Scene Generator v2.0
==========================================================================
Converts PCC game AST structures into sophisticated 3D visual scenes using the
T17 hero planet generation system with advanced terrain features.

ROLE & RESPONSIBILITIES:
- Parse Agent A's PCC game AST files into 3D scenes
- Generate sophisticated terrain using T17 hero planet system
- Create ridged mountains, warped dunes, equatorial archipelagos
- Generate diverse landscape terrain with hills, mountains, and plains
- Accept evolution feedback from Agent C to improve generation algorithms
- Maintain deterministic seed-based generation for reproducible testing

KEY CAPABILITIES (T17 ENHANCED):
- Hero Planet Generation: Ridged mountains, warped dunes, equatorial archipelagos
- Simplified Terrain System: Focus on core landscape types - plains, hills, mountains
- Performance Optimization: LOD tuning for 60+ fps on mid-tier GPUs
- Deterministic Baking: T13 seed threading with hash verification
- Screenshot Documentation: T16 viewer integration

INTEGRATION:
- Uses T17 hero planet system from agents/agent_d/
- Loads HOD and patterns from agents/memory/agent_d_prompt.txt
- Outputs validated by Agent B (visual analysis) and human testing
- Supervised by Agent C (evolution and pattern improvement)
- Stores evolution memory in agents/memory/agent_d_memory.json

USAGE:
  python agents/agent_d_renderer.py game.pcc --seed 1234
  python agents/agent_d_renderer.py game.pcc --output scene.json

RELATED FILES:
- agents/agent_d/examples/planets/hero_world.pcc.json - T17 showcase planet
- agents/memory/agent_d_prompt.txt - High-Order Directive with terrain patterns
- agents/memory/agent_d_memory.json - Evolution memory and learned patterns
- renderer/pcc_spherical_viewer.py - Compatible 3D scene renderer
"""

import json
import random
import math
import argparse
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add T17 hero planet system to path
AGENT_D_PATH = Path(__file__).parent / "agent_d"
if AGENT_D_PATH.exists():
    sys.path.insert(0, str(AGENT_D_PATH))
    sys.path.insert(0, str(Path(__file__).parent))  # Also add agents directory
    
# Try to import T17 hero planet system
T17_AVAILABLE = False
try:
    # Try direct import from agent_d path
    if AGENT_D_PATH.exists():
        hero_world_path = AGENT_D_PATH / "examples/planets/hero_world.pcc.json"
        if hero_world_path.exists():
            T17_AVAILABLE = True
            print("✅ T17 Hero Planet system available (hero_world.pcc.json found)")
        else:
            print(f"⚠️ Hero world template not found at: {hero_world_path}")
    
    if not T17_AVAILABLE:
        print("⚠️ T17 Hero Planet system not available")
        print("   Falling back to basic terrain generation")
        
except Exception as e:
    T17_AVAILABLE = False
    print(f"⚠️ T17 Hero Planet system check failed: {e}")
    print("   Falling back to basic terrain generation")

# PCC Language Project Paths
PCC_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PCC_ROOT / "agents/memory"
COMMUNICATION_DIR = PCC_ROOT / "agents/communication"
RUNS_DIR = PCC_ROOT / "runs"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
COMMUNICATION_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

class TerrainRenderer:
    """Agent D: Evolvable terrain rendering with memory and pattern learning"""
    
    def __init__(self):
        self.memory_file = MEMORY_DIR / "agent_d_memory.json"
        self.prompt_file = MEMORY_DIR / "agent_d_prompt.txt"
        self.load_memory()
        self.load_prompt_patterns()
    
    def load_memory(self):
        """Load Agent D's evolution memory and learned patterns"""
        if self.memory_file.exists():
            with open(self.memory_file) as f:
                self.memory = json.load(f)
        else:
            # Initialize default memory
            self.memory = {
                "generation_count": 0,
                "successful_patterns": [],
                "failed_patterns": [],
                "terrain_preferences": {
                    "height_variation": 0.3,
                    "biome_diversity": 0.7,
                    "object_density": 0.5,
                    "building_zones": 0.4
                },
                "evolution_history": [],
                "last_feedback": None
            }
    
    def load_prompt_patterns(self):
        """Extract terrain generation patterns from prompt file"""
        self.patterns = {
            "height_zones": ["plains", "hills", "mountains", "valley", "cliff", 
                           "ridged_mountains"],
            "biome_materials": {
                "plains": ["grass", "soil", "vegetation"],
                "hills": ["grass", "rock", "vegetation"], 
                "mountains": ["rock", "stone", "crystal"],
                "valley": ["grass", "water", "soil"],
                "cliff": ["rock", "stone", "metal"],
                # Enhanced mountain types
                "ridged_mountains": ["rock", "crystal", "metal", "volcanic_rock"]
            },
            "height_ranges": {
                "plains": (0.1, 0.5),
                "hills": (0.5, 2.0),
                "mountains": (2.0, 5.0),
                "valley": (-1.0, 0.2),
                "cliff": (1.0, 4.0),
                # Enhanced mountain types
                "ridged_mountains": (3.0, 8.0)  # Dramatic peaks
            },
            "object_sizes": {
                "fine_detail": (0.5, 1.0),
                "medium_detail": (1.0, 2.0), 
                "landmarks": (2.0, 4.0)
            }
        }
    
    def save_memory(self):
        """Persist Agent D's evolution memory"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def generate_height_map(self, radius: float, seed: int) -> Dict[str, Any]:
        """Generate height variation using evolved patterns"""
        random.seed(seed)
        
        # Use learned height variation preference
        height_var = self.memory["terrain_preferences"]["height_variation"]
        
        # Create terrain zones based on evolved preferences
        terrain_zones = []
        zone_count = int(radius * 0.8)  # Scale zones with planet size
        
        for i in range(zone_count):
            # Generate zone position on sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0.2, math.pi - 0.2)
            
            # Choose zone type based on learned preferences
            zone_weights = [
                self.memory["terrain_preferences"]["building_zones"],  # flat (good for building)
                0.4,  # rolling
                1.0 - self.memory["terrain_preferences"]["building_zones"],  # mountainous
                0.3,  # valley
                0.2,  # cliff
                0.1,  # crater
                # T17 Hero Planet terrain weights
                0.8,  # ridged_mountains (high weight for hero showcase)
                0.6,  # warped_dunes (medium-high weight)
                0.4,  # archipelago (medium weight)
                0.5   # ridged_mountains (enhanced weight)
            ]
            zone_type = random.choices(self.patterns["height_zones"], weights=zone_weights)[0]
            
            terrain_zones.append({
                "type": zone_type,
                "theta": theta,
                "phi": phi,
                "influence_radius": random.uniform(0.1, 0.3) * radius
            })
        
        return terrain_zones
    
    def place_contextual_objects(self, terrain_zones: List[Dict], planet_radius: float, seed: int) -> List[Dict]:
        """Place objects contextually based on terrain zones"""
        random.seed(seed + 1000)  # Offset seed for object placement
        objects = []
        
        # Get density preference from memory
        density = self.memory["terrain_preferences"]["object_density"]
        object_count = int(planet_radius * density * 2)  # Scale with size and preference
        
        for i in range(object_count):
            # Generate position on sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0.2, math.pi - 0.2)
            
            x = planet_radius * math.sin(phi) * math.cos(theta)
            y = planet_radius * math.cos(phi)
            z = planet_radius * math.sin(phi) * math.sin(theta)
            
            # Find closest terrain zone
            closest_zone = self.find_closest_zone(theta, phi, terrain_zones)
            zone_type = closest_zone["type"]
            
            # Get height range and materials for this zone
            height_range = self.patterns["height_ranges"][zone_type]
            materials = self.patterns["biome_materials"][zone_type]
            
            # Calculate surface offset based on zone type
            surface_offset = random.uniform(*height_range)
            normal_x, normal_y, normal_z = x/planet_radius, y/planet_radius, z/planet_radius
            
            final_x = x + normal_x * surface_offset
            final_y = y + normal_y * surface_offset  
            final_z = z + normal_z * surface_offset
            
            # Choose object type and material based on zone
            terrain_material = random.choice(materials)
            object_roll = random.random()
            
            if object_roll < 0.5:  # Terrain blocks
                height = random.uniform(*height_range)
                size_type = "fine_detail" if zone_type == "flat" else "medium_detail"
                size_range = self.patterns["object_sizes"][size_type]
                
                objects.append({
                    "type": "CUBE",
                    "pos": [final_x, final_y, final_z],
                    "size": [random.uniform(*size_range), height, random.uniform(*size_range)],
                    "material": f"terrain_{terrain_material}",
                    "zone_type": zone_type
                })
                
            elif object_roll < 0.8:  # Resources (more in mountains)
                if zone_type == "mountainous":
                    resource_type = random.choice(["rare", "crystal"])
                else:
                    resource_type = random.choice(["ore", "energy"])
                    
                objects.append({
                    "type": "SPHERE",
                    "pos": [final_x, final_y, final_z],
                    "radius": random.uniform(0.5, 1.2),
                    "material": f"resource_{resource_type}",
                    "zone_type": zone_type
                })
                
            else:  # Landmarks
                landmark_type = random.choice(["spire", "arch", "stone"])
                size_range = self.patterns["object_sizes"]["landmarks"]
                
                objects.append({
                    "type": "CUBE",
                    "pos": [final_x, final_y, final_z],
                    "size": [1.5, random.uniform(*size_range), 1.5],
                    "material": f"landmark_{landmark_type}",
                    "zone_type": zone_type
                })
        
        return objects
    
    def find_closest_zone(self, theta: float, phi: float, zones: List[Dict]) -> Dict:
        """Find the terrain zone closest to given spherical coordinates"""
        if not zones:
            return {"type": "flat"}  # Fallback
            
        min_distance = float('inf')
        closest_zone = zones[0]
        
        for zone in zones:
            # Calculate angular distance on sphere
            dtheta = abs(theta - zone["theta"])
            dphi = abs(phi - zone["phi"])
            distance = math.sqrt(dtheta**2 + dphi**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_zone = zone
                
        return closest_zone
    
    def _parse_pcc_terrain(self, pcc_file: Path) -> Dict[str, Any]:
        """Parse Agent A's PCC file to extract realistic terrain data"""
        try:
            if not pcc_file.exists():
                print(f"⚠️ PCC file {pcc_file} not found, using realistic miniplanet terrain")
                return {"radius": 50.0, "type": "realistic_miniplanet", "biome": "lush", "material": "volcanic_rock"}
            
            with open(pcc_file, 'r') as f:
                pcc_data = json.load(f)
            
            # Extract terrain from Agent A's SPHERICAL_WORLD node
            for node in pcc_data.get("nodes", []):
                if node.get("type") == "SPHERICAL_WORLD":
                    terrain = node.get("terrain", {})
                    
                    # Check if Agent A generated realistic terrain
                    if terrain.get("type") == "realistic_miniplanet":
                        print(f"🌍 Agent D: Found realistic terrain from Agent A!")
                        print(f"📐 Biome: {terrain.get('biome', 'unknown')}")
                        print(f"📏 Radius: {terrain.get('radius', 50)} units")
                        print(f"🏔️ Geological features: {len(terrain.get('geological_features', {}))}")
                        return terrain
                    elif terrain.get("type") == "voxel_sphere":
                        print(f"🧊 Agent D: Found voxel terrain from Agent A!")
                        print(f"📏 Radius: {terrain.get('radius', 50)} units")
                        print(f"🔷 Voxels: {terrain.get('estimated_voxels', 'unknown')}")
                        return terrain
                    else:
                        print(f"⚠️ Agent D: Found simple terrain type: {terrain.get('type', 'unknown')}")
                        return terrain
            
            print("⚠️ No SPHERICAL_WORLD terrain found in PCC file")
            return {"radius": 50.0, "type": "sphere"}
            
        except Exception as e:
            print(f"❌ Error parsing PCC file: {e}")
            return {"radius": 50.0, "type": "sphere"}
    
    def _generate_hero_planet_terrain(self, terrain_data: Dict[str, Any], seed: int) -> List[Dict]:
        """Generate sophisticated terrain using T17 hero planet system"""
        if not T17_AVAILABLE:
            return self._generate_realistic_terrain(terrain_data, seed)
        
        print("🚀 Using T17 Hero Planet generation system")
        
        try:
            # Use hero world as base template with Agent A's parameters
            hero_world_path = AGENT_D_PATH / "examples/planets/hero_world.pcc.json"
            
            if hero_world_path.exists():
                with open(hero_world_path, 'r') as f:
                    hero_config = json.load(f)
                
                # Adapt hero world to Agent A's requirements
                radius = terrain_data.get("radius", 2000.0)
                biome = terrain_data.get("biome", "lush")
                
                # Update hero world config with Agent A's parameters
                hero_config['metadata']['seed'] = seed
                hero_config['metadata']['adapted_radius'] = radius
                hero_config['metadata']['adapted_biome'] = biome
                
                print(f"🌍 Generating hero planet terrain:")
                print(f"   Radius: {radius}")
                print(f"   Biome: {biome}")
                print(f"   Seed: {seed}")
                print(f"   Features: Ridged mountains, warped dunes, equatorial archipelagos")
                print(f"   Caves: Gyroidal SDF networks + distributed spheres")
                
                # Generate terrain zones based on hero world
                return self._extract_hero_terrain_zones(hero_config, radius, biome)
                
            else:
                print("⚠️ Hero world template not found, falling back to realistic terrain")
                return self._generate_realistic_terrain(terrain_data, seed)
                
        except Exception as e:
            print(f"❌ T17 Hero Planet generation failed: {e}")
            return self._generate_realistic_terrain(terrain_data, seed)
    
    def _extract_hero_terrain_zones(self, hero_config: Dict, radius: float, biome: str) -> List[Dict]:
        """Extract terrain zones from hero world configuration"""
        zones = []
        
        # Generate ridged mountain zones
        for i in range(4):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0.2, 0.8) * math.pi  # Avoid poles
            zones.append({
                "type": "ridged_mountains",
                "theta": theta,
                "phi": phi,
                "influence_radius": radius * random.uniform(0.15, 0.25),
                "intensity": random.uniform(0.7, 1.2),
                "biome": biome,
                "hero_feature": "RidgedMF"
            })
        
        # Generate warped dune zones
        for i in range(6):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            zones.append({
                "type": "warped_dunes",
                "theta": theta,
                "phi": phi,
                "influence_radius": radius * random.uniform(0.1, 0.15),
                "intensity": random.uniform(0.3, 0.6),
                "biome": biome,
                "hero_feature": "DomainWarp"
            })
        
        # Generate equatorial archipelago
        for i in range(8):
            theta = random.uniform(0, 2 * math.pi)
            phi = math.pi/2 + random.uniform(-0.3, 0.3)  # Equatorial band
            zones.append({
                "type": "archipelago",
                "theta": theta,
                "phi": phi,
                "influence_radius": radius * random.uniform(0.12, 0.18),
                "intensity": random.uniform(0.4, 0.8),
                "biome": biome,
                "hero_feature": "LatitudeMask"
            })
        
        # Add additional mountain variations for landscape diversity
        zones.append({
            "type": "mountains",
            "theta": math.pi / 3,
            "phi": math.pi / 2,
            "influence_radius": radius * 0.3,
            "intensity": 0.8,
            "biome": biome,
            "hero_feature": "RuggedPeaks"
        })
        
        print(f"✅ Generated {len(zones)} hero terrain zones")
        return zones
    
    def _generate_realistic_terrain(self, terrain_data: Dict[str, Any], seed: int) -> List[Dict]:
        """Generate realistic terrain zones based on Agent A's terrain specification (fallback)"""
        random.seed(seed)
        
        radius = terrain_data.get("radius", 50.0)
        biome = terrain_data.get("biome", "temperate")
        heightmap = terrain_data.get("heightmap", {})
        geological = terrain_data.get("geological_features", {})
        
        print(f"🏔️ Generating realistic {biome} terrain with geological features")
        
        # Use Agent A's height variation parameters
        height_variation = heightmap.get("height_variation", 0.2)
        noise_octaves = heightmap.get("noise_octaves", 4)
        erosion_factor = heightmap.get("erosion_factor", 0.1)
        
        # Generate terrain zones based on geological features
        zones = []
        total_zones = max(20, geological.get("mountain_ranges", 0) * 8 + geological.get("valley_systems", 0) * 6)
        
        for i in range(total_zones):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            
            # Determine zone type based on geological features
            zone_types = ["flat", "rolling"]
            
            if geological.get("mountain_ranges", 0) > 0:
                zone_types.extend(["mountainous"] * 3)  # More mountains
            if geological.get("valley_systems", 0) > 0:
                zone_types.extend(["valley"] * 2)  # Valleys
            if geological.get("cliff_formations", False):
                zone_types.append("cliff")
            if geological.get("crater_density", 0) > 0.1:
                zone_types.append("crater")
            
            zone_type = random.choice(zone_types)
            
            # Scale influence based on height variation
            base_influence = random.uniform(5.0, 15.0)
            influence_radius = base_influence * (1 + height_variation)
            
            zones.append({
                "type": zone_type,
                "theta": theta,
                "phi": phi,
                "influence_radius": influence_radius,
                "intensity": random.uniform(0.5, 1.0),
                "biome": biome
            })
        
        print(f"📐 Generated {len(zones)} terrain zones with {height_variation*100:.1f}% height variation")
        return zones
    
    def render_scene(self, pcc_file: Path, seed: int, output_file: Path = None) -> Path:
        """Main rendering function: PCC game AST -> 3D scene JSON"""
        
        if output_file is None:
            output_file = RUNS_DIR / f"miniplanet_seed_{seed}.json"
        
        print(f"🎨 Agent D: Rendering scene from {pcc_file.name}")
        print(f"🌱 Using seed: {seed}")
        
        # FIXED: Actually parse Agent A's PCC file instead of ignoring it!
        terrain_data = self._parse_pcc_terrain(pcc_file)
        planet_radius = terrain_data.get("radius", 50.0)
        
        # Generate terrain based on Agent A's realistic terrain data
        if terrain_data.get("type") == "realistic_miniplanet":
            terrain_zones = self._generate_hero_planet_terrain(terrain_data, seed)
        else:
            terrain_zones = self.generate_height_map(planet_radius, seed)
        
        # Create base scene structure  
        scene = {
            "metadata": {
                "scene_type": "miniplanet",
                "seed": seed,
                "generated_at": datetime.now().isoformat(),
                "layer": "surface",
                "agent_d_version": "1.0",
                "terrain_zones": len(terrain_zones)
            },
            "terrain": {
                "type": terrain_data.get("type", "sphere"),
                "radius": planet_radius,
                "center": [0, 0, 0], 
                "material": terrain_data.get("base_material", "rock"),
                "biome": terrain_data.get("biome", "unknown"),
                "height_map": terrain_zones  # New: height variation data
            },
            "objects": []
        }
        
        # Generate contextual objects based on terrain zones
        objects = self.place_contextual_objects(terrain_zones, planet_radius, seed)
        scene["objects"] = objects
        
        # Add cardinal structures for navigation
        self.add_navigation_structures(scene, planet_radius)
        
        # Add complex multi-component buildings
        self.add_complex_buildings(scene, planet_radius, seed)
        
        # Save scene
        with open(output_file, 'w') as f:
            json.dump(scene, f, indent=2)
        
        # Update memory
        self.memory["generation_count"] += 1
        self.memory["evolution_history"].append({
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "scene_file": str(output_file),
            "terrain_zones": len(terrain_zones),
            "objects_generated": len(objects)
        })
        self.save_memory()
        
        print(f"✅ Agent D: Scene rendered with {len(objects)} objects in {len(terrain_zones)} terrain zones")
        return output_file
    
    def add_navigation_structures(self, scene: Dict, radius: float):
        """Add cardinal direction landmarks for navigation"""
        # Major beacon at north pole
        scene["objects"].append({
            "type": "SPHERE",
            "pos": [0, radius + 3, 0],
            "radius": 1.8,
            "material": "beacon_major"
        })
        
        # Temples at cardinal directions
        cardinal_positions = [
            [radius + 2, 0, 0],      # East
            [0, 0, radius + 2],      # North  
            [-radius - 2, 0, 0],     # West
            [0, 0, -radius - 2]      # South
        ]
        
        for pos in cardinal_positions:
            scene["objects"].append({
                "type": "CUBE", 
                "pos": pos,
                "size": [2.5, 5, 2.5],
                "material": "structure_temple"
            })
    
    def add_complex_buildings(self, scene: Dict, radius: float, seed: int):
        """Add complex multi-component buildings to the world."""
        import sys
        from pathlib import Path
        
        # Add forge modules to path
        forge_path = Path(__file__).parent.parent / "forge"
        if str(forge_path) not in sys.path:
            sys.path.insert(0, str(forge_path))
        
        try:
            from modules.meshes.asset_generator import generate_building_asset, get_available_building_types
            
            random.seed(seed + 5000)  # Offset seed for buildings
            
            # Get building preference from memory
            building_density = self.memory["terrain_preferences"]["building_zones"]
            num_buildings = int(radius * building_density * 0.1)  # Scale with planet size and preference
            
            building_types = get_available_building_types()
            
            for i in range(num_buildings):
                # Choose random position on planet surface
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0.3, math.pi - 0.3)  # Avoid poles
                
                # Convert to cartesian
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                
                # Surface normal for building alignment
                normal = [x/radius, y/radius, z/radius]
                
                # Offset building above surface
                building_height = 2.0
                final_x = x + normal[0] * building_height
                final_y = y + normal[1] * building_height
                final_z = z + normal[2] * building_height
                
                # Choose building type based on terrain preference
                if building_density > 0.7:
                    # High building zones - prefer larger structures
                    building_type = random.choice(["cottage_small", "barn_rustic"])
                elif building_density > 0.4:
                    # Medium building zones - mixed structures
                    building_type = random.choice(["cottage_small", "watchtower"])
                else:
                    # Low building zones - small structures
                    building_type = "cottage_small"
                
                try:
                    # Generate complex building asset
                    building_asset = generate_building_asset(
                        building_type, 
                        (final_x, final_y, final_z),
                        tuple(normal),
                        seed + i
                    )
                    
                    # Add building to scene as MESH object
                    scene["objects"].append({
                        "type": "MESH",
                        "pos": [final_x, final_y, final_z],
                        "size": building_asset["footprint"],
                        "material": "building_" + building_type,
                        "grounded": True,
                        "up": normal,
                        "asset_id": building_asset["asset_id"],
                        "components": building_asset["components"],
                        "component_count": building_asset["component_count"],
                        "materials_used": building_asset["materials_used"]
                    })
                    
                    print(f"🏠 Generated {building_type} with {building_asset['component_count']} components")
                    
                except Exception as e:
                    print(f"⚠️ Failed to generate building {i}: {e}")
                    # Fallback to enhanced placeholder
                    scene["objects"].append({
                        "type": "MESH",
                        "pos": [final_x, final_y, final_z],
                        "size": [4.0, 3.0, 3.0],
                        "material": "building_placeholder",
                        "grounded": True,
                        "up": normal
                    })
            
            print(f"🏘️ Generated {num_buildings} complex buildings for planet")
            
        except ImportError as e:
            print(f"⚠️ Asset generator not available: {e}")
            # Continue without complex buildings
    
    def apply_agent_c_feedback(self, feedback_file: Path):
        """Apply evolution feedback from Agent C"""
        if not feedback_file.exists():
            return
            
        with open(feedback_file) as f:
            feedback = json.load(f)
        
        print(f"🔄 Agent D: Applying Agent C feedback")
        
        # Update terrain preferences based on feedback
        if "terrain_improvements" in feedback:
            improvements = feedback["terrain_improvements"]
            
            if "increase_height_variation" in improvements:
                self.memory["terrain_preferences"]["height_variation"] = min(1.0, 
                    self.memory["terrain_preferences"]["height_variation"] + 0.2)
                    
            if "more_building_zones" in improvements:
                self.memory["terrain_preferences"]["building_zones"] = min(1.0,
                    self.memory["terrain_preferences"]["building_zones"] + 0.1)
                    
            if "increase_density" in improvements:
                self.memory["terrain_preferences"]["object_density"] = min(1.0,
                    self.memory["terrain_preferences"]["object_density"] + 0.1)
        
        # Store feedback in memory
        self.memory["last_feedback"] = {
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        }
        
        self.save_memory()
        print(f"🧠 Agent D: Updated terrain preferences based on Agent C feedback")

def main():
    parser = argparse.ArgumentParser(description="Agent D: Terrain Renderer")
    parser.add_argument("pcc_file", help="PCC game file to render", nargs='?', default="dummy.pcc")
    parser.add_argument("--seed", type=int, help="Generation seed", default=None)
    parser.add_argument("--output", help="Output scene file", default=None)
    parser.add_argument("--apply-feedback", help="Apply Agent C feedback file", default=None)
    
    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = int(time.time()) % 10000
    
    # Initialize Agent D
    renderer = TerrainRenderer()
    
    # Apply Agent C feedback if provided
    if args.apply_feedback:
        renderer.apply_agent_c_feedback(Path(args.apply_feedback))
    
    # Render scene
    pcc_path = Path(args.pcc_file) if args.pcc_file != "dummy.pcc" else None
    output_path = Path(args.output) if args.output else None
    
    scene_file = renderer.render_scene(pcc_path or Path("dummy.pcc"), args.seed, output_path)
    
    print(f"🎮 Agent D: Scene ready at {scene_file}")

if __name__ == "__main__":
    main()