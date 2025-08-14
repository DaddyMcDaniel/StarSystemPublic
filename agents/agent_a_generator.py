#!/usr/bin/env python3
"""
SUMMARY: Agent A - Primary Code Generator v1
============================================
Fast, prolific code generator specializing in PCC language features and runtime improvements.
Primary agent in the A→D→B→C evolution cycle for StarSystem development.

ROLE & RESPONSIBILITIES:
- Rapid prototyping of PCC language features and Godot integration enhancements
- Generating deterministic code changes with proper seed management
- Creating testable implementations following established patterns
- Proposing surgical edits rather than broad architectural rewrites
- Building grid-based placement logic inspired by Terraria/Eco hammer tools

KEY CAPABILITIES:
- Blueprint Chips: Generate deterministic node graphs from natural language prompts
- Replay Systems: Create reproducible test scenarios with exact seed reproduction
- Building Systems: Design grid-based placement and validation logic
- Planet Generation: Generate layered, navigatable, alien-like mini-planets
- Performance-aware: Ensures generated code meets efficiency requirements

INTEGRATION:
- Loads HOD from agents/memory/agent_a_prompt.txt
- Outputs validated by Agent B (tester/validator)
- Supervised by Agent C (governance/coordination)
- Focus on unlimited creative building time (Halo Forge mode philosophy)

USAGE:
  python agents/agent_a_generator.py
  make agents  # Run full A→B→C handshake

RELATED FILES:
- agents/memory/agent_a_prompt.txt - High-Order Directive with contracts
- Week 2+ focus on idea → playable mini-planet workflow
"""
import random
import json
from pathlib import Path
from datetime import datetime

# PCC Language Project Paths
PCC_ROOT = Path(__file__).parent.parent
PCC_SPEC = PCC_ROOT / "docs/pcc_language_spec.md"
PCC_GAMES = PCC_ROOT / "tests/examples"
PCC_LOGS = PCC_ROOT / "agents/logs"

PCC_GAMES.mkdir(parents=True, exist_ok=True)
PCC_LOGS.mkdir(parents=True, exist_ok=True)

class PCCGameGenerator:
    def __init__(self):
        self.compression_level = 1.0  # Evolution metric
        self.successful_patterns = []  # Learned from feedback
        
    def generate_spherical_planet_prompt(self):
        """Step 1: Generate spherical planet worldbuilding prompts aligned with HOD"""
        # HOD Focus: Planet Generation - layered, navigatable, alien-like mini-planets
        planet_features = ["crystalline formations", "ancient temple ruins", "resource deposits", "cave networks", "atmospheric rings"]
        environments = ["volcanic crater regions", "frozen polar zones", "lush equatorial forests", "metallic desert plains", "bioluminescent valleys"]
        building_elements = ["modular platform bases", "resource extraction points", "beacon networks", "transport hubs", "craft stations"]
        
        feature = random.choice(planet_features)
        environment = random.choice(environments)
        building = random.choice(building_elements)
        
        # Focus on spherical planet exploration and building (aligned with HOD)
        prompt = f"Create a spherical mini-planet with {feature} in {environment} where player can build {building}"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": 1,
            "prompt": prompt,
            "agent": "A",
            "hod_alignment": "spherical_planet_worldbuilding"
        }
        self._log(log_entry)
        
        return prompt
    
    def prompt_to_pcc_ast(self, prompt):
        """Step 2: Convert English prompt to spherical planet PCC AST structure"""
        # HOD-aligned: Convert prompt to spherical planet world structure
        
        # Spherical planet AST structure (aligned with Week 3 mechanics)
        # Build initial AST
        pcc_ast = {
            "type": "SPHERICAL_PLANET_GAME",
            "nodes": [
                {
                    "type": "SPHERICAL_WORLD",
                    "terrain": {
                        "type": "sphere",
                        "radius": random.uniform(20.0, 35.0),  # Varied planet sizes
                        "center": [0, 0, 0],
                        "material": self._extract_terrain_material(prompt)
                    },
                    "features": self._extract_planet_features(prompt),
                    "building_zones": self._extract_building_zones(prompt)
                },
                {
                    "type": "SPHERICAL_PLAYER", 
                    "spawn_pos": "equator_safe",  # Avoid north pole spawn issue
                    "surface_locked": True,
                    "controls": ["WASD", "mouse_look", "space_jump"]
                },
                {
                    "type": "EXPLORATION_OBJECTIVE",
                    "goal": self._extract_spherical_goal(prompt)
                }
            ],
            "compression_id": self._generate_compression_id(),
            "planet_seed": random.randint(1, 1000000)  # Deterministic seed cascade
        }
        
        # Draft asset requests for realistic meshes based on extracted features
        assets_manifest = self._build_assets_manifest_from_prompt(prompt, pcc_ast)
        if assets_manifest["assets"]:
            pcc_ast["assets"] = [a["id"] for a in assets_manifest["assets"]]
        
        # Save as .pcc file
        pcc_filename = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcc"
        pcc_path = PCC_GAMES / pcc_filename
        
        with open(pcc_path, 'w') as f:
            json.dump(pcc_ast, f, indent=2)
        
        # Save assets manifest alongside for external 3D generation pipeline
        assets_dir = PCC_ROOT / "runs" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = assets_dir / (pcc_filename.replace('.pcc', '_assets.json'))
        with open(manifest_path, 'w') as mf:
            json.dump(assets_manifest, mf, indent=2)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": 2,
            "prompt": prompt,
            "pcc_file": pcc_filename,
            "ast_nodes": len(pcc_ast["nodes"]),
            "compression_level": self.compression_level,
            "agent": "A",
            "assets_manifest": str(manifest_path)
        }
        self._log(log_entry)
        
        return pcc_path, pcc_ast

    def _build_assets_manifest_from_prompt(self, prompt: str, pcc_ast: dict) -> dict:
        """Create an assets manifest describing high-quality meshes to generate via external service."""
        def mk_asset(asset_id: str, prompt_text: str, category: str, tags: list[str], scale_hint: float = 1.0):
            return {
                "id": asset_id,
                "prompt": prompt_text,
                "category": category,
                "tags": tags,
                "format": "glb",
                "lod": [0, 1],
                "scale_hint_meters": scale_hint,
                "grounded": True,
                "pivot": "base",
                "physics": {"collision": "mesh_simplified"}
            }
        assets: list[dict] = []
        requests: list[dict] = []
        t = pcc_ast.get("nodes", [])[0].get("terrain", {})
        terrain_mat = t.get("material", "standard_rock")
        # Terrain-dependent foliage set
        if "forest" in prompt or terrain_mat in ("bio_soil",):
            assets.append(mk_asset("foliage_grass_clump_v1", "Realistic alien grass clump with bioluminescent tips, grounded at base.", "foliage", ["grass","alien","bioluminescent"], 0.5))
            assets.append(mk_asset("foliage_shrub_v1", "Low alien shrub with waxy leaves and subtle glow, grounded base.", "foliage", ["shrub","alien"], 0.8))
        if any(f.get("type") == "ANCIENT_STRUCTURE" for f in self._extract_planet_features(prompt)):
            assets.append(mk_asset("ruin_temple_column_v1", "Weathered stone column fragment with erosion, ancient temple aesthetic.", "structure", ["ruin","temple","column"], 3.0))
            assets.append(mk_asset("ruin_arch_v1", "Broken sandstone arch with chipped edges, believable wear.", "structure", ["ruin","arch"], 4.0))
        if any(f.get("type") == "CRYSTAL_FORMATION" for f in self._extract_planet_features(prompt)):
            assets.append(mk_asset("crystal_cluster_v1", "Semi-transparent crystal cluster with varied shard sizes, grounded base plate.", "resource", ["crystal","cluster"], 1.5))
        
        # Build placement intents to guide worldgen on usage
        placements: list[dict] = []
        for a in assets:
            placements.append({
                "asset_id": a["id"],
                "usage": "scatter_surface",
                "density": "medium" if a["category"] == "foliage" else "low",
                "align_to_surface": True,
                "avoid_spawn_radius_m": 10.0
            })
        
        # Baseline assets to ensure non-empty realistic set
        assets.append(mk_asset("rock_boulder_v1", "Weathered rock boulder with irregular silhouette and realistic shading, grounded base.", "terrain", ["rock","boulder"], 1.2))
        assets.append(mk_asset("ground_patch_v1", "Tileable ground patch decal matching terrain material with subtle variation.", "terrain", ["ground","decal"], 1.0))
        if terrain_mat in ("standard_rock","volcanic_rock","metal_sand"):
            assets.append(mk_asset("alien_grass_sparse_v1", "Sparse hardy alien grass tuft adapted to arid terrain, grounded clump.", "foliage", ["grass","alien"], 0.4))

        # External generation request envelope
        manifest = {
            "schema": "assets_manifest.v1",
            "game_id": pcc_ast.get("compression_id", "unknown"),
            "terrain_material": terrain_mat,
            "assets": assets,
            "placements": placements,
            "delivery": {
                "target_dir": "runs/assets/generated",
                "filenames": "{asset_id}.glb"
            }
        }
        return manifest
    
    def _extract_terrain_material(self, prompt):
        """Extract terrain material from spherical planet prompt"""
        if "volcanic" in prompt:
            return "volcanic_rock"
        elif "frozen" in prompt or "polar" in prompt:
            return "ice_crystal"
        elif "forest" in prompt or "lush" in prompt:
            return "bio_soil"
        elif "metallic" in prompt or "desert" in prompt:
            return "metal_sand"
        elif "bioluminescent" in prompt:
            return "glow_earth"
        else:
            return "standard_rock"

    def _extract_planet_features(self, prompt):
        """Extract planet surface features from prompt"""
        features = []
        
        if "crystalline" in prompt:
            features.append({"type": "CRYSTAL_FORMATION", "distribution": "scattered", "material": "resource_crystal"})
        if "temple" in prompt or "ruins" in prompt:
            features.append({"type": "ANCIENT_STRUCTURE", "style": "temple", "material": "structure_temple"})
        if "resource" in prompt or "deposits" in prompt:
            features.append({"type": "RESOURCE_NODES", "material": "resource_ore", "density": "medium"})
        if "cave" in prompt:
            features.append({"type": "CAVE_SYSTEMS", "material": "cave_entrance", "depth": "surface"})
        if "atmospheric" in prompt or "rings" in prompt:
            features.append({"type": "ATMOSPHERIC_EFFECT", "style": "orbital_rings", "visibility": "high"})
            
        # Always add beacon for navigation reference
        features.append({"type": "NAVIGATION_BEACON", "pos": "north_pole", "material": "beacon_major"})
        
        return features

    def _extract_building_zones(self, prompt):
        """Extract building zones from prompt"""
        zones = []
        
        if "platform" in prompt:
            zones.append({"type": "PLATFORM_ZONE", "size": "medium", "material": "structure_platform"})
        if "extraction" in prompt:
            zones.append({"type": "EXTRACTION_ZONE", "resource_type": "ore", "efficiency": "standard"})
        if "beacon" in prompt:
            zones.append({"type": "BEACON_NETWORK", "range": "global", "material": "beacon_minor"})
        if "transport" in prompt or "hub" in prompt:
            zones.append({"type": "TRANSPORT_HUB", "capacity": "multi_player", "material": "structure_hub"})
        if "craft" in prompt or "station" in prompt:
            zones.append({"type": "CRAFT_STATION", "recipes": "basic", "material": "structure_workbench"})
            
        return zones
    
    def _extract_spherical_goal(self, prompt):
        """Extract exploration objective from spherical planet prompt"""
        if "build" in prompt and "platform" in prompt:
            return {"type": "BUILD_STRUCTURE", "target": "platform_base", "circumnavigation_required": True}
        elif "build" in prompt and "extraction" in prompt:
            return {"type": "ESTABLISH_MINING", "target": "resource_network", "planet_coverage": 0.3}
        elif "build" in prompt and "beacon" in prompt:
            return {"type": "CREATE_NETWORK", "target": "beacon_grid", "global_visibility": True}
        elif "build" in prompt and "transport" in prompt:
            return {"type": "CONNECT_ZONES", "target": "transport_system", "efficiency": "optimal"}
        else:
            return {"type": "EXPLORE_PLANET", "coverage": 0.8, "time_limit": None}
    
    def _generate_compression_id(self):
        """Generate unique compression identifier for evolution tracking"""
        return f"C{random.randint(1000, 9999)}"
    
    def _log(self, entry):
        """Log all agent activities"""
        log_file = PCC_LOGS / "agent_a.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    generator = PCCGameGenerator()
    prompt = generator.generate_spherical_planet_prompt()
    print(f"🌍 Generated spherical planet: {prompt}")
    pcc_file, ast = generator.prompt_to_pcc_ast(prompt)
    print(f"🔧 Created: {pcc_file}")
    print(f"🌱 Planet seed: {ast.get('planet_seed', 'none')}")
    print(f"📍 HOD alignment: spherical planet worldbuilding")