#!/usr/bin/env python3
"""
Mini-Planet Creation Pipeline
Integrates all components to generate a fully navigatable mini-planet
"""

import sys
import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunkStreamingManager:
    """
    Manages on-demand loading of planet chunks based on camera position and LOD
    """
    
    def __init__(self, chunk_data: Dict[str, Any]):
        self.chunk_metadata = chunk_data['chunks']
        self.chunker = chunk_data['chunker']
        self.loaded_chunks = {}  # chunk_id -> mesh_data
        self.cache_size_limit = 100  # Maximum loaded chunks
        
    def get_visible_chunks(self, camera_pos: List[float], camera_dir: List[float], 
                          max_distance: float = 200.0, max_chunks: int = 50) -> List[Dict[str, Any]]:
        """
        Get chunks that should be visible/loaded based on camera position
        
        Args:
            camera_pos: Camera position [x, y, z]
            camera_dir: Camera direction [x, y, z] 
            max_distance: Maximum distance to load chunks
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of chunk data with meshes loaded on-demand
        """
        camera_pos = np.array(camera_pos)
        visible_chunks = []
        
        # Calculate distance and sort by closest first
        chunk_distances = []
        for i, chunk in enumerate(self.chunk_metadata):
            center = np.array(chunk['center'])
            distance = np.linalg.norm(center - camera_pos)
            radius = chunk['radius']
            
            # Basic distance culling
            if distance - radius > max_distance:
                continue
                
            # LOD selection - use higher detail for closer chunks
            target_lod = min(chunk['lod_level'], max(0, int(distance / 20)))
            
            chunk_distances.append((distance, i, target_lod))
        
        # Sort by distance and take closest chunks
        chunk_distances.sort()
        selected_chunks = chunk_distances[:max_chunks]
        
        # Load meshes for selected chunks
        for distance, chunk_idx, target_lod in selected_chunks:
            chunk = self.chunk_metadata[chunk_idx]
            chunk_id = f"{chunk['face_id']}_{chunk['node'].level}_{chunk['node'].uv_min}_{chunk['node'].uv_max}"
            
            # Load mesh if not cached
            if chunk_id not in self.loaded_chunks:
                mesh_data = self.chunker.generate_chunk_mesh(chunk['node'], target_lod)
                self.loaded_chunks[chunk_id] = mesh_data
                
                # Manage cache size
                if len(self.loaded_chunks) > self.cache_size_limit:
                    self._evict_distant_chunks(camera_pos)
            
            visible_chunks.append({
                'chunk_id': chunk_id,
                'distance': distance,
                'lod_level': target_lod,
                'mesh': self.loaded_chunks[chunk_id],
                'metadata': chunk
            })
        
        logger.info(f"Loaded {len(visible_chunks)} visible chunks (cache: {len(self.loaded_chunks)})")
        return visible_chunks
    
    def _evict_distant_chunks(self, camera_pos: np.ndarray):
        """Remove chunks that are far from camera to manage memory"""
        if len(self.loaded_chunks) <= self.cache_size_limit:
            return
            
        # Calculate distances for loaded chunks
        chunk_distances = []
        for chunk_id in self.loaded_chunks:
            # Find corresponding metadata
            for chunk in self.chunk_metadata:
                test_id = f"{chunk['face_id']}_{chunk['node'].level}_{chunk['node'].uv_min}_{chunk['node'].uv_max}"
                if test_id == chunk_id:
                    center = np.array(chunk['center'])
                    distance = np.linalg.norm(center - camera_pos)
                    chunk_distances.append((distance, chunk_id))
                    break
        
        # Sort by distance (farthest first) and remove excess
        chunk_distances.sort(reverse=True)
        chunks_to_remove = len(self.loaded_chunks) - self.cache_size_limit + 10  # Remove some extra
        
        for i in range(min(chunks_to_remove, len(chunk_distances))):
            _, chunk_id = chunk_distances[i]
            del self.loaded_chunks[chunk_id]

def generate_terrain_from_seed(seed: int) -> Dict[str, Any]:
    """
    Generate all terrain parameters deterministically from a seed
    
    Args:
        seed: Random seed for terrain generation
        
    Returns:
        Dictionary containing all terrain parameters
    """
    # Create deterministic random generator from seed
    rng = random.Random(seed)
    
    # Generate base layer parameters
    base_amplitude = rng.uniform(3.0, 8.0)
    base_frequency = rng.uniform(0.2, 0.5)
    base_octaves = rng.randint(3, 6)
    
    # Generate mountain layer parameters  
    mountain_amplitude = rng.uniform(10.0, 25.0)
    mountain_frequency = rng.uniform(0.05, 0.15)
    mountain_octaves = rng.randint(3, 5)
    mountain_sharpness = rng.uniform(0.7, 1.0)
    
    # Generate detail layer parameters
    detail_amplitude = rng.uniform(0.3, 1.0)
    detail_frequency = rng.uniform(1.5, 3.0)
    detail_octaves = rng.randint(2, 4)
    
    # Generate cave parameters
    enable_caves = rng.random() < 0.3  # 30% chance of caves
    cave_density = rng.uniform(0.2, 0.4) if enable_caves else 0.0
    
    return {
        "base_layer": {
            "type": "fbm",
            "amplitude": base_amplitude,
            "frequency": base_frequency,
            "octaves": base_octaves,
            "lacunarity": 2.0,
            "persistence": 0.5
        },
        "mountain_layer": {
            "type": "ridged",
            "amplitude": mountain_amplitude,
            "frequency": mountain_frequency,
            "octaves": mountain_octaves,
            "sharpness": mountain_sharpness
        },
        "detail_layer": {
            "type": "turbulence",
            "amplitude": detail_amplitude,
            "frequency": detail_frequency,
            "octaves": detail_octaves
        },
        "enable_caves": enable_caves,
        "cave_density": cave_density
    }


class MiniPlanetGenerator:
    """
    Complete pipeline for generating navigatable mini-planets
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration"""
        self.config = self.load_config(config_path) if config_path else self.get_default_config()
        self.setup_paths()
        
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default mini-planet configuration - terrain will be seed-based"""
        # Generate a random seed for terrain
        default_seed = random.randint(1, 999999999)
        
        return {
            "planet": {
                "name": "MiniWorld",
                "radius": 50.0,
                "seed": default_seed,
                "resolution": 32
            },
            "terrain": generate_terrain_from_seed(default_seed),
            "mesh": {
                "cubesphere_divisions": 4,
                "lod_levels": 4,
                "chunk_size": 32,
                "enable_crack_prevention": True
            },
            "rendering": {
                "enable_shadows": True,
                "enable_ssao": True,
                "enable_normal_mapping": True,
                "shadow_resolution": 2048,
                "ambient_color": [0.2, 0.25, 0.3],
                "sun_direction": [0.5, -0.7, 0.3],
                "sun_color": [1.0, 0.95, 0.8],
                "fog_enabled": True,
                "fog_density": 0.001
            },
            "optimization": {
                "enable_baking": True,
                "multi_threaded": True,
                "thread_count": 8,
                "cache_chunks": True,
                "stream_distance": 200.0
            },
            "output": {
                "format": "binary",
                "compress": True,
                "generate_preview": True,
                "output_dir": "generated_planets"
            }
        }
    
    def setup_paths(self):
        """Setup directory structure"""
        self.base_dir = Path.cwd()
        self.output_dir = self.base_dir / self.config['output']['output_dir']
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'meshes').mkdir(exist_ok=True)
        (self.output_dir / 'textures').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'previews').mkdir(exist_ok=True)
    
    def generate_pcc_definition(self) -> str:
        """Generate PCC language definition for the planet"""
        pcc_template = f"""
# PCC Planet Definition
# Generated: {datetime.now().isoformat()}

planet "{self.config['planet']['name']}" {{
    radius: {self.config['planet']['radius']}
    seed: {self.config['planet']['seed']}
    
    terrain {{
        # Base terrain layer - rolling hills
        layer "base" {{
            type: "fbm"
            amplitude: {self.config['terrain']['base_layer']['amplitude']}
            frequency: {self.config['terrain']['base_layer']['frequency']}
            octaves: {self.config['terrain']['base_layer']['octaves']}
            lacunarity: {self.config['terrain']['base_layer']['lacunarity']}
            persistence: {self.config['terrain']['base_layer']['persistence']}
        }}
        
        # Mountain layer - sharp peaks
        layer "mountains" {{
            type: "ridged"
            amplitude: {self.config['terrain']['mountain_layer']['amplitude']}
            frequency: {self.config['terrain']['mountain_layer']['frequency']}
            octaves: {self.config['terrain']['mountain_layer']['octaves']}
            mask: "height_gradient"
            blend_mode: "multiply"
        }}
        
        # Detail layer - surface roughness
        layer "detail" {{
            type: "turbulence"
            amplitude: {self.config['terrain']['detail_layer']['amplitude']}
            frequency: {self.config['terrain']['detail_layer']['frequency']}
            octaves: {self.config['terrain']['detail_layer']['octaves']}
            blend_mode: "add"
        }}
        
        # Optional cave system
        caves {{
            enabled: {str(self.config['terrain']['enable_caves']).lower()}
            density: {self.config['terrain']['cave_density']}
            min_depth: 5.0
            max_depth: 20.0
        }}
    }}
    
    materials {{
        # Define material zones based on height and slope
        zone "grass" {{
            height_range: [0.0, 0.3]
            slope_range: [0.0, 30.0]
            color: [0.3, 0.6, 0.2]
            roughness: 0.8
        }}
        
        zone "rock" {{
            height_range: [0.3, 0.7]
            slope_range: [30.0, 90.0]
            color: [0.5, 0.45, 0.4]
            roughness: 0.9
        }}
        
        zone "snow" {{
            height_range: [0.7, 1.0]
            slope_range: [0.0, 90.0]
            color: [0.95, 0.95, 1.0]
            roughness: 0.3
        }}
    }}
    
    lod {{
        levels: {self.config['mesh']['lod_levels']}
        base_chunk_size: {self.config['mesh']['chunk_size']}
        distance_multiplier: 2.0
        transition_range: 10.0
    }}
}}
"""
        return pcc_template
    
    def generate_cubesphere(self):
        """Generate the base cubesphere mesh"""
        logger.info("Generating cubesphere base mesh...")
        
        # Import required modules
        try:
            from agents.agent_d.mesh.cubesphere import CubeSphereGenerator
            
            sphere = CubeSphereGenerator(
                base_radius=self.config['planet']['radius'],
                face_res=self.config['mesh']['cubesphere_divisions']
            )
            
            # Generate mesh data
            mesh_data = sphere.generate()
            vertices = mesh_data['positions']
            faces = mesh_data['indices'] 
            uvs = mesh_data['uvs']
            
            logger.info(f"Generated {len(vertices)} vertices, {len(faces)} faces")
            return sphere
            
        except ImportError as e:
            logger.warning(f"CubeSphere module not available: {e}")
            # Fallback to basic sphere generation
            return self.generate_basic_sphere()
    
    def generate_basic_sphere(self):
        """Fallback basic sphere generation"""
        radius = self.config['planet']['radius']
        resolution = self.config['planet']['resolution']
        
        # Generate UV sphere
        theta = np.linspace(0, 2*np.pi, resolution)
        phi = np.linspace(0, np.pi, resolution//2)
        
        vertices = []
        for p in phi:
            for t in theta:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = radius * np.cos(p)
                vertices.append([x, y, z])
        
        return {"vertices": vertices, "type": "basic_sphere"}
    
    def apply_terrain(self, mesh_data):
        """Apply terrain displacement to mesh"""
        logger.info("Applying terrain displacement...")
        
        try:
            from agents.agent_d.terrain.heightfield import Heightfield
            from agents.agent_d.terrain.noise_nodes import NoiseNode
            
            # Create heightfield from config
            heightfield = Heightfield(
                seed=self.config['planet']['seed'],
                config=self.config['terrain']
            )
            
            # Apply to mesh vertices
            displaced_vertices = heightfield.displace_vertices(
                mesh_data.get('vertices', []),
                self.config['planet']['radius']
            )
            
            mesh_data['vertices'] = displaced_vertices
            logger.info("Terrain displacement applied successfully")
            
        except ImportError:
            logger.warning("Terrain modules not available, using procedural noise")
            # Fallback noise implementation
            self.apply_simple_noise(mesh_data)
        
        return mesh_data
    
    def apply_simple_noise(self, mesh_data):
        """Simple noise displacement fallback"""
        import random
        random.seed(self.config['planet']['seed'])
        
        vertices = mesh_data.get('vertices', [])
        for i, vertex in enumerate(vertices):
            # Simple random displacement
            displacement = random.uniform(-2.0, 5.0)
            # Normalize vertex and apply displacement
            length = np.linalg.norm(vertex)
            if length > 0:
                normalized = np.array(vertex) / length
                vertices[i] = normalized * (length + displacement)
        
        mesh_data['vertices'] = vertices
    
    def generate_chunks(self, mesh_data):
        """Generate LOD chunks for streaming"""
        logger.info("Generating LOD chunks...")
        
        try:
            from agents.agent_d.mesh.quadtree_chunking import QuadtreeChunker
            
            chunker = QuadtreeChunker(
                max_depth=self.config['mesh']['lod_levels'],
                chunk_res=self.config['mesh']['chunk_size']
            )
            
            # Generate quadtree structure (metadata only - no mesh generation yet)
            nodes = chunker.generate_quadtree()
            
            # Create chunk metadata for streaming system
            chunk_metadata = []
            for node in nodes:
                # Calculate bounding box for frustum culling
                bbox = chunker.compute_chunk_aabb_from_node(node)
                chunk_metadata.append({
                    'node': node,
                    'bbox': bbox,
                    'lod_level': node.level,
                    'face_id': node.face_id,
                    'center': bbox.get('center', [0, 0, 0]),
                    'radius': bbox.get('radius', 1.0),
                    'mesh': None  # Will be loaded on-demand
                })
            
            logger.info(f"Created metadata for {len(chunk_metadata)} chunks (streaming mode)")
            streaming_data = {
                'type': 'streaming_chunks',
                'chunks': chunk_metadata,
                'chunker': chunker  # Keep reference for on-demand mesh generation
            }
            
            # Create streaming manager for easy access
            streaming_data['manager'] = ChunkStreamingManager(streaming_data)
            return streaming_data
            
        except ImportError:
            logger.warning("Chunking module not available")
            return [mesh_data]  # Return single chunk
    
    def optimize_and_bake(self, chunks):
        """Optimize and bake mesh data"""
        
        # Handle streaming chunks differently - no baking needed for metadata
        if isinstance(chunks, dict) and chunks.get('type') == 'streaming_chunks':
            logger.info("Streaming chunks detected - skipping baking (on-demand loading)")
            return chunks
        
        logger.info("Optimizing and baking mesh data...")
        
        try:
            from agents.agent_d.baking.deterministic_baking import DeterministicBaker
            
            baker = DeterministicBaker(
                seed=self.config['planet']['seed'],
                multi_threaded=self.config['optimization']['multi_threaded']
            )
            
            baked_data = baker.bake_chunks(chunks)
            logger.info("Baking complete")
            return baked_data
            
        except ImportError:
            logger.warning("Baking module not available")
            return chunks
    
    def save_planet_data(self, planet_data):
        """Save generated planet data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        planet_name = self.config['planet']['name']
        
        # Handle streaming chunks
        if isinstance(planet_data, dict) and planet_data.get('type') == 'streaming_chunks':
            # Save in chunked planet format for the viewer
            manifest_path = self.output_dir / f"{planet_name}_{timestamp}_streaming.json"
            
            # Create viewer-compatible format
            viewer_data = {
                "planet": {
                    "type": "chunked_quadtree",
                    "name": planet_name,
                    "radius": self.config['planet']['radius'],
                    "seed": self.config['planet']['seed'],
                    "streaming": True,
                    "chunk_count": len(planet_data['chunks']),
                    "max_depth": self.config['mesh']['lod_levels'],
                    "chunk_resolution": self.config['mesh']['chunk_size']
                },
                "chunks": [],  # Chunks loaded on-demand
                "config": self.config,
                "timestamp": timestamp,
                "streaming_data": {
                    "chunk_metadata": [
                        {
                            "face_id": chunk['face_id'],
                            "lod_level": chunk['lod_level'],
                            "center": chunk['center'],
                            "radius": chunk['radius'],
                            "bbox": chunk['bbox']
                        } for chunk in planet_data['chunks']
                    ]
                }
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(viewer_data, f, indent=2)
            
            logger.info(f"Streaming planet data saved to {manifest_path}")
            return manifest_path
        
        else:
            # Original format for non-streaming data
            metadata_path = self.output_dir / 'metadata' / f"{planet_name}_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "config": self.config,
                    "timestamp": timestamp,
                    "stats": {
                        "vertex_count": len(planet_data.get('vertices', [])),
                        "chunk_count": len(planet_data.get('chunks', [])),
                        "radius": self.config['planet']['radius']
                    }
                }, f, indent=2)
            
            logger.info(f"Planet data saved to {metadata_path}")
            
            # Save PCC definition
            pcc_path = self.output_dir / f"{planet_name}_{timestamp}.pcc"
            with open(pcc_path, 'w') as f:
                f.write(self.generate_pcc_definition())
            
            return metadata_path
    
    def launch_viewer(self, planet_data, planet_file_path):
        """Launch the interactive viewer"""
        logger.info("Launching planet viewer...")
        
        try:
            from renderer.pcc_game_viewer import render_pcc_game_interactive
            
            # Launch the viewer with the saved planet file
            render_pcc_game_interactive(str(planet_file_path))
            
        except ImportError:
            logger.warning("Viewer not available, trying alternative...")
            self.launch_simple_viewer(planet_data)
    
    def launch_simple_viewer(self, planet_data):
        """Launch simple viewer as fallback"""
        try:
            from renderer.pcc_simple_viewer import SimpleViewer
            
            viewer = SimpleViewer(planet_data)
            viewer.run()
            
        except ImportError:
            logger.error("No viewer available. Planet data saved to disk.")
    
    def generate(self):
        """Main generation pipeline"""
        logger.info(f"Starting mini-planet generation: {self.config['planet']['name']}")
        
        # Step 1: Generate base mesh
        mesh_data = self.generate_cubesphere()
        
        # Step 2: Apply terrain
        mesh_data = self.apply_terrain(mesh_data)
        
        # Step 3: Generate chunks
        chunks = self.generate_chunks(mesh_data)
        
        # Step 4: Optimize and bake
        if self.config['optimization']['enable_baking']:
            planet_data = self.optimize_and_bake(chunks)
        else:
            # For streaming chunks, pass through the streaming data directly
            if isinstance(chunks, dict) and chunks.get('type') == 'streaming_chunks':
                planet_data = chunks
            else:
                planet_data = {"chunks": chunks, "mesh": mesh_data}
        
        # Step 5: Save data
        saved_path = self.save_planet_data(planet_data)
        
        logger.info(f"Planet generation complete! Data saved to: {saved_path}")
        
        return planet_data, saved_path
    
    def generate_and_view(self):
        """Generate planet and launch viewer"""
        planet_data, planet_file_path = self.generate()
        self.launch_viewer(planet_data, planet_file_path)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a navigatable mini-planet')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--name', type=str, default='MiniWorld', help='Planet name')
    parser.add_argument('--radius', type=float, default=50.0, help='Planet radius')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (auto-generated if not specified)')
    parser.add_argument('--view', action='store_true', help='Launch viewer after generation')
    parser.add_argument('--quick', action='store_true', help='Quick generation (no baking)')
    
    args = parser.parse_args()
    
    # Create generator
    if args.config:
        generator = MiniPlanetGenerator(args.config)
    else:
        generator = MiniPlanetGenerator()
        
        # Override with command line arguments
        if args.name:
            generator.config['planet']['name'] = args.name
        if args.radius:
            generator.config['planet']['radius'] = args.radius
        # Generate or use provided seed
        if args.seed:
            generator.config['planet']['seed'] = args.seed
            # Regenerate terrain based on new seed
            generator.config['terrain'] = generate_terrain_from_seed(args.seed)
            logger.info(f"Using seed: {args.seed}")
        else:
            # Generate random seed
            new_seed = random.randint(1, 999999999)
            generator.config['planet']['seed'] = new_seed
            generator.config['terrain'] = generate_terrain_from_seed(new_seed)
            logger.info(f"Generated random seed: {new_seed}")
        if args.quick:
            generator.config['optimization']['enable_baking'] = False
            # Keep normal chunk counts but disable expensive operations
    
    # Generate planet
    if args.view:
        generator.generate_and_view()
    else:
        planet_data, saved_path = generator.generate()
        print(f"Planet '{generator.config['planet']['name']}' generated successfully!")
        print(f"Output directory: {generator.output_dir}")


if __name__ == "__main__":
    main()
