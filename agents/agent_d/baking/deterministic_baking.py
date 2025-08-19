#!/usr/bin/env python3
"""
Deterministic Baking System - T13
=================================

Provides bake command that writes buffers and manifest.sha256.json for
deterministic terrain generation. Ensures same PCC seed produces identical
buffers and hashes with complete provenance tracking.

Features:
- Deterministic buffer generation and serialization
- SHA256 hash verification for all generated data
- Provenance logging (PCC graph hash, params, code version)
- Manifest generation with complete metadata
- Buffer integrity validation and verification

Usage:
    from deterministic_baking import DeterministicBaker
    
    baker = DeterministicBaker()
    manifest = baker.bake_planet("sample_planet", seed=12345)
"""

import numpy as np
import json
import hashlib
import time
import struct
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import pickle
import gzip

# Import T13 seed system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'determinism'))
from seed_threading import DeterministicSeedManager, SeedDomain, get_global_seed_manager

# Import generation systems
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'noise'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fusion'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marching_cubes'))

try:
    from noise_field_3d import NoiseField3D
    from sdf_evaluator import SDFEvaluator, ChunkBounds
    from surface_sdf_fusion import TerrainSdfFusion, FusedMeshData
    from marching_cubes import MarchingCubesVertex
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False


class BufferType(Enum):
    """Types of buffers that can be baked"""
    HEIGHTFIELD = "heightfield"
    NOISE_FIELD = "noise_field"
    SDF_FIELD = "sdf_field"
    VERTEX_BUFFER = "vertex_buffer"
    INDEX_BUFFER = "index_buffer"
    NORMAL_BUFFER = "normal_buffer"
    TANGENT_BUFFER = "tangent_buffer"
    UV_BUFFER = "uv_buffer"
    MATERIAL_BUFFER = "material_buffer"
    FUSED_MESH = "fused_mesh"


@dataclass
class BufferMetadata:
    """Metadata for a baked buffer"""
    buffer_type: BufferType
    data_type: str  # numpy dtype string
    shape: Tuple[int, ...]
    size_bytes: int
    sha256_hash: str
    compression: Optional[str] = None
    chunk_id: Optional[str] = None
    lod_level: Optional[int] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    seed_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceRecord:
    """Complete provenance record for baked data"""
    pcc_graph_hash: str
    master_seed: int
    generation_params: Dict[str, Any]
    code_version: str
    system_info: Dict[str, Any]
    timestamp: float
    git_commit: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)


@dataclass
class BakeManifest:
    """Complete manifest for baked planet data"""
    planet_id: str
    provenance: ProvenanceRecord
    buffers: Dict[str, BufferMetadata]
    chunks: Dict[str, Dict[str, Any]]
    overall_hash: str
    bake_duration: float
    total_size_bytes: int


class DeterministicHasher:
    """Deterministic hash computation for various data types"""
    
    @staticmethod
    def hash_numpy_array(array: np.ndarray) -> str:
        """Compute deterministic SHA256 hash of numpy array"""
        # Ensure consistent byte order and format
        array_bytes = array.tobytes(order='C')  # C-contiguous order
        return hashlib.sha256(array_bytes).hexdigest()
    
    @staticmethod
    def hash_vertices(vertices: List[Any]) -> str:
        """Hash list of vertices deterministically"""
        # Convert vertices to consistent binary format
        vertex_data = []
        for vertex in vertices:
            if hasattr(vertex, 'position'):
                vertex_data.extend(vertex.position.astype(np.float32).tobytes())
            if hasattr(vertex, 'normal'):
                vertex_data.extend(vertex.normal.astype(np.float32).tobytes())
            if hasattr(vertex, 'tangent'):
                vertex_data.extend(vertex.tangent.astype(np.float32).tobytes())
        
        combined_bytes = b''.join(vertex_data)
        return hashlib.sha256(combined_bytes).hexdigest()
    
    @staticmethod
    def hash_dict(data: Dict[str, Any]) -> str:
        """Hash dictionary deterministically"""
        # Sort keys for consistent ordering
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_pcc_graph(pcc_data: Dict[str, Any]) -> str:
        """Compute hash of PCC graph structure"""
        # Extract relevant parts for hashing
        graph_structure = {
            'nodes': pcc_data.get('nodes', {}),
            'connections': pcc_data.get('connections', {}),
            'parameters': pcc_data.get('parameters', {})
        }
        return DeterministicHasher.hash_dict(graph_structure)


class BufferSerializer:
    """Handles deterministic buffer serialization"""
    
    @staticmethod
    def serialize_numpy_array(array: np.ndarray, compress: bool = True) -> bytes:
        """Serialize numpy array deterministically"""
        if compress:
            # Use gzip compression
            array_bytes = array.tobytes(order='C')
            return gzip.compress(array_bytes)
        else:
            return array.tobytes(order='C')
    
    @staticmethod
    def deserialize_numpy_array(data: bytes, dtype: np.dtype, shape: Tuple[int, ...], 
                               compressed: bool = True) -> np.ndarray:
        """Deserialize numpy array"""
        if compressed:
            decompressed = gzip.decompress(data)
        else:
            decompressed = data
        
        array = np.frombuffer(decompressed, dtype=dtype)
        return array.reshape(shape)
    
    @staticmethod
    def serialize_vertex_list(vertices: List[Any]) -> bytes:
        """Serialize vertex list deterministically"""
        # Convert to structured array for consistent serialization
        if not vertices:
            return b''
        
        # Determine vertex structure from first vertex
        first_vertex = vertices[0]
        vertex_dtype = []
        
        if hasattr(first_vertex, 'position'):
            vertex_dtype.append(('position', 'f4', (3,)))
        if hasattr(first_vertex, 'normal'):
            vertex_dtype.append(('normal', 'f4', (3,)))
        if hasattr(first_vertex, 'tangent'):
            vertex_dtype.append(('tangent', 'f4', (4,)))  # Include handedness
        if hasattr(first_vertex, 'uv'):
            vertex_dtype.append(('uv', 'f4', (2,)))
        
        # Create structured array
        structured_array = np.zeros(len(vertices), dtype=vertex_dtype)
        
        for i, vertex in enumerate(vertices):
            if hasattr(vertex, 'position'):
                structured_array[i]['position'] = vertex.position
            if hasattr(vertex, 'normal'):
                structured_array[i]['normal'] = vertex.normal
            if hasattr(vertex, 'tangent'):
                structured_array[i]['tangent'] = vertex.tangent
            if hasattr(vertex, 'uv'):
                structured_array[i]['uv'] = vertex.uv
        
        return gzip.compress(structured_array.tobytes())


class DeterministicBaker:
    """Handles deterministic baking of terrain data"""
    
    def __init__(self, output_dir: Union[str, Path] = None):
        """Initialize deterministic baker"""
        self.output_dir = Path(output_dir) if output_dir else Path("baked_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize seed manager
        self.seed_manager = get_global_seed_manager()
        
        # Generation systems
        if GENERATION_AVAILABLE:
            self.noise_generator = NoiseField3D()
            self.sdf_evaluator = SDFEvaluator() 
            self.terrain_fusion = TerrainSdfFusion()
        
        # Baking statistics
        self.bake_stats = {
            'total_bakes': 0,
            'total_buffers': 0,
            'total_bytes': 0,
            'total_time': 0.0
        }
    
    def bake_planet(self, planet_id: str, seed: int, 
                   generation_params: Optional[Dict[str, Any]] = None) -> BakeManifest:
        """Bake complete planet with deterministic generation"""
        print(f"🔥 Baking planet '{planet_id}' with seed {seed}")
        start_time = time.time()
        
        # Set deterministic seed
        self.seed_manager = DeterministicSeedManager(seed)
        self.seed_manager.enable_global_override()
        
        if generation_params is None:
            generation_params = self._get_default_generation_params()
        
        try:
            # Create provenance record
            provenance = self._create_provenance_record(seed, generation_params)
            
            # Generate and bake buffers
            buffers = {}
            chunks = {}
            total_size = 0
            
            # Bake terrain chunks
            chunk_results = self._bake_terrain_chunks(planet_id, generation_params)
            buffers.update(chunk_results['buffers'])
            chunks.update(chunk_results['chunks'])
            total_size += chunk_results['total_size']
            
            # Bake noise fields
            noise_results = self._bake_noise_fields(planet_id, generation_params)
            buffers.update(noise_results['buffers'])
            total_size += noise_results['total_size']
            
            # Bake SDF fields
            sdf_results = self._bake_sdf_fields(planet_id, generation_params)
            buffers.update(sdf_results['buffers'])
            total_size += sdf_results['total_size']
            
            # Compute overall hash
            all_hashes = [buffer_meta.sha256_hash for buffer_meta in buffers.values()]
            overall_hash_input = ''.join(sorted(all_hashes))
            overall_hash = hashlib.sha256(overall_hash_input.encode('utf-8')).hexdigest()
            
            bake_duration = time.time() - start_time
            
            # Create manifest
            manifest = BakeManifest(
                planet_id=planet_id,
                provenance=provenance,
                buffers=buffers,
                chunks=chunks,
                overall_hash=overall_hash,
                bake_duration=bake_duration,
                total_size_bytes=total_size
            )
            
            # Write manifest
            manifest_path = self.output_dir / f"{planet_id}_manifest.sha256.json"
            self._write_manifest(manifest, manifest_path)
            
            # Update statistics
            self.bake_stats['total_bakes'] += 1
            self.bake_stats['total_buffers'] += len(buffers)
            self.bake_stats['total_bytes'] += total_size
            self.bake_stats['total_time'] += bake_duration
            
            print(f"   ✅ Baked {len(buffers)} buffers ({total_size / 1024 / 1024:.2f} MB) in {bake_duration:.2f}s")
            print(f"   📋 Manifest: {manifest_path}")
            print(f"   🔐 Overall hash: {overall_hash[:16]}...")
            
            return manifest
            
        finally:
            self.seed_manager.disable_global_override()
    
    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters for testing"""
        return {
            'terrain': {
                'resolution': 32,
                'size': 8.0,
                'height_scale': 2.0,
                'noise_octaves': 4,
                'noise_frequency': 0.1
            },
            'caves': {
                'enabled': True,
                'density': 0.3,
                'size_scale': 1.5
            },
            'chunks': {
                'count_x': 2,
                'count_z': 2,
                'chunk_size': 4.0
            },
            'lod': {
                'levels': [0, 1, 2],
                'distance_factors': [1.0, 2.0, 4.0]
            }
        }
    
    def _create_provenance_record(self, seed: int, params: Dict[str, Any]) -> ProvenanceRecord:
        """Create complete provenance record"""
        import platform
        import subprocess
        
        # Get git commit if available
        git_commit = None
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except:
            pass
        
        # System information
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'architecture': platform.architecture()[0]
        }
        
        # Code version (simplified)
        code_version = f"T13_v1.0_{int(time.time())}"
        
        # Create fake PCC graph hash for testing
        pcc_graph = {
            'terrain_node': params.get('terrain', {}),
            'cave_node': params.get('caves', {}),
            'fusion_node': {'enabled': True}
        }
        
        return ProvenanceRecord(
            pcc_graph_hash=DeterministicHasher.hash_pcc_graph(pcc_graph),
            master_seed=seed,
            generation_params=params,
            code_version=code_version,
            system_info=system_info,
            timestamp=time.time(),
            git_commit=git_commit
        )
    
    def _bake_terrain_chunks(self, planet_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Bake terrain chunk buffers"""
        print("   🗻 Baking terrain chunks...")
        
        buffers = {}
        chunks = {}
        total_size = 0
        
        terrain_params = params.get('terrain', {})
        chunk_params = params.get('chunks', {})
        
        count_x = chunk_params.get('count_x', 2)
        count_z = chunk_params.get('count_z', 2)
        chunk_size = chunk_params.get('chunk_size', 4.0)
        resolution = terrain_params.get('resolution', 32)
        
        for x in range(count_x):
            for z in range(count_z):
                chunk_id = f"{x}_{z}"
                
                # Generate deterministic heightfield
                chunk_rng = self.seed_manager.get_rng(
                    SeedDomain.TERRAIN_HEIGHTFIELD, 
                    "heightfield_generation",
                    chunk_id=chunk_id
                )
                
                # Create heightfield with seeded noise
                heightfield = self._generate_deterministic_heightfield(
                    chunk_rng, resolution, terrain_params
                )
                
                # Serialize heightfield
                heightfield_data = BufferSerializer.serialize_numpy_array(heightfield)
                heightfield_hash = DeterministicHasher.hash_numpy_array(heightfield)
                
                buffer_key = f"{chunk_id}_heightfield"
                buffers[buffer_key] = BufferMetadata(
                    buffer_type=BufferType.HEIGHTFIELD,
                    data_type=str(heightfield.dtype),
                    shape=heightfield.shape,
                    size_bytes=len(heightfield_data),
                    sha256_hash=heightfield_hash,
                    compression="gzip",
                    chunk_id=chunk_id,
                    generation_params=terrain_params,
                    seed_info={
                        'domain': SeedDomain.TERRAIN_HEIGHTFIELD.value,
                        'context': 'heightfield_generation',
                        'chunk_id': chunk_id
                    }
                )
                
                # Write buffer to disk
                buffer_path = self.output_dir / f"{planet_id}_{buffer_key}.bin"
                with open(buffer_path, 'wb') as f:
                    f.write(heightfield_data)
                
                # Track chunk metadata
                chunks[chunk_id] = {
                    'position_x': x * chunk_size - (count_x * chunk_size) / 2,
                    'position_z': z * chunk_size - (count_z * chunk_size) / 2,
                    'size': chunk_size,
                    'resolution': resolution,
                    'heightfield_buffer': buffer_key
                }
                
                total_size += len(heightfield_data)
        
        print(f"      ✅ Generated {len(buffers)} heightfield buffers")
        return {
            'buffers': buffers,
            'chunks': chunks,
            'total_size': total_size
        }
    
    def _bake_noise_fields(self, planet_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Bake noise field buffers"""
        print("   🌊 Baking noise fields...")
        
        buffers = {}
        total_size = 0
        
        terrain_params = params.get('terrain', {})
        
        # Generate deterministic 3D noise field
        noise_rng = self.seed_manager.get_rng(
            SeedDomain.NOISE_FIELD,
            "perlin_3d"
        )
        
        # Create test noise field
        resolution = terrain_params.get('resolution', 32)
        noise_field = self._generate_deterministic_noise_field(noise_rng, resolution)
        
        # Serialize noise field
        noise_data = BufferSerializer.serialize_numpy_array(noise_field)
        noise_hash = DeterministicHasher.hash_numpy_array(noise_field)
        
        buffer_key = "noise_field_3d"
        buffers[buffer_key] = BufferMetadata(
            buffer_type=BufferType.NOISE_FIELD,
            data_type=str(noise_field.dtype),
            shape=noise_field.shape,
            size_bytes=len(noise_data),
            sha256_hash=noise_hash,
            compression="gzip",
            generation_params=terrain_params,
            seed_info={
                'domain': SeedDomain.NOISE_FIELD.value,
                'context': 'perlin_3d'
            }
        )
        
        # Write buffer to disk
        buffer_path = self.output_dir / f"{planet_id}_{buffer_key}.bin"
        with open(buffer_path, 'wb') as f:
            f.write(noise_data)
        
        total_size += len(noise_data)
        
        print(f"      ✅ Generated {len(buffers)} noise field buffers")
        return {
            'buffers': buffers,
            'total_size': total_size
        }
    
    def _bake_sdf_fields(self, planet_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Bake SDF field buffers"""
        print("   🕳️  Baking SDF fields...")
        
        buffers = {}
        total_size = 0
        
        cave_params = params.get('caves', {})
        
        if not cave_params.get('enabled', True):
            return {'buffers': buffers, 'total_size': total_size}
        
        # Generate deterministic SDF field
        sdf_rng = self.seed_manager.get_rng(
            SeedDomain.CAVE_SDF,
            "sphere_caves"
        )
        
        # Create test SDF field
        resolution = 32
        sdf_field = self._generate_deterministic_sdf_field(sdf_rng, resolution, cave_params)
        
        # Serialize SDF field
        sdf_data = BufferSerializer.serialize_numpy_array(sdf_field)
        sdf_hash = DeterministicHasher.hash_numpy_array(sdf_field)
        
        buffer_key = "sdf_field_caves"
        buffers[buffer_key] = BufferMetadata(
            buffer_type=BufferType.SDF_FIELD,
            data_type=str(sdf_field.dtype),
            shape=sdf_field.shape,
            size_bytes=len(sdf_data),
            sha256_hash=sdf_hash,
            compression="gzip",
            generation_params=cave_params,
            seed_info={
                'domain': SeedDomain.CAVE_SDF.value,
                'context': 'sphere_caves'
            }
        )
        
        # Write buffer to disk
        buffer_path = self.output_dir / f"{planet_id}_{buffer_key}.bin"
        with open(buffer_path, 'wb') as f:
            f.write(sdf_data)
        
        total_size += len(sdf_data)
        
        print(f"      ✅ Generated {len(buffers)} SDF field buffers")
        return {
            'buffers': buffers,
            'total_size': total_size
        }
    
    def _generate_deterministic_heightfield(self, rng, resolution: int, 
                                          params: Dict[str, Any]) -> np.ndarray:
        """Generate deterministic heightfield using seeded RNG"""
        heightfield = np.zeros((resolution, resolution), dtype=np.float32)
        
        height_scale = params.get('height_scale', 2.0)
        noise_frequency = params.get('noise_frequency', 0.1)
        
        # Generate with seeded noise
        for y in range(resolution):
            for x in range(resolution):
                # Use seeded RNG for deterministic noise
                noise_x = x * noise_frequency
                noise_y = y * noise_frequency
                
                # Simple deterministic noise approximation
                noise_val = (rng.normal(0, 1) + 
                           rng.normal(noise_x * 0.1, 0.5) + 
                           rng.normal(noise_y * 0.1, 0.3)) / 3.0
                
                heightfield[y, x] = noise_val * height_scale
        
        return heightfield
    
    def _generate_deterministic_noise_field(self, rng, resolution: int) -> np.ndarray:
        """Generate deterministic 3D noise field"""
        noise_field = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Generate with seeded RNG
        for z in range(resolution):
            for y in range(resolution):
                for x in range(resolution):
                    # Deterministic 3D noise
                    noise_val = (rng.normal(0, 1) + 
                               rng.normal(x * 0.01, 0.3) + 
                               rng.normal(y * 0.01, 0.3) +
                               rng.normal(z * 0.01, 0.3)) / 4.0
                    
                    noise_field[z, y, x] = noise_val
        
        return noise_field
    
    def _generate_deterministic_sdf_field(self, rng, resolution: int, 
                                        params: Dict[str, Any]) -> np.ndarray:
        """Generate deterministic SDF field"""
        sdf_field = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        density = params.get('density', 0.3)
        size_scale = params.get('size_scale', 1.5)
        
        center = resolution // 2
        
        # Generate sphere-like SDF with seeded variations
        for z in range(resolution):
            for y in range(resolution):
                for x in range(resolution):
                    # Distance from center
                    dx = x - center
                    dy = y - center  
                    dz = z - center
                    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Base sphere radius with seeded variation
                    base_radius = resolution * 0.3 * size_scale
                    radius_variation = rng.normal(0, base_radius * 0.1)
                    radius = base_radius + radius_variation
                    
                    # SDF value (negative inside, positive outside)
                    sdf_val = distance - radius
                    
                    # Add density-based variation
                    if rng.random() > density:
                        sdf_val = abs(sdf_val)  # Make it "solid" in low-density areas
                    
                    sdf_field[z, y, x] = sdf_val
        
        return sdf_field
    
    def _write_manifest(self, manifest: BakeManifest, path: Path):
        """Write manifest to JSON file"""
        manifest_dict = asdict(manifest)
        
        # Convert dataclass fields to serializable format
        manifest_dict['buffers'] = {k: asdict(v) for k, v in manifest.buffers.items()}
        manifest_dict['provenance'] = asdict(manifest.provenance)
        
        with open(path, 'w') as f:
            json.dump(manifest_dict, f, indent=2, separators=(',', ': '), sort_keys=True)
    
    def verify_manifest(self, manifest_path: Union[str, Path]) -> Dict[str, Any]:
        """Verify integrity of baked data using manifest"""
        print(f"🔍 Verifying manifest: {manifest_path}")
        
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return {'valid': False, 'error': 'Manifest file not found'}
        
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            planet_id = manifest_data['planet_id']
            buffers = manifest_data['buffers']
            
            verification_results = {
                'valid': True,
                'verified_buffers': 0,
                'total_buffers': len(buffers),
                'mismatched_hashes': [],
                'missing_files': [],
                'size_mismatches': []
            }
            
            for buffer_key, buffer_meta in buffers.items():
                buffer_path = manifest_path.parent / f"{planet_id}_{buffer_key}.bin"
                
                if not buffer_path.exists():
                    verification_results['missing_files'].append(buffer_key)
                    verification_results['valid'] = False
                    continue
                
                # Check file size
                actual_size = buffer_path.stat().st_size
                expected_size = buffer_meta['size_bytes']
                
                if actual_size != expected_size:
                    verification_results['size_mismatches'].append({
                        'buffer': buffer_key,
                        'expected': expected_size,
                        'actual': actual_size
                    })
                    verification_results['valid'] = False
                    continue
                
                # Verify hash
                with open(buffer_path, 'rb') as f:
                    buffer_data = f.read()
                
                actual_hash = hashlib.sha256(buffer_data).hexdigest()
                
                # For compressed data, we need to hash the raw data
                if buffer_meta.get('compression') == 'gzip':
                    try:
                        decompressed = gzip.decompress(buffer_data)
                        # Create array to verify hash matches original data
                        dtype = np.dtype(buffer_meta['data_type'])
                        shape = tuple(buffer_meta['shape'])
                        array = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
                        actual_hash = DeterministicHasher.hash_numpy_array(array)
                    except Exception as e:
                        verification_results['mismatched_hashes'].append({
                            'buffer': buffer_key,
                            'error': f'Decompression failed: {e}'
                        })
                        verification_results['valid'] = False
                        continue
                
                expected_hash = buffer_meta['sha256_hash']
                
                if actual_hash != expected_hash:
                    verification_results['mismatched_hashes'].append({
                        'buffer': buffer_key,
                        'expected': expected_hash[:16] + '...',
                        'actual': actual_hash[:16] + '...'
                    })
                    verification_results['valid'] = False
                else:
                    verification_results['verified_buffers'] += 1
            
            if verification_results['valid']:
                print(f"   ✅ All {verification_results['verified_buffers']} buffers verified")
            else:
                print(f"   ❌ Verification failed:")
                if verification_results['missing_files']:
                    print(f"      Missing files: {verification_results['missing_files']}")
                if verification_results['size_mismatches']:
                    print(f"      Size mismatches: {len(verification_results['size_mismatches'])}")
                if verification_results['mismatched_hashes']:
                    print(f"      Hash mismatches: {len(verification_results['mismatched_hashes'])}")
            
            return verification_results
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def get_bake_statistics(self) -> Dict[str, Any]:
        """Get baking statistics"""
        return self.bake_stats.copy()


if __name__ == "__main__":
    # Test deterministic baking system
    print("🚀 T13 Deterministic Baking System")
    print("=" * 60)
    
    # Create baker
    baker = DeterministicBaker()
    
    # Test baking small sample planet
    manifest = baker.bake_planet("test_planet", seed=12345)
    
    print(f"\n📊 Bake Results:")
    print(f"   Planet ID: {manifest.planet_id}")
    print(f"   Master seed: {manifest.provenance.master_seed}")
    print(f"   Buffers: {len(manifest.buffers)}")
    print(f"   Chunks: {len(manifest.chunks)}")
    print(f"   Total size: {manifest.total_size_bytes / 1024 / 1024:.2f} MB")
    print(f"   Duration: {manifest.bake_duration:.2f}s")
    print(f"   Overall hash: {manifest.overall_hash[:16]}...")
    
    # Test verification
    manifest_path = baker.output_dir / f"{manifest.planet_id}_manifest.sha256.json"
    verification = baker.verify_manifest(manifest_path)
    
    print(f"\n🔐 Verification Results:")
    print(f"   Valid: {verification['valid']}")
    print(f"   Verified buffers: {verification['verified_buffers']}/{verification['total_buffers']}")
    
    # Test determinism - bake same planet again
    print(f"\n🔄 Testing determinism...")
    manifest2 = baker.bake_planet("test_planet", seed=12345)
    
    deterministic = manifest.overall_hash == manifest2.overall_hash
    print(f"   Same overall hash: {deterministic}")
    print(f"   Hash 1: {manifest.overall_hash[:16]}...")
    print(f"   Hash 2: {manifest2.overall_hash[:16]}...")
    
    if deterministic:
        print("✅ Deterministic baking system functional")
    else:
        print("❌ Determinism failed - hashes differ")
    
    # Display statistics
    stats = baker.get_bake_statistics()
    print(f"\n📈 Baker Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")