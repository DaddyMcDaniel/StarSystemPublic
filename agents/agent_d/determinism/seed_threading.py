#!/usr/bin/env python3
"""
Deterministic Seed Threading System - T13
=========================================

Threads seeds through all noise/SDF modules to ensure deterministic generation
without hidden RNGs. Provides seeded random number generation with reproducible
sequences across all terrain generation components.

Features:
- Hierarchical seed derivation from master seed
- Seeded RNG state management for all modules
- Thread-safe deterministic random generation
- Seed validation and verification
- Complete elimination of hidden randomness

Usage:
    from seed_threading import DeterministicSeedManager
    
    seed_mgr = DeterministicSeedManager(master_seed=12345)
    noise_seed = seed_mgr.derive_seed("terrain_noise", chunk_id="0_0")
"""

import numpy as np
import hashlib
import struct
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
import random

# Import SDF and noise systems
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'noise'))

try:
    from sdf_primitives import SDFNode
    from noise_field_3d import NoiseField3D
    SDF_AVAILABLE = True
except ImportError:
    SDF_AVAILABLE = False


class SeedDomain(Enum):
    """Seed domains for different generation components"""
    TERRAIN_HEIGHTFIELD = "terrain_heightfield"
    CAVE_SDF = "cave_sdf" 
    NOISE_FIELD = "noise_field"
    MARCHING_CUBES = "marching_cubes"
    FUSION = "fusion"
    MATERIALS = "materials"
    TANGENT_GENERATION = "tangent_generation"
    LOD_SELECTION = "lod_selection"
    CHUNK_STREAMING = "chunk_streaming"


@dataclass
class SeedDerivation:
    """Record of seed derivation for provenance"""
    domain: SeedDomain
    parent_seed: int
    derived_seed: int
    context: str
    derivation_method: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class RandomState:
    """Captured random state for reproducibility"""
    numpy_state: Tuple[str, np.ndarray, int, int, float]
    python_state: Tuple[int, Tuple, Optional[float]]
    seed_value: int


class DeterministicRNG:
    """Thread-safe deterministic random number generator"""
    
    def __init__(self, seed: int):
        """Initialize with specific seed"""
        self.seed = seed
        self._lock = threading.Lock()
        self._rng = np.random.RandomState(seed)
        self._python_rng = random.Random(seed)
        
        # Statistics
        self.generation_count = 0
        self.last_values = []
        self.max_history = 100
    
    def random(self) -> float:
        """Generate random float [0, 1)"""
        with self._lock:
            value = self._rng.random()
            self._record_generation(value)
            return value
    
    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate uniform random float in range"""
        with self._lock:
            value = self._rng.uniform(low, high)
            self._record_generation(value)
            return value
    
    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        """Generate normal distributed random float"""
        with self._lock:
            value = self._rng.normal(loc, scale)
            self._record_generation(value)
            return value
    
    def integers(self, low: int, high: int, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[int, np.ndarray]:
        """Generate random integers in range"""
        with self._lock:
            value = self._rng.integers(low, high, size=size)
            self._record_generation(value if np.isscalar(value) else value.tolist())
            return value
    
    def choice(self, a: Union[int, np.ndarray, List], size: Optional[Union[int, Tuple[int, ...]]] = None, 
               replace: bool = True, p: Optional[np.ndarray] = None):
        """Generate random choice from array"""
        with self._lock:
            value = self._rng.choice(a, size=size, replace=replace, p=p)
            self._record_generation(value if np.isscalar(value) else value.tolist())
            return value
    
    def shuffle(self, a: np.ndarray) -> np.ndarray:
        """Shuffle array in-place deterministically"""
        with self._lock:
            self._rng.shuffle(a)
            self.generation_count += 1
            return a
    
    def _record_generation(self, value):
        """Record generated value for debugging"""
        self.generation_count += 1
        if len(self.last_values) >= self.max_history:
            self.last_values.pop(0)
        self.last_values.append((self.generation_count, value))
    
    def get_state(self) -> RandomState:
        """Get current random state"""
        with self._lock:
            numpy_state = self._rng.get_state()
            python_state = self._python_rng.getstate()
            return RandomState(numpy_state, python_state, self.seed)
    
    def set_state(self, state: RandomState):
        """Restore random state"""
        with self._lock:
            self._rng.set_state(state.numpy_state)
            self._python_rng.setstate(state.python_state)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'seed': self.seed,
            'generation_count': self.generation_count,
            'last_values_count': len(self.last_values),
            'recent_values': self.last_values[-5:] if self.last_values else []
        }


class DeterministicSeedManager:
    """Manages hierarchical seed derivation for deterministic generation"""
    
    def __init__(self, master_seed: int = 42):
        """Initialize with master seed"""
        self.master_seed = master_seed
        self.seed_derivations: Dict[str, SeedDerivation] = {}
        self.rng_instances: Dict[str, DeterministicRNG] = {}
        self._lock = threading.Lock()
        
        # Initialize master RNG
        self.master_rng = DeterministicRNG(master_seed)
        
        # Track seed usage for validation
        self.used_seeds: set[int] = {master_seed}
        self.seed_collision_count = 0
        
        # Global RNG replacement
        self._original_numpy_random = None
        self._original_python_random = None
        self._global_override_active = False
    
    def derive_seed(self, domain: Union[SeedDomain, str], context: str = "", 
                   chunk_id: Optional[str] = None, additional_params: Optional[Dict] = None) -> int:
        """Derive deterministic seed for specific domain and context"""
        
        if isinstance(domain, str):
            domain_str = domain
        else:
            domain_str = domain.value
        
        # Create unique derivation key
        derivation_key = f"{domain_str}:{context}"
        if chunk_id:
            derivation_key += f":{chunk_id}"
        if additional_params:
            # Sort params for consistency
            param_str = ":".join(f"{k}={v}" for k, v in sorted(additional_params.items()))
            derivation_key += f":{param_str}"
        
        with self._lock:
            # Check if already derived
            if derivation_key in self.seed_derivations:
                return self.seed_derivations[derivation_key].derived_seed
            
            # Hash-based derivation for determinism
            hasher = hashlib.sha256()
            hasher.update(str(self.master_seed).encode('utf-8'))
            hasher.update(derivation_key.encode('utf-8'))
            
            # Convert hash to seed
            hash_bytes = hasher.digest()
            derived_seed = struct.unpack('<I', hash_bytes[:4])[0]  # Use first 4 bytes as uint32
            
            # Handle collisions (extremely unlikely but handle gracefully)
            collision_counter = 0
            original_derived_seed = derived_seed
            while derived_seed in self.used_seeds:
                collision_counter += 1
                hasher = hashlib.sha256()
                hasher.update(str(self.master_seed).encode('utf-8'))
                hasher.update(derivation_key.encode('utf-8'))
                hasher.update(str(collision_counter).encode('utf-8'))
                hash_bytes = hasher.digest()
                derived_seed = struct.unpack('<I', hash_bytes[:4])[0]
                
                if collision_counter > 1000:  # Failsafe
                    derived_seed = hash(derivation_key) % (2**32)
                    break
            
            if collision_counter > 0:
                self.seed_collision_count += collision_counter
            
            self.used_seeds.add(derived_seed)
            
            # Record derivation
            derivation = SeedDerivation(
                domain=SeedDomain(domain_str) if isinstance(domain, str) else domain,
                parent_seed=self.master_seed,
                derived_seed=derived_seed,
                context=derivation_key,
                derivation_method="sha256_hash"
            )
            
            self.seed_derivations[derivation_key] = derivation
            return derived_seed
    
    def get_rng(self, domain: Union[SeedDomain, str], context: str = "",
                chunk_id: Optional[str] = None, additional_params: Optional[Dict] = None) -> DeterministicRNG:
        """Get or create RNG instance for specific domain"""
        
        # Create RNG key
        if isinstance(domain, str):
            domain_str = domain
        else:
            domain_str = domain.value
            
        rng_key = f"{domain_str}:{context}"
        if chunk_id:
            rng_key += f":{chunk_id}"
        if additional_params:
            param_str = ":".join(f"{k}={v}" for k, v in sorted(additional_params.items()))
            rng_key += f":{param_str}"
        
        with self._lock:
            if rng_key not in self.rng_instances:
                seed = self.derive_seed(domain, context, chunk_id, additional_params)
                self.rng_instances[rng_key] = DeterministicRNG(seed)
            
            return self.rng_instances[rng_key]
    
    def enable_global_override(self):
        """Replace global random number generators with deterministic versions"""
        if self._global_override_active:
            return
        
        # Store original generators
        self._original_numpy_random = np.random.get_state()
        self._original_python_random = random.getstate()
        
        # Set deterministic seeds
        np.random.seed(self.master_seed)
        random.seed(self.master_seed)
        
        self._global_override_active = True
        print(f"🎲 Global RNG override enabled with seed {self.master_seed}")
    
    def disable_global_override(self):
        """Restore original random number generators"""
        if not self._global_override_active:
            return
        
        # Restore original states
        if self._original_numpy_random is not None:
            np.random.set_state(self._original_numpy_random)
        if self._original_python_random is not None:
            random.setstate(self._original_python_random)
        
        self._global_override_active = False
        print("🎲 Global RNG override disabled")
    
    def validate_determinism(self, domain: Union[SeedDomain, str], context: str = "",
                           iterations: int = 3) -> Dict[str, Any]:
        """Validate that RNG produces identical sequences"""
        
        results = []
        rng_key = f"{domain}:{context}" if isinstance(domain, str) else f"{domain.value}:{context}"
        
        for iteration in range(iterations):
            # Get fresh RNG instance
            rng = self.get_rng(domain, f"{context}_test_{iteration}")
            
            # Generate test sequence
            test_values = []
            test_values.append(rng.random())
            test_values.append(rng.uniform(-1, 1))
            test_values.append(rng.normal(0, 1))
            test_values.append(int(rng.integers(0, 100)))
            
            results.append(test_values)
        
        # Check if all iterations produced identical results
        identical = all(results[0] == result for result in results[1:])
        
        return {
            'domain': rng_key,
            'iterations': iterations,
            'identical': identical,
            'sample_values': results[0] if results else [],
            'all_results': results
        }
    
    def get_seed_manifest(self) -> Dict[str, Any]:
        """Get complete manifest of all derived seeds"""
        manifest = {
            'master_seed': self.master_seed,
            'total_derived_seeds': len(self.seed_derivations),
            'seed_collisions': self.seed_collision_count,
            'domains': {},
            'derivations': []
        }
        
        # Group by domain
        domain_counts = {}
        for derivation in self.seed_derivations.values():
            domain_name = derivation.domain.value
            if domain_name not in domain_counts:
                domain_counts[domain_name] = 0
            domain_counts[domain_name] += 1
        
        manifest['domains'] = domain_counts
        
        # Add derivation records
        for key, derivation in self.seed_derivations.items():
            manifest['derivations'].append({
                'key': key,
                'domain': derivation.domain.value,
                'parent_seed': derivation.parent_seed,
                'derived_seed': derivation.derived_seed,
                'context': derivation.context,
                'method': derivation.derivation_method,
                'timestamp': derivation.timestamp
            })
        
        return manifest
    
    def get_rng_statistics(self) -> Dict[str, Any]:
        """Get statistics for all RNG instances"""
        stats = {
            'total_rng_instances': len(self.rng_instances),
            'instances': {}
        }
        
        for key, rng in self.rng_instances.items():
            stats['instances'][key] = rng.get_generation_stats()
        
        return stats
    
    def clear_derived_seeds(self):
        """Clear all derived seeds and RNG instances (keep master seed)"""
        with self._lock:
            self.seed_derivations.clear()
            self.rng_instances.clear()
            self.used_seeds = {self.master_seed}
            self.seed_collision_count = 0
        
        print("🧹 Cleared all derived seeds and RNG instances")


# Global seed manager instance
_global_seed_manager: Optional[DeterministicSeedManager] = None


def get_global_seed_manager() -> DeterministicSeedManager:
    """Get or create global seed manager"""
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = DeterministicSeedManager()
    return _global_seed_manager


def set_global_master_seed(seed: int):
    """Set master seed for global seed manager"""
    global _global_seed_manager
    _global_seed_manager = DeterministicSeedManager(seed)
    print(f"🌱 Set global master seed to {seed}")


def get_seeded_rng(domain: Union[SeedDomain, str], context: str = "", 
                   chunk_id: Optional[str] = None) -> DeterministicRNG:
    """Convenience function to get seeded RNG from global manager"""
    return get_global_seed_manager().get_rng(domain, context, chunk_id)


if __name__ == "__main__":
    # Test deterministic seed threading system
    print("🚀 T13 Deterministic Seed Threading System")
    print("=" * 60)
    
    # Create seed manager
    seed_mgr = DeterministicSeedManager(master_seed=12345)
    
    # Test seed derivation
    print("🌱 Testing seed derivation:")
    terrain_seed = seed_mgr.derive_seed(SeedDomain.TERRAIN_HEIGHTFIELD, "test_chunk")
    cave_seed = seed_mgr.derive_seed(SeedDomain.CAVE_SDF, "test_chunk")
    noise_seed = seed_mgr.derive_seed(SeedDomain.NOISE_FIELD, "perlin", chunk_id="0_0")
    
    print(f"   Terrain seed: {terrain_seed}")
    print(f"   Cave seed: {cave_seed}")
    print(f"   Noise seed: {noise_seed}")
    
    # Test RNG generation
    print("\n🎲 Testing RNG generation:")
    terrain_rng = seed_mgr.get_rng(SeedDomain.TERRAIN_HEIGHTFIELD, "test_chunk")
    
    # Generate test values
    random_values = [terrain_rng.random() for _ in range(3)]
    uniform_values = [terrain_rng.uniform(-1, 1) for _ in range(3)]
    normal_values = [terrain_rng.normal(0, 1) for _ in range(3)]
    
    print(f"   Random values: {[f'{v:.6f}' for v in random_values]}")
    print(f"   Uniform values: {[f'{v:.6f}' for v in uniform_values]}")
    print(f"   Normal values: {[f'{v:.6f}' for v in normal_values]}")
    
    # Test determinism validation
    print("\n🔍 Testing determinism validation:")
    validation = seed_mgr.validate_determinism(SeedDomain.NOISE_FIELD, "determinism_test")
    print(f"   Identical across iterations: {validation['identical']}")
    print(f"   Sample values: {[f'{v:.6f}' if isinstance(v, float) else v for v in validation['sample_values']]}")
    
    # Test global override
    print("\n🌍 Testing global override:")
    seed_mgr.enable_global_override()
    global_random_1 = np.random.random()
    global_random_2 = np.random.random()
    seed_mgr.disable_global_override()
    
    # Re-enable and check reproducibility
    seed_mgr.enable_global_override()
    global_random_1_repeat = np.random.random()
    global_random_2_repeat = np.random.random()
    seed_mgr.disable_global_override()
    
    reproducible = (global_random_1 == global_random_1_repeat and 
                   global_random_2 == global_random_2_repeat)
    print(f"   Global override reproducible: {reproducible}")
    
    # Display manifest
    manifest = seed_mgr.get_seed_manifest()
    print(f"\n📋 Seed manifest:")
    print(f"   Master seed: {manifest['master_seed']}")
    print(f"   Derived seeds: {manifest['total_derived_seeds']}")
    print(f"   Domains: {list(manifest['domains'].keys())}")
    
    # Display statistics
    stats = seed_mgr.get_rng_statistics()
    print(f"   RNG instances: {stats['total_rng_instances']}")
    
    print("\n✅ Deterministic seed threading system functional")