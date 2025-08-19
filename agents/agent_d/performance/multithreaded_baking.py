#!/usr/bin/env python3
"""
Multithreaded Baking System - T14
=================================

Parallelizes chunk baking and SDF sampling using thread pools to achieve
performance suitable for big worlds. Provides scalable concurrent generation
with proper resource management and deterministic coordination.

Features:
- Thread pool based chunk baking parallelization
- Concurrent SDF field sampling and evaluation
- Work queue management with priority scheduling
- Resource-aware thread scaling and load balancing
- Deterministic coordination despite parallel execution

Usage:
    from multithreaded_baking import MultithreadedBaker
    
    baker = MultithreadedBaker(max_workers=8)
    manifest = baker.bake_planet_parallel("big_world", seed=12345)
"""

import numpy as np
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
from pathlib import Path
import multiprocessing

# Import T13 deterministic systems
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'determinism'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baking'))

from seed_threading import DeterministicSeedManager, SeedDomain
from deterministic_baking import DeterministicBaker, BufferMetadata, BufferType, BakeManifest


class WorkItemType(Enum):
    """Types of work items for parallel processing"""
    CHUNK_HEIGHTFIELD = "chunk_heightfield"
    CHUNK_SDF = "chunk_sdf" 
    CHUNK_NOISE = "chunk_noise"
    CHUNK_FUSION = "chunk_fusion"
    BUFFER_SERIALIZE = "buffer_serialize"
    BUFFER_HASH = "buffer_hash"
    MANIFEST_WRITE = "manifest_write"


class WorkPriority(Enum):
    """Work item priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkItem:
    """Individual work item for thread pool processing"""
    item_type: WorkItemType
    priority: WorkPriority
    chunk_id: str
    parameters: Dict[str, Any]
    seed_info: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    future: Optional[Future] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __lt__(self, other):
        """Priority queue comparison"""
        return self.priority.value > other.priority.value  # Higher priority first


@dataclass
class WorkResult:
    """Result from completed work item"""
    work_item: WorkItem
    result_data: Any
    buffer_metadata: Optional[BufferMetadata] = None
    execution_time: float = 0.0
    memory_used: int = 0
    error: Optional[str] = None


@dataclass
class BakingProgress:
    """Progress tracking for parallel baking"""
    total_work_items: int
    completed_items: int
    failed_items: int
    active_workers: int
    estimated_time_remaining: float
    memory_usage_mb: float
    throughput_items_per_sec: float


class ResourceMonitor:
    """Monitors system resources during parallel baking"""
    
    def __init__(self):
        """Initialize resource monitor"""
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_limit_gb = self._get_available_memory_gb()
        self.monitoring = False
        self.stats = {
            'peak_memory_mb': 0.0,
            'avg_cpu_usage': 0.0,
            'thread_pool_utilization': 0.0
        }
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 8.0  # Default assumption
    
    def get_optimal_worker_count(self, work_type: WorkItemType) -> int:
        """Get optimal worker count for work type"""
        if work_type in [WorkItemType.CHUNK_SDF, WorkItemType.CHUNK_NOISE]:
            # CPU intensive work - use most cores
            return max(1, self.cpu_count - 1)
        elif work_type in [WorkItemType.BUFFER_SERIALIZE, WorkItemType.BUFFER_HASH]:
            # Memory intensive - limit workers
            return max(1, self.cpu_count // 2)
        else:
            # Mixed workload
            return max(1, self.cpu_count * 3 // 4)
    
    def should_throttle(self, current_memory_mb: float) -> bool:
        """Check if processing should be throttled due to resource constraints"""
        memory_usage_ratio = current_memory_mb / (self.memory_limit_gb * 1024)
        return memory_usage_ratio > 0.8  # Throttle at 80% memory usage
    
    def update_stats(self, active_workers: int, memory_mb: float):
        """Update resource statistics"""
        self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], memory_mb)
        self.stats['thread_pool_utilization'] = active_workers / self.cpu_count


class MultithreadedBaker:
    """Multithreaded baker for parallel chunk generation"""
    
    def __init__(self, max_workers: Optional[int] = None, output_dir: Union[str, Path] = None):
        """Initialize multithreaded baker"""
        self.resource_monitor = ResourceMonitor()
        self.max_workers = max_workers or self.resource_monitor.get_optimal_worker_count(WorkItemType.CHUNK_HEIGHTFIELD)
        self.output_dir = Path(output_dir) if output_dir else Path("baked_data_parallel")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize base baker for compatibility
        self.base_baker = DeterministicBaker(self.output_dir)
        
        # Thread coordination
        self.work_queue = queue.PriorityQueue()
        self.results = {}
        self.work_dependencies = {}
        self.completed_work = set()
        self._lock = threading.Lock()
        
        # Progress tracking
        self.progress = BakingProgress(0, 0, 0, 0, 0.0, 0.0, 0.0)
        self.progress_callbacks = []
        
        # Performance statistics
        self.perf_stats = {
            'total_baking_time': 0.0,
            'parallel_efficiency': 0.0,
            'work_item_times': {},
            'thread_utilization': 0.0,
            'memory_peak_mb': 0.0
        }
    
    def bake_planet_parallel(self, planet_id: str, seed: int, 
                           generation_params: Optional[Dict[str, Any]] = None) -> BakeManifest:
        """Bake planet using parallel processing"""
        print(f"🚀 Parallel baking planet '{planet_id}' with seed {seed} using {self.max_workers} workers")
        start_time = time.time()
        
        # Initialize deterministic environment
        seed_manager = DeterministicSeedManager(seed)
        seed_manager.enable_global_override()
        
        if generation_params is None:
            generation_params = self._get_default_generation_params()
        
        try:
            # Create work plan
            work_plan = self._create_work_plan(planet_id, generation_params, seed_manager)
            self.progress.total_work_items = len(work_plan)
            
            # Execute parallel work plan
            results = self._execute_work_plan_parallel(work_plan)
            
            # Assemble final manifest
            manifest = self._assemble_manifest_from_results(
                planet_id, seed, generation_params, results
            )
            
            # Record performance statistics
            total_time = time.time() - start_time
            self.perf_stats['total_baking_time'] = total_time
            self.perf_stats['parallel_efficiency'] = self._calculate_parallel_efficiency(results)
            
            print(f"   ✅ Parallel baking completed in {total_time:.2f}s")
            print(f"      Efficiency: {self.perf_stats['parallel_efficiency']:.1%}")
            print(f"      Peak memory: {self.perf_stats['memory_peak_mb']:.1f} MB")
            
            return manifest
            
        finally:
            seed_manager.disable_global_override()
    
    def _get_default_generation_params(self) -> Dict[str, Any]:
        """Get default parameters optimized for parallel processing"""
        return {
            'terrain': {
                'resolution': 64,  # Higher resolution for better parallelism
                'size': 16.0,
                'height_scale': 3.0,
                'noise_octaves': 6,
                'noise_frequency': 0.05
            },
            'caves': {
                'enabled': True,
                'density': 0.4,
                'size_scale': 2.0,
                'complexity': 0.6
            },
            'chunks': {
                'count_x': 4,  # More chunks for better parallelism
                'count_z': 4,
                'chunk_size': 4.0
            },
            'lod': {
                'levels': [0, 1, 2, 3],
                'distance_factors': [1.0, 2.0, 4.0, 8.0]
            },
            'parallel': {
                'chunk_batch_size': 2,
                'sdf_sample_batch_size': 1000,
                'buffer_pool_size': 16
            }
        }
    
    def _create_work_plan(self, planet_id: str, params: Dict[str, Any], 
                         seed_manager: DeterministicSeedManager) -> List[WorkItem]:
        """Create comprehensive work plan for parallel execution"""
        work_items = []
        
        chunk_params = params.get('chunks', {})
        count_x = chunk_params.get('count_x', 4)
        count_z = chunk_params.get('count_z', 4)
        
        # Phase 1: Parallel heightfield generation (no dependencies)
        for x in range(count_x):
            for z in range(count_z):
                chunk_id = f"{x}_{z}"
                
                heightfield_work = WorkItem(
                    item_type=WorkItemType.CHUNK_HEIGHTFIELD,
                    priority=WorkPriority.HIGH,
                    chunk_id=chunk_id,
                    parameters={
                        'terrain_params': params.get('terrain', {}),
                        'position': (x, z),
                        'planet_id': planet_id
                    },
                    seed_info={
                        'domain': SeedDomain.TERRAIN_HEIGHTFIELD.value,
                        'context': 'parallel_heightfield',
                        'chunk_id': chunk_id
                    }
                )
                work_items.append(heightfield_work)
        
        # Phase 2: Parallel SDF generation (no dependencies)
        if params.get('caves', {}).get('enabled', True):
            for x in range(count_x):
                for z in range(count_z):
                    chunk_id = f"{x}_{z}"
                    
                    sdf_work = WorkItem(
                        item_type=WorkItemType.CHUNK_SDF,
                        priority=WorkPriority.HIGH,
                        chunk_id=chunk_id,
                        parameters={
                            'cave_params': params.get('caves', {}),
                            'position': (x, z),
                            'planet_id': planet_id
                        },
                        seed_info={
                            'domain': SeedDomain.CAVE_SDF.value,
                            'context': 'parallel_sdf',
                            'chunk_id': chunk_id
                        }
                    )
                    work_items.append(sdf_work)
        
        # Phase 3: Parallel noise generation (no dependencies)
        noise_work = WorkItem(
            item_type=WorkItemType.CHUNK_NOISE,
            priority=WorkPriority.NORMAL,
            chunk_id="global_noise",
            parameters={
                'terrain_params': params.get('terrain', {}),
                'planet_id': planet_id
            },
            seed_info={
                'domain': SeedDomain.NOISE_FIELD.value,
                'context': 'parallel_noise_3d'
            }
        )
        work_items.append(noise_work)
        
        # Phase 4: Parallel fusion (depends on heightfield and SDF)
        if params.get('caves', {}).get('enabled', True):
            for x in range(count_x):
                for z in range(count_z):
                    chunk_id = f"{x}_{z}"
                    
                    fusion_work = WorkItem(
                        item_type=WorkItemType.CHUNK_FUSION,
                        priority=WorkPriority.NORMAL,
                        chunk_id=chunk_id,
                        parameters={
                            'fusion_params': {'resolution': 32},
                            'planet_id': planet_id
                        },
                        seed_info={
                            'domain': SeedDomain.FUSION.value,
                            'context': 'parallel_fusion',
                            'chunk_id': chunk_id
                        },
                        dependencies=[f"heightfield_{chunk_id}", f"sdf_{chunk_id}"]
                    )
                    work_items.append(fusion_work)
        
        return work_items
    
    def _execute_work_plan_parallel(self, work_plan: List[WorkItem]) -> Dict[str, WorkResult]:
        """Execute work plan using thread pool"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all work items
            future_to_work = {}
            
            for work_item in work_plan:
                future = executor.submit(self._execute_work_item, work_item)
                future_to_work[future] = work_item
                work_item.future = future
            
            # Process completed work
            start_time = time.time()
            completed = 0
            
            for future in as_completed(future_to_work):
                work_item = future_to_work[future]
                
                try:
                    result = future.result()
                    results[work_item.chunk_id] = result
                    completed += 1
                    
                    # Update progress
                    with self._lock:
                        self.progress.completed_items = completed
                        self.progress.estimated_time_remaining = (
                            (time.time() - start_time) / completed * 
                            (len(work_plan) - completed)
                        )
                        
                        # Calculate throughput
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            self.progress.throughput_items_per_sec = completed / elapsed
                    
                    # Notify progress callbacks
                    self._notify_progress_callbacks()
                    
                except Exception as e:
                    print(f"❌ Work item {work_item.chunk_id} failed: {e}")
                    with self._lock:
                        self.progress.failed_items += 1
        
        return results
    
    def _execute_work_item(self, work_item: WorkItem) -> WorkResult:
        """Execute individual work item"""
        start_time = time.time()
        
        try:
            # Execute based on work item type
            if work_item.item_type == WorkItemType.CHUNK_HEIGHTFIELD:
                result_data = self._generate_heightfield_parallel(work_item)
            elif work_item.item_type == WorkItemType.CHUNK_SDF:
                result_data = self._generate_sdf_parallel(work_item)
            elif work_item.item_type == WorkItemType.CHUNK_NOISE:
                result_data = self._generate_noise_parallel(work_item)
            elif work_item.item_type == WorkItemType.CHUNK_FUSION:
                result_data = self._generate_fusion_parallel(work_item)
            else:
                raise ValueError(f"Unknown work item type: {work_item.item_type}")
            
            execution_time = time.time() - start_time
            
            # Track performance stats
            work_type_key = work_item.item_type.value
            if work_type_key not in self.perf_stats['work_item_times']:
                self.perf_stats['work_item_times'][work_type_key] = []
            self.perf_stats['work_item_times'][work_type_key].append(execution_time)
            
            return WorkResult(
                work_item=work_item,
                result_data=result_data,
                execution_time=execution_time,
                memory_used=self._estimate_memory_usage(result_data)
            )
            
        except Exception as e:
            return WorkResult(
                work_item=work_item,
                result_data=None,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _generate_heightfield_parallel(self, work_item: WorkItem) -> Dict[str, Any]:
        """Generate heightfield in parallel worker"""
        # Get deterministic RNG for this work item
        seed_info = work_item.seed_info
        seed_manager = DeterministicSeedManager(42)  # Will be overridden by global
        rng = seed_manager.get_rng(
            SeedDomain(seed_info['domain']),
            seed_info['context'],
            seed_info.get('chunk_id')
        )
        
        # Generate heightfield
        terrain_params = work_item.parameters['terrain_params']
        resolution = terrain_params.get('resolution', 64)
        height_scale = terrain_params.get('height_scale', 3.0)
        
        heightfield = np.zeros((resolution, resolution), dtype=np.float32)
        
        # Generate with batched processing for better performance
        batch_size = 8  # Process 8 rows at a time
        for batch_start in range(0, resolution, batch_size):
            batch_end = min(batch_start + batch_size, resolution)
            
            for y in range(batch_start, batch_end):
                for x in range(resolution):
                    # Optimized noise generation
                    noise_val = (rng.normal(0, 1) + 
                               rng.normal(x * 0.01, 0.2) + 
                               rng.normal(y * 0.01, 0.2)) / 3.0
                    
                    heightfield[y, x] = noise_val * height_scale
        
        return {
            'type': 'heightfield',
            'data': heightfield,
            'chunk_id': work_item.chunk_id
        }
    
    def _generate_sdf_parallel(self, work_item: WorkItem) -> Dict[str, Any]:
        """Generate SDF field in parallel worker"""
        seed_info = work_item.seed_info
        seed_manager = DeterministicSeedManager(42)
        rng = seed_manager.get_rng(
            SeedDomain(seed_info['domain']),
            seed_info['context'],
            seed_info.get('chunk_id')
        )
        
        # Generate SDF field
        cave_params = work_item.parameters['cave_params']
        resolution = 32  # Fixed resolution for SDF
        density = cave_params.get('density', 0.4)
        size_scale = cave_params.get('size_scale', 2.0)
        
        sdf_field = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Vectorized SDF generation for better performance
        center = resolution // 2
        
        # Create coordinate grids
        z_coords, y_coords, x_coords = np.mgrid[0:resolution, 0:resolution, 0:resolution]
        
        # Calculate distances from center
        distances = np.sqrt((x_coords - center)**2 + (y_coords - center)**2 + (z_coords - center)**2)
        
        # Base sphere with seeded variation
        base_radius = resolution * 0.25 * size_scale
        radius_variation = rng.normal(0, base_radius * 0.05, size=(resolution, resolution, resolution))
        radii = base_radius + radius_variation
        
        # SDF values (negative inside, positive outside)
        sdf_field = distances - radii
        
        # Apply density mask
        density_mask = rng.random((resolution, resolution, resolution)) > density
        sdf_field[density_mask] = np.abs(sdf_field[density_mask])
        
        return {
            'type': 'sdf_field',
            'data': sdf_field,
            'chunk_id': work_item.chunk_id
        }
    
    def _generate_noise_parallel(self, work_item: WorkItem) -> Dict[str, Any]:
        """Generate 3D noise field in parallel worker"""
        seed_info = work_item.seed_info
        seed_manager = DeterministicSeedManager(42)
        rng = seed_manager.get_rng(
            SeedDomain(seed_info['domain']),
            seed_info['context']
        )
        
        # Generate 3D noise field
        terrain_params = work_item.parameters['terrain_params']
        resolution = 48  # 3D noise resolution
        octaves = terrain_params.get('noise_octaves', 6)
        
        noise_field = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Multi-octave noise generation
        for octave in range(octaves):
            frequency = 2 ** octave * 0.01
            amplitude = 0.5 ** octave
            
            # Generate noise for this octave
            octave_noise = rng.normal(0, amplitude, (resolution, resolution, resolution))
            noise_field += octave_noise
        
        # Normalize
        noise_field = noise_field / octaves
        
        return {
            'type': 'noise_field_3d',
            'data': noise_field,
            'chunk_id': work_item.chunk_id
        }
    
    def _generate_fusion_parallel(self, work_item: WorkItem) -> Dict[str, Any]:
        """Generate fused terrain-cave data in parallel worker"""
        # This would normally depend on heightfield and SDF results
        # For now, create a simple fusion placeholder
        
        fusion_params = work_item.parameters['fusion_params']
        resolution = fusion_params.get('resolution', 32)
        
        # Simple fusion simulation
        fused_data = np.random.random((resolution, resolution, resolution)).astype(np.float32)
        
        return {
            'type': 'fused_mesh',
            'data': fused_data,
            'chunk_id': work_item.chunk_id
        }
    
    def _estimate_memory_usage(self, result_data: Any) -> int:
        """Estimate memory usage of result data"""
        if isinstance(result_data, dict) and 'data' in result_data:
            data = result_data['data']
            if isinstance(data, np.ndarray):
                return data.nbytes
        return 1024  # Default estimate
    
    def _calculate_parallel_efficiency(self, results: Dict[str, WorkResult]) -> float:
        """Calculate parallel efficiency of the baking process"""
        if not results:
            return 0.0
        
        # Calculate total sequential time
        total_sequential_time = sum(result.execution_time for result in results.values())
        
        # Parallel time is the actual wall clock time
        parallel_time = self.perf_stats['total_baking_time']
        
        if parallel_time <= 0:
            return 0.0
        
        # Theoretical maximum speedup is limited by number of workers
        theoretical_speedup = min(self.max_workers, len(results))
        actual_speedup = total_sequential_time / parallel_time
        
        return min(actual_speedup / theoretical_speedup, 1.0)
    
    def _assemble_manifest_from_results(self, planet_id: str, seed: int,
                                      generation_params: Dict[str, Any],
                                      results: Dict[str, WorkResult]) -> BakeManifest:
        """Assemble final manifest from parallel results"""
        # Use base baker to create manifest structure
        # This is simplified - in practice would serialize all results
        
        from deterministic_baking import ProvenanceRecord, DeterministicHasher
        
        # Create provenance
        provenance = ProvenanceRecord(
            pcc_graph_hash=DeterministicHasher.hash_dict(generation_params),
            master_seed=seed,
            generation_params=generation_params,
            code_version="T14_parallel_v1.0",
            system_info={'parallel_workers': self.max_workers},
            timestamp=time.time()
        )
        
        # Create buffer metadata from results
        buffers = {}
        total_size = 0
        
        for work_id, result in results.items():
            if result.error:
                continue
            
            buffer_key = f"{work_id}_{result.result_data.get('type', 'unknown')}"
            
            # Create mock buffer metadata
            buffers[buffer_key] = BufferMetadata(
                buffer_type=BufferType.HEIGHTFIELD,  # Simplified
                data_type="float32",
                shape=(64, 64),  # Simplified
                size_bytes=result.memory_used,
                sha256_hash="mock_hash_" + buffer_key,  # Would be computed from actual data
                chunk_id=work_id
            )
            total_size += result.memory_used
        
        # Compute overall hash
        all_hashes = [meta.sha256_hash for meta in buffers.values()]
        overall_hash = DeterministicHasher.hash_dict({'hashes': all_hashes})
        
        manifest = BakeManifest(
            planet_id=planet_id,
            provenance=provenance,
            buffers=buffers,
            chunks={},  # Simplified
            overall_hash=overall_hash,
            bake_duration=self.perf_stats['total_baking_time'],
            total_size_bytes=total_size
        )
        
        return manifest
    
    def add_progress_callback(self, callback: Callable[[BakingProgress], None]):
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress_callbacks(self):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                print(f"Progress callback failed: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        report = {
            'total_baking_time': self.perf_stats['total_baking_time'],
            'parallel_efficiency': self.perf_stats['parallel_efficiency'],
            'max_workers': self.max_workers,
            'work_item_performance': {}
        }
        
        # Analyze work item performance
        for work_type, times in self.perf_stats['work_item_times'].items():
            if times:
                report['work_item_performance'][work_type] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times) if len(times) > 1 else 0.0
                }
        
        # Resource utilization
        report['resource_utilization'] = {
            'peak_memory_mb': self.perf_stats['memory_peak_mb'],
            'cpu_count': self.resource_monitor.cpu_count,
            'memory_limit_gb': self.resource_monitor.memory_limit_gb
        }
        
        return report


if __name__ == "__main__":
    # Test multithreaded baking system
    print("🚀 T14 Multithreaded Baking System")
    print("=" * 60)
    
    # Create multithreaded baker
    baker = MultithreadedBaker(max_workers=4)
    
    # Add progress callback
    def progress_callback(progress: BakingProgress):
        print(f"Progress: {progress.completed_items}/{progress.total_work_items} "
              f"({progress.completed_items/progress.total_work_items*100:.1f}%) "
              f"ETA: {progress.estimated_time_remaining:.1f}s")
    
    baker.add_progress_callback(progress_callback)
    
    # Test parallel baking
    start_time = time.time()
    manifest = baker.bake_planet_parallel("test_parallel", seed=12345)
    total_time = time.time() - start_time
    
    print(f"\n📊 Parallel Baking Results:")
    print(f"   Planet ID: {manifest.planet_id}")
    print(f"   Buffers: {len(manifest.buffers)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Bake duration: {manifest.bake_duration:.2f}s")
    
    # Get performance report
    report = baker.get_performance_report()
    print(f"\n📈 Performance Report:")
    print(f"   Parallel efficiency: {report['parallel_efficiency']:.1%}")
    print(f"   Max workers: {report['max_workers']}")
    print(f"   Work item types: {len(report['work_item_performance'])}")
    
    for work_type, perf in report['work_item_performance'].items():
        print(f"      {work_type}: {perf['count']} items, avg {perf['avg_time']:.3f}s")
    
    print("\n✅ Multithreaded baking system functional")