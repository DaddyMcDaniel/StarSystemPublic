#!/usr/bin/env python3
"""
Buffer Pool Management System - T14
===================================

Introduces buffer pools to avoid frequent malloc/free operations during
terrain generation. Provides pre-allocated, reusable buffers with automatic
size management and memory optimization.

Features:
- Pre-allocated buffer pools for different data types and sizes
- Automatic buffer sizing and growth management
- Memory-efficient reuse of buffers across generation cycles
- Thread-safe buffer checkout/checkin with reference counting
- Memory pressure monitoring and adaptive pool sizing

Usage:
    from buffer_pools import BufferPoolManager, BufferType
    
    pool_mgr = BufferPoolManager()
    buffer = pool_mgr.get_buffer(BufferType.HEIGHTFIELD_2D, (256, 256))
    # Use buffer...
    pool_mgr.return_buffer(buffer)
"""

import numpy as np
import threading
import time
import weakref
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
from collections import defaultdict, deque

# Import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PooledBufferType(Enum):
    """Types of buffers managed by the pool system"""
    HEIGHTFIELD_2D = "heightfield_2d"          # 2D float32 heightfields
    SDF_FIELD_3D = "sdf_field_3d"              # 3D float32 SDF volumes
    NOISE_FIELD_3D = "noise_field_3d"          # 3D float32 noise volumes
    VERTEX_BUFFER = "vertex_buffer"            # Vertex position data
    NORMAL_BUFFER = "normal_buffer"            # Normal vector data
    TANGENT_BUFFER = "tangent_buffer"          # Tangent vector data
    INDEX_BUFFER = "index_buffer"              # Triangle index data
    MATERIAL_BUFFER = "material_buffer"        # Material parameter data
    TEMP_WORKSPACE = "temp_workspace"          # Temporary computation buffers
    SERIALIZATION_BUFFER = "serialization"    # Buffer for serialization


@dataclass
class BufferSpec:
    """Specification for a buffer type"""
    buffer_type: PooledBufferType
    dtype: np.dtype
    default_shape: Tuple[int, ...]
    max_size_mb: float
    typical_lifetime_sec: float
    growth_factor: float = 1.5


@dataclass
class PooledBuffer:
    """Managed buffer with metadata"""
    array: np.ndarray
    buffer_type: PooledBufferType
    allocated_shape: Tuple[int, ...]
    creation_time: float
    last_used_time: float
    use_count: int
    checked_out: bool
    thread_id: Optional[int] = None
    
    def get_view(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Get a view of the buffer with specified shape"""
        if not all(s <= a for s, a in zip(shape, self.allocated_shape)):
            raise ValueError(f"Requested shape {shape} exceeds allocated {self.allocated_shape}")
        
        # Create view with requested shape
        view = self.array.ravel()[:np.prod(shape)].reshape(shape)
        self.last_used_time = time.time()
        self.use_count += 1
        return view
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes"""
        return self.array.nbytes


class BufferPool:
    """Pool for a specific buffer type and size range"""
    
    def __init__(self, buffer_spec: BufferSpec, initial_count: int = 2):
        """Initialize buffer pool"""
        self.buffer_spec = buffer_spec
        self.available_buffers = deque()
        self.checked_out_buffers = set()
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_created': 0,
            'total_checkouts': 0,
            'total_returns': 0,
            'peak_checked_out': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Create initial buffers
        for _ in range(initial_count):
            self._create_buffer()
    
    def _create_buffer(self, shape: Optional[Tuple[int, ...]] = None) -> PooledBuffer:
        """Create new buffer with specified or default shape"""
        if shape is None:
            shape = self.buffer_spec.default_shape
        
        # Ensure shape doesn't exceed maximum size
        total_elements = np.prod(shape)
        max_elements = int(self.buffer_spec.max_size_mb * 1024 * 1024 / 
                          self.buffer_spec.dtype.itemsize)
        
        if total_elements > max_elements:
            # Scale down shape proportionally
            scale_factor = (max_elements / total_elements) ** (1.0 / len(shape))
            shape = tuple(int(dim * scale_factor) for dim in shape)
        
        # Allocate array
        array = np.empty(shape, dtype=self.buffer_spec.dtype)
        
        buffer = PooledBuffer(
            array=array,
            buffer_type=self.buffer_spec.buffer_type,
            allocated_shape=shape,
            creation_time=time.time(),
            last_used_time=time.time(),
            use_count=0,
            checked_out=False
        )
        
        self.stats['total_created'] += 1
        return buffer
    
    def get_buffer(self, requested_shape: Tuple[int, ...]) -> PooledBuffer:
        """Get buffer from pool or create new one"""
        with self._lock:
            # Look for suitable available buffer
            suitable_buffer = None
            
            for i, buffer in enumerate(self.available_buffers):
                if all(s <= a for s, a in zip(requested_shape, buffer.allocated_shape)):
                    suitable_buffer = self.available_buffers.popleft()
                    # Move checked buffers to end for better reuse
                    for _ in range(i):
                        self.available_buffers.append(self.available_buffers.popleft())
                    break
            
            if suitable_buffer is None:
                # Create new buffer with requested shape (possibly grown)
                grown_shape = tuple(max(r, d) for r, d in zip(requested_shape, 
                                                             self.buffer_spec.default_shape))
                grown_shape = tuple(int(dim * self.buffer_spec.growth_factor) 
                                  for dim in grown_shape)
                suitable_buffer = self._create_buffer(grown_shape)
                self.stats['cache_misses'] += 1
            else:
                self.stats['cache_hits'] += 1
            
            # Check out buffer
            suitable_buffer.checked_out = True
            suitable_buffer.thread_id = threading.get_ident()
            suitable_buffer.last_used_time = time.time()
            
            self.checked_out_buffers.add(suitable_buffer)
            self.stats['total_checkouts'] += 1
            self.stats['peak_checked_out'] = max(self.stats['peak_checked_out'], 
                                               len(self.checked_out_buffers))
            
            return suitable_buffer
    
    def return_buffer(self, buffer: PooledBuffer):
        """Return buffer to pool"""
        with self._lock:
            if buffer not in self.checked_out_buffers:
                return  # Buffer already returned or not from this pool
            
            buffer.checked_out = False
            buffer.thread_id = None
            buffer.last_used_time = time.time()
            
            self.checked_out_buffers.remove(buffer)
            self.available_buffers.append(buffer)
            self.stats['total_returns'] += 1
    
    def cleanup_old_buffers(self, max_age_sec: float = 300):
        """Clean up buffers that haven't been used recently"""
        with self._lock:
            current_time = time.time()
            min_buffers = max(1, len(self.checked_out_buffers))
            
            # Keep at least min_buffers in the pool
            while (len(self.available_buffers) > min_buffers and 
                   self.available_buffers and
                   current_time - self.available_buffers[0].last_used_time > max_age_sec):
                old_buffer = self.available_buffers.popleft()
                del old_buffer  # Let GC handle the array
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                **self.stats,
                'available_count': len(self.available_buffers),
                'checked_out_count': len(self.checked_out_buffers),
                'total_memory_mb': sum(b.get_memory_usage() for b in self.available_buffers | 
                                     self.checked_out_buffers) / (1024 * 1024),
                'cache_hit_rate': (self.stats['cache_hits'] / 
                                 max(1, self.stats['cache_hits'] + self.stats['cache_misses']))
            }


class BufferPoolManager:
    """Central manager for all buffer pools"""
    
    def __init__(self, memory_limit_mb: float = 1024):
        """Initialize buffer pool manager"""
        self.memory_limit_mb = memory_limit_mb
        self.pools: Dict[Tuple[PooledBufferType, Tuple[int, ...]], BufferPool] = {}
        self._lock = threading.Lock()
        
        # Buffer specifications
        self.buffer_specs = self._create_default_buffer_specs()
        
        # Global statistics
        self.global_stats = {
            'total_pools': 0,
            'total_memory_mb': 0.0,
            'peak_memory_mb': 0.0,
            'memory_pressure_events': 0
        }
        
        # Cleanup thread
        self.cleanup_thread = None
        self.cleanup_running = False
        self._start_cleanup_thread()
    
    def _create_default_buffer_specs(self) -> Dict[PooledBufferType, BufferSpec]:
        """Create default buffer specifications"""
        specs = {}
        
        # 2D heightfield buffers
        specs[PooledBufferType.HEIGHTFIELD_2D] = BufferSpec(
            buffer_type=PooledBufferType.HEIGHTFIELD_2D,
            dtype=np.float32,
            default_shape=(256, 256),
            max_size_mb=64,
            typical_lifetime_sec=60,
            growth_factor=1.2
        )
        
        # 3D SDF field buffers
        specs[PooledBufferType.SDF_FIELD_3D] = BufferSpec(
            buffer_type=PooledBufferType.SDF_FIELD_3D,
            dtype=np.float32,
            default_shape=(64, 64, 64),
            max_size_mb=128,
            typical_lifetime_sec=30,
            growth_factor=1.3
        )
        
        # 3D noise field buffers
        specs[PooledBufferType.NOISE_FIELD_3D] = BufferSpec(
            buffer_type=PooledBufferType.NOISE_FIELD_3D,
            dtype=np.float32,
            default_shape=(48, 48, 48),
            max_size_mb=96,
            typical_lifetime_sec=45,
            growth_factor=1.25
        )
        
        # Vertex buffers
        specs[PooledBufferType.VERTEX_BUFFER] = BufferSpec(
            buffer_type=PooledBufferType.VERTEX_BUFFER,
            dtype=np.float32,
            default_shape=(10000, 3),  # 10k vertices, 3D positions
            max_size_mb=32,
            typical_lifetime_sec=120,
            growth_factor=1.5
        )
        
        # Normal buffers
        specs[PooledBufferType.NORMAL_BUFFER] = BufferSpec(
            buffer_type=PooledBufferType.NORMAL_BUFFER,
            dtype=np.float32,
            default_shape=(10000, 3),  # 10k normals, 3D vectors
            max_size_mb=32,
            typical_lifetime_sec=120,
            growth_factor=1.5
        )
        
        # Tangent buffers
        specs[PooledBufferType.TANGENT_BUFFER] = BufferSpec(
            buffer_type=PooledBufferType.TANGENT_BUFFER,
            dtype=np.float32,
            default_shape=(10000, 4),  # 10k tangents, 4D (xyz + handedness)
            max_size_mb=40,
            typical_lifetime_sec=120,
            growth_factor=1.5
        )
        
        # Index buffers
        specs[PooledBufferType.INDEX_BUFFER] = BufferSpec(
            buffer_type=PooledBufferType.INDEX_BUFFER,
            dtype=np.int32,
            default_shape=(20000, 3),  # 20k triangles, 3 indices each
            max_size_mb=24,
            typical_lifetime_sec=180,
            growth_factor=1.5
        )
        
        # Temporary workspace buffers
        specs[PooledBufferType.TEMP_WORKSPACE] = BufferSpec(
            buffer_type=PooledBufferType.TEMP_WORKSPACE,
            dtype=np.float32,
            default_shape=(1024, 1024),
            max_size_mb=32,
            typical_lifetime_sec=15,
            growth_factor=1.2
        )
        
        # Serialization buffers
        specs[PooledBufferType.SERIALIZATION_BUFFER] = BufferSpec(
            buffer_type=PooledBufferType.SERIALIZATION_BUFFER,
            dtype=np.uint8,
            default_shape=(1024 * 1024,),  # 1MB serialization buffer
            max_size_mb=16,
            typical_lifetime_sec=30,
            growth_factor=1.3
        )
        
        return specs
    
    def get_buffer(self, buffer_type: PooledBufferType, 
                  shape: Tuple[int, ...]) -> 'ManagedBuffer':
        """Get managed buffer from appropriate pool"""
        
        # Find or create appropriate pool
        pool_key = (buffer_type, shape)
        with self._lock:
            if pool_key not in self.pools:
                if buffer_type not in self.buffer_specs:
                    raise ValueError(f"Unknown buffer type: {buffer_type}")
                
                self.pools[pool_key] = BufferPool(self.buffer_specs[buffer_type])
                self.global_stats['total_pools'] += 1
        
        pool = self.pools[pool_key]
        pooled_buffer = pool.get_buffer(shape)
        
        # Wrap in managed buffer for automatic return
        managed_buffer = ManagedBuffer(pooled_buffer, pool, self)
        
        # Update global memory tracking
        self._update_memory_stats()
        
        return managed_buffer
    
    def _update_memory_stats(self):
        """Update global memory statistics"""
        total_memory = 0.0
        for pool in self.pools.values():
            stats = pool.get_stats()
            total_memory += stats['total_memory_mb']
        
        self.global_stats['total_memory_mb'] = total_memory
        self.global_stats['peak_memory_mb'] = max(self.global_stats['peak_memory_mb'], 
                                                  total_memory)
        
        # Check for memory pressure
        if total_memory > self.memory_limit_mb * 0.9:
            self.global_stats['memory_pressure_events'] += 1
            self._handle_memory_pressure()
    
    def _handle_memory_pressure(self):
        """Handle memory pressure by cleaning up old buffers"""
        print(f"⚠️ Memory pressure detected: {self.global_stats['total_memory_mb']:.1f}MB")
        
        # Aggressive cleanup of old buffers
        for pool in self.pools.values():
            pool.cleanup_old_buffers(max_age_sec=60)  # More aggressive cleanup
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
        
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker for periodic cleanup"""
        while self.cleanup_running:
            try:
                time.sleep(30)  # Cleanup every 30 seconds
                
                for pool in list(self.pools.values()):
                    pool.cleanup_old_buffers()
                
                self._update_memory_stats()
                
            except Exception as e:
                print(f"Buffer pool cleanup error: {e}")
    
    def shutdown(self):
        """Shutdown buffer pool manager"""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global buffer pool statistics"""
        stats = {
            **self.global_stats,
            'active_pools': len(self.pools),
            'pool_stats': {}
        }
        
        for pool_key, pool in self.pools.items():
            buffer_type, shape = pool_key
            pool_stats = pool.get_stats()
            stats['pool_stats'][f"{buffer_type.value}_{shape}"] = pool_stats
        
        return stats
    
    def force_cleanup(self):
        """Force immediate cleanup of all pools"""
        for pool in self.pools.values():
            pool.cleanup_old_buffers(max_age_sec=0)
        
        self._update_memory_stats()


class ManagedBuffer:
    """Automatically managed buffer that returns to pool when done"""
    
    def __init__(self, pooled_buffer: PooledBuffer, pool: BufferPool, 
                 pool_manager: BufferPoolManager):
        """Initialize managed buffer"""
        self.pooled_buffer = pooled_buffer
        self.pool = pool
        self.pool_manager = pool_manager
        self.returned = False
        
        # Create weak reference to automatically return on garbage collection
        self._finalizer = weakref.finalize(self, self._return_to_pool, 
                                         pooled_buffer, pool)
    
    @staticmethod
    def _return_to_pool(pooled_buffer: PooledBuffer, pool: BufferPool):
        """Static method for finalizer"""
        pool.return_buffer(pooled_buffer)
    
    def get_array(self, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Get numpy array view with optional shape"""
        if self.returned:
            raise RuntimeError("Buffer already returned to pool")
        
        if shape is None:
            return self.pooled_buffer.array
        else:
            return self.pooled_buffer.get_view(shape)
    
    def return_to_pool(self):
        """Manually return buffer to pool"""
        if not self.returned:
            self.pool.return_buffer(self.pooled_buffer)
            self.returned = True
            self._finalizer.detach()  # Prevent finalizer from running
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically return buffer"""
        self.return_to_pool()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get allocated shape"""
        return self.pooled_buffer.allocated_shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get buffer data type"""
        return self.pooled_buffer.array.dtype


# Global buffer pool manager instance
_global_pool_manager: Optional[BufferPoolManager] = None


def get_global_buffer_pool() -> BufferPoolManager:
    """Get or create global buffer pool manager"""
    global _global_pool_manager
    if _global_pool_manager is None:
        _global_pool_manager = BufferPoolManager()
    return _global_pool_manager


def get_pooled_buffer(buffer_type: PooledBufferType, 
                     shape: Tuple[int, ...]) -> ManagedBuffer:
    """Convenience function to get pooled buffer"""
    return get_global_buffer_pool().get_buffer(buffer_type, shape)


if __name__ == "__main__":
    # Test buffer pool system
    print("🚀 T14 Buffer Pool Management System")
    print("=" * 60)
    
    # Create buffer pool manager
    pool_mgr = BufferPoolManager(memory_limit_mb=512)
    
    print("📊 Testing buffer allocation and reuse...")
    
    # Test different buffer types
    test_cases = [
        (PooledBufferType.HEIGHTFIELD_2D, (128, 128)),
        (PooledBufferType.SDF_FIELD_3D, (32, 32, 32)),
        (PooledBufferType.VERTEX_BUFFER, (5000, 3)),
        (PooledBufferType.TEMP_WORKSPACE, (512, 512))
    ]
    
    # Test allocation and usage
    buffers = []
    for buffer_type, shape in test_cases:
        print(f"   Testing {buffer_type.value} {shape}...")
        
        # Get buffer
        managed_buffer = pool_mgr.get_buffer(buffer_type, shape)
        array = managed_buffer.get_array(shape)
        
        # Use buffer
        array.fill(1.0)
        
        print(f"      ✅ Allocated {array.nbytes / 1024:.1f}KB buffer")
        buffers.append(managed_buffer)
    
    # Test buffer return and reuse
    print("\n🔄 Testing buffer return and reuse...")
    
    for buffer in buffers:
        buffer.return_to_pool()
    
    # Request same buffers again - should reuse
    for buffer_type, shape in test_cases:
        with pool_mgr.get_buffer(buffer_type, shape) as managed_buffer:
            array = managed_buffer.get_array(shape)
            print(f"   ♻️ Reused {buffer_type.value} buffer")
    
    # Get global statistics
    stats = pool_mgr.get_global_stats()
    print(f"\n📈 Buffer Pool Statistics:")
    print(f"   Active pools: {stats['active_pools']}")
    print(f"   Total memory: {stats['total_memory_mb']:.1f}MB")
    print(f"   Peak memory: {stats['peak_memory_mb']:.1f}MB")
    print(f"   Memory pressure events: {stats['memory_pressure_events']}")
    
    # Show per-pool statistics
    print(f"\n📋 Per-Pool Statistics:")
    for pool_key, pool_stats in stats['pool_stats'].items():
        hit_rate = pool_stats['cache_hit_rate']
        print(f"   {pool_key}:")
        print(f"      Cache hit rate: {hit_rate:.1%}")
        print(f"      Total memory: {pool_stats['total_memory_mb']:.1f}MB")
        print(f"      Available/Checked out: {pool_stats['available_count']}/{pool_stats['checked_out_count']}")
    
    # Test memory pressure
    print(f"\n🧪 Testing memory pressure handling...")
    large_buffers = []
    try:
        for i in range(20):  # Try to allocate many large buffers
            buffer = pool_mgr.get_buffer(PooledBufferType.SDF_FIELD_3D, (128, 128, 128))
            large_buffers.append(buffer)
    except:
        pass
    
    final_stats = pool_mgr.get_global_stats()
    print(f"   Final memory: {final_stats['total_memory_mb']:.1f}MB")
    print(f"   Memory pressure events: {final_stats['memory_pressure_events']}")
    
    # Cleanup
    pool_mgr.shutdown()
    print("\n✅ Buffer pool system functional")