#!/usr/bin/env python3
"""
Async Streaming System - T14
============================

Implements background I/O for chunk manifests and buffers with async streaming
to enable smooth runtime performance for big worlds. Provides non-blocking
loading, caching, and progressive streaming of terrain data.

Features:
- Asynchronous chunk manifest and buffer loading
- Background I/O thread pool with priority queuing
- Smart caching with LRU eviction and preloading
- Progressive streaming based on camera position
- Non-blocking API with callback-based completion

Usage:
    from async_streaming import AsyncStreamingManager
    
    streamer = AsyncStreamingManager()
    streamer.stream_chunk_async("chunk_0_0", callback=on_chunk_loaded)
"""

import asyncio
import threading
import queue
import time
import json
import gzip
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import os
import weakref
from collections import OrderedDict

# Import buffer pool system
sys.path.append(os.path.dirname(__file__))
from buffer_pools import BufferPoolManager, PooledBufferType, get_global_buffer_pool

# Import T13 deterministic systems  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baking'))
from deterministic_baking import BufferMetadata, BakeManifest


class StreamingPriority(Enum):
    """Priority levels for streaming operations"""
    IMMEDIATE = 4      # Currently visible chunks
    HIGH = 3          # About to be visible
    NORMAL = 2        # Nearby chunks for preloading  
    LOW = 1           # Distant chunks for background loading
    BACKGROUND = 0    # Cleanup and maintenance


class StreamingOperation(Enum):
    """Types of streaming operations"""
    LOAD_MANIFEST = "load_manifest"
    LOAD_BUFFER = "load_buffer"
    DECOMPRESS_BUFFER = "decompress_buffer"
    PRELOAD_CHUNK = "preload_chunk"
    EVICT_CHUNK = "evict_chunk"
    VALIDATE_BUFFER = "validate_buffer"


@dataclass
class StreamRequest:
    """Request for streaming operation"""
    operation: StreamingOperation
    priority: StreamingPriority
    chunk_id: str
    buffer_id: Optional[str] = None
    file_path: Optional[Path] = None
    callback: Optional[Callable] = None
    user_data: Any = None
    submit_time: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """Priority queue comparison - higher priority first"""
        return self.priority.value > other.priority.value


@dataclass
class ChunkData:
    """Complete chunk data in memory"""
    chunk_id: str
    manifest: Optional[Dict[str, Any]] = None
    buffers: Dict[str, Any] = field(default_factory=dict)
    load_time: float = field(default_factory=time.time)
    access_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    fully_loaded: bool = False
    loading_in_progress: Set[str] = field(default_factory=set)


@dataclass
class StreamingStats:
    """Statistics for streaming operations"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_loaded: int = 0
    bytes_evicted: int = 0
    avg_load_time_ms: float = 0.0
    background_queue_size: int = 0
    active_operations: int = 0


class LRUCache:
    """Least Recently Used cache for chunk data"""
    
    def __init__(self, max_size_mb: float = 512):
        """Initialize LRU cache"""
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.current_size_bytes = 0
        self.cache: OrderedDict[str, ChunkData] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, chunk_id: str) -> Optional[ChunkData]:
        """Get chunk data and mark as recently used"""
        with self._lock:
            if chunk_id in self.cache:
                # Move to end (most recently used)
                chunk_data = self.cache.pop(chunk_id)
                chunk_data.access_time = time.time()
                self.cache[chunk_id] = chunk_data
                return chunk_data
            return None
    
    def put(self, chunk_id: str, chunk_data: ChunkData):
        """Put chunk data in cache, evicting old entries if needed"""
        with self._lock:
            # Remove existing entry if present
            if chunk_id in self.cache:
                old_data = self.cache.pop(chunk_id)
                self.current_size_bytes -= old_data.size_bytes
            
            # Add new entry
            self.cache[chunk_id] = chunk_data
            self.current_size_bytes += chunk_data.size_bytes
            
            # Evict old entries if over size limit
            self._evict_if_needed()
    
    def _evict_if_needed(self):
        """Evict least recently used entries if over size limit"""
        evicted_bytes = 0
        
        while (self.current_size_bytes > self.max_size_bytes and 
               len(self.cache) > 1):  # Keep at least one entry
            
            # Remove least recently used (first in OrderedDict)
            chunk_id, chunk_data = self.cache.popitem(last=False)
            evicted_size = chunk_data.size_bytes
            self.current_size_bytes -= evicted_size
            evicted_bytes += evicted_size
            
            print(f"🗑️ Evicted chunk {chunk_id} ({evicted_size / 1024:.1f}KB)")
        
        return evicted_bytes
    
    def remove(self, chunk_id: str) -> bool:
        """Remove specific chunk from cache"""
        with self._lock:
            if chunk_id in self.cache:
                chunk_data = self.cache.pop(chunk_id)
                self.current_size_bytes -= chunk_data.size_bytes
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'cached_chunks': len(self.cache),
                'current_size_mb': self.current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self.current_size_bytes / self.max_size_bytes
            }


class AsyncStreamingManager:
    """Manages asynchronous streaming of terrain data"""
    
    def __init__(self, data_dir: Union[str, Path] = None, 
                 max_cache_mb: float = 512, max_workers: int = 4):
        """Initialize async streaming manager"""
        self.data_dir = Path(data_dir) if data_dir else Path("baked_data")
        self.cache = LRUCache(max_cache_mb)
        self.buffer_pool = get_global_buffer_pool()
        
        # Threading infrastructure
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.request_queue = queue.PriorityQueue()
        self.active_futures: Dict[str, Future] = {}
        
        # Background processing
        self.background_thread = None
        self.background_running = False
        self._start_background_processor()
        
        # Streaming state
        self.camera_position = (0.0, 0.0, 0.0)
        self.stream_radius = 200.0
        self.preload_radius = 300.0
        self.chunk_size = 4.0  # Size of each chunk in world units
        
        # Statistics
        self.stats = StreamingStats()
        self._lock = threading.RLock()
        
        # Callbacks
        self.chunk_loaded_callbacks = []
        self.chunk_evicted_callbacks = []
    
    def _start_background_processor(self):
        """Start background request processing thread"""
        if self.background_thread and self.background_thread.is_alive():
            return
        
        self.background_running = True
        self.background_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.background_thread.start()
    
    def _background_processor(self):
        """Background thread for processing streaming requests"""
        while self.background_running:
            try:
                # Get request with timeout
                try:
                    request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process request
                future = self.executor.submit(self._process_request, request)
                
                with self._lock:
                    self.active_futures[request.chunk_id] = future
                    self.stats.active_operations += 1
                
                # Handle completion in background
                def on_complete(fut):
                    with self._lock:
                        self.active_futures.pop(request.chunk_id, None)
                        self.stats.active_operations -= 1
                    
                    try:
                        result = fut.result()
                        self.stats.completed_requests += 1
                        
                        if request.callback:
                            request.callback(result, request.user_data)
                    except Exception as e:
                        self.stats.failed_requests += 1
                        print(f"❌ Streaming request failed: {e}")
                
                future.add_done_callback(on_complete)
                
            except Exception as e:
                print(f"Background processor error: {e}")
    
    def _process_request(self, request: StreamRequest) -> Any:
        """Process individual streaming request"""
        start_time = time.time()
        
        try:
            if request.operation == StreamingOperation.LOAD_MANIFEST:
                result = self._load_manifest(request.chunk_id)
            elif request.operation == StreamingOperation.LOAD_BUFFER:
                result = self._load_buffer(request.chunk_id, request.buffer_id)
            elif request.operation == StreamingOperation.PRELOAD_CHUNK:
                result = self._preload_chunk(request.chunk_id)
            elif request.operation == StreamingOperation.EVICT_CHUNK:
                result = self._evict_chunk(request.chunk_id)
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
            
            # Update performance stats
            load_time_ms = (time.time() - start_time) * 1000
            with self._lock:
                self.stats.avg_load_time_ms = (
                    (self.stats.avg_load_time_ms * self.stats.completed_requests + load_time_ms) /
                    (self.stats.completed_requests + 1)
                )
            
            return result
            
        except Exception as e:
            print(f"Request processing error: {e}")
            raise
    
    def _load_manifest(self, chunk_id: str) -> Dict[str, Any]:
        """Load chunk manifest from disk"""
        manifest_path = self.data_dir / f"{chunk_id}_manifest.sha256.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        # Load and parse manifest
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        # Cache manifest
        chunk_data = self.cache.get(chunk_id)
        if chunk_data is None:
            chunk_data = ChunkData(chunk_id=chunk_id)
        
        chunk_data.manifest = manifest_data
        chunk_data.size_bytes += len(json.dumps(manifest_data).encode('utf-8'))
        self.cache.put(chunk_id, chunk_data)
        
        return manifest_data
    
    def _load_buffer(self, chunk_id: str, buffer_id: str) -> Any:
        """Load specific buffer from disk"""
        buffer_path = self.data_dir / f"{chunk_id}_{buffer_id}.bin"
        
        if not buffer_path.exists():
            raise FileNotFoundError(f"Buffer not found: {buffer_path}")
        
        # Load buffer data
        with open(buffer_path, 'rb') as f:
            buffer_data = f.read()
        
        # Update statistics
        with self._lock:
            self.stats.bytes_loaded += len(buffer_data)
        
        # Decompress if needed (simplified)
        if buffer_id.endswith('_compressed'):
            buffer_data = gzip.decompress(buffer_data)
        
        # Cache buffer
        chunk_data = self.cache.get(chunk_id)
        if chunk_data is None:
            chunk_data = ChunkData(chunk_id=chunk_id)
        
        chunk_data.buffers[buffer_id] = buffer_data
        chunk_data.size_bytes += len(buffer_data)
        self.cache.put(chunk_id, chunk_data)
        
        return buffer_data
    
    def _preload_chunk(self, chunk_id: str) -> ChunkData:
        """Preload complete chunk data"""
        # Load manifest first
        manifest = self._load_manifest(chunk_id)
        
        # Load all buffers
        chunk_data = self.cache.get(chunk_id)
        if chunk_data is None:
            chunk_data = ChunkData(chunk_id=chunk_id, manifest=manifest)
        
        # Load each buffer mentioned in manifest
        buffers = manifest.get('buffers', {})
        for buffer_id in buffers.keys():
            if buffer_id not in chunk_data.buffers:
                try:
                    buffer_data = self._load_buffer(chunk_id, buffer_id)
                    chunk_data.buffers[buffer_id] = buffer_data
                except FileNotFoundError:
                    print(f"⚠️ Buffer not found: {buffer_id}")
        
        chunk_data.fully_loaded = True
        self.cache.put(chunk_id, chunk_data)
        
        # Notify callbacks
        for callback in self.chunk_loaded_callbacks:
            try:
                callback(chunk_id, chunk_data)
            except Exception as e:
                print(f"Chunk loaded callback error: {e}")
        
        return chunk_data
    
    def _evict_chunk(self, chunk_id: str) -> bool:
        """Evict chunk from cache"""
        removed = self.cache.remove(chunk_id)
        
        if removed:
            # Notify callbacks
            for callback in self.chunk_evicted_callbacks:
                try:
                    callback(chunk_id)
                except Exception as e:
                    print(f"Chunk evicted callback error: {e}")
        
        return removed
    
    def stream_chunk_async(self, chunk_id: str, priority: StreamingPriority = StreamingPriority.NORMAL,
                          callback: Optional[Callable] = None, user_data: Any = None) -> bool:
        """Request asynchronous chunk streaming"""
        
        # Check if already in cache
        chunk_data = self.cache.get(chunk_id)
        if chunk_data and chunk_data.fully_loaded:
            if callback:
                callback(chunk_data, user_data)
            with self._lock:
                self.stats.cache_hits += 1
            return True
        
        # Submit streaming request
        request = StreamRequest(
            operation=StreamingOperation.PRELOAD_CHUNK,
            priority=priority,
            chunk_id=chunk_id,
            callback=callback,
            user_data=user_data
        )
        
        self.request_queue.put(request)
        
        with self._lock:
            self.stats.total_requests += 1
            self.stats.cache_misses += 1
            self.stats.background_queue_size = self.request_queue.qsize()
        
        return False  # Not immediately available
    
    def get_chunk_sync(self, chunk_id: str, timeout: float = 5.0) -> Optional[ChunkData]:
        """Get chunk synchronously with timeout"""
        # Check cache first
        chunk_data = self.cache.get(chunk_id)
        if chunk_data and chunk_data.fully_loaded:
            return chunk_data
        
        # Load synchronously
        try:
            return self._preload_chunk(chunk_id)
        except Exception as e:
            print(f"Synchronous chunk load failed: {e}")
            return None
    
    def update_camera_position(self, position: Tuple[float, float, float]):
        """Update camera position for streaming priority"""
        self.camera_position = position
        
        # Trigger adaptive streaming based on new position
        self._update_streaming_priorities()
    
    def _update_streaming_priorities(self):
        """Update streaming priorities based on camera position"""
        cx, cy, cz = self.camera_position
        
        # Calculate which chunks should be loaded
        chunk_x_start = int((cx - self.stream_radius) // self.chunk_size)
        chunk_x_end = int((cx + self.stream_radius) // self.chunk_size) + 1
        chunk_z_start = int((cz - self.stream_radius) // self.chunk_size)
        chunk_z_end = int((cz + self.stream_radius) // self.chunk_size) + 1
        
        # Queue high-priority chunks
        for x in range(chunk_x_start, chunk_x_end):
            for z in range(chunk_z_start, chunk_z_end):
                chunk_id = f"{x}_{z}"
                
                # Calculate distance to determine priority
                chunk_center_x = x * self.chunk_size + self.chunk_size / 2
                chunk_center_z = z * self.chunk_size + self.chunk_size / 2
                distance = ((cx - chunk_center_x)**2 + (cz - chunk_center_z)**2)**0.5
                
                # Determine priority based on distance
                if distance < self.stream_radius * 0.3:
                    priority = StreamingPriority.IMMEDIATE
                elif distance < self.stream_radius * 0.6:
                    priority = StreamingPriority.HIGH
                elif distance < self.stream_radius:
                    priority = StreamingPriority.NORMAL
                else:
                    continue  # Too far, don't load
                
                # Check if already loaded or loading
                chunk_data = self.cache.get(chunk_id)
                if chunk_data and chunk_data.fully_loaded:
                    continue
                
                # Queue for loading
                self.stream_chunk_async(chunk_id, priority)
        
        # Queue eviction of distant chunks
        self._queue_distant_chunk_eviction(cx, cz)
    
    def _queue_distant_chunk_eviction(self, cx: float, cz: float):
        """Queue eviction of chunks that are too distant"""
        eviction_radius = self.preload_radius * 1.5
        
        # Check cached chunks for eviction
        for chunk_id in list(self.cache.cache.keys()):
            try:
                # Parse chunk coordinates
                x_str, z_str = chunk_id.split('_')
                chunk_x, chunk_z = int(x_str), int(z_str)
                
                # Calculate distance
                chunk_center_x = chunk_x * self.chunk_size + self.chunk_size / 2
                chunk_center_z = chunk_z * self.chunk_size + self.chunk_size / 2
                distance = ((cx - chunk_center_x)**2 + (cz - chunk_center_z)**2)**0.5
                
                # Queue for eviction if too far
                if distance > eviction_radius:
                    request = StreamRequest(
                        operation=StreamingOperation.EVICT_CHUNK,
                        priority=StreamingPriority.BACKGROUND,
                        chunk_id=chunk_id
                    )
                    self.request_queue.put(request)
                    
            except ValueError:
                # Skip chunks with non-standard IDs
                continue
    
    def add_chunk_loaded_callback(self, callback: Callable[[str, ChunkData], None]):
        """Add callback for when chunks are loaded"""
        self.chunk_loaded_callbacks.append(callback)
    
    def add_chunk_evicted_callback(self, callback: Callable[[str], None]):
        """Add callback for when chunks are evicted"""
        self.chunk_evicted_callbacks.append(callback)
    
    def get_streaming_stats(self) -> StreamingStats:
        """Get current streaming statistics"""
        with self._lock:
            self.stats.background_queue_size = self.request_queue.qsize()
            return StreamingStats(**self.stats.__dict__)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def shutdown(self):
        """Shutdown streaming manager"""
        self.background_running = False
        
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        # Cancel active futures
        for future in self.active_futures.values():
            future.cancel()
        
        self.executor.shutdown(wait=False)


if __name__ == "__main__":
    # Test async streaming system
    print("🚀 T14 Async Streaming System")
    print("=" * 60)
    
    # Create streaming manager
    streamer = AsyncStreamingManager(max_cache_mb=128, max_workers=2)
    
    # Test callback
    loaded_chunks = []
    def on_chunk_loaded(chunk_data, user_data):
        loaded_chunks.append(chunk_data.chunk_id)
        print(f"   ✅ Chunk loaded: {chunk_data.chunk_id}")
    
    print("📊 Testing async chunk streaming...")
    
    # Test streaming requests (these will fail without actual data files)
    test_chunks = ["0_0", "0_1", "1_0", "1_1"]
    
    for chunk_id in test_chunks:
        print(f"   Requesting chunk {chunk_id}...")
        streamer.stream_chunk_async(
            chunk_id, 
            StreamingPriority.HIGH, 
            callback=on_chunk_loaded,
            user_data={'test': True}
        )
    
    # Wait a bit for background processing
    time.sleep(2)
    
    # Test camera position update
    print("\n🎥 Testing camera position update...")
    streamer.update_camera_position((10.0, 0.0, 10.0))
    
    # Get statistics
    stats = streamer.get_streaming_stats()
    cache_stats = streamer.get_cache_stats()
    
    print(f"\n📈 Streaming Statistics:")
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Completed: {stats.completed_requests}")
    print(f"   Failed: {stats.failed_requests}")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache misses: {stats.cache_misses}")
    print(f"   Background queue: {stats.background_queue_size}")
    print(f"   Active operations: {stats.active_operations}")
    
    print(f"\n💾 Cache Statistics:")
    print(f"   Cached chunks: {cache_stats['cached_chunks']}")
    print(f"   Cache size: {cache_stats['current_size_mb']:.1f}MB")
    print(f"   Cache utilization: {cache_stats['utilization']:.1%}")
    
    print(f"\n🎯 Loaded chunks: {loaded_chunks}")
    
    # Cleanup
    streamer.shutdown()
    print("\n✅ Async streaming system functional (with expected failures for missing data)")