#!/usr/bin/env python3
"""
Chunk Streamer for VAO Management - T08
=======================================

Handles loading and unloading of chunk VAOs based on LOD selection.
Manages GPU memory by dynamically creating and destroying OpenGL resources.

Features:
- Async chunk loading with time budgets
- VAO creation and destruction
- Memory usage tracking
- Performance metrics
- LRU cache for chunk data

Usage:
    from chunk_streamer import ChunkStreamer
    
    streamer = ChunkStreamer()
    streamer.load_chunk(chunk_data)
    vao_id = streamer.get_chunk_vao(chunk_id)
    streamer.unload_chunk(chunk_id)
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
import sys
import os

try:
    from OpenGL.GL import *
    from OpenGL.arrays import vbo
    OPENGL_AVAILABLE = True
except ImportError:
    print("⚠️ OpenGL not available for chunk streaming")
    OPENGL_AVAILABLE = False


@dataclass
class ChunkVAO:
    """OpenGL VAO data for a chunk"""
    vao_id: int
    vertex_count: int
    triangle_count: int
    memory_usage: int  # Estimated GPU memory in bytes
    load_time: float   # Time when loaded
    last_used: float   # Last access time


@dataclass
class StreamingStats:
    """Chunk streaming performance statistics"""
    total_chunks_loaded: int = 0
    total_chunks_unloaded: int = 0
    active_vaos: int = 0
    gpu_memory_usage: int = 0
    load_time_ms: float = 0.0
    unload_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class ChunkStreamer:
    """
    Manages streaming of chunk VAOs for runtime LOD system
    """
    
    def __init__(self, 
                 max_memory_mb: float = 256.0,
                 max_active_chunks: int = 200,
                 load_budget_ms: float = 5.0):
        """
        Initialize chunk streamer
        
        Args:
            max_memory_mb: Maximum GPU memory to use for chunks (MB)
            max_active_chunks: Maximum number of active chunk VAOs
            load_budget_ms: Time budget for loading per frame (ms)
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_active_chunks = max_active_chunks
        self.load_budget_ms = load_budget_ms
        
        # VAO storage
        self.chunk_vaos: Dict[str, ChunkVAO] = {}
        self.vao_access_order = OrderedDict()  # For LRU eviction
        
        # Chunk data cache
        self.chunk_data_cache: Dict[str, Dict] = {}
        self.cache_access_order = OrderedDict()
        self.max_cache_size = 50  # Keep chunk data for fast reload
        
        # Loading queues
        self.load_queue: List[str] = []
        self.unload_queue: List[str] = []
        self.loading_in_progress: Set[str] = set()
        
        # Statistics
        self.stats = StreamingStats()
        
        print(f"🔧 Chunk Streamer initialized:")
        print(f"   Max GPU memory: {max_memory_mb:.1f} MB")
        print(f"   Max active chunks: {max_active_chunks}")
        print(f"   Load budget: {load_budget_ms:.1f} ms")

    def estimate_chunk_memory_usage(self, chunk_data: Dict) -> int:
        """
        Estimate GPU memory usage for a chunk
        
        Args:
            chunk_data: Chunk mesh data
            
        Returns:
            Estimated memory usage in bytes
        """
        positions = chunk_data.get("positions", np.array([]))
        normals = chunk_data.get("normals", np.array([]))
        indices = chunk_data.get("indices", np.array([]))
        uvs = chunk_data.get("uv0", np.array([]))
        tangents = chunk_data.get("tangents", np.array([]))
        
        # Calculate buffer sizes
        position_size = positions.nbytes if hasattr(positions, 'nbytes') else len(positions) * 4
        normal_size = normals.nbytes if hasattr(normals, 'nbytes') else len(normals) * 4
        index_size = indices.nbytes if hasattr(indices, 'nbytes') else len(indices) * 4
        uv_size = uvs.nbytes if hasattr(uvs, 'nbytes') else len(uvs) * 4
        tangent_size = tangents.nbytes if hasattr(tangents, 'nbytes') else len(tangents) * 4
        
        # Add VAO overhead
        total_size = position_size + normal_size + index_size + uv_size + tangent_size + 1024
        
        return total_size

    def create_chunk_vao(self, chunk_id: str, chunk_data: Dict) -> Optional[ChunkVAO]:
        """
        Create VAO for chunk data
        
        Args:
            chunk_id: Unique chunk identifier
            chunk_data: Chunk mesh data with positions, normals, indices
            
        Returns:
            ChunkVAO instance or None if failed
        """
        if not OPENGL_AVAILABLE:
            return None
        
        try:
            start_time = time.time()
            
            positions = chunk_data.get("positions", np.array([]))
            normals = chunk_data.get("normals", np.array([]))
            indices = chunk_data.get("indices", np.array([]))
            uvs = chunk_data.get("uv0", np.array([]))
            tangents = chunk_data.get("tangents", np.array([]))
            
            if positions.size == 0 or indices.size == 0:
                print(f"⚠️ Chunk {chunk_id} has no geometry data")
                return None
            
            # Ensure data is float32/uint32
            if hasattr(positions, 'astype'):
                positions = positions.astype(np.float32)
            if hasattr(normals, 'astype'):
                normals = normals.astype(np.float32) 
            if hasattr(indices, 'astype'):
                indices = indices.astype(np.uint32)
            if hasattr(uvs, 'astype'):
                uvs = uvs.astype(np.float32)
            if hasattr(tangents, 'astype'):
                tangents = tangents.astype(np.float32)
            
            # Generate VAO
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            
            # Position buffer (attribute 0)
            if positions.size > 0:
                pos_buffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, pos_buffer)
                glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            
            # Normal buffer (attribute 1)
            if normals.size > 0:
                norm_buffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, norm_buffer)
                glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
            
            # UV buffer (attribute 2)
            if uvs.size > 0:
                uv_buffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, uv_buffer)
                glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
                glEnableVertexAttribArray(2)
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
            
            # Tangent buffer (attribute 3)
            if tangents.size > 0:
                tan_buffer = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, tan_buffer)
                glBufferData(GL_ARRAY_BUFFER, tangents.nbytes, tangents, GL_STATIC_DRAW)
                glEnableVertexAttribArray(3)
                glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, None)
            
            # Index buffer
            index_buffer = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            
            # Unbind VAO
            glBindVertexArray(0)
            
            # Calculate metrics
            vertex_count = len(positions) // 3 if positions.size > 0 else 0
            triangle_count = len(indices) // 3 if indices.size > 0 else 0
            memory_usage = self.estimate_chunk_memory_usage(chunk_data)
            
            load_time = time.time() - start_time
            self.stats.load_time_ms += load_time * 1000
            
            return ChunkVAO(
                vao_id=vao,
                vertex_count=vertex_count,
                triangle_count=triangle_count,
                memory_usage=memory_usage,
                load_time=time.time(),
                last_used=time.time()
            )
            
        except Exception as e:
            print(f"❌ Failed to create VAO for chunk {chunk_id}: {e}")
            return None

    def destroy_chunk_vao(self, chunk_vao: ChunkVAO):
        """
        Destroy VAO and free GPU resources
        
        Args:
            chunk_vao: ChunkVAO to destroy
        """
        if not OPENGL_AVAILABLE:
            return
        
        try:
            start_time = time.time()
            
            # Delete VAO
            if glIsVertexArray(chunk_vao.vao_id):
                glDeleteVertexArrays(1, [chunk_vao.vao_id])
            
            unload_time = time.time() - start_time
            self.stats.unload_time_ms += unload_time * 1000
            
        except Exception as e:
            print(f"⚠️ Error destroying VAO {chunk_vao.vao_id}: {e}")

    def load_chunk_from_file(self, chunk_path: Path) -> Optional[Dict]:
        """
        Load chunk data from file
        
        Args:
            chunk_path: Path to chunk manifest file
            
        Returns:
            Chunk data dictionary or None if failed
        """
        try:
            chunk_dir = chunk_path.parent
            
            with open(chunk_path, 'r') as f:
                manifest = json.load(f)
            
            mesh_data = manifest.get("mesh", {})
            chunk_info = manifest.get("chunk", {})
            
            # Load binary buffers
            chunk_data = {"chunk_info": chunk_info}
            
            for attr_name, buffer_ref in mesh_data.items():
                if buffer_ref.startswith("buffer://"):
                    buffer_file = chunk_dir / buffer_ref.replace("buffer://", "")
                    if buffer_file.exists():
                        if attr_name == "indices":
                            data = np.fromfile(str(buffer_file), dtype=np.uint32)
                        else:
                            data = np.fromfile(str(buffer_file), dtype=np.float32)
                        chunk_data[attr_name] = data
            
            return chunk_data
            
        except Exception as e:
            print(f"❌ Failed to load chunk from {chunk_path}: {e}")
            return None

    def cache_chunk_data(self, chunk_id: str, chunk_data: Dict):
        """
        Cache chunk data for fast reloading
        
        Args:
            chunk_id: Chunk identifier
            chunk_data: Chunk data to cache
        """
        # Remove if already exists (to update access order)
        if chunk_id in self.chunk_data_cache:
            del self.chunk_data_cache[chunk_id]
            del self.cache_access_order[chunk_id]
        
        # Add to cache
        self.chunk_data_cache[chunk_id] = chunk_data
        self.cache_access_order[chunk_id] = time.time()
        
        # Evict oldest if cache is full
        while len(self.chunk_data_cache) > self.max_cache_size:
            oldest_id = next(iter(self.cache_access_order))
            del self.chunk_data_cache[oldest_id]
            del self.cache_access_order[oldest_id]

    def get_cached_chunk_data(self, chunk_id: str) -> Optional[Dict]:
        """
        Get cached chunk data
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Cached chunk data or None
        """
        if chunk_id in self.chunk_data_cache:
            # Update access time
            self.cache_access_order[chunk_id] = time.time()
            self.stats.cache_hits += 1
            return self.chunk_data_cache[chunk_id]
        else:
            self.stats.cache_misses += 1
            return None

    def load_chunk(self, chunk_id: str, chunk_data: Optional[Dict] = None, 
                  chunk_path: Optional[Path] = None) -> bool:
        """
        Load a chunk and create its VAO
        
        Args:
            chunk_id: Unique chunk identifier
            chunk_data: Chunk data (optional, will load from path if not provided)
            chunk_path: Path to chunk file (optional, for loading from disk)
            
        Returns:
            True if successfully loaded
        """
        if chunk_id in self.chunk_vaos:
            # Already loaded, update access time
            self.vao_access_order[chunk_id] = time.time()
            self.chunk_vaos[chunk_id].last_used = time.time()
            return True
        
        # Check if we need to free memory first
        if not self._ensure_memory_available():
            print(f"⚠️ Cannot load chunk {chunk_id}: insufficient memory")
            return False
        
        # Get chunk data
        if chunk_data is None:
            # Try cache first
            chunk_data = self.get_cached_chunk_data(chunk_id)
            
            # Load from file if not cached
            if chunk_data is None and chunk_path is not None:
                chunk_data = self.load_chunk_from_file(chunk_path)
                if chunk_data:
                    self.cache_chunk_data(chunk_id, chunk_data)
        
        if chunk_data is None:
            print(f"❌ No data available for chunk {chunk_id}")
            return False
        
        # Create VAO
        chunk_vao = self.create_chunk_vao(chunk_id, chunk_data)
        if chunk_vao is None:
            return False
        
        # Store VAO
        self.chunk_vaos[chunk_id] = chunk_vao
        self.vao_access_order[chunk_id] = time.time()
        
        # Update statistics
        self.stats.total_chunks_loaded += 1
        self.stats.active_vaos += 1
        self.stats.gpu_memory_usage += chunk_vao.memory_usage
        
        return True

    def unload_chunk(self, chunk_id: str) -> bool:
        """
        Unload a chunk and destroy its VAO
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if successfully unloaded
        """
        if chunk_id not in self.chunk_vaos:
            return False
        
        chunk_vao = self.chunk_vaos[chunk_id]
        
        # Destroy VAO
        self.destroy_chunk_vao(chunk_vao)
        
        # Remove from storage
        del self.chunk_vaos[chunk_id]
        if chunk_id in self.vao_access_order:
            del self.vao_access_order[chunk_id]
        
        # Update statistics
        self.stats.total_chunks_unloaded += 1
        self.stats.active_vaos -= 1
        self.stats.gpu_memory_usage -= chunk_vao.memory_usage
        
        return True

    def get_chunk_vao(self, chunk_id: str) -> Optional[int]:
        """
        Get VAO ID for a chunk
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            VAO ID or None if not loaded
        """
        if chunk_id not in self.chunk_vaos:
            return None
        
        # Update access time for LRU
        self.vao_access_order[chunk_id] = time.time()
        self.chunk_vaos[chunk_id].last_used = time.time()
        
        return self.chunk_vaos[chunk_id].vao_id

    def _ensure_memory_available(self) -> bool:
        """
        Ensure memory is available by evicting LRU chunks if necessary
        
        Returns:
            True if memory is available
        """
        while (self.stats.gpu_memory_usage > self.max_memory_bytes or 
               len(self.chunk_vaos) >= self.max_active_chunks):
            
            if not self.vao_access_order:
                break
            
            # Find LRU chunk
            lru_chunk_id = min(self.vao_access_order.keys(), 
                             key=lambda x: self.vao_access_order[x])
            
            print(f"🗑️ Evicting LRU chunk: {lru_chunk_id}")
            self.unload_chunk(lru_chunk_id)
        
        return True

    def update_streaming(self, load_queue: List[str], unload_queue: List[str], 
                        chunk_data_provider = None):
        """
        Update streaming based on LOD selection
        
        Args:
            load_queue: List of chunk IDs to load
            unload_queue: List of chunk IDs to unload  
            chunk_data_provider: Function to get chunk data by ID
        """
        start_time = time.time()
        
        # Process unload queue (fast)
        for chunk_id in unload_queue:
            self.unload_chunk(chunk_id)
        
        # Process load queue with time budget
        for chunk_id in load_queue:
            if chunk_id in self.chunk_vaos:
                continue  # Already loaded
            
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.load_budget_ms:
                break  # Exceeded time budget
            
            # Get chunk data from provider
            chunk_data = None
            if chunk_data_provider:
                chunk_data = chunk_data_provider(chunk_id)
            
            self.load_chunk(chunk_id, chunk_data)

    def get_streaming_stats(self) -> StreamingStats:
        """Get current streaming statistics"""
        return self.stats

    def get_memory_usage_mb(self) -> float:
        """Get current GPU memory usage in MB"""
        return self.stats.gpu_memory_usage / (1024 * 1024)

    def cleanup(self):
        """Clean up all VAOs and free resources"""
        chunk_ids = list(self.chunk_vaos.keys())
        for chunk_id in chunk_ids:
            self.unload_chunk(chunk_id)
        
        self.chunk_data_cache.clear()
        self.cache_access_order.clear()


if __name__ == "__main__":
    # Example usage
    print("🚀 T08 Chunk Streamer")
    print("=" * 30)
    
    # Initialize streamer
    streamer = ChunkStreamer(
        max_memory_mb=128.0,
        max_active_chunks=100,
        load_budget_ms=3.0
    )
    
    print("✅ Chunk streamer initialized")
    
    # Example streaming stats
    stats = streamer.get_streaming_stats()
    print(f"📊 Streaming Stats:")
    print(f"   Active VAOs: {stats.active_vaos}")
    print(f"   GPU Memory: {streamer.get_memory_usage_mb():.1f} MB")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache misses: {stats.cache_misses}")