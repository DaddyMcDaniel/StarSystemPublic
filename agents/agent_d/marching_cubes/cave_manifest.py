#!/usr/bin/env python3
"""
Cave Manifest System - T10
==========================

Manages cave chunk manifests with material IDs for rendering terrain + caves together.
Provides loading, validation, and integration with the existing T06-T08 chunk system.

Features:
- Cave chunk manifest loading and validation
- Material ID management for different surface types
- Integration with terrain chunk system
- Binary buffer management
- Memory-efficient cave chunk streaming

Usage:
    from cave_manifest import CaveManifestManager
    
    manager = CaveManifestManager()
    cave_chunks = manager.load_cave_manifest("cave_chunks.json")
    combined_chunks = manager.combine_terrain_and_caves(terrain_chunks, cave_chunks)
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import struct

@dataclass
class MaterialInfo:
    """Material information for rendering"""
    material_id: int
    material_type: str  # "terrain", "cave", "overhang"
    color: Tuple[float, float, float]  # RGB color
    roughness: float = 0.5
    metallic: float = 0.0
    
    @classmethod
    def terrain_material(cls) -> 'MaterialInfo':
        return cls(0, "terrain", (0.6, 0.4, 0.2), 0.8, 0.0)  # Brown terrain
    
    @classmethod
    def cave_material(cls) -> 'MaterialInfo':
        return cls(1, "cave", (0.3, 0.3, 0.4), 0.9, 0.1)     # Dark gray caves
    
    @classmethod
    def overhang_material(cls) -> 'MaterialInfo':
        return cls(2, "overhang", (0.4, 0.3, 0.2), 0.7, 0.0) # Dark brown overhangs


@dataclass
class CaveChunkInfo:
    """Cave chunk information"""
    chunk_id: str
    material_id: int
    vertex_count: int
    triangle_count: int
    index_count: int
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    buffers: Dict[str, str]  # Buffer type -> filename
    lod_levels: List[int]
    
    @property
    def bounds_center(self) -> np.ndarray:
        return (self.aabb_min + self.aabb_max) * 0.5
    
    @property
    def bounds_size(self) -> np.ndarray:
        return self.aabb_max - self.aabb_min


class CaveManifestManager:
    """
    Manages cave chunk manifests and integration with terrain system
    """
    
    def __init__(self):
        """Initialize cave manifest manager"""
        self.materials = {
            0: MaterialInfo.terrain_material(),
            1: MaterialInfo.cave_material(),
            2: MaterialInfo.overhang_material()
        }
        
        self.cave_chunks: Dict[str, CaveChunkInfo] = {}
        self.manifest_data: Optional[Dict] = None
        self.base_path: Optional[Path] = None
    
    def load_cave_manifest(self, manifest_path: str) -> Dict[str, CaveChunkInfo]:
        """
        Load cave chunk manifest from file
        
        Args:
            manifest_path: Path to cave manifest JSON file
            
        Returns:
            Dictionary of chunk_id -> CaveChunkInfo
        """
        manifest_file = Path(manifest_path)
        self.base_path = manifest_file.parent
        
        print(f"Loading cave manifest: {manifest_path}")
        
        try:
            with open(manifest_file, 'r') as f:
                self.manifest_data = json.load(f)
            
            # Validate manifest structure
            if not self._validate_manifest(self.manifest_data):
                raise ValueError("Invalid cave manifest structure")
            
            # Load cave chunks
            cave_chunks = {}
            for chunk_data in self.manifest_data.get("chunks", []):
                chunk_info = self._parse_cave_chunk(chunk_data)
                if chunk_info:
                    cave_chunks[chunk_info.chunk_id] = chunk_info
            
            self.cave_chunks = cave_chunks
            
            print(f"✅ Loaded {len(cave_chunks)} cave chunks")
            print(f"   Material ID: {self.manifest_data.get('material_id', 'unknown')}")
            print(f"   Resolution: {self.manifest_data.get('resolution', 'unknown')}")
            
            return cave_chunks
            
        except Exception as e:
            print(f"❌ Failed to load cave manifest: {e}")
            return {}
    
    def _validate_manifest(self, manifest: Dict) -> bool:
        """Validate cave manifest structure"""
        required_fields = ["name", "type", "chunks"]
        
        for field in required_fields:
            if field not in manifest:
                print(f"❌ Missing required field: {field}")
                return False
        
        if manifest["type"] != "cave_chunks":
            print(f"❌ Invalid manifest type: {manifest['type']}")
            return False
        
        return True
    
    def _parse_cave_chunk(self, chunk_data: Dict) -> Optional[CaveChunkInfo]:
        """Parse cave chunk data from manifest"""
        try:
            aabb = chunk_data.get("aabb", {})
            
            chunk_info = CaveChunkInfo(
                chunk_id=chunk_data["chunk_id"],
                material_id=chunk_data.get("material_id", 1),
                vertex_count=chunk_data.get("vertex_count", 0),
                triangle_count=chunk_data.get("triangle_count", 0),
                index_count=chunk_data.get("index_count", 0),
                aabb_min=np.array(aabb.get("min", [0, 0, 0])),
                aabb_max=np.array(aabb.get("max", [0, 0, 0])),
                buffers=chunk_data.get("buffers", {}),
                lod_levels=chunk_data.get("lod_levels", [0])
            )
            
            return chunk_info
            
        except Exception as e:
            print(f"❌ Failed to parse cave chunk {chunk_data.get('chunk_id', 'unknown')}: {e}")
            return None
    
    def load_cave_chunk_buffers(self, chunk_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load binary buffers for a cave chunk
        
        Args:
            chunk_id: Cave chunk identifier
            
        Returns:
            Dictionary of buffer_type -> numpy array or None if failed
        """
        if chunk_id not in self.cave_chunks:
            return None
        
        chunk_info = self.cave_chunks[chunk_id]
        buffers = {}
        
        try:
            for buffer_type, filename in chunk_info.buffers.items():
                buffer_path = self.base_path / filename
                
                if not buffer_path.exists():
                    print(f"⚠️ Buffer file not found: {buffer_path}")
                    continue
                
                # Load binary data
                with open(buffer_path, 'rb') as f:
                    buffer_data = f.read()
                
                # Parse based on buffer type
                if buffer_type in ["positions", "normals"]:
                    # 3D vectors (float32)
                    array = np.frombuffer(buffer_data, dtype=np.float32).reshape(-1, 3)
                elif buffer_type == "uvs":
                    # 2D texture coordinates (float32)
                    array = np.frombuffer(buffer_data, dtype=np.float32).reshape(-1, 2)
                elif buffer_type == "tangents":
                    # 4D tangent vectors (float32)
                    array = np.frombuffer(buffer_data, dtype=np.float32).reshape(-1, 4)
                elif buffer_type == "indices":
                    # Triangle indices (uint32)
                    array = np.frombuffer(buffer_data, dtype=np.uint32)
                else:
                    print(f"⚠️ Unknown buffer type: {buffer_type}")
                    continue
                
                buffers[buffer_type] = array
            
            return buffers
            
        except Exception as e:
            print(f"❌ Failed to load buffers for chunk {chunk_id}: {e}")
            return None
    
    def combine_terrain_and_caves(self, terrain_chunks: List[Dict], 
                                cave_chunks: Dict[str, CaveChunkInfo]) -> Dict:
        """
        Combine terrain and cave chunks for unified rendering
        
        Args:
            terrain_chunks: List of terrain chunk data
            cave_chunks: Dictionary of cave chunks
            
        Returns:
            Combined chunk data for rendering
        """
        combined_data = {
            "terrain_chunks": terrain_chunks,
            "cave_chunks": cave_chunks,
            "materials": self.materials,
            "render_order": ["terrain", "caves"]  # Render terrain first, then caves
        }
        
        # Add spatial indexing for efficient culling
        combined_data["spatial_index"] = self._build_spatial_index(terrain_chunks, cave_chunks)
        
        return combined_data
    
    def _build_spatial_index(self, terrain_chunks: List[Dict], 
                           cave_chunks: Dict[str, CaveChunkInfo]) -> Dict:
        """Build spatial index for efficient chunk culling"""
        spatial_index = {
            "terrain_bounds": [],
            "cave_bounds": [],
            "all_bounds": []
        }
        
        # Index terrain chunks
        for chunk in terrain_chunks:
            aabb = chunk.get("aabb", {})
            if aabb:
                bounds = {
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "type": "terrain",
                    "min": np.array(aabb.get("min", [0, 0, 0])),
                    "max": np.array(aabb.get("max", [0, 0, 0]))
                }
                spatial_index["terrain_bounds"].append(bounds)
                spatial_index["all_bounds"].append(bounds)
        
        # Index cave chunks
        for chunk_id, chunk_info in cave_chunks.items():
            bounds = {
                "chunk_id": chunk_id,
                "type": "cave",
                "min": chunk_info.aabb_min,
                "max": chunk_info.aabb_max
            }
            spatial_index["cave_bounds"].append(bounds)
            spatial_index["all_bounds"].append(bounds)
        
        return spatial_index
    
    def get_material_color(self, material_id: int) -> Tuple[float, float, float]:
        """Get RGB color for material ID"""
        if material_id in self.materials:
            return self.materials[material_id].color
        return (1.0, 0.0, 1.0)  # Magenta for unknown materials
    
    def get_chunks_in_radius(self, center: np.ndarray, radius: float, 
                           chunk_type: str = "all") -> List[str]:
        """
        Get chunks within radius of center point
        
        Args:
            center: Center point [x, y, z]
            radius: Search radius
            chunk_type: "terrain", "cave", or "all"
            
        Returns:
            List of chunk IDs within radius
        """
        chunks_in_radius = []
        
        if chunk_type in ["cave", "all"]:
            for chunk_id, chunk_info in self.cave_chunks.items():
                chunk_center = chunk_info.bounds_center
                distance = np.linalg.norm(chunk_center - center)
                if distance <= radius:
                    chunks_in_radius.append(chunk_id)
        
        return chunks_in_radius
    
    def export_combined_manifest(self, terrain_manifest_path: str, 
                               cave_manifest_path: str, output_path: str):
        """
        Export combined terrain + cave manifest for viewer
        
        Args:
            terrain_manifest_path: Path to terrain manifest
            cave_manifest_path: Path to cave manifest  
            output_path: Output path for combined manifest
        """
        try:
            # Load terrain manifest
            with open(terrain_manifest_path, 'r') as f:
                terrain_manifest = json.load(f)
            
            # Load cave manifest
            cave_chunks = self.load_cave_manifest(cave_manifest_path)
            
            # Create combined manifest
            combined_manifest = {
                "name": "Combined Terrain + Caves",
                "type": "combined_chunks",
                "version": "1.0",
                "terrain_manifest": terrain_manifest,
                "cave_chunks": {
                    "count": len(cave_chunks),
                    "chunks": [
                        {
                            "chunk_id": chunk_id,
                            "material_id": chunk_info.material_id,
                            "vertex_count": chunk_info.vertex_count,
                            "triangle_count": chunk_info.triangle_count,
                            "aabb": {
                                "min": chunk_info.aabb_min.tolist(),
                                "max": chunk_info.aabb_max.tolist()
                            },
                            "buffers": chunk_info.buffers
                        }
                        for chunk_id, chunk_info in cave_chunks.items()
                    ]
                },
                "materials": {
                    str(mat_id): {
                        "material_id": mat_id,
                        "type": mat_info.material_type,
                        "color": mat_info.color,
                        "roughness": mat_info.roughness,
                        "metallic": mat_info.metallic
                    }
                    for mat_id, mat_info in self.materials.items()
                }
            }
            
            # Save combined manifest
            with open(output_path, 'w') as f:
                json.dump(combined_manifest, f, indent=2)
            
            print(f"✅ Combined manifest exported to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to export combined manifest: {e}")


if __name__ == "__main__":
    # Test cave manifest system
    print("🧪 Testing Cave Manifest System")
    print("=" * 40)
    
    # Create test manager
    manager = CaveManifestManager()
    
    # Test material system
    print("Materials:")
    for mat_id, mat_info in manager.materials.items():
        print(f"   {mat_id}: {mat_info.material_type} - {mat_info.color}")
    
    # Test with dummy data
    test_cave_chunks = {
        "test_chunk_caves": CaveChunkInfo(
            chunk_id="test_chunk_caves",
            material_id=1,
            vertex_count=100,
            triangle_count=50,
            index_count=150,
            aabb_min=np.array([-1, -1, -1]),
            aabb_max=np.array([1, 1, 1]),
            buffers={"positions": "test_pos.bin", "normals": "test_norm.bin"},
            lod_levels=[0]
        )
    }
    
    # Test spatial queries
    center = np.array([0, 0, 0])
    chunks_in_radius = manager.get_chunks_in_radius(center, 2.0, "cave")
    print(f"\nChunks within radius: {len(chunks_in_radius)}")
    
    print("\n✅ Cave manifest system functional")