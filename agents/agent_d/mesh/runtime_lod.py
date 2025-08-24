#!/usr/bin/env python3
"""
Runtime LOD Selection and Frustum Culling System - T08
======================================================

Implements dynamic Level of Detail selection based on:
1. Distance-based LOD bands
2. Screen-space error metrics  
3. Frustum culling with bounding sphere/AABB tests
4. Chunk streaming for VAO load/unload management

Features:
- Per-chunk bounding sphere and AABB calculations
- Camera-relative distance LOD selection
- Screen-space error estimation for smooth transitions
- Frustum plane extraction and intersection testing
- Simple chunk streamer for memory management
- Performance metrics and debugging tools

Usage:
    from runtime_lod import RuntimeLODManager
    
    lod_manager = RuntimeLODManager()
    active_chunks = lod_manager.select_lod_chunks(all_chunks, camera_pos, view_matrix, proj_matrix)
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import T06 and T07 dependencies  
sys.path.append(os.path.dirname(__file__))
from quadtree_chunking import QuadtreeNode


class LODLevel(Enum):
    """LOD level enumeration"""
    LOD0 = 0  # Highest detail
    LOD1 = 1  # Medium detail
    LOD2 = 2  # Low detail
    LOD3 = 3  # Lowest detail


@dataclass
class BoundingSphere:
    """Bounding sphere for chunk culling"""
    center: np.ndarray  # 3D center position
    radius: float       # Sphere radius


@dataclass  
class AABB:
    """Axis-aligned bounding box for chunk culling"""
    min_bounds: np.ndarray  # Minimum corner [x, y, z]
    max_bounds: np.ndarray  # Maximum corner [x, y, z]
    
    @property
    def center(self) -> np.ndarray:
        """Get AABB center"""
        return (self.min_bounds + self.max_bounds) * 0.5
    
    @property
    def size(self) -> np.ndarray:
        """Get AABB size"""
        return self.max_bounds - self.min_bounds
    
    def diagonal_length(self) -> float:
        """Get diagonal length for distance calculations"""
        return np.linalg.norm(self.size)


@dataclass
class FrustumPlane:
    """Frustum plane for culling tests"""
    normal: np.ndarray  # Plane normal vector
    distance: float     # Distance from origin


@dataclass  
class ChunkLODInfo:
    """LOD information for a chunk"""
    chunk_id: str
    lod_level: LODLevel
    distance_to_camera: float
    screen_space_error: float
    bounding_sphere: BoundingSphere
    aabb: AABB
    is_visible: bool = True
    should_load: bool = True


class RuntimeLODManager:
    """
    Manages runtime LOD selection and frustum culling for chunked terrain
    """
    
    def __init__(self, 
                 lod_distance_bands: List[float] = None,
                 screen_error_thresholds: List[float] = None,
                 max_chunks_per_frame: int = 100,
                 chunk_load_budget_ms: float = 2.0,
                 chunk_res_per_lod: Dict[LODLevel, int] = None):
        """
        Initialize runtime LOD manager
        
        Args:
            lod_distance_bands: Distance thresholds for LOD levels [LOD0->LOD1, LOD1->LOD2, LOD2->LOD3]
            screen_error_thresholds: Screen error thresholds for LOD transitions
            max_chunks_per_frame: Maximum chunks to render per frame
            chunk_load_budget_ms: Time budget for chunk loading per frame (ms)
            chunk_res_per_lod: Resolution per LOD level for mesh density control
        """
        # Tighter LOD for nearby-only rendering (R ≈ 50, so 0.7R ≈ 35, 1.5R ≈ 75, 3R ≈ 150)
        self.lod_distance_bands = lod_distance_bands or [35.0, 75.0, 150.0, 300.0]  # near≈0.7R, mid≈1.5R, far≈3R
        
        # Very strict screen error thresholds (≤ 1.5px)
        self.screen_error_thresholds = screen_error_thresholds or [0.5, 1.0, 1.5, 3.0]  # Tight screen error control
        
        # T19: Per-LOD chunk resolution for increased triangle density
        self.chunk_res_per_lod = chunk_res_per_lod or {
            LODLevel.LOD0: 128,  # Highest detail: 128x128 grid
            LODLevel.LOD1: 128,  # Still high detail: 128x128 grid  
            LODLevel.LOD2: 64,   # Medium detail: 64x64 grid
            LODLevel.LOD3: 32    # Low detail: 32x32 grid
        }
        
        self.max_chunks_per_frame = max_chunks_per_frame
        self.chunk_load_budget_ms = chunk_load_budget_ms
        
        # Cached frustum planes
        self.frustum_planes: List[FrustumPlane] = []
        
        # Chunk streaming state
        self.loaded_chunks: Set[str] = set()
        self.loading_queue: List[str] = []
        self.unloading_queue: List[str] = []
        
        # Performance metrics
        self.stats = {
            'total_chunks': 0,
            'visible_chunks': 0,
            'culled_chunks': 0,
            'active_lod_levels': {},
            'frame_time_ms': 0.0,
            'cull_time_ms': 0.0,
            'lod_time_ms': 0.0,
            'total_triangles': 0,
            'median_triangle_count': 0,
            'lod_histogram': {lod: 0 for lod in LODLevel}
        }

    def compute_bounding_sphere(self, chunk_data: Dict) -> BoundingSphere:
        """
        Compute bounding sphere for a chunk
        
        Args:
            chunk_data: Chunk mesh data with positions
            
        Returns:
            BoundingSphere for the chunk
        """
        positions = chunk_data.get("positions", np.array([]))
        
        if positions.size == 0:
            return BoundingSphere(np.zeros(3), 0.0)
        
        # Reshape to vertex array
        if positions.size % 3 != 0:
            return BoundingSphere(np.zeros(3), 0.0)
            
        vertices = positions.reshape(-1, 3)
        
        # Compute center as average of all vertices
        center = np.mean(vertices, axis=0)
        
        # Compute radius as maximum distance from center
        distances = np.linalg.norm(vertices - center, axis=1)
        radius = np.max(distances)
        
        return BoundingSphere(center, radius)

    def compute_aabb(self, chunk_data: Dict) -> AABB:
        """
        Compute axis-aligned bounding box for a chunk
        
        Args:
            chunk_data: Chunk mesh data with positions
            
        Returns:
            AABB for the chunk
        """
        positions = chunk_data.get("positions", np.array([]))
        
        if positions.size == 0:
            return AABB(np.zeros(3), np.zeros(3))
        
        # Use existing AABB if available
        chunk_info = chunk_data.get("chunk_info", {})
        if "aabb" in chunk_info:
            aabb_data = chunk_info["aabb"]
            min_bounds = np.array(aabb_data["min"])
            max_bounds = np.array(aabb_data["max"])
            return AABB(min_bounds, max_bounds)
        
        # Compute from positions
        if positions.size % 3 != 0:
            return AABB(np.zeros(3), np.zeros(3))
            
        vertices = positions.reshape(-1, 3)
        
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        return AABB(min_bounds, max_bounds)

    def extract_frustum_planes(self, view_matrix: np.ndarray, proj_matrix: np.ndarray) -> List[FrustumPlane]:
        """
        Extract frustum planes from view-projection matrix
        
        Args:
            view_matrix: 4x4 view matrix
            proj_matrix: 4x4 projection matrix
            
        Returns:
            List of 6 frustum planes [left, right, bottom, top, near, far]
        """
        # Combine view and projection matrices
        vp_matrix = proj_matrix @ view_matrix
        
        planes = []
        
        # Extract planes using standard method
        # Left plane: row4 + row1
        left_plane = vp_matrix[3, :] + vp_matrix[0, :]
        planes.append(self._normalize_plane(left_plane))
        
        # Right plane: row4 - row1
        right_plane = vp_matrix[3, :] - vp_matrix[0, :]
        planes.append(self._normalize_plane(right_plane))
        
        # Bottom plane: row4 + row2
        bottom_plane = vp_matrix[3, :] + vp_matrix[1, :]
        planes.append(self._normalize_plane(bottom_plane))
        
        # Top plane: row4 - row2
        top_plane = vp_matrix[3, :] - vp_matrix[1, :]
        planes.append(self._normalize_plane(top_plane))
        
        # Near plane: row4 + row3
        near_plane = vp_matrix[3, :] + vp_matrix[2, :]
        planes.append(self._normalize_plane(near_plane))
        
        # Far plane: row4 - row3
        far_plane = vp_matrix[3, :] - vp_matrix[2, :]
        planes.append(self._normalize_plane(far_plane))
        
        return planes

    def _normalize_plane(self, plane: np.ndarray) -> FrustumPlane:
        """Normalize a plane equation"""
        normal = plane[:3]
        length = np.linalg.norm(normal)
        
        if length < 1e-8:
            return FrustumPlane(np.array([0, 1, 0]), 0.0)
        
        normal = normal / length
        distance = plane[3] / length
        
        return FrustumPlane(normal, distance)

    def test_sphere_frustum(self, sphere: BoundingSphere, frustum_planes: List[FrustumPlane]) -> bool:
        """
        Test if bounding sphere intersects frustum
        
        Args:
            sphere: Bounding sphere to test
            frustum_planes: List of frustum planes
            
        Returns:
            True if sphere is visible (intersects or inside frustum)
        """
        for plane in frustum_planes:
            # Distance from sphere center to plane
            distance = np.dot(plane.normal, sphere.center) + plane.distance
            
            # If sphere is completely outside any plane, it's culled
            if distance < -sphere.radius:
                return False
        
        return True

    def test_aabb_frustum(self, aabb: AABB, frustum_planes: List[FrustumPlane]) -> bool:
        """
        Test if AABB intersects frustum
        
        Args:
            aabb: AABB to test
            frustum_planes: List of frustum planes
            
        Returns:
            True if AABB is visible (intersects or inside frustum)
        """
        for plane in frustum_planes:
            # Find AABB corner closest to plane (in negative normal direction)
            closest_corner = np.where(plane.normal < 0, aabb.max_bounds, aabb.min_bounds)
            
            # Distance from closest corner to plane
            distance = np.dot(plane.normal, closest_corner) + plane.distance
            
            # If closest corner is outside plane, entire AABB is outside
            if distance < 0:
                return False
        
        return True

    def calculate_distance_to_camera(self, chunk_center: np.ndarray, camera_pos: np.ndarray) -> float:
        """Calculate distance from chunk center to camera"""
        return np.linalg.norm(chunk_center - camera_pos)

    def calculate_screen_space_error(self, chunk_aabb: AABB, distance: float, 
                                   fov_y: float, screen_height: int) -> float:
        """
        Calculate screen-space error for chunk
        
        Args:
            chunk_aabb: Chunk bounding box
            distance: Distance to camera
            fov_y: Field of view in radians
            screen_height: Screen height in pixels
            
        Returns:
            Estimated screen-space error in pixels
        """
        if distance < 1e-6:
            return float('inf')
        
        # Estimate chunk size as diagonal of AABB
        chunk_size = chunk_aabb.diagonal_length()
        
        # Project size to screen space
        # screen_size = (chunk_size / distance) * (screen_height / (2 * tan(fov_y / 2)))
        tan_half_fov = math.tan(fov_y * 0.5)
        screen_size = (chunk_size / distance) * (screen_height / (2 * tan_half_fov))
        
        return screen_size

    def select_lod_level(self, distance: float, screen_error: float) -> LODLevel:
        """
        Select appropriate LOD level based on distance and screen error with improved thresholds
        
        Args:
            distance: Distance to camera
            screen_error: Screen-space error in pixels
            
        Returns:
            Selected LOD level
        """
        # T19: Use screen error as primary metric with improved thresholds
        # Prioritize screen-space error for perceptually-driven LOD selection
        
        if screen_error > self.screen_error_thresholds[3]:
            return LODLevel.LOD3
        elif screen_error > self.screen_error_thresholds[2]:
            return LODLevel.LOD2  
        elif screen_error > self.screen_error_thresholds[1]:
            return LODLevel.LOD1
        else:
            return LODLevel.LOD0
        
        # Distance-based fallback (secondary, with tighter bands)
        if distance > self.lod_distance_bands[3]:
            return LODLevel.LOD3
        elif distance > self.lod_distance_bands[2]:
            return LODLevel.LOD2
        elif distance > self.lod_distance_bands[1]:
            return LODLevel.LOD1
        else:
            return LODLevel.LOD0
    
    def get_chunk_resolution(self, lod_level: LODLevel) -> int:
        """
        Get chunk resolution for given LOD level
        
        Args:
            lod_level: LOD level to get resolution for
            
        Returns:
            Grid resolution (N×N) for this LOD level
        """
        return self.chunk_res_per_lod.get(lod_level, 32)

    def select_active_chunks(self, all_chunks: List[Dict], camera_pos: np.ndarray,
                           view_matrix: np.ndarray, proj_matrix: np.ndarray,
                           fov_y: float = math.radians(60), screen_height: int = 600) -> List[ChunkLODInfo]:
        """
        Select active chunks with LOD and frustum culling
        
        Args:
            all_chunks: List of all available chunks
            camera_pos: Camera position in world space
            view_matrix: 4x4 view matrix
            proj_matrix: 4x4 projection matrix  
            fov_y: Field of view in radians
            screen_height: Screen height in pixels
            
        Returns:
            List of selected chunk LOD info
        """
        import time
        start_time = time.time()
        
        # Extract frustum planes
        self.frustum_planes = self.extract_frustum_planes(view_matrix, proj_matrix)
        
        cull_start = time.time()
        
        selected_chunks = []
        lod_counts = {lod: 0 for lod in LODLevel}
        
        for chunk in all_chunks:
            chunk_info = chunk.get("chunk_info", {})
            # Handle both unified format (chunk_id at top level) and legacy format (nested in chunk_info)
            chunk_id = chunk.get("chunk_id") or chunk_info.get("chunk_id", "unknown")
            
            # Compute bounding volumes
            bounding_sphere = self.compute_bounding_sphere(chunk)
            aabb = self.compute_aabb(chunk)
            
            # Frustum culling using sphere test (faster)
            is_visible = self.test_sphere_frustum(bounding_sphere, self.frustum_planes)
            
            if not is_visible:
                continue
            
            # Calculate distance and screen error
            distance = self.calculate_distance_to_camera(bounding_sphere.center, camera_pos)
            screen_error = self.calculate_screen_space_error(aabb, distance, fov_y, screen_height)
            
            # Select LOD level
            lod_level = self.select_lod_level(distance, screen_error)
            lod_counts[lod_level] += 1
            
            # Create chunk LOD info
            chunk_lod_info = ChunkLODInfo(
                chunk_id=chunk_id,
                lod_level=lod_level,
                distance_to_camera=distance,
                screen_space_error=screen_error,
                bounding_sphere=bounding_sphere,
                aabb=aabb,
                is_visible=is_visible,
                should_load=True
            )
            
            selected_chunks.append(chunk_lod_info)
        
        cull_time = time.time() - cull_start
        
        # Sort by distance (render closest first)
        selected_chunks.sort(key=lambda x: x.distance_to_camera)
        
        # Apply chunk limit
        if len(selected_chunks) > self.max_chunks_per_frame:
            selected_chunks = selected_chunks[:self.max_chunks_per_frame]
        
        # T19: Calculate triangle statistics for HUD display
        total_triangles = 0
        triangle_counts = []
        
        for chunk_info in selected_chunks:
            # Estimate triangles based on LOD resolution
            chunk_res = self.get_chunk_resolution(chunk_info.lod_level)
            triangles_per_chunk = (chunk_res - 1) * (chunk_res - 1) * 2  # 2 triangles per quad
            total_triangles += triangles_per_chunk
            triangle_counts.append(triangles_per_chunk)
        
        # Calculate median triangle count
        median_triangles = 0
        if triangle_counts:
            triangle_counts.sort()
            n = len(triangle_counts)
            median_triangles = triangle_counts[n // 2] if n % 2 == 1 else (triangle_counts[n // 2 - 1] + triangle_counts[n // 2]) // 2
        
        # Update performance stats with triangle information
        total_time = time.time() - start_time
        self.stats.update({
            'total_chunks': len(all_chunks),
            'visible_chunks': len(selected_chunks),
            'culled_chunks': len(all_chunks) - len(selected_chunks),
            'active_lod_levels': dict(lod_counts),
            'cull_time_ms': cull_time * 1000,
            'lod_time_ms': total_time * 1000,
            'total_triangles': total_triangles,
            'median_triangle_count': median_triangles,
            'lod_histogram': dict(lod_counts)
        })
        
        return selected_chunks

    def get_lod_statistics(self) -> Dict:
        """Get current LOD selection statistics"""
        return self.stats.copy()

    def should_stream_chunk(self, chunk_lod_info: ChunkLODInfo) -> Tuple[bool, bool]:
        """
        Determine if chunk should be loaded or unloaded
        
        Args:
            chunk_lod_info: Chunk LOD information
            
        Returns:
            Tuple of (should_load, should_unload)
        """
        chunk_id = chunk_lod_info.chunk_id
        is_loaded = chunk_id in self.loaded_chunks
        
        if chunk_lod_info.is_visible and chunk_lod_info.should_load:
            # Should load if visible and not loaded
            return not is_loaded, False
        else:
            # Should unload if not visible but loaded
            return False, is_loaded

    def update_chunk_streaming(self, selected_chunks: List[ChunkLODInfo]):
        """
        Update chunk streaming queues based on selected chunks
        
        Args:
            selected_chunks: List of selected chunks with LOD info
        """
        # Clear previous queues
        self.loading_queue.clear()
        self.unloading_queue.clear()
        
        # Build set of chunks that should be loaded
        should_be_loaded = {chunk.chunk_id for chunk in selected_chunks if chunk.should_load}
        
        # Find chunks to load
        for chunk in selected_chunks:
            if chunk.chunk_id not in self.loaded_chunks:
                self.loading_queue.append(chunk.chunk_id)
        
        # Find chunks to unload
        for loaded_chunk in self.loaded_chunks:
            if loaded_chunk not in should_be_loaded:
                self.unloading_queue.append(loaded_chunk)

    def mark_chunk_loaded(self, chunk_id: str):
        """Mark a chunk as loaded"""
        self.loaded_chunks.add(chunk_id)

    def mark_chunk_unloaded(self, chunk_id: str):
        """Mark a chunk as unloaded"""
        self.loaded_chunks.discard(chunk_id)


def create_view_matrix(camera_pos: np.ndarray, camera_target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Create view matrix from camera parameters
    
    Args:
        camera_pos: Camera position
        camera_target: Camera look-at target
        up: Up vector
        
    Returns:
        4x4 view matrix
    """
    forward = camera_target - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    view_matrix = np.eye(4)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up
    view_matrix[2, :3] = -forward
    view_matrix[3, :3] = -camera_pos
    
    return view_matrix


def create_projection_matrix(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Create perspective projection matrix
    
    Args:
        fov_y: Field of view in radians
        aspect: Aspect ratio (width/height)
        near: Near clipping plane
        far: Far clipping plane
        
    Returns:
        4x4 projection matrix
    """
    f = 1.0 / math.tan(fov_y * 0.5)
    
    proj_matrix = np.zeros((4, 4))
    proj_matrix[0, 0] = f / aspect
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2 * far * near / (far - near)
    proj_matrix[3, 2] = -1
    
    return proj_matrix


if __name__ == "__main__":
    # Example usage
    print("🚀 T08 Runtime LOD Selection System")
    print("=" * 50)
    
    # Initialize LOD manager
    lod_manager = RuntimeLODManager(
        lod_distance_bands=[3.0, 10.0, 25.0, 60.0],
        screen_error_thresholds=[0.5, 2.0, 6.0, 24.0]
    )
    
    print("✅ Runtime LOD manager initialized")
    print(f"   LOD distance bands: {lod_manager.lod_distance_bands}")
    print(f"   Screen error thresholds: {lod_manager.screen_error_thresholds}")
    print(f"   Max chunks per frame: {lod_manager.max_chunks_per_frame}")