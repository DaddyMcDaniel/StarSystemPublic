#!/usr/bin/env python3
"""
Marching Cubes Debug Visualization - T20
=======================================

Debug tools for T20 Marching Cubes improvements including:
- Voxel cell size visualization
- Cave seam strip visualization
- Normal quality visualization
- Chunk overlap verification

Features:
- Debug toggle for voxel cell rendering
- Seam alignment verification
- Normal direction visualization
- Resolution comparison tools
- Cave quality metrics

Usage:
    from marching_cubes_debug import MarchingCubesDebugger
    
    debugger = MarchingCubesDebugger()
    debugger.toggle_voxel_visualization(True)
    debugger.render_debug_overlays(cave_mesh, screen)
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import sys
import os

# Import marching cubes system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marching_cubes'))
from marching_cubes import MarchingCubesVertex, CaveMeshData


class DebugMode(Enum):
    """Debug visualization modes"""
    VOXEL_CELLS = "voxel_cells"
    SEAM_STRIPS = "seam_strips"  
    NORMAL_VECTORS = "normal_vectors"
    RESOLUTION_HEATMAP = "resolution_heatmap"
    OVERLAP_ZONES = "overlap_zones"


class MarchingCubesDebugger:
    """
    Debug visualization system for T20 Marching Cubes improvements
    """
    
    def __init__(self):
        """Initialize Marching Cubes debugger"""
        self.enabled_modes: Dict[DebugMode, bool] = {
            DebugMode.VOXEL_CELLS: False,
            DebugMode.SEAM_STRIPS: False,
            DebugMode.NORMAL_VECTORS: False,
            DebugMode.RESOLUTION_HEATMAP: False,
            DebugMode.OVERLAP_ZONES: False
        }
        
        # Debug colors
        self.debug_colors = {
            'voxel_grid': (100, 100, 255, 128),      # Semi-transparent blue
            'seam_strip': (255, 100, 100, 200),      # Red highlight
            'normal_good': (0, 255, 0, 255),         # Green normals
            'normal_bad': (255, 0, 0, 255),          # Red normals
            'overlap_zone': (255, 255, 0, 150),      # Yellow overlay
            'high_res': (0, 255, 0, 200),            # Green for high resolution
            'low_res': (255, 0, 0, 200),             # Red for low resolution
        }
        
        # Statistics
        self.debug_stats = {
            'voxel_cells_rendered': 0,
            'seam_strips_rendered': 0,
            'normal_vectors_rendered': 0,
            'overlap_zones_rendered': 0,
            'total_debug_draw_calls': 0
        }
        
        # Initialize pygame font for debug text
        pygame.font.init()
        self.font = pygame.font.Font(None, 12)
    
    def toggle_debug_mode(self, mode: DebugMode, enabled: bool = None):
        """
        Toggle or set debug visualization mode
        
        Args:
            mode: Debug mode to toggle
            enabled: Force enable/disable (None toggles current state)
        """
        if enabled is None:
            self.enabled_modes[mode] = not self.enabled_modes[mode]
        else:
            self.enabled_modes[mode] = enabled
            
        print(f"🔧 Debug mode {mode.value}: {'ON' if self.enabled_modes[mode] else 'OFF'}")
    
    def is_enabled(self, mode: DebugMode) -> bool:
        """Check if debug mode is enabled"""
        return self.enabled_modes.get(mode, False)
    
    def render_voxel_cells(self, surface: pygame.Surface, voxel_grid_info: Dict[str, Any], 
                          camera_transform: np.ndarray):
        """
        T20: Render voxel cell grid for size visualization
        
        Args:
            surface: Pygame surface to render to
            voxel_grid_info: Voxel grid information (bounds, resolution, cell_size)
            camera_transform: 4x4 camera transformation matrix
        """
        if not self.is_enabled(DebugMode.VOXEL_CELLS):
            return
        
        resolution = voxel_grid_info.get('resolution', 32)
        cell_size = voxel_grid_info.get('cell_size', 0.1)
        bounds_min = np.array(voxel_grid_info.get('bounds_min', [-1, -1, -1]))
        bounds_max = np.array(voxel_grid_info.get('bounds_max', [1, 1, 1]))
        
        # Render subset of voxel cells (every Nth cell to avoid performance issues)
        step = max(1, resolution // 8)  # Show ~8x8x8 grid maximum
        cells_drawn = 0
        
        for i in range(0, resolution, step):
            for j in range(0, resolution, step):
                for k in range(0, resolution, step):
                    # Calculate voxel center
                    voxel_center = bounds_min + np.array([
                        (i + 0.5) * cell_size,
                        (j + 0.5) * cell_size,
                        (k + 0.5) * cell_size
                    ])
                    
                    # Transform to screen space
                    screen_pos = self._world_to_screen(voxel_center, camera_transform, surface)
                    
                    if screen_pos is not None:
                        # Draw voxel cell as wireframe cube
                        self._draw_wireframe_cube(surface, screen_pos, cell_size * 0.8, 
                                                self.debug_colors['voxel_grid'])
                        cells_drawn += 1
        
        self.debug_stats['voxel_cells_rendered'] = cells_drawn
    
    def render_seam_strips(self, surface: pygame.Surface, cave_mesh: CaveMeshData,
                          overlap_info: Dict[str, Any], camera_transform: np.ndarray):
        """
        T20: Render seam strips to verify alignment between chunks
        
        Args:
            surface: Pygame surface to render to
            cave_mesh: Cave mesh data
            overlap_info: Information about chunk overlaps
            camera_transform: Camera transformation matrix
        """
        if not self.is_enabled(DebugMode.SEAM_STRIPS):
            return
            
        if not cave_mesh or len(cave_mesh.vertices) == 0:
            return
        
        overlap_width = overlap_info.get('overlap_voxels', 1) * overlap_info.get('voxel_size', 0.1)
        chunk_bounds = cave_mesh.bounds
        
        # Find vertices in overlap regions (near chunk boundaries)
        seam_vertices = []
        
        for vertex in cave_mesh.vertices:
            pos = vertex.position
            
            # Check if vertex is near any chunk boundary (within overlap width)
            near_boundary = False
            for axis in range(3):
                if (abs(pos[axis] - chunk_bounds.min_point[axis]) < overlap_width or 
                    abs(pos[axis] - chunk_bounds.max_point[axis]) < overlap_width):
                    near_boundary = True
                    break
            
            if near_boundary:
                seam_vertices.append(vertex)
        
        # Render seam vertices with special highlighting
        strips_drawn = 0
        for vertex in seam_vertices:
            screen_pos = self._world_to_screen(vertex.position, camera_transform, surface)
            if screen_pos is not None:
                # Draw highlighted vertex
                pygame.draw.circle(surface, self.debug_colors['seam_strip'], 
                                 screen_pos, 3)
                
                # Draw normal vector from vertex
                if len(vertex.normal) == 3:
                    normal_end = vertex.position + vertex.normal * 0.1
                    normal_screen = self._world_to_screen(normal_end, camera_transform, surface)
                    if normal_screen is not None:
                        pygame.draw.line(surface, self.debug_colors['seam_strip'],
                                       screen_pos, normal_screen, 2)
                strips_drawn += 1
        
        self.debug_stats['seam_strips_rendered'] = strips_drawn
    
    def render_normal_vectors(self, surface: pygame.Surface, cave_mesh: CaveMeshData,
                            camera_transform: np.ndarray, quality_threshold: float = 0.9):
        """
        Render vertex normals with quality color coding
        
        Args:
            surface: Pygame surface to render to
            cave_mesh: Cave mesh data
            camera_transform: Camera transformation matrix
            quality_threshold: Threshold for "good" normal quality
        """
        if not self.is_enabled(DebugMode.NORMAL_VECTORS):
            return
            
        if not cave_mesh or len(cave_mesh.vertices) == 0:
            return
        
        normals_drawn = 0
        normal_length = 0.15  # Visual length of normal vectors
        
        for vertex in cave_mesh.vertices:
            screen_pos = self._world_to_screen(vertex.position, camera_transform, surface)
            if screen_pos is None:
                continue
            
            # Calculate normal quality (length should be ~1.0 for good normals)
            normal_quality = np.linalg.norm(vertex.normal)
            
            # Choose color based on quality
            color = (self.debug_colors['normal_good'] if normal_quality > quality_threshold 
                    else self.debug_colors['normal_bad'])
            
            # Calculate normal end position
            normal_end = vertex.position + vertex.normal * normal_length
            normal_screen = self._world_to_screen(normal_end, camera_transform, surface)
            
            if normal_screen is not None:
                # Draw normal vector as line
                pygame.draw.line(surface, color, screen_pos, normal_screen, 2)
                
                # Draw arrow head
                self._draw_arrow_head(surface, screen_pos, normal_screen, color)
                normals_drawn += 1
        
        self.debug_stats['normal_vectors_rendered'] = normals_drawn
    
    def render_resolution_heatmap(self, surface: pygame.Surface, chunks_info: List[Dict[str, Any]],
                                camera_transform: np.ndarray):
        """
        Render resolution heatmap showing voxel density across chunks
        
        Args:
            surface: Pygame surface to render to
            chunks_info: List of chunk information with resolution data
            camera_transform: Camera transformation matrix
        """
        if not self.is_enabled(DebugMode.RESOLUTION_HEATMAP):
            return
        
        for chunk_info in chunks_info:
            resolution = chunk_info.get('voxel_resolution', 32)
            bounds = chunk_info.get('bounds', {})
            chunk_center = np.array(bounds.get('center', [0, 0, 0]))
            
            screen_pos = self._world_to_screen(chunk_center, camera_transform, surface)
            if screen_pos is None:
                continue
            
            # Color based on resolution (high res = green, low res = red)
            resolution_factor = min(1.0, resolution / 64.0)  # Normalize to 64 as max
            color_r = int(255 * (1 - resolution_factor))
            color_g = int(255 * resolution_factor)
            color = (color_r, color_g, 0, 200)
            
            # Draw resolution indicator
            radius = max(5, int(10 * resolution_factor))
            pygame.draw.circle(surface, color, screen_pos, radius)
            
            # Draw resolution text
            res_text = self.font.render(f"{resolution}", True, (255, 255, 255))
            text_pos = (screen_pos[0] - res_text.get_width()//2, 
                       screen_pos[1] + radius + 2)
            surface.blit(res_text, text_pos)
    
    def render_overlap_zones(self, surface: pygame.Surface, chunks_info: List[Dict[str, Any]],
                           camera_transform: np.ndarray):
        """
        Render chunk overlap zones for seam prevention verification
        
        Args:
            surface: Pygame surface to render to
            chunks_info: List of chunk information
            camera_transform: Camera transformation matrix
        """
        if not self.is_enabled(DebugMode.OVERLAP_ZONES):
            return
        
        zones_drawn = 0
        
        for chunk_info in chunks_info:
            overlap_enabled = chunk_info.get('vertex_overlap_enabled', False)
            if not overlap_enabled:
                continue
            
            bounds = chunk_info.get('bounds', {})
            min_bounds = np.array(bounds.get('min', [-1, -1, -1]))
            max_bounds = np.array(bounds.get('max', [1, 1, 1]))
            
            # Draw overlap zone as wireframe box
            corners = self._get_box_corners(min_bounds, max_bounds)
            screen_corners = []
            
            for corner in corners:
                screen_pos = self._world_to_screen(corner, camera_transform, surface)
                if screen_pos is not None:
                    screen_corners.append(screen_pos)
            
            # Draw wireframe box for overlap zone
            if len(screen_corners) >= 4:
                self._draw_wireframe_box(surface, screen_corners, 
                                       self.debug_colors['overlap_zone'])
                zones_drawn += 1
        
        self.debug_stats['overlap_zones_rendered'] = zones_drawn
    
    def render_debug_info_panel(self, surface: pygame.Surface):
        """
        Render debug information panel with T20 statistics
        
        Args:
            surface: Pygame surface to render to
        """
        panel_x, panel_y = 10, surface.get_height() - 150
        panel_width, panel_height = 250, 140
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(surface, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(surface, (100, 100, 100), panel_rect, 2)
        
        # Render debug information
        y_offset = panel_y + 5
        
        title = self.font.render("T20 Debug Info", True, (255, 255, 255))
        surface.blit(title, (panel_x + 5, y_offset))
        y_offset += 15
        
        # Active modes
        active_modes = [mode.value for mode, enabled in self.enabled_modes.items() if enabled]
        modes_text = f"Active: {', '.join(active_modes) if active_modes else 'None'}"
        modes_surface = self.font.render(modes_text, True, (200, 200, 200))
        surface.blit(modes_surface, (panel_x + 5, y_offset))
        y_offset += 12
        
        # Statistics
        stats_lines = [
            f"Voxel Cells: {self.debug_stats['voxel_cells_rendered']}",
            f"Seam Strips: {self.debug_stats['seam_strips_rendered']}",
            f"Normals: {self.debug_stats['normal_vectors_rendered']}",
            f"Overlaps: {self.debug_stats['overlap_zones_rendered']}",
        ]
        
        for line in stats_lines:
            text_surface = self.font.render(line, True, (180, 180, 180))
            surface.blit(text_surface, (panel_x + 5, y_offset))
            y_offset += 12
    
    def _world_to_screen(self, world_pos: np.ndarray, camera_transform: np.ndarray, 
                        surface: pygame.Surface) -> Optional[Tuple[int, int]]:
        """
        Transform world position to screen coordinates
        
        Args:
            world_pos: 3D world position
            camera_transform: 4x4 camera transformation matrix
            surface: Screen surface for bounds checking
            
        Returns:
            Screen coordinates (x, y) or None if behind camera/outside screen
        """
        # Simple orthographic projection for debugging
        # In real implementation, this would use proper view-projection matrix
        
        # Apply camera transform (simplified)
        screen_x = int(surface.get_width() // 2 + world_pos[0] * 100)
        screen_y = int(surface.get_height() // 2 - world_pos[1] * 100)  # Y inverted
        
        # Check bounds
        if (0 <= screen_x < surface.get_width() and 
            0 <= screen_y < surface.get_height()):
            return (screen_x, screen_y)
        
        return None
    
    def _draw_wireframe_cube(self, surface: pygame.Surface, center: Tuple[int, int], 
                           size: float, color: Tuple[int, int, int, int]):
        """Draw wireframe cube for voxel visualization"""
        half_size = int(size * 50)  # Scale for screen space
        
        # Draw simple square for cube representation
        rect = pygame.Rect(center[0] - half_size, center[1] - half_size,
                          half_size * 2, half_size * 2)
        pygame.draw.rect(surface, color[:3], rect, 1)
    
    def _draw_arrow_head(self, surface: pygame.Surface, start: Tuple[int, int], 
                        end: Tuple[int, int], color: Tuple[int, int, int, int]):
        """Draw arrow head for normal vector visualization"""
        # Calculate arrow head points
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 1:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Arrow head size
        head_size = 5
        
        # Arrow head points
        p1 = (int(end[0] - head_size * dx + head_size * dy * 0.5),
              int(end[1] - head_size * dy - head_size * dx * 0.5))
        p2 = (int(end[0] - head_size * dx - head_size * dy * 0.5),
              int(end[1] - head_size * dy + head_size * dx * 0.5))
        
        # Draw arrow head
        pygame.draw.polygon(surface, color[:3], [end, p1, p2])
    
    def _get_box_corners(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> List[np.ndarray]:
        """Get 8 corner positions of a bounding box"""
        return [
            np.array([min_bounds[0], min_bounds[1], min_bounds[2]]),
            np.array([max_bounds[0], min_bounds[1], min_bounds[2]]),
            np.array([max_bounds[0], max_bounds[1], min_bounds[2]]),
            np.array([min_bounds[0], max_bounds[1], min_bounds[2]]),
            np.array([min_bounds[0], min_bounds[1], max_bounds[2]]),
            np.array([max_bounds[0], min_bounds[1], max_bounds[2]]),
            np.array([max_bounds[0], max_bounds[1], max_bounds[2]]),
            np.array([min_bounds[0], max_bounds[1], max_bounds[2]]),
        ]
    
    def _draw_wireframe_box(self, surface: pygame.Surface, corners: List[Tuple[int, int]], 
                           color: Tuple[int, int, int, int]):
        """Draw wireframe bounding box"""
        if len(corners) < 8:
            return
        
        # Draw edges of the box (simplified 2D projection)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for edge in edges:
            if edge[0] < len(corners) and edge[1] < len(corners):
                pygame.draw.line(surface, color[:3], corners[edge[0]], corners[edge[1]], 1)


def create_test_debug_scene():
    """Create test scene for debugging T20 improvements"""
    
    # Create sample cave mesh data
    vertices = [
        MarchingCubesVertex(
            position=np.array([0.1, 0.2, 0.3]),
            normal=np.array([0.0, 1.0, 0.0]),
            material_id=1
        ),
        MarchingCubesVertex(
            position=np.array([0.5, 0.1, 0.2]),
            normal=np.array([1.0, 0.0, 0.0]),
            material_id=1
        )
    ]
    
    triangles = np.array([[0, 1, 0]], dtype=np.int32)
    
    # Mock chunk bounds
    from marching_cubes import ChunkBounds
    bounds = ChunkBounds(
        min_point=np.array([0, 0, 0]),
        max_point=np.array([1, 1, 1])
    )
    
    cave_mesh = CaveMeshData(
        vertices=vertices,
        triangles=triangles,
        chunk_id="test_chunk",
        material_id=1,
        bounds=bounds
    )
    
    return cave_mesh


if __name__ == "__main__":
    # Test the debug system
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("T20 Marching Cubes Debug Test")
    clock = pygame.time.Clock()
    
    # Create debugger and test scene
    debugger = MarchingCubesDebugger()
    cave_mesh = create_test_debug_scene()
    
    # Enable all debug modes for testing
    for mode in DebugMode:
        debugger.toggle_debug_mode(mode, True)
    
    # Simple camera transform (identity for testing)
    camera_transform = np.eye(4)
    
    print("🔧 T20 Debug System Test")
    print("Controls:")
    print("  - All debug modes are enabled")
    print("  - Close window to exit")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    debugger.toggle_debug_mode(DebugMode.VOXEL_CELLS)
                elif event.key == pygame.K_2:
                    debugger.toggle_debug_mode(DebugMode.SEAM_STRIPS)
                elif event.key == pygame.K_3:
                    debugger.toggle_debug_mode(DebugMode.NORMAL_VECTORS)
                elif event.key == pygame.K_4:
                    debugger.toggle_debug_mode(DebugMode.RESOLUTION_HEATMAP)
                elif event.key == pygame.K_5:
                    debugger.toggle_debug_mode(DebugMode.OVERLAP_ZONES)
        
        # Clear screen
        screen.fill((30, 30, 50))
        
        # Render debug visualizations
        voxel_info = {
            'resolution': 64,
            'cell_size': 0.03,
            'bounds_min': [-1, -1, -1],
            'bounds_max': [1, 1, 1]
        }
        
        overlap_info = {
            'overlap_voxels': 1,
            'voxel_size': 0.03
        }
        
        chunks_info = [{
            'voxel_resolution': 64,
            'vertex_overlap_enabled': True,
            'bounds': {
                'center': [0, 0, 0],
                'min': [-0.5, -0.5, -0.5],
                'max': [0.5, 0.5, 0.5]
            }
        }]
        
        # Render debug overlays
        debugger.render_voxel_cells(screen, voxel_info, camera_transform)
        debugger.render_seam_strips(screen, cave_mesh, overlap_info, camera_transform)
        debugger.render_normal_vectors(screen, cave_mesh, camera_transform)
        debugger.render_resolution_heatmap(screen, chunks_info, camera_transform)
        debugger.render_overlap_zones(screen, chunks_info, camera_transform)
        
        # Render debug info panel
        debugger.render_debug_info_panel(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("✅ T20 Debug System test completed")