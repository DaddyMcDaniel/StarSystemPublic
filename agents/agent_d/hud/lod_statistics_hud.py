#!/usr/bin/env python3
"""
LOD Statistics HUD Display - T19
================================

Displays active LOD histogram and median triangle count for terrain LOD debugging.
Shows real-time LOD distribution as camera moves from orbit to ground.

Features:
- Live LOD level histogram display
- Median triangle count tracking
- Frame time and performance metrics
- Chunk loading/culling statistics
- Screen-space error visualization

Usage:
    from lod_statistics_hud import LODStatisticsHUD
    
    hud = LODStatisticsHUD()
    hud.update(lod_manager.get_lod_statistics())
    hud.render(screen)
"""

import pygame
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import math

# Import LOD types
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mesh'))
from runtime_lod import LODLevel


class HUDPosition(Enum):
    """HUD positioning options"""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class LODStatisticsHUD:
    """
    Real-time LOD statistics display for T19 terrain improvements
    """
    
    def __init__(self, position: HUDPosition = HUDPosition.TOP_RIGHT, 
                 width: int = 300, height: int = 200,
                 font_size: int = 14, background_alpha: int = 180):
        """
        Initialize LOD statistics HUD
        
        Args:
            position: HUD position on screen
            width: HUD panel width in pixels
            height: HUD panel height in pixels
            font_size: Text font size
            background_alpha: Background transparency (0-255)
        """
        self.position = position
        self.width = width
        self.height = height
        self.font_size = font_size
        self.background_alpha = background_alpha
        
        # Initialize pygame font
        pygame.font.init()
        self.font = pygame.font.Font(None, font_size)
        self.title_font = pygame.font.Font(None, font_size + 2)
        
        # Color scheme
        self.colors = {
            'background': (0, 0, 0, background_alpha),
            'border': (100, 100, 100, 255),
            'text': (255, 255, 255, 255),
            'title': (200, 200, 255, 255),
            'lod0': (0, 255, 0, 255),    # Green - highest detail
            'lod1': (255, 255, 0, 255),  # Yellow - high detail
            'lod2': (255, 128, 0, 255),  # Orange - medium detail
            'lod3': (255, 0, 0, 255),    # Red - low detail
            'stats': (128, 255, 128, 255) # Light green - statistics
        }
        
        self.lod_colors = {
            LODLevel.LOD0: self.colors['lod0'],
            LODLevel.LOD1: self.colors['lod1'],
            LODLevel.LOD2: self.colors['lod2'],
            LODLevel.LOD3: self.colors['lod3']
        }
        
        # Statistics tracking
        self.current_stats: Dict[str, Any] = {}
        self.history_length = 60  # Keep 60 frames of history
        self.triangle_history: List[int] = []
        self.frame_time_history: List[float] = []
        
    def get_position(self, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """
        Get HUD position based on screen size and position setting
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            (x, y) position for top-left corner of HUD
        """
        margin = 10
        
        if self.position == HUDPosition.TOP_LEFT:
            return (margin, margin)
        elif self.position == HUDPosition.TOP_RIGHT:
            return (screen_width - self.width - margin, margin)
        elif self.position == HUDPosition.BOTTOM_LEFT:
            return (margin, screen_height - self.height - margin)
        else:  # BOTTOM_RIGHT
            return (screen_width - self.width - margin, screen_height - self.height - margin)
    
    def update(self, lod_stats: Dict[str, Any]):
        """
        Update HUD with new LOD statistics
        
        Args:
            lod_stats: Statistics dictionary from RuntimeLODManager.get_lod_statistics()
        """
        self.current_stats = lod_stats.copy()
        
        # Track triangle count history
        total_triangles = lod_stats.get('total_triangles', 0)
        self.triangle_history.append(total_triangles)
        if len(self.triangle_history) > self.history_length:
            self.triangle_history.pop(0)
        
        # Track frame time history
        frame_time = lod_stats.get('lod_time_ms', 0.0)
        self.frame_time_history.append(frame_time)
        if len(self.frame_time_history) > self.history_length:
            self.frame_time_history.pop(0)
    
    def render(self, screen: pygame.Surface):
        """
        Render HUD to screen
        
        Args:
            screen: Pygame screen surface to render to
        """
        if not self.current_stats:
            return
        
        screen_width, screen_height = screen.get_size()
        hud_x, hud_y = self.get_position(screen_width, screen_height)
        
        # Create HUD surface with alpha
        hud_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw background
        background_color = (*self.colors['background'][:3], self.background_alpha)
        hud_surface.fill(background_color)
        
        # Draw border
        pygame.draw.rect(hud_surface, self.colors['border'], 
                        (0, 0, self.width, self.height), 2)
        
        # Render content
        y_offset = 10
        y_offset = self._render_title(hud_surface, y_offset)
        y_offset = self._render_lod_histogram(hud_surface, y_offset)
        y_offset = self._render_statistics(hud_surface, y_offset)
        y_offset = self._render_performance(hud_surface, y_offset)
        
        # Blit to main screen
        screen.blit(hud_surface, (hud_x, hud_y))
    
    def _render_title(self, surface: pygame.Surface, y_offset: int) -> int:
        """Render HUD title"""
        title_text = "LOD Statistics (T19)"
        title_surface = self.title_font.render(title_text, True, self.colors['title'])
        surface.blit(title_surface, (10, y_offset))
        return y_offset + title_surface.get_height() + 5
    
    def _render_lod_histogram(self, surface: pygame.Surface, y_offset: int) -> int:
        """Render LOD level histogram"""
        lod_histogram = self.current_stats.get('lod_histogram', {})
        
        if not lod_histogram:
            return y_offset
        
        # Title
        hist_title = self.font.render("Active LOD Histogram:", True, self.colors['text'])
        surface.blit(hist_title, (10, y_offset))
        y_offset += hist_title.get_height() + 3
        
        # Calculate bar dimensions
        bar_width = self.width - 40
        bar_height = 12
        max_count = max(lod_histogram.values()) if lod_histogram.values() else 1
        
        # Render each LOD level
        for i, lod_level in enumerate([LODLevel.LOD0, LODLevel.LOD1, LODLevel.LOD2, LODLevel.LOD3]):
            count = lod_histogram.get(lod_level, 0)
            color = self.lod_colors[lod_level]
            
            # LOD label
            lod_text = f"LOD{lod_level.value}: {count}"
            text_surface = self.font.render(lod_text, True, self.colors['text'])
            surface.blit(text_surface, (10, y_offset))
            
            # Bar
            if max_count > 0:
                fill_width = int((count / max_count) * bar_width)
                bar_rect = (110, y_offset + 2, fill_width, bar_height)
                pygame.draw.rect(surface, color, bar_rect)
            
            # Bar outline
            outline_rect = (110, y_offset + 2, bar_width, bar_height)
            pygame.draw.rect(surface, self.colors['border'], outline_rect, 1)
            
            y_offset += bar_height + 4
        
        return y_offset + 5
    
    def _render_statistics(self, surface: pygame.Surface, y_offset: int) -> int:
        """Render triangle and chunk statistics"""
        stats_lines = [
            f"Total Triangles: {self.current_stats.get('total_triangles', 0):,}",
            f"Median Tri Count: {self.current_stats.get('median_triangle_count', 0):,}",
            f"Visible Chunks: {self.current_stats.get('visible_chunks', 0)}",
            f"Culled Chunks: {self.current_stats.get('culled_chunks', 0)}"
        ]
        
        for line in stats_lines:
            text_surface = self.font.render(line, True, self.colors['stats'])
            surface.blit(text_surface, (10, y_offset))
            y_offset += text_surface.get_height() + 2
        
        return y_offset + 5
    
    def _render_performance(self, surface: pygame.Surface, y_offset: int) -> int:
        """Render performance metrics"""
        cull_time = self.current_stats.get('cull_time_ms', 0.0)
        lod_time = self.current_stats.get('lod_time_ms', 0.0)
        
        perf_lines = [
            f"Cull Time: {cull_time:.2f}ms",
            f"LOD Time: {lod_time:.2f}ms"
        ]
        
        # Add triangle rate if we have history
        if len(self.triangle_history) >= 2:
            avg_triangles = sum(self.triangle_history[-10:]) / min(10, len(self.triangle_history))
            perf_lines.append(f"Avg Triangles: {avg_triangles:,.0f}")
        
        for line in perf_lines:
            text_surface = self.font.render(line, True, self.colors['text'])
            surface.blit(text_surface, (10, y_offset))
            y_offset += text_surface.get_height() + 2
        
        return y_offset
    
    def render_mini_graph(self, surface: pygame.Surface, y_offset: int) -> int:
        """Render mini triangle count graph"""
        if len(self.triangle_history) < 2:
            return y_offset
        
        graph_width = self.width - 20
        graph_height = 30
        
        # Graph background
        graph_rect = (10, y_offset, graph_width, graph_height)
        pygame.draw.rect(surface, (20, 20, 20, 255), graph_rect)
        pygame.draw.rect(surface, self.colors['border'], graph_rect, 1)
        
        # Draw triangle count line
        if len(self.triangle_history) > 1:
            min_triangles = min(self.triangle_history)
            max_triangles = max(self.triangle_history)
            triangle_range = max_triangles - min_triangles
            
            if triangle_range > 0:
                points = []
                for i, count in enumerate(self.triangle_history):
                    x = int(10 + (i / (len(self.triangle_history) - 1)) * graph_width)
                    y = int(y_offset + graph_height - ((count - min_triangles) / triangle_range) * graph_height)
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(surface, self.colors['lod0'], False, points, 2)
        
        return y_offset + graph_height + 10
    
    def get_lod_transition_quality_score(self) -> float:
        """
        Calculate a quality score for LOD transitions based on triangle count stability
        
        Returns:
            Quality score between 0.0 (poor) and 1.0 (excellent)
        """
        if len(self.triangle_history) < 10:
            return 0.5
        
        recent_history = self.triangle_history[-10:]
        
        # Calculate coefficient of variation (stability measure)
        mean_triangles = np.mean(recent_history)
        std_triangles = np.std(recent_history)
        
        if mean_triangles == 0:
            return 0.0
        
        cv = std_triangles / mean_triangles
        
        # Convert to quality score (lower variation = higher quality)
        # Good LOD should have smooth transitions with CV < 0.1
        quality = max(0.0, 1.0 - cv * 5.0)  # Scale so CV of 0.2 = quality 0.0
        
        return min(1.0, quality)


def create_test_hud():
    """Create test HUD with sample data"""
    hud = LODStatisticsHUD()
    
    # Sample statistics
    test_stats = {
        'total_chunks': 150,
        'visible_chunks': 85,
        'culled_chunks': 65,
        'lod_histogram': {
            LODLevel.LOD0: 15,
            LODLevel.LOD1: 25,
            LODLevel.LOD2: 30,
            LODLevel.LOD3: 15
        },
        'total_triangles': 125000,
        'median_triangle_count': 1800,
        'cull_time_ms': 0.8,
        'lod_time_ms': 1.2
    }
    
    hud.update(test_stats)
    return hud


if __name__ == "__main__":
    # Test the HUD system
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("LOD Statistics HUD Test")
    clock = pygame.time.Clock()
    
    hud = create_test_hud()
    
    # Simulate changing LOD statistics
    frame = 0
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Simulate dynamic LOD changes
        distance_factor = math.sin(frame * 0.02) * 0.5 + 0.5  # 0 to 1
        
        # Create dynamic statistics based on simulated camera distance
        dynamic_stats = {
            'total_chunks': 150,
            'visible_chunks': int(60 + distance_factor * 50),
            'culled_chunks': int(90 - distance_factor * 50),
            'lod_histogram': {
                LODLevel.LOD0: int(30 * (1 - distance_factor)),
                LODLevel.LOD1: int(25 * (1 - distance_factor * 0.5)),
                LODLevel.LOD2: int(20 + distance_factor * 15),
                LODLevel.LOD3: int(distance_factor * 25)
            },
            'total_triangles': int(50000 + (1 - distance_factor) * 100000),
            'median_triangle_count': int(800 + (1 - distance_factor) * 2000),
            'cull_time_ms': 0.5 + distance_factor * 0.8,
            'lod_time_ms': 0.8 + distance_factor * 1.0
        }
        
        hud.update(dynamic_stats)
        
        # Render
        screen.fill((50, 50, 80))
        
        # Draw some fake terrain visualization
        pygame.draw.circle(screen, (100, 150, 100), (400, 300), 200, 3)
        
        # Render HUD
        hud.render(screen)
        
        pygame.display.flip()
        clock.tick(60)
        frame += 1
    
    pygame.quit()
    print("✅ LOD Statistics HUD test completed")