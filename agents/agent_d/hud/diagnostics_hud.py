#!/usr/bin/env python3
"""
Diagnostics HUD System - T16
============================

Real-time diagnostics HUD for PCC terrain viewer showing performance metrics,
memory usage, LOD information, and rendering statistics. Designed for 
debugging content generation and LOD performance.

Features:
- FPS and frame timing with history
- Draw calls and rendering statistics
- Active chunk count and memory usage
- VRAM and mesh memory tracking
- LOD level histogram with visual distribution
- Real-time performance monitoring

Usage:
    from hud.diagnostics_hud import DiagnosticsHUD
    
    hud = DiagnosticsHUD()
    hud.update(fps=60.0, draw_calls=120, active_chunks=25)
    hud.render(screen_width=1920, screen_height=1080)
"""

import time
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import sys
import os

# Import performance profiling from T14
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'performance'))


@dataclass
class FrameStats:
    """Statistics for a single frame"""
    timestamp: float
    fps: float
    frame_time_ms: float
    draw_calls: int
    triangles: int
    vertices: int
    active_chunks: int
    visible_chunks: int
    memory_usage_mb: float
    vram_usage_mb: float


@dataclass
class LODStats:
    """Level of Detail statistics"""
    lod_histogram: Dict[int, int] = field(default_factory=dict)  # LOD level -> chunk count
    total_chunks: int = 0
    highest_lod: int = 0
    lowest_lod: int = 0
    average_lod: float = 0.0
    lod_switches_per_second: float = 0.0


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    mesh_memory_mb: float = 0.0
    texture_memory_mb: float = 0.0
    buffer_memory_mb: float = 0.0
    total_vram_mb: float = 0.0
    system_memory_mb: float = 0.0
    memory_pressure: float = 0.0  # 0.0 to 1.0


@dataclass 
class RenderingStats:
    """Rendering pipeline statistics"""
    draw_calls: int = 0
    triangles_drawn: int = 0
    vertices_processed: int = 0
    texture_switches: int = 0
    shader_switches: int = 0
    culled_chunks: int = 0
    frustum_culled: int = 0
    occlusion_culled: int = 0


class DiagnosticsHUD:
    """Real-time diagnostics HUD for terrain viewer"""
    
    def __init__(self, history_size: int = 120):
        """Initialize diagnostics HUD"""
        self.history_size = history_size
        self.frame_history = deque(maxlen=history_size)
        
        # Current frame statistics
        self.current_stats = FrameStats(
            timestamp=time.time(),
            fps=0.0,
            frame_time_ms=0.0,
            draw_calls=0,
            triangles=0,
            vertices=0,
            active_chunks=0,
            visible_chunks=0,
            memory_usage_mb=0.0,
            vram_usage_mb=0.0
        )
        
        # Specialized statistics
        self.lod_stats = LODStats()
        self.memory_stats = MemoryStats()
        self.rendering_stats = RenderingStats()
        
        # HUD state
        self.visible = True
        self.detailed_view = False
        self.hud_scale = 1.0
        self.hud_opacity = 0.85
        
        # Performance tracking
        self.last_update_time = time.time()
        self.frame_count = 0
        
        # Colors for different metrics (RGB tuples)
        self.colors = {
            'good': (0.2, 0.8, 0.2),      # Green
            'warning': (0.9, 0.7, 0.2),   # Yellow
            'critical': (0.9, 0.2, 0.2),  # Red
            'info': (0.2, 0.6, 0.9),      # Blue
            'neutral': (0.8, 0.8, 0.8),   # Light gray
            'background': (0.0, 0.0, 0.0, 0.7)  # Semi-transparent black
        }
    
    def update_frame_stats(self, fps: float, frame_time_ms: float, draw_calls: int,
                          triangles: int = 0, vertices: int = 0, active_chunks: int = 0,
                          visible_chunks: int = 0, memory_usage_mb: float = 0.0,
                          vram_usage_mb: float = 0.0):
        """Update frame statistics"""
        current_time = time.time()
        
        self.current_stats = FrameStats(
            timestamp=current_time,
            fps=fps,
            frame_time_ms=frame_time_ms,
            draw_calls=draw_calls,
            triangles=triangles,
            vertices=vertices,
            active_chunks=active_chunks,
            visible_chunks=visible_chunks,
            memory_usage_mb=memory_usage_mb,
            vram_usage_mb=vram_usage_mb
        )
        
        # Add to history
        self.frame_history.append(self.current_stats)
        self.frame_count += 1
        self.last_update_time = current_time
    
    def update_lod_stats(self, lod_histogram: Dict[int, int], lod_switches_per_second: float = 0.0):
        """Update LOD statistics"""
        self.lod_stats.lod_histogram = lod_histogram.copy()
        self.lod_stats.total_chunks = sum(lod_histogram.values())
        self.lod_stats.lod_switches_per_second = lod_switches_per_second
        
        if lod_histogram:
            self.lod_stats.highest_lod = max(lod_histogram.keys())
            self.lod_stats.lowest_lod = min(lod_histogram.keys())
            
            # Calculate weighted average LOD
            total_weighted = sum(lod * count for lod, count in lod_histogram.items())
            self.lod_stats.average_lod = total_weighted / self.lod_stats.total_chunks if self.lod_stats.total_chunks > 0 else 0.0
        else:
            self.lod_stats.highest_lod = 0
            self.lod_stats.lowest_lod = 0
            self.lod_stats.average_lod = 0.0
    
    def update_memory_stats(self, mesh_memory_mb: float, texture_memory_mb: float = 0.0,
                           buffer_memory_mb: float = 0.0, system_memory_mb: float = 0.0):
        """Update memory usage statistics"""
        self.memory_stats.mesh_memory_mb = mesh_memory_mb
        self.memory_stats.texture_memory_mb = texture_memory_mb
        self.memory_stats.buffer_memory_mb = buffer_memory_mb
        self.memory_stats.system_memory_mb = system_memory_mb
        
        # Calculate total VRAM
        self.memory_stats.total_vram_mb = (mesh_memory_mb + texture_memory_mb + buffer_memory_mb)
        
        # Calculate memory pressure (simplified)
        max_vram = 8192.0  # Assume 8GB VRAM limit
        self.memory_stats.memory_pressure = min(1.0, self.memory_stats.total_vram_mb / max_vram)
    
    def update_rendering_stats(self, draw_calls: int, triangles: int, vertices: int,
                              texture_switches: int = 0, shader_switches: int = 0,
                              culled_chunks: int = 0, frustum_culled: int = 0,
                              occlusion_culled: int = 0):
        """Update rendering pipeline statistics"""
        self.rendering_stats.draw_calls = draw_calls
        self.rendering_stats.triangles_drawn = triangles
        self.rendering_stats.vertices_processed = vertices
        self.rendering_stats.texture_switches = texture_switches
        self.rendering_stats.shader_switches = shader_switches
        self.rendering_stats.culled_chunks = culled_chunks
        self.rendering_stats.frustum_culled = frustum_culled
        self.rendering_stats.occlusion_culled = occlusion_culled
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last few seconds"""
        if not self.frame_history:
            return {}
        
        recent_frames = [f for f in self.frame_history if time.time() - f.timestamp < 3.0]
        if not recent_frames:
            return {}
        
        fps_values = [f.fps for f in recent_frames]
        frame_times = [f.frame_time_ms for f in recent_frames]
        draw_calls = [f.draw_calls for f in recent_frames]
        
        return {
            'avg_fps': statistics.mean(fps_values),
            'min_fps': min(fps_values),
            'max_fps': max(fps_values),
            'avg_frame_time': statistics.mean(frame_times),
            'frame_time_variance': statistics.variance(frame_times) if len(frame_times) > 1 else 0.0,
            'avg_draw_calls': statistics.mean(draw_calls),
            'frame_count': len(recent_frames)
        }
    
    def get_color_for_value(self, value: float, good_threshold: float, 
                           warning_threshold: float, reverse: bool = False) -> Tuple[float, float, float]:
        """Get color based on value thresholds"""
        if reverse:
            # For metrics where lower is better (frame time, memory usage)
            if value <= good_threshold:
                return self.colors['good']
            elif value <= warning_threshold:
                return self.colors['warning']
            else:
                return self.colors['critical']
        else:
            # For metrics where higher is better (FPS)
            if value >= good_threshold:
                return self.colors['good']
            elif value >= warning_threshold:
                return self.colors['warning']
            else:
                return self.colors['critical']
    
    def generate_hud_text(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Generate HUD text lines with colors"""
        lines = []
        
        # Performance section
        fps_color = self.get_color_for_value(self.current_stats.fps, 55.0, 30.0)
        frame_time_color = self.get_color_for_value(self.current_stats.frame_time_ms, 16.67, 33.33, reverse=True)
        
        lines.append((f"FPS: {self.current_stats.fps:.1f}", fps_color))
        lines.append((f"Frame Time: {self.current_stats.frame_time_ms:.2f}ms", frame_time_color))
        
        # Rendering section
        draw_calls_color = self.get_color_for_value(self.current_stats.draw_calls, 100, 200, reverse=True)
        lines.append((f"Draw Calls: {self.current_stats.draw_calls}", draw_calls_color))
        lines.append((f"Triangles: {self.current_stats.triangles:,}", self.colors['info']))
        
        # Chunk information
        chunk_color = self.colors['info']
        lines.append((f"Active Chunks: {self.current_stats.active_chunks}", chunk_color))
        lines.append((f"Visible Chunks: {self.current_stats.visible_chunks}", chunk_color))
        
        # Memory section
        vram_color = self.get_color_for_value(self.memory_stats.total_vram_mb, 2048, 4096, reverse=True)
        lines.append((f"VRAM: {self.memory_stats.total_vram_mb:.1f}MB", vram_color))
        lines.append((f"Mesh Memory: {self.memory_stats.mesh_memory_mb:.1f}MB", self.colors['neutral']))
        
        # LOD section
        if self.lod_stats.total_chunks > 0:
            lines.append(("", self.colors['neutral']))  # Spacer
            lines.append((f"LOD Average: {self.lod_stats.average_lod:.1f}", self.colors['info']))
            lines.append((f"LOD Range: {self.lod_stats.lowest_lod}-{self.lod_stats.highest_lod}", self.colors['info']))
            lines.append((f"LOD Switches: {self.lod_stats.lod_switches_per_second:.1f}/s", self.colors['neutral']))
        
        # Detailed view additional information
        if self.detailed_view:
            perf_summary = self.get_performance_summary()
            if perf_summary:
                lines.append(("", self.colors['neutral']))  # Spacer
                lines.append((f"Avg FPS: {perf_summary['avg_fps']:.1f}", self.colors['neutral']))
                lines.append((f"Min/Max FPS: {perf_summary['min_fps']:.1f}/{perf_summary['max_fps']:.1f}", self.colors['neutral']))
                lines.append((f"Frame Variance: {perf_summary['frame_time_variance']:.2f}", self.colors['neutral']))
            
            # Culling statistics
            if self.rendering_stats.culled_chunks > 0:
                lines.append(("", self.colors['neutral']))  # Spacer
                lines.append((f"Culled Chunks: {self.rendering_stats.culled_chunks}", self.colors['neutral']))
                lines.append((f"Frustum Culled: {self.rendering_stats.frustum_culled}", self.colors['neutral']))
                lines.append((f"Occlusion Culled: {self.rendering_stats.occlusion_culled}", self.colors['neutral']))
        
        return lines
    
    def generate_lod_histogram_text(self) -> List[str]:
        """Generate LOD histogram visualization"""
        if not self.lod_stats.lod_histogram:
            return ["LOD Histogram: No data"]
        
        lines = ["LOD Histogram:"]
        max_count = max(self.lod_stats.lod_histogram.values())
        
        for lod in sorted(self.lod_stats.lod_histogram.keys()):
            count = self.lod_stats.lod_histogram[lod]
            bar_length = int((count / max_count) * 20) if max_count > 0 else 0
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"LOD {lod:2d}: {bar} {count:3d}")
        
        return lines
    
    def toggle_visibility(self):
        """Toggle HUD visibility"""
        self.visible = not self.visible
    
    def toggle_detailed_view(self):
        """Toggle detailed view mode"""
        self.detailed_view = not self.detailed_view
    
    def set_scale(self, scale: float):
        """Set HUD scale factor"""
        self.hud_scale = max(0.5, min(2.0, scale))
    
    def set_opacity(self, opacity: float):
        """Set HUD opacity"""
        self.hud_opacity = max(0.1, min(1.0, opacity))
    
    def get_hud_data(self) -> Dict[str, Any]:
        """Get all HUD data for rendering"""
        if not self.visible:
            return {'visible': False}
        
        return {
            'visible': True,
            'scale': self.hud_scale,
            'opacity': self.hud_opacity,
            'text_lines': self.generate_hud_text(),
            'lod_histogram': self.generate_lod_histogram_text(),
            'detailed_view': self.detailed_view,
            'colors': self.colors,
            'frame_count': self.frame_count,
            'last_update': self.last_update_time
        }
    
    def export_performance_data(self, filename: str):
        """Export performance data to file"""
        import json
        
        export_data = {
            'timestamp': time.time(),
            'frame_count': self.frame_count,
            'current_stats': {
                'fps': self.current_stats.fps,
                'frame_time_ms': self.current_stats.frame_time_ms,
                'draw_calls': self.current_stats.draw_calls,
                'triangles': self.current_stats.triangles,
                'active_chunks': self.current_stats.active_chunks,
                'memory_usage_mb': self.current_stats.memory_usage_mb
            },
            'lod_stats': {
                'lod_histogram': self.lod_stats.lod_histogram,
                'average_lod': self.lod_stats.average_lod,
                'lod_switches_per_second': self.lod_stats.lod_switches_per_second
            },
            'memory_stats': {
                'mesh_memory_mb': self.memory_stats.mesh_memory_mb,
                'total_vram_mb': self.memory_stats.total_vram_mb,
                'memory_pressure': self.memory_stats.memory_pressure
            },
            'performance_summary': self.get_performance_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Performance data exported to {filename}")


if __name__ == "__main__":
    # Test diagnostics HUD
    print("🚀 T16 Diagnostics HUD System")
    print("=" * 60)
    
    # Create HUD
    hud = DiagnosticsHUD()
    
    print("📊 Testing HUD functionality...")
    
    # Simulate frame updates
    import random
    
    for frame in range(10):
        # Simulate frame data
        fps = 55 + random.uniform(-10, 15)
        frame_time = 1000.0 / fps if fps > 0 else 100.0
        draw_calls = random.randint(80, 150)
        triangles = random.randint(50000, 200000)
        active_chunks = random.randint(20, 50)
        visible_chunks = random.randint(15, active_chunks)
        memory_usage = random.uniform(100, 500)
        vram_usage = random.uniform(200, 1000)
        
        hud.update_frame_stats(
            fps=fps,
            frame_time_ms=frame_time,
            draw_calls=draw_calls,
            triangles=triangles,
            active_chunks=active_chunks,
            visible_chunks=visible_chunks,
            memory_usage_mb=memory_usage,
            vram_usage_mb=vram_usage
        )
        
        # Simulate LOD histogram
        lod_histogram = {
            0: random.randint(5, 15),
            1: random.randint(8, 20),
            2: random.randint(3, 12),
            3: random.randint(1, 6)
        }
        
        hud.update_lod_stats(lod_histogram, random.uniform(0.5, 3.0))
        
        # Simulate memory stats
        hud.update_memory_stats(
            mesh_memory_mb=random.uniform(200, 800),
            texture_memory_mb=random.uniform(100, 400),
            buffer_memory_mb=random.uniform(50, 200)
        )
        
        time.sleep(0.016)  # Simulate 60 FPS
    
    # Get HUD data
    hud_data = hud.get_hud_data()
    print(f"   ✅ HUD data generated: {len(hud_data['text_lines'])} text lines")
    
    # Test performance summary
    perf_summary = hud.get_performance_summary()
    print(f"   ✅ Performance summary: Avg FPS {perf_summary.get('avg_fps', 0):.1f}")
    
    # Test LOD histogram
    lod_histogram_text = hud.generate_lod_histogram_text()
    print(f"   ✅ LOD histogram: {len(lod_histogram_text)} lines")
    
    # Test detailed view toggle
    hud.toggle_detailed_view()
    detailed_data = hud.get_hud_data()
    print(f"   ✅ Detailed view: {detailed_data['detailed_view']}")
    
    # Export performance data
    hud.export_performance_data("test_performance_data.json")
    
    print(f"\n📈 Sample HUD Text Lines:")
    for line_text, color in hud_data['text_lines'][:8]:
        if line_text:  # Skip empty spacers
            print(f"   {line_text}")
    
    print(f"\n📊 Sample LOD Histogram:")
    for line in lod_histogram_text[:5]:
        print(f"   {line}")
    
    print(f"\n✅ Diagnostics HUD system functional")