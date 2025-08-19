#!/usr/bin/env python3
"""
SDF Debug Overlay System - T11
===============================

Provides debug visualization of SDF iso-contours on terrain surfaces
to help visualize cave/terrain fusion boundaries and validate blending.

Features:
- SDF iso-contour generation on surface meshes
- Multi-level contour visualization 
- OpenGL rendering integration
- Real-time SDF value sampling
- Visual fusion boundary validation

Usage:
    from sdf_debug_overlay import SdfDebugOverlay
    
    overlay = SdfDebugOverlay()
    contours = overlay.generate_surface_contours(fusion_sdf, mesh_data)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marching_cubes'))
sys.path.append(os.path.dirname(__file__))

from sdf_primitives import SDFNode
from surface_sdf_fusion import FusedMeshData
from sdf_evaluator import ChunkBounds

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


@dataclass
class IsoContour:
    """Single iso-contour line at specific SDF value"""
    iso_value: float
    points: np.ndarray  # Nx3 contour points
    color: Tuple[float, float, float]  # RGB color
    line_width: float = 2.0
    
    def __len__(self) -> int:
        return len(self.points)


@dataclass 
class SdfContourSet:
    """Set of iso-contours for SDF visualization"""
    contours: List[IsoContour]
    bounds: ChunkBounds
    mesh_id: str
    
    def get_contour(self, iso_value: float) -> Optional[IsoContour]:
        """Get contour for specific iso-value"""
        for contour in self.contours:
            if abs(contour.iso_value - iso_value) < 1e-6:
                return contour
        return None
    
    @property
    def total_points(self) -> int:
        return sum(len(contour) for contour in self.contours)


class SdfDebugOverlay:
    """Debug overlay for SDF iso-contour visualization"""
    
    def __init__(self):
        """Initialize SDF debug overlay system"""
        self.contour_sets: Dict[str, SdfContourSet] = {}
        self.debug_enabled = True
        
        # Default iso-value levels for visualization
        self.iso_levels = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]
        
        # Color scheme for different iso-values
        self.iso_colors = {
            -2.0: (0.0, 0.0, 1.0),    # Deep blue (far inside)
            -1.0: (0.0, 0.5, 1.0),    # Blue
            -0.5: (0.0, 1.0, 1.0),    # Cyan
            -0.1: (0.0, 1.0, 0.0),    # Green (near surface inside)
            0.0: (1.0, 1.0, 0.0),     # Yellow (surface)
            0.1: (1.0, 0.5, 0.0),     # Orange (near surface outside)  
            0.5: (1.0, 0.0, 0.0),     # Red
            1.0: (1.0, 0.0, 1.0),     # Magenta
            2.0: (0.5, 0.0, 0.5),     # Purple (far outside)
        }
    
    def generate_surface_contours(self, fusion_sdf: SDFNode, mesh_data: FusedMeshData,
                                surface_sample_density: int = 64) -> SdfContourSet:
        """Generate iso-contours on the mesh surface"""
        print(f"🔍 Generating SDF contours for {mesh_data.chunk_id}...")
        
        contours = []
        bounds = mesh_data.bounds
        
        # Sample SDF values across the surface using mesh vertices as base
        surface_points = self._extract_surface_sample_points(mesh_data, surface_sample_density)
        
        # Generate contours for each iso-level
        for iso_value in self.iso_levels:
            contour_points = self._find_iso_contour_points(
                fusion_sdf, surface_points, iso_value, bounds
            )
            
            if len(contour_points) > 0:
                color = self.iso_colors.get(iso_value, (1.0, 1.0, 1.0))
                line_width = 3.0 if abs(iso_value) < 0.1 else 2.0  # Thicker line for surface
                
                contour = IsoContour(
                    iso_value=iso_value,
                    points=contour_points,
                    color=color,
                    line_width=line_width
                )
                contours.append(contour)
        
        contour_set = SdfContourSet(
            contours=contours,
            bounds=bounds,
            mesh_id=mesh_data.chunk_id
        )
        
        self.contour_sets[mesh_data.chunk_id] = contour_set
        
        print(f"   ✅ Generated {len(contours)} contours with {contour_set.total_points} points")
        return contour_set
    
    def _extract_surface_sample_points(self, mesh_data: FusedMeshData, 
                                     sample_density: int) -> np.ndarray:
        """Extract sample points from mesh surface and regular grid"""
        points = []
        
        # Add mesh vertices as sample points
        for vertex in mesh_data.vertices:
            points.append(vertex.position)
        
        # Add regular grid sample points across the chunk bounds
        bounds = mesh_data.bounds
        x_coords = np.linspace(bounds.min_point[0], bounds.max_point[0], sample_density)
        y_coords = np.linspace(bounds.min_point[1], bounds.max_point[1], sample_density // 2)
        z_coords = np.linspace(bounds.min_point[2], bounds.max_point[2], sample_density)
        
        for x in x_coords[::4]:  # Subsample for performance
            for y in y_coords[::2]:
                for z in z_coords[::4]:
                    points.append(np.array([x, y, z]))
        
        return np.array(points)
    
    def _find_iso_contour_points(self, fusion_sdf: SDFNode, surface_points: np.ndarray,
                               iso_value: float, bounds: ChunkBounds) -> np.ndarray:
        """Find points near the specified iso-value"""
        contour_points = []
        tolerance = 0.2  # Distance tolerance for iso-contour inclusion
        
        # Evaluate SDF at all surface points
        sdf_values = np.array([fusion_sdf.evaluate(point) for point in surface_points])
        
        # Find points near the iso-value
        near_iso = np.abs(sdf_values - iso_value) < tolerance
        near_points = surface_points[near_iso]
        
        # For each pair of nearby points, check for iso-value crossing
        for i in range(len(surface_points) - 1):
            p1, p2 = surface_points[i], surface_points[i + 1]
            v1, v2 = sdf_values[i], sdf_values[i + 1]
            
            # Check for sign change across iso-value
            if ((v1 - iso_value) * (v2 - iso_value)) < 0:
                # Linear interpolation to find iso-crossing
                t = (iso_value - v1) / (v2 - v1) if abs(v2 - v1) > 1e-8 else 0.5
                t = np.clip(t, 0.0, 1.0)
                
                iso_point = p1 + t * (p2 - p1)
                
                # Verify point is within chunk bounds
                if (np.all(iso_point >= bounds.min_point) and 
                    np.all(iso_point <= bounds.max_point)):
                    contour_points.append(iso_point)
        
        # Add points directly near the iso-value
        contour_points.extend(near_points)
        
        return np.array(contour_points) if contour_points else np.empty((0, 3))
    
    def render_debug_contours(self, mesh_id: str):
        """Render debug contours using OpenGL (if available)"""
        if not OPENGL_AVAILABLE or not self.debug_enabled:
            return
        
        if mesh_id not in self.contour_sets:
            return
        
        contour_set = self.contour_sets[mesh_id]
        
        # Render each contour
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        for contour in contour_set.contours:
            if len(contour.points) == 0:
                continue
            
            glColor3f(*contour.color)
            glLineWidth(contour.line_width)
            
            # Render contour as line segments
            glBegin(GL_LINES)
            for i in range(len(contour.points) - 1):
                p1, p2 = contour.points[i], contour.points[i + 1]
                glVertex3f(*p1)
                glVertex3f(*p2)
            glEnd()
            
            # Render contour points
            glPointSize(4.0)
            glBegin(GL_POINTS)
            for point in contour.points:
                glVertex3f(*point)
            glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def render_all_contours(self):
        """Render contours for all registered meshes"""
        for mesh_id in self.contour_sets.keys():
            self.render_debug_contours(mesh_id)
    
    def export_contours_to_obj(self, mesh_id: str, output_path: str):
        """Export contours to OBJ file for external visualization"""
        if mesh_id not in self.contour_sets:
            print(f"❌ No contours found for mesh {mesh_id}")
            return
        
        contour_set = self.contour_sets[mesh_id]
        
        try:
            with open(output_path, 'w') as f:
                f.write(f"# SDF Iso-Contours for {mesh_id}\n")
                f.write(f"# Generated by T11 SDF Debug Overlay\n\n")
                
                vertex_index = 1
                
                for contour in contour_set.contours:
                    if len(contour.points) == 0:
                        continue
                    
                    f.write(f"# Iso-contour at value {contour.iso_value}\n")
                    f.write(f"# Color: {contour.color}\n")
                    
                    # Write vertices
                    start_index = vertex_index
                    for point in contour.points:
                        f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                        vertex_index += 1
                    
                    # Write lines
                    for i in range(len(contour.points) - 1):
                        f.write(f"l {start_index + i} {start_index + i + 1}\n")
                    f.write("\n")
            
            print(f"✅ Exported contours to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to export contours: {e}")
    
    def analyze_sdf_distribution(self, mesh_id: str) -> Dict[str, Any]:
        """Analyze SDF value distribution across contours"""
        if mesh_id not in self.contour_sets:
            return {}
        
        contour_set = self.contour_sets[mesh_id]
        
        analysis = {
            'mesh_id': mesh_id,
            'contour_count': len(contour_set.contours),
            'total_points': contour_set.total_points,
            'iso_values': [c.iso_value for c in contour_set.contours],
            'point_distribution': {},
            'coverage_bounds': {
                'min': contour_set.bounds.min_point.tolist(),
                'max': contour_set.bounds.max_point.tolist()
            }
        }
        
        # Analyze point distribution per iso-value
        for contour in contour_set.contours:
            analysis['point_distribution'][str(contour.iso_value)] = len(contour.points)
        
        return analysis
    
    def clear_contours(self, mesh_id: Optional[str] = None):
        """Clear contours for specific mesh or all meshes"""
        if mesh_id is not None:
            if mesh_id in self.contour_sets:
                del self.contour_sets[mesh_id]
                print(f"✅ Cleared contours for {mesh_id}")
        else:
            self.contour_sets.clear()
            print("✅ Cleared all contours")
    
    def toggle_debug_overlay(self) -> bool:
        """Toggle debug overlay visibility"""
        self.debug_enabled = not self.debug_enabled
        print(f"SDF Debug Overlay: {'ON' if self.debug_enabled else 'OFF'}")
        return self.debug_enabled


class FusionDebugHUD:
    """HUD for displaying fusion and SDF debug information"""
    
    def __init__(self):
        """Initialize fusion debug HUD"""
        self.visible = True
        self.stats = {}
    
    def update_stats(self, fusion_stats: Dict[str, Any], contour_stats: Dict[str, Any]):
        """Update HUD statistics"""
        self.stats.update({
            'fusion': fusion_stats,
            'contours': contour_stats
        })
    
    def render_hud(self):
        """Render debug HUD (simplified text display)"""
        if not OPENGL_AVAILABLE or not self.visible:
            return
        
        # Render semi-transparent background
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 800, 0, 600, -1, 1)  # Assume 800x600 viewport
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Draw HUD background
        glColor4f(0.0, 0.0, 0.0, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(10, 550)   # Top-left
        glVertex2f(300, 550)  # Top-right
        glVertex2f(300, 450)  # Bottom-right
        glVertex2f(10, 450)   # Bottom-left
        glEnd()
        
        # Draw colored bars for visual indicators (simplified)
        if 'contours' in self.stats:
            contour_stats = self.stats['contours']
            contour_count = contour_stats.get('contour_count', 0)
            
            # Contour count bar
            glColor3f(0.0, 1.0, 1.0)  # Cyan
            bar_width = min(200, contour_count * 20)
            glBegin(GL_QUADS)
            glVertex2f(20, 520)
            glVertex2f(20 + bar_width, 520)
            glVertex2f(20 + bar_width, 530)
            glVertex2f(20, 530)
            glEnd()
            
            # Point count bar
            total_points = contour_stats.get('total_points', 0)
            glColor3f(1.0, 1.0, 0.0)  # Yellow
            bar_width = min(200, total_points // 10)
            glBegin(GL_QUADS)
            glVertex2f(20, 500)
            glVertex2f(20 + bar_width, 500)
            glVertex2f(20 + bar_width, 510)
            glVertex2f(20, 510)
            glEnd()
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
    
    def toggle_visibility(self) -> bool:
        """Toggle HUD visibility"""
        self.visible = not self.visible
        return self.visible


if __name__ == "__main__":
    # Test SDF debug overlay system
    print("🧪 Testing SDF Debug Overlay System")
    print("=" * 50)
    
    # Create test SDF
    from sdf_primitives import SDFSphere
    test_sdf = SDFSphere(center=[0, 0, 0], radius=1.0, seed=42)
    
    # Create mock mesh data
    from marching_cubes import MarchingCubesVertex
    
    test_vertices = [
        MarchingCubesVertex(position=np.array([0, 0, 0]), normal=np.array([0, 1, 0])),
        MarchingCubesVertex(position=np.array([1, 0, 0]), normal=np.array([1, 0, 0])),
        MarchingCubesVertex(position=np.array([0, 0, 1]), normal=np.array([0, 0, 1]))
    ]
    
    test_bounds = ChunkBounds(
        min_point=np.array([-1.5, -1.5, -1.5]),
        max_point=np.array([1.5, 1.5, 1.5])
    )
    
    test_mesh = FusedMeshData(
        vertices=test_vertices,
        triangles=np.array([[0, 1, 2]]),
        chunk_id="test_mesh",
        bounds=test_bounds,
        has_caves=True,
        fusion_stats={}
    )
    
    # Test overlay system
    overlay = SdfDebugOverlay()
    contour_set = overlay.generate_surface_contours(test_sdf, test_mesh)
    
    print(f"✅ Generated contour set:")
    print(f"   Contours: {len(contour_set.contours)}")
    print(f"   Total points: {contour_set.total_points}")
    
    # Test analysis
    analysis = overlay.analyze_sdf_distribution("test_mesh")
    print(f"\n📊 SDF Distribution Analysis:")
    for key, value in analysis.items():
        if key != 'point_distribution':
            print(f"   {key}: {value}")
    
    # Test HUD
    hud = FusionDebugHUD()
    hud.update_stats({}, analysis)
    
    print("\n✅ SDF debug overlay system functional")