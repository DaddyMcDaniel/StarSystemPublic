#!/usr/bin/env python3
"""
Cave Viewer Extension - T10
============================

Extends the PCC game viewer to render terrain + caves together.
Adds cave chunk loading, material-based rendering, and cave-specific debug visualizations.

Features:
- Combined terrain + cave rendering
- Material-based color coding
- Cave chunk streaming with terrain chunks
- Cave-specific debug visualizations
- Integration with T08 runtime LOD system

Usage:
    Import this module and call the extended rendering functions from pcc_game_viewer.py
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import numpy as np
except ImportError:
    print("❌ OpenGL not available")
    sys.exit(1)

# Import cave system
sys.path.append(os.path.dirname(__file__))
from cave_manifest import CaveManifestManager, CaveChunkInfo, MaterialInfo

# Global cave rendering state
cave_manager = None
cave_chunk_vaos = {}  # Cache for cave VAOs
debug_cave_materials = False
debug_cave_aabb = False
cave_render_enabled = True

# Cave rendering statistics
cave_stats = {
    'caves_rendered': 0,
    'cave_triangles': 0,
    'cave_vertices': 0,
    'cave_memory_mb': 0.0
}


def InitializeCaveSystem():
    """Initialize the cave rendering system"""
    global cave_manager
    cave_manager = CaveManifestManager()
    print("✅ Cave rendering system initialized")


def LoadCaveManifest(cave_manifest_path: str) -> bool:
    """
    Load cave manifest for rendering
    
    Args:
        cave_manifest_path: Path to cave chunk manifest
        
    Returns:
        True if loaded successfully
    """
    global cave_manager
    
    if cave_manager is None:
        InitializeCaveSystem()
    
    try:
        cave_chunks = cave_manager.load_cave_manifest(cave_manifest_path)
        print(f"✅ Loaded {len(cave_chunks)} cave chunks for rendering")
        return True
    except Exception as e:
        print(f"❌ Failed to load cave manifest: {e}")
        return False


def CreateCaveVAO(chunk_id: str) -> Optional[int]:
    """
    Create OpenGL VAO for cave chunk
    
    Args:
        chunk_id: Cave chunk identifier
        
    Returns:
        VAO ID or None if failed
    """
    global cave_manager, cave_chunk_vaos
    
    if cave_manager is None or chunk_id not in cave_manager.cave_chunks:
        return None
    
    # Check if already cached
    if chunk_id in cave_chunk_vaos:
        return cave_chunk_vaos[chunk_id]
    
    try:
        # Load cave chunk buffers
        buffers = cave_manager.load_cave_chunk_buffers(chunk_id)
        if not buffers:
            return None
        
        # Create VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        
        # Position buffer (attribute 0)
        if "positions" in buffers:
            pos_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, pos_buffer)
            positions = buffers["positions"]
            glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        # Normal buffer (attribute 1)
        if "normals" in buffers:
            normal_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, normal_buffer)
            normals = buffers["normals"]
            glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        # UV buffer (attribute 2)
        if "uvs" in buffers:
            uv_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, uv_buffer)
            uvs = buffers["uvs"]
            glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        
        # Tangent buffer (attribute 3)
        if "tangents" in buffers:
            tangent_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, tangent_buffer)
            tangents = buffers["tangents"]
            glBufferData(GL_ARRAY_BUFFER, tangents.nbytes, tangents, GL_STATIC_DRAW)
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, None)
        
        # Index buffer
        if "indices" in buffers:
            index_buffer = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)
            indices = buffers["indices"]
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        
        # Cache VAO
        cave_chunk_vaos[chunk_id] = vao
        
        print(f"✅ Created VAO for cave chunk {chunk_id}")
        return vao
        
    except Exception as e:
        print(f"❌ Failed to create cave VAO for {chunk_id}: {e}")
        return None


def RenderCaveChunks(camera_pos: np.ndarray, frustum_cull: bool = True):
    """
    Render all visible cave chunks
    
    Args:
        camera_pos: Camera position for distance culling
        frustum_cull: Whether to perform frustum culling
    """
    global cave_manager, cave_stats
    
    if cave_manager is None or not cave_render_enabled:
        return
    
    caves_rendered = 0
    total_triangles = 0
    total_vertices = 0
    
    try:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Render each cave chunk
        for chunk_id, chunk_info in cave_manager.cave_chunks.items():
            # Distance culling (simple sphere test)
            if frustum_cull:
                chunk_center = chunk_info.bounds_center
                distance = np.linalg.norm(chunk_center - camera_pos)
                
                # Skip if too far away
                if distance > 50.0:  # Adjust render distance as needed
                    continue
            
            # Get or create VAO
            vao_id = CreateCaveVAO(chunk_id)
            if vao_id is None:
                continue
            
            # Set material color
            material_color = cave_manager.get_material_color(chunk_info.material_id)
            
            if debug_cave_materials:
                # Debug: Use material colors
                glColor3f(*material_color)
            else:
                # Normal rendering: slightly darker than terrain
                glColor3f(0.7, 0.6, 0.5)  # Cave color
            
            # Bind and render VAO
            glBindVertexArray(vao_id)
            
            # Render triangles
            glDrawElements(GL_TRIANGLES, chunk_info.index_count, GL_UNSIGNED_INT, None)
            
            caves_rendered += 1
            total_triangles += chunk_info.triangle_count
            total_vertices += chunk_info.vertex_count
            
            glBindVertexArray(0)
        
        # Update statistics
        cave_stats.update({
            'caves_rendered': caves_rendered,
            'cave_triangles': total_triangles,
            'cave_vertices': total_vertices
        })
        
    except Exception as e:
        print(f"❌ Cave rendering error: {e}")


def RenderCombinedTerrainAndCaves(chunked_planet: dict, cave_manifest_path: str = None):
    """
    Render terrain and caves together
    
    Args:
        chunked_planet: Terrain chunk data
        cave_manifest_path: Optional path to cave manifest
    """
    # Load cave manifest if provided
    if cave_manifest_path and cave_manager is None:
        LoadCaveManifest(cave_manifest_path)
    
    # Import viewer functions (dynamic import to avoid circular dependency)
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / "renderer"))
        if 'pcc_game_viewer' not in sys.modules:
            import pcc_game_viewer
        else:
            pcc_game_viewer = sys.modules['pcc_game_viewer']
        
        # Get camera position from viewer
        camera_pos = np.array([
            pcc_game_viewer.camera_x, 
            pcc_game_viewer.camera_y, 
            pcc_game_viewer.camera_z
        ])
        
        # Render terrain first
        if hasattr(pcc_game_viewer, 'RenderChunkedPlanetWithLOD'):
            pcc_game_viewer.RenderChunkedPlanetWithLOD(chunked_planet)
        else:
            pcc_game_viewer.RenderChunkedPlanet(chunked_planet)
        
        # Render caves on top
        RenderCaveChunks(camera_pos, frustum_cull=True)
        
        # Debug visualizations
        if debug_cave_aabb:
            DrawCaveAABBs()
        
    except Exception as e:
        print(f"❌ Combined rendering error: {e}")


def DrawCaveAABBs():
    """Draw AABB wireframes for cave chunks"""
    global cave_manager
    
    if cave_manager is None:
        return
    
    glDisable(GL_DEPTH_TEST)
    glColor3f(0.0, 1.0, 1.0)  # Cyan for cave AABBs
    
    for chunk_id, chunk_info in cave_manager.cave_chunks.items():
        min_pt = chunk_info.aabb_min
        max_pt = chunk_info.aabb_max
        
        # Draw wireframe box
        glBegin(GL_LINES)
        
        # Bottom face
        glVertex3f(min_pt[0], min_pt[1], min_pt[2])
        glVertex3f(max_pt[0], min_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], min_pt[2])
        glVertex3f(max_pt[0], min_pt[1], max_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], max_pt[2])
        glVertex3f(min_pt[0], min_pt[1], max_pt[2])
        
        glVertex3f(min_pt[0], min_pt[1], max_pt[2])
        glVertex3f(min_pt[0], min_pt[1], min_pt[2])
        
        # Top face
        glVertex3f(min_pt[0], max_pt[1], min_pt[2])
        glVertex3f(max_pt[0], max_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], max_pt[1], min_pt[2])
        glVertex3f(max_pt[0], max_pt[1], max_pt[2])
        
        glVertex3f(max_pt[0], max_pt[1], max_pt[2])
        glVertex3f(min_pt[0], max_pt[1], max_pt[2])
        
        glVertex3f(min_pt[0], max_pt[1], max_pt[2])
        glVertex3f(min_pt[0], max_pt[1], min_pt[2])
        
        # Vertical edges
        glVertex3f(min_pt[0], min_pt[1], min_pt[2])
        glVertex3f(min_pt[0], max_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], min_pt[2])
        glVertex3f(max_pt[0], max_pt[1], min_pt[2])
        
        glVertex3f(max_pt[0], min_pt[1], max_pt[2])
        glVertex3f(max_pt[0], max_pt[1], max_pt[2])
        
        glVertex3f(min_pt[0], min_pt[1], max_pt[2])
        glVertex3f(min_pt[0], max_pt[1], max_pt[2])
        
        glEnd()
    
    glEnable(GL_DEPTH_TEST)


def DrawCaveRenderingHUD():
    """Draw cave rendering statistics HUD"""
    global cave_stats
    
    try:
        from OpenGL.GL import *
        
        # Draw semi-transparent background
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Position HUD in bottom-right corner
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        
        # Import window dimensions from viewer
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent / "renderer"))
            import pcc_game_viewer
            width = pcc_game_viewer.window_width
            height = pcc_game_viewer.window_height
        except:
            width, height = 800, 600
        
        glOrtho(0, width, 0, height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Draw background
        x, y = width - 250, 10
        w, h = 240, 80
        
        glColor4f(0.0, 0.0, 0.0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()
        
        # Draw cave statistics (text rendering would require additional setup)
        # For now, just draw colored bars to indicate cave activity
        
        # Cave count bar
        glColor3f(0.0, 1.0, 1.0)  # Cyan
        bar_width = min(200, cave_stats['caves_rendered'] * 10)
        glBegin(GL_QUADS)
        glVertex2f(x + 10, y + 60)
        glVertex2f(x + 10 + bar_width, y + 60)
        glVertex2f(x + 10 + bar_width, y + 70)
        glVertex2f(x + 10, y + 70)
        glEnd()
        
        # Triangle count bar (scaled)
        glColor3f(1.0, 0.0, 1.0)  # Magenta
        bar_width = min(200, cave_stats['cave_triangles'] // 100)
        glBegin(GL_QUADS)
        glVertex2f(x + 10, y + 40)
        glVertex2f(x + 10 + bar_width, y + 40)
        glVertex2f(x + 10 + bar_width, y + 50)
        glVertex2f(x + 10, y + 50)
        glEnd()
        
        # Vertex count bar (scaled)
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        bar_width = min(200, cave_stats['cave_vertices'] // 200)
        glBegin(GL_QUADS)
        glVertex2f(x + 10, y + 20)
        glVertex2f(x + 10 + bar_width, y + 20)
        glVertex2f(x + 10 + bar_width, y + 30)
        glVertex2f(x + 10, y + 30)
        glEnd()
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
    except Exception as e:
        pass  # HUD is optional


def HandleCaveDebugKeys(key: str):
    """
    Handle debug keys for cave rendering
    
    Args:
        key: Key pressed (uppercase)
    """
    global debug_cave_materials, debug_cave_aabb, cave_render_enabled
    
    if key == 'M':
        debug_cave_materials = not debug_cave_materials
        print(f"Cave material debug: {'ON' if debug_cave_materials else 'OFF'}")
    
    elif key == 'B':
        debug_cave_aabb = not debug_cave_aabb
        print(f"Cave AABB debug: {'ON' if debug_cave_aabb else 'OFF'}")
    
    elif key == 'C':
        cave_render_enabled = not cave_render_enabled
        print(f"Cave rendering: {'ON' if cave_render_enabled else 'OFF'}")


def CleanupCaveResources():
    """Cleanup cave rendering resources"""
    global cave_chunk_vaos
    
    try:
        # Delete VAOs
        for vao_id in cave_chunk_vaos.values():
            glDeleteVertexArrays(1, [vao_id])
        
        cave_chunk_vaos.clear()
        print("✅ Cave rendering resources cleaned up")
        
    except Exception as e:
        print(f"⚠️ Error cleaning up cave resources: {e}")


if __name__ == "__main__":
    # Test cave viewer extension
    print("🧪 Testing Cave Viewer Extension")
    print("=" * 40)
    
    # Initialize cave system
    InitializeCaveSystem()
    
    # Test material colors
    print("Material colors:")
    if cave_manager:
        for mat_id in [0, 1, 2]:
            color = cave_manager.get_material_color(mat_id)
            print(f"   Material {mat_id}: {color}")
    
    print("\n✅ Cave viewer extension ready")
    print("\nDebug controls:")
    print("   M - Toggle material debug colors")
    print("   B - Toggle cave AABB wireframes") 
    print("   C - Toggle cave rendering on/off")