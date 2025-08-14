#!/usr/bin/env python3
"""
SUMMARY: Spherical Planet Viewer with True Surface Walking
==========================================================
OpenGL viewer implementing proper tiny planet mechanics where player walks on curved surface.
Implements spherical gravity, surface-locked movement, and curved horizon rendering.

KEY FEATURES:
- True spherical surface walking (player locked to planet surface)
- Dynamic gravity orientation (up vector always away from planet center)
- Visible planet curvature and curved horizons
- Surface-relative movement (WASD moves along planet surface)
- Circumnavigation support (walk full circles around planet)
- Enhanced Valheim-style lighting and materials

PLANET MECHANICS:
- Player position locked to planet surface at radius distance
- Camera orientation follows planet surface normal
- Movement vectors projected onto sphere surface
- Horizon shows visible curvature of tiny planet
- Full 360° navigation around planet with seamless transitions

CONTROLS:
- WASD: Move along planet surface (relative to current orientation)
- Mouse: Look around (first-person, surface-relative)
- Space: Jump slightly off surface
- Tab: Toggle mouse capture
- R: Reset to north pole position
- ESC: Exit

USAGE:
  python renderer/pcc_spherical_viewer.py scene.json
  python renderer/pcc_spherical_viewer.py scene.json --bridge-port 8765

RELATED FILES:
- scripts/run_gl.py - Launcher with mini-planet generation
- Week 3 requirement for true spherical world navigation
"""

import json
import sys
import math
import argparse
import threading
import socket
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except ImportError as e:
    print(f"❌ OpenGL not available: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

class Vector3:
    """Simple 3D vector class for planet calculations"""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def normalized(self):
        l = self.length()
        if l > 0:
            return Vector3(self.x/l, self.y/l, self.z/l)
        return Vector3(0, 1, 0)
    
    def to_list(self):
        return [self.x, self.y, self.z]

class SphericalPlanetViewer:
    def __init__(self, scene_file, bridge_port=None):
        self.scene_file = scene_file
        self.scene_data = self.load_scene()
        self.bridge_port = bridge_port
        
        # Planet properties
        terrain = self.scene_data.get('terrain', {})
        self.planet_radius = terrain.get('radius', 15.0)
        self.planet_center = Vector3(*terrain.get('center', [0, 0, 0]))
        
        # Player position on sphere surface (start at equator to avoid north pole beacon)
        self.surface_height_offset = 2.5  # Player height above surface (increased)
        # Start at equator position to avoid spawning inside north pole beacon
        self.player_sphere_pos = Vector3(self.planet_radius, 0, 0)  # Equator (east)
        surface_normal = self.get_surface_normal(self.player_sphere_pos)
        self.player_world_pos = self.planet_center + self.player_sphere_pos + surface_normal * self.surface_height_offset
        
        # Surface-relative camera orientation
        self.surface_yaw = 0.0    # Rotation around surface normal (left/right)
        self.surface_pitch = 0.0  # Pitch relative to surface (up/down look)
        
        # Movement properties
        self.movement_speed = 3.0  # Units per second along surface
        self.mouse_sensitivity = 0.15
        self.jump_velocity = 0.0
        self.gravity_strength = 9.8
        
        # Mouse control
        self.mouse_captured = True
        self.last_mouse_x = 400
        self.last_mouse_y = 300
        self.first_mouse = True
        self.window_width = 1024
        self.window_height = 768
        
        # Input state
        self.keys_pressed = set()
        self.last_frame_time = time.time()
        
        # Enhanced graphics
        self.fog_enabled = True
        self.detail_level = "high"
        
        # Bridge server for Agent B
        self.bridge_server = None
        if bridge_port:
            self.setup_bridge_server(bridge_port)
        
        # Ensure we don't spawn intersecting any placed object
        self._ensure_safe_spawn_position()
    
    def load_scene(self):
        """Load 3D scene data"""
        try:
            with open(self.scene_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load scene: {e}")
            return {"objects": [], "terrain": {"type": "sphere", "radius": 15.0, "center": [0, 0, 0]}}
    
    def setup_bridge_server(self, port):
        """Setup bridge server for Agent B communication"""
        def bridge_handler():
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(('localhost', port))
                server.listen(1)
                print(f"🌉 Bridge server listening on port {port}")
                
                while True:
                    client, addr = server.accept()
                    data = client.recv(1024).decode('utf-8')
                    
                    # Send detailed planet state for Agent B
                    response = {
                        "status": "spherical_viewer_active",
                        "player_world_pos": self.player_world_pos.to_list(),
                        "player_sphere_pos": self.player_sphere_pos.to_list(),
                        "surface_yaw": self.surface_yaw,
                        "surface_pitch": self.surface_pitch,
                        "planet_radius": self.planet_radius,
                        "planet_center": self.planet_center.to_list(),
                        "can_circumnavigate": True
                    }
                    client.send(json.dumps(response).encode('utf-8'))
                    client.close()
            except Exception as e:
                print(f"Bridge server error: {e}")
        
        self.bridge_server = threading.Thread(target=bridge_handler, daemon=True)
        self.bridge_server.start()
    
    def get_surface_normal(self, sphere_pos):
        """Get surface normal at position on sphere"""
        return sphere_pos.normalized()
    
    def get_surface_basis(self, sphere_pos, yaw):
        """Get local coordinate system for surface at position"""
        # Surface normal (up direction)
        up = self.get_surface_normal(sphere_pos)
        
        # Calculate north direction (toward +Y pole, projected onto surface)
        north_world = Vector3(0, 1, 0)
        north = north_world - up * up.dot(north_world)
        if north.length() < 0.001:  # At poles, use arbitrary north
            north = Vector3(1, 0, 0) - up * up.dot(Vector3(1, 0, 0))
        north = north.normalized()
        
        # Calculate east direction (right when facing north)
        east = north.cross(up).normalized()
        
        # Apply yaw rotation around up vector
        cos_yaw = math.cos(math.radians(yaw))
        sin_yaw = math.sin(math.radians(yaw))
        
        forward = north * cos_yaw + east * sin_yaw
        right = forward.cross(up).normalized()
        
        return forward, right, up
    
    def update_player_position(self, delta_time):
        """Update player position with surface-locked movement"""
        if not self.keys_pressed:
            return
        
        # Get current surface basis vectors
        forward, right, up = self.get_surface_basis(self.player_sphere_pos, self.surface_yaw)
        
        # Calculate movement in surface coordinates
        move_forward = 0.0
        move_right = 0.0
        
        if ord('w') in self.keys_pressed or ord('W') in self.keys_pressed:
            move_forward += 1.0
        if ord('s') in self.keys_pressed or ord('S') in self.keys_pressed:
            move_forward -= 1.0
        if ord('d') in self.keys_pressed or ord('D') in self.keys_pressed:
            move_right += 1.0
        if ord('a') in self.keys_pressed or ord('A') in self.keys_pressed:
            move_right -= 1.0
        
        # Apply movement along surface
        if move_forward != 0.0 or move_right != 0.0:
            # Calculate movement vector in world space
            movement = forward * move_forward + right * move_right
            movement = movement.normalized() * self.movement_speed * delta_time
            
            # Project movement onto sphere surface
            new_sphere_pos = self.player_sphere_pos + movement
            
            # Normalize to keep on sphere surface
            self.player_sphere_pos = new_sphere_pos.normalized() * self.planet_radius
            
            # Update world position
            surface_normal = self.get_surface_normal(self.player_sphere_pos)
            self.player_world_pos = self.planet_center + self.player_sphere_pos + surface_normal * self.surface_height_offset
        
        # Handle jumping
        if ord(' ') in self.keys_pressed and self.jump_velocity <= 0:
            self.jump_velocity = 5.0
        
        # Apply gravity and jumping
        if self.jump_velocity > 0:
            surface_normal = self.get_surface_normal(self.player_sphere_pos)
            self.player_world_pos = self.player_world_pos + surface_normal * self.jump_velocity * delta_time
            self.jump_velocity -= self.gravity_strength * delta_time
            
            # Land back on surface
            if self.jump_velocity <= 0:
                self.jump_velocity = 0
                surface_normal = self.get_surface_normal(self.player_sphere_pos)
                self.player_world_pos = self.planet_center + self.player_sphere_pos + surface_normal * self.surface_height_offset
    
    def init_gl(self):
        """Initialize OpenGL with enhanced Valheim-style settings"""
        # Sky gradient colors
        glClearColor(0.53, 0.81, 0.92, 1.0)  # Light blue sky
        
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        
        # Enhanced lighting setup
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)  # Secondary light
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Main sun light (directional)
        sun_direction = [0.3, 0.8, 0.5, 0.0]  # Directional light
        sun_color = [1.0, 0.95, 0.8, 1.0]     # Warm sunlight
        glLightfv(GL_LIGHT0, GL_POSITION, sun_direction)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, sun_color)
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        
        # Ambient sky light
        sky_ambient = [0.4, 0.5, 0.7, 1.0]    # Cool ambient
        glLightfv(GL_LIGHT0, GL_AMBIENT, sky_ambient)
        
        # Secondary fill light (opposite direction)
        fill_direction = [-0.2, 0.3, -0.4, 0.0]
        fill_color = [0.3, 0.4, 0.6, 1.0]     # Cool fill light
        glLightfv(GL_LIGHT1, GL_POSITION, fill_direction)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, fill_color)
        
        # Fog for atmospheric perspective
        if self.fog_enabled:
            glEnable(GL_FOG)
            glFogi(GL_FOG_MODE, GL_EXP2)
            glFogfv(GL_FOG_COLOR, [0.6, 0.7, 0.8, 1.0])  # Hazy blue
            glFogf(GL_FOG_DENSITY, 0.015)  # Moderate fog
            glFogf(GL_FOG_START, 10.0)
            glFogf(GL_FOG_END, 50.0)
        
        # Enable smooth shading and anti-aliasing
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        
        # Enable blending for transparency effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    def draw_planet_surface(self):
        """Draw detailed planet surface with curvature"""
        glPushMatrix()
        glTranslatef(*self.planet_center.to_list())
        
        # Draw main planet surface with texture-like appearance
        glColor3f(0.4, 0.25, 0.15)  # Rich brown soil
        
        # High-detail sphere for close viewing
        glEnable(GL_LIGHTING)
        glutSolidSphere(self.planet_radius - 0.1, 32, 32)
        
        # Draw surface details in concentric rings for curvature effect
        for ring in range(3):
            radius = self.planet_radius + ring * 0.05
            alpha = 0.6 - ring * 0.2
            
            if ring == 0:
                glColor4f(0.5, 0.3, 0.2, alpha)  # Surface layer
            elif ring == 1:
                glColor4f(0.3, 0.5, 0.3, alpha)  # Grass patches
            else:
                glColor4f(0.2, 0.3, 0.2, alpha)  # Vegetation
            
            # Wire sphere to show surface detail
            glutWireSphere(radius, 24, 24)
        
        glPopMatrix()
    
    def draw_horizon_atmosphere(self):
        """Draw atmospheric effects showing planet curvature"""
        player_to_center = self.player_world_pos - self.planet_center
        distance_from_center = player_to_center.length()
        
        # Only draw if player is close to surface
        if distance_from_center < self.planet_radius + 10:
            glPushMatrix()
            glTranslatef(*self.planet_center.to_list())
            
            # Atmospheric halo around planet
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glColor4f(0.7, 0.8, 1.0, 0.2)  # Atmospheric blue
            glutWireSphere(self.planet_radius + 2, 16, 16)
            
            # Outer atmosphere
            glColor4f(0.5, 0.6, 0.9, 0.1)
            glutWireSphere(self.planet_radius + 4, 12, 12)
            
            glEnable(GL_LIGHTING)
            glPopMatrix()
    
    def get_material_color(self, material, obj_type):
        """Get color based on material type with Agent B feedback enhancements"""
        material_colors = {
            # Legacy materials
            "terrain": (0.6, 0.4, 0.2),      # Brown
            "resource": (1.0, 1.0, 0.0),     # Yellow
            "structure": (0.7, 0.7, 0.7),    # Gray
            "beacon": (0.0, 1.0, 1.0),       # Cyan
            
            # Agent B feedback: Enhanced terrain variety
            "terrain_rock": (0.5, 0.4, 0.3),     # Dark brown rock
            "terrain_grass": (0.2, 0.6, 0.2),    # Green grass
            "terrain_sand": (0.8, 0.7, 0.4),     # Sandy beige
            "terrain_crystal": (0.7, 0.5, 0.9),  # Purple crystal
            "terrain_metal": (0.6, 0.6, 0.7),    # Metallic gray
            
            # Enhanced resource variety
            "resource_ore": (0.8, 0.6, 0.2),     # Bronze ore
            "resource_crystal": (0.4, 0.8, 1.0), # Blue crystal
            "resource_energy": (1.0, 0.4, 0.8),  # Pink energy
            "resource_rare": (0.9, 0.1, 0.9),    # Rare purple
            
            # Landmark materials
            "landmark_stone": (0.4, 0.4, 0.4),   # Dark stone
            "landmark_crater": (0.3, 0.2, 0.2),  # Dark crater rim
            "landmark_pillar": (0.8, 0.8, 0.9),  # Light pillar
            "landmark_arch": (0.6, 0.5, 0.4),    # Sandstone arch
            "landmark_spire": (0.9, 0.9, 1.0),   # White spire
            
            # Enhanced structures
            "structure_temple": (0.8, 0.7, 0.5), # Golden temple
            "structure_monument": (0.4, 0.4, 0.5), # Dark monument
            "beacon_major": (0.2, 1.0, 1.0),     # Bright cyan beacon
            "cave_entrance": (0.1, 0.1, 0.1),    # Dark cave
            
            # Legacy
            "boss": (1.0, 0.0, 0.0),         # Red
            "enemy": (0.8, 0.2, 0.2),        # Dark red
            "collectible": (1.0, 1.0, 0.0),  # Yellow
            "player": (0.0, 1.0, 0.0),       # Green
        }
        
        if material in material_colors:
            return material_colors[material]
        elif obj_type == "CUBE":
            return (0.5, 0.5, 1.0)  # Default blue for cubes
        else:
            return (1.0, 1.0, 1.0)  # Default white for spheres

    def _is_position_colliding(self, world_pos: "Vector3", clearance: float = 1.5) -> bool:
        """Approximate collision test against scene objects using spherical bounds."""
        for obj in self.scene_data.get("objects", []):
            pos = obj.get("pos", [0, 0, 0])
            dx = world_pos.x - float(pos[0])
            dy = world_pos.y - float(pos[1])
            dz = world_pos.z - float(pos[2])
            dist_sq = dx*dx + dy*dy + dz*dz
            if obj.get("type") == "SPHERE":
                r = max(float(obj.get("radius", 0.5)), 0.3)
                if dist_sq < (r + clearance) * (r + clearance):
                    return True
            elif obj.get("type") == "CUBE":
                size = obj.get("size", [1, 1, 1])
                r = 0.5 * float(max(size[0], size[1], size[2]))
                if dist_sq < (r + clearance) * (r + clearance):
                    return True
        return False

    def _ensure_safe_spawn_position(self):
        """Rotate around equator to find a non-colliding spawn position, else raise height a bit."""
        for angle_deg in range(0, 360, 10):
            ang = math.radians(angle_deg)
            candidate_sphere = Vector3(self.planet_radius * math.cos(ang), 0.0, self.planet_radius * math.sin(ang))
            nrm = self.get_surface_normal(candidate_sphere)
            candidate_world = self.planet_center + candidate_sphere + nrm * self.surface_height_offset
            if not self._is_position_colliding(candidate_world, clearance=1.0):
                self.player_sphere_pos = candidate_sphere
                self.player_world_pos = candidate_world
                return
        # Fallback: lift up slightly if all equator spots are blocked
        nrm = self.get_surface_normal(self.player_sphere_pos)
        self.player_world_pos = self.planet_center + self.player_sphere_pos + nrm * (self.surface_height_offset + 1.0)

    def draw_enhanced_objects(self):
        """Draw scene objects with enhanced Valheim-style materials"""
        for obj in self.scene_data.get("objects", []):
            obj_type = obj.get("type", "")
            pos = obj.get("pos", [0, 0, 0])
            material = obj.get("material", "")
            mesh_file = obj.get("mesh_file") if obj_type == "MESH" else None
            
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            
            if obj_type == "CUBE":
                size = obj.get("size", [1, 1, 1])
                # Align cube so its local +Y points along provided up vector (toward surface normal)
                up = obj.get("up")
                if up:
                    ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
                    # Build rotation matrix to map (0,1,0) to (ux,uy,uz)
                    # Axis = cross((0,1,0), up), angle = arccos(dot)
                    dot = max(-1.0, min(1.0, uy))
                    angle_deg = math.degrees(math.acos(dot))
                    ax = -uz
                    ay = 0.0
                    az = ux
                    al = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
                    ax, ay, az = ax/al, ay/al, az/al
                    if angle_deg > 1e-3:
                        glRotatef(angle_deg, ax, ay, az)
                glScalef(size[0], size[1], size[2])
                
                # Get enhanced material color
                color = self.get_material_color(material, obj_type)
                glColor3f(*color)
                glutSolidCube(1.0)
                
                # Add texture effects for special materials
                if material.startswith("terrain_crystal"):
                    # Add crystal sparkle effect
                    glColor3f(min(1.0, color[0] + 0.3), min(1.0, color[1] + 0.3), min(1.0, color[2] + 0.3))
                    glutWireCube(1.02)
                elif material.startswith("landmark_"):
                    # Add detailed lines for landmarks
                    glColor3f(max(0.0, color[0] - 0.2), max(0.0, color[1] - 0.2), max(0.0, color[2] - 0.2))
                    glutWireCube(1.01)
                elif material.startswith("structure_"):
                    # Add masonry lines for structures
                    glColor3f(max(0.0, color[0] - 0.15), max(0.0, color[1] - 0.15), max(0.0, color[2] - 0.15))
                    glutWireCube(1.005)
                    
            elif obj_type == "SPHERE":
                radius = max(obj.get("radius", 0.5), 0.3)
                
                # Get enhanced material color
                color = self.get_material_color(material, obj_type)
                glColor3f(*color)
                glutSolidSphere(radius, 16, 16)
                
                # Add special effects for specific materials
                if material.startswith("resource_crystal"):
                    # Add crystal sparkle effect
                    glColor3f(min(1.0, color[0] + 0.4), min(1.0, color[1] + 0.4), min(1.0, color[2] + 0.4))
                    glutWireSphere(radius * 1.1, 12, 12)
                elif material.startswith("beacon"):
                    # Animated beacon with glow
                    glow = 0.5 + 0.5 * math.sin(time.time() * 3)
                    glColor4f(color[0], color[1], color[2], 0.7)
                    glutWireSphere(radius * 1.3, 16, 16)
                    
                    # Pulsing outer glow
                    glColor4f(color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 0.3 * glow)
                    glutWireSphere(radius * 1.6, 12, 12)
                elif material == "cave_entrance":
                    # Dark cave with mysterious glow
                    glColor4f(0.3, 0.2, 0.6, 0.5)  # Purple mysterious glow
                    glutWireSphere(radius * 1.2, 10, 10)
            elif obj_type == "MESH":
                # Complex multi-component mesh objects
                components = obj.get("components", [])
                mesh_file = obj.get("mesh_file")
                size = obj.get("size", [1.0, 1.0, 1.0])
                
                # Apply surface alignment
                up = obj.get("up")
                if up:
                    ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
                    dot = max(-1.0, min(1.0, uy))
                    angle_deg = math.degrees(math.acos(dot))
                    ax = -uz
                    ay = 0.0
                    az = ux
                    al = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
                    ax, ay, az = ax/al, ay/al, az/al
                    if angle_deg > 1e-3:
                        glRotatef(angle_deg, ax, ay, az)
                
                if components:
                    # Render multi-component asset (house made of 2x4s, etc.)
                    self._render_multi_component_asset(components)
                elif mesh_file:
                    # Render single mesh file
                    self._render_mesh_file(mesh_file, size, material)
                else:
                    # Fallback: render as enhanced placeholder
                    glScalef(size[0], size[1], size[2])
                    
                    # Enhanced building-like placeholder
                    color = self.get_material_color(material, obj_type)
                    glColor3f(*color)
                    
                    # Main structure
                    glutSolidCube(0.8)
                    
                    # Add architectural details
                    glColor3f(max(0.0, color[0] - 0.2), max(0.0, color[1] - 0.2), max(0.0, color[2] - 0.2))
                    
                    # Roof
                    glPushMatrix()
                    glTranslatef(0, 0.5, 0)
                    glScalef(0.9, 0.3, 0.9)
                    glutSolidCube(1.0)
                    glPopMatrix()
                    
                    # Foundation
                    glPushMatrix()
                    glTranslatef(0, -0.5, 0)
                    glScalef(1.1, 0.2, 1.1)
                    glutSolidCube(1.0)
                    glPopMatrix()
                    
                    # Framework outline
                    glColor3f(min(1.0, color[0] + 0.3), min(1.0, color[1] + 0.3), min(1.0, color[2] + 0.3))
                    glutWireCube(0.85)
            
            glPopMatrix()
    
    def _render_multi_component_asset(self, components: List[Dict]):
        """Render complex asset built from multiple component meshes."""
        # Group components by material for efficient rendering
        material_groups = {}
        for comp in components:
            material = comp.get("material", "default")
            if material not in material_groups:
                material_groups[material] = []
            material_groups[material].append(comp)
        
        # Render each material group
        for material, comps in material_groups.items():
            color = self._get_component_material_color(material)
            glColor3f(*color)
            
            for comp in comps:
                comp_pos = comp.get("position", [0, 0, 0])
                comp_dims = comp.get("dimensions", [0.1, 0.1, 0.1])
                comp_type = comp.get("component_type", "unknown")
                
                glPushMatrix()
                
                # Position component relative to asset origin
                glTranslatef(comp_pos[0], comp_pos[1], comp_pos[2])
                
                # Scale to component dimensions  
                glScalef(comp_dims[0], comp_dims[1], comp_dims[2])
                
                # Render component based on type
                if comp_type in ["beam", "post", "joist", "rafter"]:
                    # Structural lumber - render as solid beam
                    glutSolidCube(1.0)
                    
                    # Add wood grain lines
                    glColor3f(max(0.0, color[0] - 0.1), max(0.0, color[1] - 0.1), max(0.0, color[2] - 0.1))
                    glutWireCube(1.02)
                    
                elif comp_type in ["siding", "wall_board", "plank"]:
                    # Siding/boards - render as flat planks
                    glScalef(1.0, 0.3, 1.0)  # Thinner profile
                    glutSolidCube(1.0)
                    
                    # Add board lines
                    glColor3f(max(0.0, color[0] - 0.15), max(0.0, color[1] - 0.15), max(0.0, color[2] - 0.15))
                    glutWireCube(1.01)
                    
                elif comp_type in ["foundation", "foundation_wall"]:
                    # Foundation blocks - render as masonry
                    glutSolidCube(1.0)
                    
                    # Add mortar lines
                    glColor3f(max(0.0, color[0] - 0.2), max(0.0, color[1] - 0.2), max(0.0, color[2] - 0.2))
                    glutWireCube(1.05)
                    
                elif comp_type in ["shingles", "roofing"]:
                    # Roofing - render as overlapped scales
                    glScalef(1.0, 0.1, 1.0)  # Very thin
                    glutSolidCube(1.0)
                    
                else:
                    # Default component rendering
                    glutSolidCube(1.0)
                
                glPopMatrix()
    
    def _render_mesh_file(self, mesh_file: str, size: List[float], material: str):
        """Render mesh from GLB file (placeholder for now)."""
        # TODO: Implement actual GLB loading with trimesh/pygltflib
        # For now, render as enhanced placeholder
        
        glScalef(size[0], size[1], size[2])
        color = self.get_material_color(material, "MESH")
        glColor3f(*color)
        
        # Render as detailed proxy mesh
        glutSolidCube(0.9)
        
        # Add mesh wireframe overlay
        glColor3f(min(1.0, color[0] + 0.2), min(1.0, color[1] + 0.2), min(1.0, color[2] + 0.2))
        glutWireCube(0.95)
        
        # Add file reference indicator
        glColor3f(1.0, 1.0, 0.0)  # Yellow indicator
        glPushMatrix()
        glTranslatef(0, 0.6, 0)
        glScalef(0.1, 0.1, 0.1)
        glutSolidSphere(1.0, 8, 8)
        glPopMatrix()
    
    def _get_component_material_color(self, material: str) -> Tuple[float, float, float]:
        """Get color for component materials."""
        component_colors = {
            # Wood materials
            "wood_frame": (0.6, 0.4, 0.2),       # Natural lumber
            "wood_siding": (0.5, 0.35, 0.15),    # Weathered siding
            "wood_planks": (0.7, 0.45, 0.25),    # Fresh planks
            
            # Masonry materials  
            "concrete": (0.7, 0.7, 0.7),         # Gray concrete
            "concrete_block": (0.6, 0.6, 0.6),   # CMU blocks
            "brick": (0.7, 0.3, 0.2),            # Red brick
            
            # Roofing materials
            "roofing_asphalt": (0.2, 0.2, 0.3),  # Dark shingles
            "roofing_metal": (0.5, 0.5, 0.6),    # Metal roofing
            "roofing_tile": (0.8, 0.4, 0.3),     # Clay tile
            
            # Default
            "default": (0.5, 0.5, 0.5)
        }
        
        return component_colors.get(material, component_colors["default"])
    
    def display(self):
        """Main display function with spherical camera"""
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Update player position
        self.update_player_position(delta_time)
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Calculate camera orientation based on surface position
        surface_normal = self.get_surface_normal(self.player_sphere_pos)
        forward, right, up = self.get_surface_basis(self.player_sphere_pos, self.surface_yaw)
        
        # Apply pitch to forward vector
        pitch_rad = math.radians(self.surface_pitch)
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        
        look_forward = forward * cos_pitch + up * sin_pitch
        look_target = self.player_world_pos + look_forward
        
        # Set up camera with surface-relative orientation
        gluLookAt(
            self.player_world_pos.x, self.player_world_pos.y, self.player_world_pos.z,  # Eye
            look_target.x, look_target.y, look_target.z,                              # Target
            up.x, up.y, up.z                                                          # Up
        )
        
        # Draw world
        self.draw_planet_surface()
        self.draw_horizon_atmosphere()
        self.draw_enhanced_objects()
        
        # Draw coordinate reference
        self.draw_coordinate_axes()
        
        glutSwapBuffers()
    
    def draw_coordinate_axes(self):
        """Draw world coordinate axes for reference"""
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        
        # X-axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(5, 0, 0)
        
        # Y-axis (green)  
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 5, 0)
        
        # Z-axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 5)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def keyboard(self, key, x, y):
        """Handle keyboard input"""
        key_code = ord(key) if isinstance(key, bytes) else key
        self.keys_pressed.add(key_code)
        
        if key == b'\x1b':  # Escape
            print("👋 Spherical viewer closed")
            print(f"📍 Final position: {self.player_world_pos.to_list()}")
            print(f"🌍 Planet position: {self.player_sphere_pos.to_list()}")
            glutLeaveMainLoop()
        elif key == b'\t':  # Tab - toggle mouse capture
            self.mouse_captured = not self.mouse_captured
            if self.mouse_captured:
                glutSetCursor(GLUT_CURSOR_NONE)
                glutWarpPointer(self.window_width // 2, self.window_height // 2)
                print("🔒 Mouse captured - spherical look")
            else:
                glutSetCursor(GLUT_CURSOR_INHERIT)
                print("🔓 Mouse released")
        elif key == b'r' or key == b'R':  # Reset to equator (safe spawn)
            # Try equator positions for a safe spawn
            self.surface_yaw = 0.0
            self.surface_pitch = 0.0
            self.player_sphere_pos = Vector3(self.planet_radius, 0, 0)
            self._ensure_safe_spawn_position()
            print("🔄 Reset to equator (safe spawn position)")
        elif key == b'+' or key == b'=':
            self.movement_speed = min(self.movement_speed * 1.2, 15.0)
            print(f"🏃 Speed: {self.movement_speed:.1f}")
        elif key == b'-':
            self.movement_speed = max(self.movement_speed * 0.8, 0.5)
            print(f"🚶 Speed: {self.movement_speed:.1f}")
        
        glutPostRedisplay()
    
    def keyboard_up(self, key, x, y):
        """Handle key release"""
        key_code = ord(key) if isinstance(key, bytes) else key
        self.keys_pressed.discard(key_code)
    
    def mouse_motion(self, x, y):
        """Handle mouse motion for surface-relative look"""
        if not self.mouse_captured:
            return
            
        # Use delta from window center and recenter to allow infinite turning
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        delta_x = x - center_x
        delta_y = y - center_y
        if abs(delta_x) > 1 or abs(delta_y) > 1:
            self.surface_yaw += delta_x * self.mouse_sensitivity
            self.surface_pitch -= delta_y * self.mouse_sensitivity
            self.surface_pitch = max(-89.0, min(89.0, self.surface_pitch))
            self.surface_yaw = self.surface_yaw % 360.0
            glutPostRedisplay()
            glutWarpPointer(center_x, center_y)
    
    def mouse_passive_motion(self, x, y):
        """Handle passive mouse motion"""
        self.mouse_motion(x, y)
    
    def reshape(self, width, height):
        """Handle window reshape"""
        self.window_width = max(1, int(width))
        self.window_height = max(1, int(height))
        glViewport(0, 0, self.window_width, self.window_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Wide FOV to see planet curvature
        aspect = self.window_width / max(1, self.window_height)
        gluPerspective(75, aspect, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
    
    def run(self):
        """Run the spherical planet viewer"""
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        
        # Check for fullscreen mode (for auto loop screenshots)
        fullscreen_mode = os.getenv('GL_FULLSCREEN', '0') == '1'
        
        if fullscreen_mode:
            # Fullscreen mode for auto loop screenshots
            glutGameModeString("1920x1080:32@60")
            if glutGameModeGet(GLUT_GAME_MODE_POSSIBLE):
                glutEnterGameMode()
                print("🖥️ Fullscreen mode enabled for screenshot capture")
            else:
                # Fallback to large window
                glutInitWindowSize(1920, 1080)
                glutInitWindowPosition(0, 0)
                print("⚠️ Fullscreen not available, using large window")
        else:
            # Normal windowed mode for manual testing
            glutInitWindowSize(self.window_width, self.window_height)
            glutInitWindowPosition(100, 100)
        
        scene_name = Path(self.scene_file).name
        window_title = f"StarSystem Spherical Planet - {scene_name}"
        if self.bridge_port:
            window_title += f" (Bridge: {self.bridge_port})"
        glutCreateWindow(window_title)
        
        self.init_gl()
        
        # Set up callbacks
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutKeyboardUpFunc(self.keyboard_up)
        glutMotionFunc(self.mouse_motion)
        glutPassiveMotionFunc(self.mouse_passive_motion)
        
        # Continuous updates for smooth movement
        glutIdleFunc(self.display)
        
        # Start with mouse captured
        glutSetCursor(GLUT_CURSOR_NONE)
        glutWarpPointer(self.window_width // 2, self.window_height // 2)
        
        print("🌍 Spherical Planet Viewer - Week 3 Enhanced")
        print("📋 True Planet Surface Navigation:")
        print("   WASD - Walk along planet surface")
        print("   Mouse - Look around (surface-relative)")
        print("   Space - Jump off surface")
        print("   Tab - Toggle mouse capture")
        print("   R - Reset to north pole")
        print("   +/- - Change movement speed")
        print("   ESC - Quit")
        print()
        print(f"🌍 Planet radius: {self.planet_radius:.1f} units")
        print(f"🌍 Planet circumference: {2 * math.pi * self.planet_radius:.1f} units")
        print("🚶 Walk a full circle to return to start!")
        if self.bridge_port:
            print(f"🌉 Agent B bridge: localhost:{self.bridge_port}")
        print("✨ Enhanced graphics with atmospheric effects")
        
        glutMainLoop()

def main():
    parser = argparse.ArgumentParser(description="Spherical Planet Viewer with True Surface Walking")
    parser.add_argument("scene_file", help="Scene file to load")
    parser.add_argument("--bridge-port", type=int, help="Port for Agent B bridge protocol")
    
    args = parser.parse_args()
    
    if not Path(args.scene_file).exists():
        print(f"❌ Scene file not found: {args.scene_file}")
        return 1
    
    viewer = SphericalPlanetViewer(args.scene_file, args.bridge_port)
    viewer.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())