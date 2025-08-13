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
from pathlib import Path

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except ImportError:
    print("❌ OpenGL not available")
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
        
        # Player position on sphere surface (start at north pole)
        self.surface_height_offset = 1.8  # Player height above surface
        self.player_sphere_pos = Vector3(0, self.planet_radius, 0)  # North pole
        self.player_world_pos = self.planet_center + self.player_sphere_pos + Vector3(0, self.surface_height_offset, 0)
        
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
    
    def draw_enhanced_objects(self):
        """Draw scene objects with enhanced Valheim-style materials"""
        for obj in self.scene_data.get("objects", []):
            obj_type = obj.get("type", "")
            pos = obj.get("pos", [0, 0, 0])
            material = obj.get("material", "")
            
            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])
            
            if obj_type == "CUBE":
                size = obj.get("size", [1, 1, 1])
                glScalef(size[0], size[1], size[2])
                
                # Enhanced material colors
                if material == "terrain":
                    glColor3f(0.6, 0.4, 0.3)  # Rich earth
                    glutSolidCube(1.0)
                    # Add stone texture lines
                    glColor3f(0.4, 0.3, 0.2)
                    glutWireCube(1.02)
                elif material == "structure":
                    glColor3f(0.7, 0.6, 0.5)  # Stone building
                    glutSolidCube(1.0)
                    # Add masonry lines
                    glColor3f(0.5, 0.4, 0.3)
                    glutWireCube(1.01)
                else:
                    glColor3f(0.5, 0.5, 0.8)  # Default blue-gray
                    glutSolidCube(1.0)
                    
            elif obj_type == "SPHERE":
                radius = max(obj.get("radius", 0.5), 0.3)
                
                if material == "resource":
                    glColor3f(1.0, 0.8, 0.2)  # Golden ore
                    glutSolidSphere(radius, 12, 12)
                    # Add sparkle effect
                    glColor3f(1.0, 1.0, 0.5)
                    glutWireSphere(radius * 1.1, 8, 8)
                elif material == "beacon":
                    # Animated beacon with glow
                    glow = 0.5 + 0.5 * math.sin(time.time() * 3)
                    glColor3f(0.2 + glow * 0.8, 1.0, 1.0)
                    glutSolidSphere(radius, 16, 16)
                    
                    # Glow halo
                    glColor4f(0.0, 1.0, 1.0, 0.3)
                    glutWireSphere(radius * 1.5, 12, 12)
                else:
                    glColor3f(0.8, 0.8, 0.9)  # Stone
                    glutSolidSphere(radius, 12, 12)
            
            glPopMatrix()
    
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
                glutWarpPointer(400, 300)
                print("🔒 Mouse captured - spherical look")
            else:
                glutSetCursor(GLUT_CURSOR_INHERIT)
                print("🔓 Mouse released")
        elif key == b'r' or key == b'R':  # Reset to north pole
            self.player_sphere_pos = Vector3(0, self.planet_radius, 0)
            self.surface_yaw = 0.0
            self.surface_pitch = 0.0
            surface_normal = self.get_surface_normal(self.player_sphere_pos)
            self.player_world_pos = self.planet_center + self.player_sphere_pos + surface_normal * self.surface_height_offset
            print("🔄 Reset to north pole")
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
            
        if self.first_mouse:
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.first_mouse = False
            return
        
        # Calculate mouse movement
        delta_x = x - self.last_mouse_x
        delta_y = y - self.last_mouse_y
        
        # Update surface-relative angles
        self.surface_yaw += delta_x * self.mouse_sensitivity
        self.surface_pitch -= delta_y * self.mouse_sensitivity  # Inverted Y
        
        # Clamp pitch
        self.surface_pitch = max(-89.0, min(89.0, self.surface_pitch))
        
        # Keep yaw in range
        self.surface_yaw = self.surface_yaw % 360.0
        
        self.last_mouse_x = x
        self.last_mouse_y = y
        
        glutPostRedisplay()
    
    def mouse_passive_motion(self, x, y):
        """Handle passive mouse motion"""
        self.mouse_motion(x, y)
    
    def reshape(self, width, height):
        """Handle window reshape"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Wide FOV to see planet curvature
        gluPerspective(75, width/height, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
    
    def run(self):
        """Run the spherical planet viewer"""
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(1024, 768)  # Larger window for better experience
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
        glutWarpPointer(400, 300)
        
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