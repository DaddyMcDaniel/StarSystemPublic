#!/usr/bin/env python3
"""
SUMMARY: Enhanced PCC 3D Viewer with FP Navigation
==================================================
OpenGL viewer with first-person mouse-look, mini-planet support, and Agent B bridge.
Implements Week 3 alpha proof requirements for seamless 3D navigation.

KEY FEATURES:
- True first-person navigation with WASD+mouse-look
- Mini-planet terrain rendering with spherical worlds
- Agent B bridge protocol for automated testing
- Layer-aware rendering (orbit/surface/subsurface)
- Building preview and interaction support

CONTROLS:
- WASD: Movement
- Mouse: Look around (first-person)
- Space/C: Up/Down movement
- Tab: Toggle mouse capture
- ESC: Exit

USAGE:
  python renderer/pcc_simple_viewer.py scene.json
  python renderer/pcc_simple_viewer.py scene.json --bridge-port 8765

RELATED FILES:
- scripts/run_gl.py - Quick launcher with mini-planet generation
- Week 3 requirement for alpha proof with FP navigation
"""
import json
import sys
import math
import argparse
import threading
import socket
from pathlib import Path

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except ImportError:
    print("❌ OpenGL not available")
    sys.exit(1)

class EnhancedPCCViewer:
    def __init__(self, scene_file, bridge_port=None):
        self.scene_file = scene_file
        self.scene_data = self.load_scene()
        self.bridge_port = bridge_port
        
        # First-person camera with mouse look
        self.camera_pos = [0.0, 3.0, 8.0]
        self.camera_pitch = 0.0  # X-axis rotation (up/down)
        self.camera_yaw = 0.0    # Y-axis rotation (left/right)
        self.movement_speed = 5.0
        self.mouse_sensitivity = 0.1
        
        # Mouse control state
        self.mouse_captured = True
        self.last_mouse_x = 400
        self.last_mouse_y = 300
        self.first_mouse = True
        
        # Input state
        self.keys_pressed = set()
        
        # Mini-planet support
        self.planet_radius = self.scene_data.get('terrain', {}).get('radius', 15.0)
        self.planet_center = self.scene_data.get('terrain', {}).get('center', [0, -15, 0])
        
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
            return {"objects": []}
    
    def init_gl(self):
        """Initialize OpenGL with basic settings"""
        glClearColor(0.4, 0.6, 0.9, 1.0)  # Sky blue
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Simple lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        
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
                    response = {"status": "viewer_active", "camera_pos": self.camera_pos}
                    client.send(json.dumps(response).encode('utf-8'))
                    client.close()
            except Exception as e:
                print(f"Bridge server error: {e}")
        
        self.bridge_server = threading.Thread(target=bridge_handler, daemon=True)
        self.bridge_server.start()
    
    def handle_movement(self):
        """Handle first-person WASD movement with proper frame timing"""
        # Calculate movement vectors based on camera yaw/pitch
        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)
        
        # Forward/backward (affected by pitch for true FP movement)
        forward_x = math.sin(yaw_rad) * math.cos(pitch_rad)
        forward_y = -math.sin(pitch_rad)
        forward_z = math.cos(yaw_rad) * math.cos(pitch_rad)
        
        # Right/left (perpendicular to forward, not affected by pitch)
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        # Movement speed adjusted for frame rate
        frame_speed = self.movement_speed * 0.016  # ~60fps timing
        
        # Apply movement
        if ord('w') in self.keys_pressed or ord('W') in self.keys_pressed:
            self.camera_pos[0] += forward_x * frame_speed
            self.camera_pos[1] += forward_y * frame_speed
            self.camera_pos[2] += forward_z * frame_speed
        if ord('s') in self.keys_pressed or ord('S') in self.keys_pressed:
            self.camera_pos[0] -= forward_x * frame_speed
            self.camera_pos[1] -= forward_y * frame_speed
            self.camera_pos[2] -= forward_z * frame_speed
        if ord('a') in self.keys_pressed or ord('A') in self.keys_pressed:
            self.camera_pos[0] -= right_x * frame_speed
            self.camera_pos[2] -= right_z * frame_speed
        if ord('d') in self.keys_pressed or ord('D') in self.keys_pressed:
            self.camera_pos[0] += right_x * frame_speed
            self.camera_pos[2] += right_z * frame_speed
        if ord(' ') in self.keys_pressed:  # Space for up
            self.camera_pos[1] += frame_speed
        if ord('c') in self.keys_pressed or ord('C') in self.keys_pressed:  # C for down
            self.camera_pos[1] -= frame_speed
            
        # Keep camera above ground (mini-planet aware)
        min_height = self.get_ground_height_at(self.camera_pos[0], self.camera_pos[2]) + 1.5
        self.camera_pos[1] = max(min_height, self.camera_pos[1])
        
        # Keep camera within reasonable bounds
        max_distance = self.planet_radius + 50
        for i in [0, 2]:  # X and Z bounds
            self.camera_pos[i] = max(-max_distance, min(max_distance, self.camera_pos[i]))
    
    def get_ground_height_at(self, x, z):
        """Get ground height at position for mini-planet"""
        # Distance from planet center
        dx = x - self.planet_center[0]
        dz = z - self.planet_center[2]
        distance_from_center = math.sqrt(dx*dx + dz*dz)
        
        if distance_from_center <= self.planet_radius:
            # On the planet surface
            height_offset = math.sqrt(max(0, self.planet_radius**2 - distance_from_center**2))
            return self.planet_center[1] + height_offset
        else:
            # Off the planet
            return self.planet_center[1]
    
    def draw_miniplanet_terrain(self):
        """Draw mini-planet spherical terrain"""
        terrain = self.scene_data.get('terrain', {})
        if terrain.get('type') == 'sphere':
            glPushMatrix()
            center = terrain.get('center', [0, -15, 0])
            radius = terrain.get('radius', 15.0)
            glTranslatef(*center)
            glColor3f(0.4, 0.3, 0.2)  # Brown planet surface
            
            # Draw wireframe sphere for planet
            glutWireSphere(radius, 20, 20)
            
            # Draw solid sphere with transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.4, 0.3, 0.2, 0.3)
            glutSolidSphere(radius, 12, 12)
            glDisable(GL_BLEND)
            
            glPopMatrix()
        else:
            # Fallback to flat ground
            self.draw_flat_ground()
        
        glColor3f(1, 1, 1)  # Reset color
    
    def draw_flat_ground(self):
        """Draw simple flat ground plane"""
        glColor3f(0.2, 0.6, 0.2)  # Green
        
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-50, 0, -50)
        glVertex3f(50, 0, -50)
        glVertex3f(50, 0, 50)
        glVertex3f(-50, 0, 50)
        glEnd()
    
    def draw_cube(self, pos, size, color):
        """Draw cube with proper size scaling"""
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glScalef(size[0], size[1], size[2])
        glColor3f(*color)
        glutSolidCube(1.0)
        glPopMatrix()
    
    def draw_sphere(self, pos, radius, color):
        """Draw simple sphere"""
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glColor3f(*color)
        glutSolidSphere(radius, 12, 12)
        glPopMatrix()
    
    def display(self):
        """Main display function with first-person camera"""
        # Handle movement
        self.handle_movement()
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # First-person camera setup
        glRotatef(self.camera_pitch, 1, 0, 0)  # Pitch (up/down)
        glRotatef(self.camera_yaw, 0, 1, 0)    # Yaw (left/right)
        glTranslatef(-self.camera_pos[0], -self.camera_pos[1], -self.camera_pos[2])
        
        # Draw terrain (mini-planet or flat)
        self.draw_miniplanet_terrain()
        
        # Draw scene objects with enhanced materials
        for obj in self.scene_data.get("objects", []):
            obj_type = obj.get("type", "")
            pos = obj.get("pos", [0, 0, 0])
            material = obj.get("material", "")
            
            if obj_type == "CUBE":
                size = obj.get("size", [1, 1, 1])
                color = self.get_material_color(material, "cube")
                self.draw_cube(pos, size, color)
                
            elif obj_type == "SPHERE":
                radius = max(obj.get("radius", 0.5), 0.3)
                color = self.get_material_color(material, "sphere")
                self.draw_sphere(pos, radius, color)
        
        # Draw coordinate axes for reference
        self.draw_coordinate_axes()
        
        # Draw HUD info
        self.draw_hud_info()
        
        glutSwapBuffers()
    
    def get_material_color(self, material, obj_type):
        """Get color based on material type"""
        material_colors = {
            "terrain": (0.6, 0.4, 0.2),      # Brown
            "resource": (1.0, 1.0, 0.0),     # Yellow
            "structure": (0.7, 0.7, 0.7),    # Gray
            "beacon": (0.0, 1.0, 1.0),       # Cyan
            "boss": (1.0, 0.0, 0.0),         # Red
            "enemy": (0.8, 0.2, 0.2),        # Dark red
            "collectible": (1.0, 1.0, 0.0),  # Yellow
            "player": (0.0, 1.0, 0.0),       # Green
        }
        
        if material in material_colors:
            return material_colors[material]
        elif obj_type == "cube":
            return (0.5, 0.5, 1.0)  # Default blue for cubes
        else:
            return (1.0, 1.0, 1.0)  # Default white for spheres
    
    def draw_coordinate_axes(self):
        """Draw coordinate axes for spatial reference"""
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)  # X-axis (red)
        glVertex3f(0, 0.1, 0)
        glVertex3f(5, 0.1, 0)
        glColor3f(0.0, 1.0, 0.0)  # Y-axis (green)
        glVertex3f(0, 0.1, 0)
        glVertex3f(0, 5.1, 0)
        glColor3f(0.0, 0.0, 1.0)  # Z-axis (blue)
        glVertex3f(0, 0.1, 0)
        glVertex3f(0, 0.1, 5)
        glEnd()
    
    def draw_hud_info(self):
        """Draw HUD information overlay"""
        # Save current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 800, 0, 600, -1, 1)  # 2D overlay
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for overlay
        glDisable(GL_DEPTH_TEST)
        
        # Draw simple position indicator
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(10, 580)
        glVertex2f(200, 580)
        glVertex2f(200, 590)
        glVertex2f(10, 590)
        glEnd()
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def keyboard(self, key, x, y):
        """Handle key press with FP navigation controls"""
        key_code = ord(key) if isinstance(key, bytes) else key
        self.keys_pressed.add(key_code)
        
        if key == b'\x1b':  # Escape
            print("👋 Enhanced viewer closed by user")
            print(f"📍 Final position: ({self.camera_pos[0]:.1f}, {self.camera_pos[1]:.1f}, {self.camera_pos[2]:.1f})")
            print(f"📍 Final view angle: pitch={self.camera_pitch:.1f}°, yaw={self.camera_yaw:.1f}°")
            glutLeaveMainLoop()
        elif key == b'\t':  # Tab - toggle mouse capture
            self.mouse_captured = not self.mouse_captured
            if self.mouse_captured:
                glutSetCursor(GLUT_CURSOR_NONE)
                glutWarpPointer(400, 300)
                print("🔒 Mouse captured - look around")
            else:
                glutSetCursor(GLUT_CURSOR_INHERIT)
                print("🔓 Mouse released")
        elif key == b'+' or key == b'=':
            self.movement_speed = min(self.movement_speed * 1.2, 20.0)
            print(f"🏃 Movement speed: {self.movement_speed:.1f}")
        elif key == b'-':
            self.movement_speed = max(self.movement_speed * 0.8, 0.5)
            print(f"🚶 Movement speed: {self.movement_speed:.1f}")
        elif key == b'r' or key == b'R':  # Reset position
            self.camera_pos = [0.0, 3.0, 8.0]
            self.camera_pitch = 0.0
            self.camera_yaw = 0.0
            print("🔄 Camera reset")
        
        glutPostRedisplay()
    
    def mouse_motion(self, x, y):
        """Handle mouse motion for first-person look"""
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
        
        # Update camera angles
        self.camera_yaw += delta_x * self.mouse_sensitivity
        self.camera_pitch -= delta_y * self.mouse_sensitivity  # Inverted Y
        
        # Clamp pitch to prevent flipping
        self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
        
        # Keep yaw in 0-360 range
        self.camera_yaw = self.camera_yaw % 360.0
        
        self.last_mouse_x = x
        self.last_mouse_y = y
        
        glutPostRedisplay()
    
    def mouse_passive_motion(self, x, y):
        """Handle passive mouse motion"""
        self.mouse_motion(x, y)
    
    def keyboard_up(self, key, x, y):
        """Handle key release"""
        key_code = ord(key) if isinstance(key, bytes) else key
        self.keys_pressed.discard(key_code)
    
    def reshape(self, width, height):
        """Handle window reshape"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, width/height, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
    
    def run(self):
        """Run the enhanced viewer with FP navigation"""
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutInitWindowPosition(100, 100)
        
        scene_name = Path(self.scene_file).name
        window_title = f"StarSystem Enhanced Viewer - {scene_name}"
        if self.bridge_port:
            window_title += f" (Bridge: {self.bridge_port})"
        glutCreateWindow(window_title)
        
        self.init_gl()
        
        # Set up all callbacks including mouse
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutKeyboardUpFunc(self.keyboard_up)
        glutMotionFunc(self.mouse_motion)
        glutPassiveMotionFunc(self.mouse_passive_motion)
        
        # Start with mouse captured for FP experience
        glutSetCursor(GLUT_CURSOR_NONE)
        glutWarpPointer(400, 300)
        
        # Continuous redraw for smooth movement
        glutIdleFunc(self.display)
        
        print("🎮 Enhanced PCC 3D Viewer - Week 3 Alpha")
        print("📋 First-Person Controls:")
        print("   WASD - Move around")
        print("   Mouse - Look around (first-person)")
        print("   Space/C - Move up/down")
        print("   Tab - Toggle mouse capture")
        print("   +/- - Change movement speed")
        print("   R - Reset camera position")
        print("   ESC - Quit")
        print()
        print(f"🌍 Scene: {scene_name}")
        print(f"🌍 Planet radius: {self.planet_radius:.1f}")
        if self.bridge_port:
            print(f"🌉 Bridge server: localhost:{self.bridge_port}")
        print("💡 True first-person navigation enabled!")
        
        glutMainLoop()

def main():
    parser = argparse.ArgumentParser(description="Enhanced PCC 3D Viewer with FP Navigation")
    parser.add_argument("scene_file", help="Scene file to load")
    parser.add_argument("--bridge-port", type=int, help="Port for Agent B bridge protocol")
    
    args = parser.parse_args()
    
    if not Path(args.scene_file).exists():
        print(f"❌ Scene file not found: {args.scene_file}")
        return 1
    
    viewer = EnhancedPCCViewer(args.scene_file, args.bridge_port)
    viewer.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())
