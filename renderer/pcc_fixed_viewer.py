#!/usr/bin/env python3
"""
PCC Fixed 3D Viewer - Proper window focus and input handling
"""
import json
import sys
import math
from pathlib import Path

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except ImportError:
    print("❌ OpenGL not available")
    sys.exit(1)

class FixedPCCViewer:
    def __init__(self, scene_file):
        self.scene_file = scene_file
        self.scene_data = self.load_scene()
        
        # Camera settings
        self.camera_pos = [0.0, 2.0, 8.0]
        self.camera_rotation_x = -15.0
        self.camera_rotation_y = 0.0
        self.movement_speed = 0.3
        
        # Input state
        self.keys_pressed = set()
        self.window_focused = False
        
    def load_scene(self):
        """Load 3D scene data"""
        try:
            with open(self.scene_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load scene: {e}")
            return {"objects": []}
    
    def init_gl(self):
        """Initialize OpenGL"""
        glClearColor(0.4, 0.6, 0.9, 1.0)  # Sky blue
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        
    def handle_movement(self):
        """Handle WASD movement"""
        if not self.window_focused:
            return
            
        yaw_rad = math.radians(self.camera_rotation_y)
        forward_x = math.sin(yaw_rad)
        forward_z = math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        # Apply movement
        if ord('w') in self.keys_pressed or ord('W') in self.keys_pressed:
            self.camera_pos[0] += forward_x * self.movement_speed
            self.camera_pos[2] += forward_z * self.movement_speed
        if ord('s') in self.keys_pressed or ord('S') in self.keys_pressed:
            self.camera_pos[0] -= forward_x * self.movement_speed
            self.camera_pos[2] -= forward_z * self.movement_speed
        if ord('a') in self.keys_pressed or ord('A') in self.keys_pressed:
            self.camera_pos[0] -= right_x * self.movement_speed
            self.camera_pos[2] -= right_z * self.movement_speed
        if ord('d') in self.keys_pressed or ord('D') in self.keys_pressed:
            self.camera_pos[0] += right_x * self.movement_speed
            self.camera_pos[2] += right_z * self.movement_speed
        if ord(' ') in self.keys_pressed:  # Space for up
            self.camera_pos[1] += self.movement_speed
        if ord('c') in self.keys_pressed or ord('C') in self.keys_pressed:  # C for down
            self.camera_pos[1] -= self.movement_speed
            
        # Keep camera within bounds
        self.camera_pos[0] = max(-50, min(50, self.camera_pos[0]))
        self.camera_pos[1] = max(0.5, min(20, self.camera_pos[1]))
        self.camera_pos[2] = max(-50, min(50, self.camera_pos[2]))
    
    def draw_ground(self):
        """Draw ground plane"""
        glColor3f(0.2, 0.6, 0.2)  # Green
        
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-20, 0, -20)
        glVertex3f(20, 0, -20)
        glVertex3f(20, 0, 20)
        glVertex3f(-20, 0, 20)
        glEnd()
        
        glColor3f(1, 1, 1)  # Reset color
    
    def draw_cube(self, pos, size, color):
        """Draw cube"""
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glColor3f(*color)
        glutSolidCube(max(size))
        glPopMatrix()
    
    def draw_sphere(self, pos, radius, color):
        """Draw sphere"""
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glColor3f(*color)
        glutSolidSphere(radius, 12, 12)
        glPopMatrix()
    
    def display(self):
        """Main display function"""
        self.handle_movement()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set up camera
        glTranslatef(0, 0, -8)
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        glTranslatef(-self.camera_pos[0], -self.camera_pos[1], -self.camera_pos[2])
        
        # Draw ground
        self.draw_ground()
        
        # Draw scene objects
        for obj in self.scene_data.get("objects", []):
            obj_type = obj.get("type", "")
            pos = obj.get("pos", [0, 0, 0])
            material = obj.get("material", "")
            
            if obj_type == "CUBE":
                size = obj.get("size", [1, 1, 1])
                if material == "boss":
                    color = (1.0, 0.0, 0.0)  # Red
                elif material == "enemy":
                    color = (0.8, 0.2, 0.2)  # Dark red
                else:
                    color = (0.5, 0.5, 1.0)  # Blue
                self.draw_cube(pos, size, color)
                
            elif obj_type == "SPHERE":
                radius = max(obj.get("radius", 0.5), 0.3)
                if material == "collectible":
                    color = (1.0, 1.0, 0.0)  # Yellow
                elif material == "player":
                    color = (0.0, 1.0, 0.0)  # Green
                else:
                    color = (1.0, 1.0, 1.0)  # White
                self.draw_sphere(pos, radius, color)
        
        # Draw target cube
        glPushMatrix()
        glTranslatef(9, 1, 0)
        glColor3f(0.0, 0.0, 1.0)  # Blue target
        glutSolidCube(1.5)
        glPopMatrix()
        
        # Draw coordinate axes
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)  # X-axis (red)
        glVertex3f(0, 0.1, 0)
        glVertex3f(3, 0.1, 0)
        glColor3f(0.0, 1.0, 0.0)  # Y-axis (green)
        glVertex3f(0, 0.1, 0)
        glVertex3f(0, 3.1, 0)
        glColor3f(0.0, 0.0, 1.0)  # Z-axis (blue)
        glVertex3f(0, 0.1, 0)
        glVertex3f(0, 0.1, 3)
        glEnd()
        
        glutSwapBuffers()
    
    def keyboard(self, key, x, y):
        """Handle key press"""
        key_code = ord(key) if isinstance(key, bytes) else key
        self.keys_pressed.add(key_code)
        
        if key == b'\x1b':  # Escape
            print("👋 Game viewer closed by user")
            print(f"📍 Final position: ({self.camera_pos[0]:.1f}, {self.camera_pos[1]:.1f}, {self.camera_pos[2]:.1f})")
            glutLeaveMainLoop()
        elif key == b'q' or key == b'Q':
            self.camera_rotation_y -= 5
        elif key == b'e' or key == b'E':
            self.camera_rotation_y += 5
        elif key == b'r' or key == b'R':
            self.camera_rotation_x -= 5
        elif key == b'f' or key == b'F':
            self.camera_rotation_x += 5
        elif key == b'+' or key == b'=':
            self.movement_speed = min(self.movement_speed * 1.2, 2.0)
            print(f"🏃 Movement speed: {self.movement_speed:.2f}")
        elif key == b'-':
            self.movement_speed = max(self.movement_speed * 0.8, 0.1)
            print(f"🚶 Movement speed: {self.movement_speed:.2f}")
        
        glutPostRedisplay()
    
    def keyboard_up(self, key, x, y):
        """Handle key release"""
        key_code = ord(key) if isinstance(key, bytes) else key
        self.keys_pressed.discard(key_code)
    
    def mouse(self, button, state, x, y):
        """Handle mouse clicks to focus window"""
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            self.window_focused = True
            print("🎯 Window focused - controls active!")
        glutPostRedisplay()
    
    def entry(self, state):
        """Handle window entry/exit"""
        if state == GLUT_ENTERED:
            self.window_focused = True
            print("🎯 Window focused - controls active!")
        elif state == GLUT_LEFT:
            self.window_focused = False
            print("🖱️ Window lost focus")
        glutPostRedisplay()
    
    def reshape(self, width, height):
        """Handle window reshape"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, width/height, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
    
    def run(self):
        """Run the fixed viewer"""
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutInitWindowPosition(100, 100)
        glutCreateWindow(f"PCC Fixed 3D Viewer - {Path(self.scene_file).name}")
        
        self.init_gl()
        
        # Set up callbacks
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutKeyboardUpFunc(self.keyboard_up)
        glutMouseFunc(self.mouse)
        glutEntryFunc(self.entry)
        
        print("🎮 PCC Fixed 3D Viewer")
        print("📋 Controls:")
        print("   Click in window to focus")
        print("   WASD - Move around")
        print("   Q/E - Rotate camera left/right")
        print("   R/F - Rotate camera up/down")
        print("   Space/C - Move up/down")
        print("   +/- - Change movement speed")
        print("   ESC - Quit")
        print()
        print("🎲 Game Objective: Find the blue target cube at (9,0,0)")
        print("💡 Click in the window to enable controls!")
        
        glutMainLoop()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pcc_fixed_viewer.py <scene_file>")
        return
    
    viewer = FixedPCCViewer(sys.argv[1])
    viewer.run()

if __name__ == "__main__":
    main() 