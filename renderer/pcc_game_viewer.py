#!/usr/bin/env python3
"""
PCC Game Viewer - Proper User Controls
- Click window to capture mouse/keyboard
- Press TAB to release mouse capture
- Better camera controls and movement
"""

import json
import sys
import time
import math
import threading
import socket
import struct
import base64
from collections import deque
from pathlib import Path

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except ImportError:
    print("❌ OpenGL not available")
    sys.exit(1)

# Global variables
scene_data = None
window_title = ""

# Camera and movement variables
camera_x, camera_y, camera_z = 0.0, 1.6, 5.0
camera_pitch, camera_yaw = -10.0, 0.0
movement_speed = 0.15
mouse_sensitivity = 0.2

# Simple grounded movement physics
base_eye_height = 1.6
gravity = -0.02
velocity_y = 0.0

# Input state
keys_pressed = set()
mouse_captured = False
last_mouse_x, last_mouse_y = 400, 300
window_width, window_height = 800, 600

# Bridge state (step-driven control)
bridge_enabled = False
bridge_port = 0
server_thread = None
input_queue = deque()
scheduled_key_releases = []  # list of tuples (release_step_id, key_code)
step_id = 0
sim_time_ms = 0
frame_id = 0
dt_ms_default = 16
last_step_dt_ms = dt_ms_default

# Telemetry state
visited_cells: set[tuple[int, int]] = set()
coverage_cell_size = 1.0
coverage_origin = (0.0, 0.0, 0.0)
coverage_grid_span = 40  # +-20 cells each axis (x/z), 40x40 grid → 1600 cells

stall_eps = 0.05
stall_start_ms: int | None = None
stall_secs = 0.0
_last_planar_pos = (0.0, 0.0)

# Frame ring buffer (~12s @ 10 FPS)
frame_ring_buffer = deque(maxlen=12 * 10)
frame_ring_lock = threading.Lock()
last_capture_ms = 0
capture_interval_ms = 100
captures_dir = (Path(__file__).parent / "captures")
captures_dir.mkdir(exist_ok=True)
per_run_budget_bytes = 200 * 1024 * 1024
_active_flush_thread: threading.Thread | None = None
last_screenshot_b64 = ""
last_screenshot_ms = 0

def _estimate_captures_dir_bytes() -> int:
    try:
        return sum(p.stat().st_size for p in captures_dir.glob("*.mp4"))
    except Exception:
        return 0

def _gl_capture_rgb_frame(width: int, height: int) -> bytes:
    # Read from bottom-left origin; we'll flip in ffmpeg via vf=vflip
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    # PyOpenGL returns a bytes-like object
    return bytes(data)

def _gl_capture_png_b64(width: int, height: int) -> str:
    # Capture and convert to PNG in-memory
    try:
        from PIL import Image
    except Exception:
        return ""
    rgb = _gl_capture_rgb_frame(width, height)
    img = Image.frombytes('RGB', (width, height), rgb)
    # Flip vertically to correct orientation
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')

def _flush_ring_to_mp4_async(width: int, height: int, fps: int = 10) -> str | None:
    global _active_flush_thread
    # Copy frames under lock to avoid blocking timer
    with frame_ring_lock:
        frames = list(frame_ring_buffer)
    if not frames:
        return None
    out_path = captures_dir / f"capture_{int(time.time())}.mp4"

    def _writer(frames_local: list[bytes], out_file: Path):
        try:
            # Use ffmpeg rawvideo pipe
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s:v", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-vf", "vflip",
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "28",
                str(out_file),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for rgb in frames_local:
                proc.stdin.write(rgb)
            proc.stdin.close()
            proc.wait(timeout=60)
        except Exception:
            # Best-effort only
            try:
                if out_file.exists():
                    out_file.unlink(missing_ok=True)
            except Exception:
                pass

    # Enforce per-run budget before starting
    if _estimate_captures_dir_bytes() >= per_run_budget_bytes:
        return None
    _active_flush_thread = threading.Thread(target=_writer, args=(frames, out_path), daemon=True)
    _active_flush_thread.start()
    return str(out_path)

def _keycode_from_name(name: str) -> int:
    if not name:
        return 0
    if len(name) == 1:
        return ord(name)
    mapping = {
        'SPACE': ord(' '),
        'TAB': ord('\t'),
        'ESC': 27,
        'C': ord('c'),
        'W': ord('w'), 'A': ord('a'), 'S': ord('s'), 'D': ord('d'),
    }
    return mapping.get(name.upper(), 0)

def init_gl():
    """Initialize OpenGL settings"""
    glClearColor(0.2, 0.4, 0.8, 1.0)  # Nice sky blue background
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    # Better lighting setup
    glLightfv(GL_LIGHT0, GL_POSITION, [5, 10, 5, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 0.8, 1])
    
    # Material properties
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

def get_camera_vectors():
    """Get camera forward, right, and up vectors"""
    # Convert angles to radians
    pitch_rad = math.radians(camera_pitch)
    yaw_rad = math.radians(camera_yaw)
    
    # Forward vector
    forward_x = math.sin(yaw_rad) * math.cos(pitch_rad)
    forward_y = math.sin(pitch_rad)
    forward_z = math.cos(yaw_rad) * math.cos(pitch_rad)
    
    # Right vector (perpendicular to forward in horizontal plane)
    right_x = math.cos(yaw_rad)
    right_z = -math.sin(yaw_rad)
    
    return (forward_x, forward_y, forward_z), (right_x, 0, right_z), (0, 1, 0)

def handle_movement():
    """Handle WASD movement with grounded jump/crouch physics"""
    global camera_x, camera_y, camera_z, velocity_y

    forward, right, _ = get_camera_vectors()

    # Horizontal movement
    if ord('w') in keys_pressed or ord('W') in keys_pressed:
        camera_x += forward[0] * movement_speed
        camera_z += forward[2] * movement_speed
    if ord('s') in keys_pressed or ord('S') in keys_pressed:
        camera_x -= forward[0] * movement_speed
        camera_z -= forward[2] * movement_speed
    if ord('a') in keys_pressed or ord('A') in keys_pressed:
        camera_x -= right[0] * movement_speed
        camera_z -= right[2] * movement_speed
    if ord('d') in keys_pressed or ord('D') in keys_pressed:
        camera_x += right[0] * movement_speed
        camera_z += right[2] * movement_speed

    # Grounded physics (jump/crouch)
    on_ground = camera_y <= base_eye_height + 1e-3
    if on_ground:
        camera_y = base_eye_height
        velocity_y = 0.0
        if ord(' ') in keys_pressed:  # jump
            velocity_y = 0.35
        if ord('c') in keys_pressed or ord('C') in keys_pressed:  # crouch
            camera_y = base_eye_height * 0.6
    else:
        velocity_y += gravity

    camera_y += velocity_y

    # Update stall metric (planar movement only)
    global stall_start_ms, stall_secs, _last_planar_pos
    dx = camera_x - _last_planar_pos[0]
    dz = camera_z - _last_planar_pos[1]
    if (dx * dx + dz * dz) ** 0.5 < stall_eps:
        if stall_start_ms is None:
            stall_start_ms = sim_time_ms
        else:
            stall_secs = max(0.0, (sim_time_ms - stall_start_ms) / 1000.0)
    else:
        stall_start_ms = None
    _last_planar_pos = (camera_x, camera_z)

def display():
    """Main display function"""
    global mouse_captured
    
    # Handle movement
    handle_movement()
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Calculate look-at point
    forward, _, _ = get_camera_vectors()
    look_x = camera_x + forward[0]
    look_y = camera_y + forward[1] 
    look_z = camera_z + forward[2]

    # Set up camera
    gluLookAt(camera_x, camera_y, camera_z, look_x, look_y, look_z, 0, 1, 0)

    # Draw large ground plane
    glColor3f(0.3, 0.7, 0.3)  # Nice green
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)
    glVertex3f(-50, 0, -50)
    glVertex3f(-50, 0, 50)
    glVertex3f(50, 0, 50)
    glVertex3f(50, 0, -50)
    glEnd()

    # Draw grid lines for reference
    glColor3f(0.2, 0.5, 0.2)
    glBegin(GL_LINES)
    for i in range(-20, 21, 2):
        glVertex3f(i, 0.01, -20)
        glVertex3f(i, 0.01, 20)
        glVertex3f(-20, 0.01, i)
        glVertex3f(20, 0.01, i)
    glEnd()

    # Remove hardcoded walls; rely solely on scene_data

    # Draw objects from scene data
    if scene_data:
        for obj in scene_data.get("objects", []):
            obj_type = obj.get("type", "")
            pos = obj.get("pos", [0, 0, 0])
            material = obj.get("material", "")

            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])

            if obj_type == "CUBE":
                size = obj.get("size", [1, 1, 1])
                if material == "boss":
                    glColor3f(1.0, 0.0, 0.0)  # Bright red for boss
                elif material == "enemy":
                    glColor3f(0.8, 0.2, 0.2)  # Dark red for enemies
                else:
                    glColor3f(0.5, 0.5, 1.0)  # Blue for other cubes
                glutSolidCube(max(size))
                
            elif obj_type == "SPHERE":
                radius = max(obj.get("radius", 0.5), 0.3)
                if material == "collectible":
                    glColor3f(1.0, 1.0, 0.0)  # Bright yellow for collectibles
                elif material == "player":
                    glColor3f(0.0, 1.0, 0.0)  # Bright green for player
                else:
                    glColor3f(1.0, 1.0, 1.0)  # White for other spheres
                glutSolidSphere(radius, 16, 16)

            glPopMatrix()

    # Draw target at (9,1,0)
    glPushMatrix()
    glTranslatef(9, 1, 0)
    glColor3f(0.0, 0.3, 1.0)  # Bright blue target
    glutSolidCube(2.0)
    glPopMatrix()

    # Draw coordinate axes at origin
    glLineWidth(3.0)
    glBegin(GL_LINES)
    # X-axis (red)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0, 0.5, 0)
    glVertex3f(3, 0.5, 0)
    # Z-axis (blue)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0, 0.5, 0)
    glVertex3f(0, 0.5, 3)
    # Y-axis (green)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0, 0.5, 0)
    glVertex3f(0, 3.5, 0)
    glEnd()
    glLineWidth(1.0)

    # Show mouse capture status
    if not mouse_captured:
        # Draw text indicator (simplified)
        glColor3f(1.0, 1.0, 0.0)
        glRasterPos3f(camera_x + 2, camera_y + 2, camera_z)
        
    # Opportunistically capture a PNG screenshot in the GL thread (~10 FPS)
    global last_screenshot_b64, last_screenshot_ms
    if sim_time_ms - last_screenshot_ms >= 100:
        try:
            last_screenshot_b64 = _gl_capture_png_b64(window_width, window_height)
            last_screenshot_ms = sim_time_ms
        except Exception:
            pass

    glutSwapBuffers()

def reshape(width, height):
    """Handle window reshape"""
    global window_width, window_height, last_mouse_x, last_mouse_y
    
    window_width, window_height = width, height
    last_mouse_x, last_mouse_y = width // 2, height // 2
    
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    """Handle key press"""
    global movement_speed, mouse_captured, camera_x, camera_y, camera_z
    
    key_code = ord(key) if isinstance(key, bytes) else key
    keys_pressed.add(key_code)
    
    if key == b'\x1b':  # Escape key
        print("👋 Game closed by user")
        print(f"📍 Final position: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})")
        glutLeaveMainLoop()
    elif key == b'\t':  # Tab key - toggle mouse capture
        mouse_captured = not mouse_captured
        if mouse_captured:
            glutSetCursor(GLUT_CURSOR_NONE)
            glutWarpPointer(window_width // 2, window_height // 2)
            print("🎯 Mouse captured - move mouse to look around")
        else:
            glutSetCursor(GLUT_CURSOR_INHERIT)
            print("🖱️ Mouse released - click window to recapture")
    elif key == b'+' or key == b'=':  # Speed up
        movement_speed = min(movement_speed * 1.5, 1.0)
        print(f"🏃 Movement speed: {movement_speed:.2f}")
    elif key == b'-':  # Slow down
        movement_speed = max(movement_speed * 0.7, 0.02)
        print(f"🚶 Movement speed: {movement_speed:.2f}")
    elif key == b'r' or key == b'R':  # Reset camera
        camera_x, camera_y, camera_z = 0.0, 3.0, 5.0
        camera_pitch, camera_yaw = -20.0, 0.0
        print(f"🔄 Camera reset to start position")

def keyboard_up(key, x, y):
    """Handle key release"""
    key_code = ord(key) if isinstance(key, bytes) else key
    keys_pressed.discard(key_code)

def mouse_click(button, state, x, y):
    """Handle mouse clicks"""
    global mouse_captured, last_mouse_x, last_mouse_y
    
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        mouse_captured = True
        glutSetCursor(GLUT_CURSOR_NONE)
        last_mouse_x, last_mouse_y = x, y
        glutWarpPointer(window_width // 2, window_height // 2)
        print("🎯 Mouse captured - move to look around (TAB to release)")

def mouse_motion(x, y):
    """Handle mouse movement for camera control"""
    global camera_pitch, camera_yaw, last_mouse_x, last_mouse_y
    
    if not mouse_captured:
        return
    
    # Calculate mouse delta from center
    center_x, center_y = window_width // 2, window_height // 2
    dx = x - center_x
    dy = y - center_y
    
    # Only process if there's significant movement
    if abs(dx) > 2 or abs(dy) > 2:
        # Update camera rotation
        camera_yaw -= dx * mouse_sensitivity
        camera_pitch -= dy * mouse_sensitivity
        
        # Clamp pitch to prevent over-rotation
        camera_pitch = max(-85, min(85, camera_pitch))
        
        # Re-center mouse
        glutWarpPointer(center_x, center_y)

def timer(value):
    """Timer for continuous updates"""
    global step_id, sim_time_ms, last_step_dt_ms, frame_id

    # Apply any scheduled key releases at this step
    if scheduled_key_releases:
        still_scheduled = []
        for rel_step, code in scheduled_key_releases:
            if rel_step <= step_id:
                keys_pressed.discard(code)
            else:
                still_scheduled.append((rel_step, code))
        scheduled_key_releases[:] = still_scheduled

    # Advance simulation one fixed step
    dt = last_step_dt_ms / 1000.0
    sim_time_ms += last_step_dt_ms
    step_id += 1
    glutPostRedisplay()
    frame_id += 1
    glutTimerFunc(last_step_dt_ms, timer, 0)

    # Coverage grid update (x/z plane)
    try:
        gx = int(math.floor((camera_x - coverage_origin[0]) / max(coverage_cell_size, 1e-6)))
        gz = int(math.floor((camera_z - coverage_origin[2]) / max(coverage_cell_size, 1e-6)))
        # Limit to bounded grid to keep denominator meaningful
        if -coverage_grid_span // 2 <= gx <= coverage_grid_span // 2 and -coverage_grid_span // 2 <= gz <= coverage_grid_span // 2:
            visited_cells.add((gx, gz))
    except Exception:
        pass

    # Frame capture at ~10 FPS
    global last_capture_ms
    if sim_time_ms - last_capture_ms >= capture_interval_ms:
        try:
            with frame_ring_lock:
                frame_ring_buffer.append(
                    _gl_capture_rgb_frame(window_width, window_height)
                )
            last_capture_ms = sim_time_ms
        except Exception:
            # Ignore capture errors (e.g., context not ready)
            pass

def render_pcc_game_interactive(scene_file):
    """Render PCC game with proper user controls"""
    global scene_data, window_title
    
    try:
        # Load scene data
        with open(scene_file, 'r') as f:
            scene_data = json.load(f)

        window_title = f"PCC Game - {Path(scene_file).name}"
        
        print(f"🎮 Opening PCC game: {scene_file}")
        print(f"📊 Scene objects: {len(scene_data.get('objects', []))}")
        print()
        print("🎯 CONTROLS:")
        print("   CLICK WINDOW - Capture mouse for camera control")
        print("   TAB - Release/capture mouse")
        print("   WASD - Move around")
        print("   SPACE - Move up")
        print("   C - Move down") 
        print("   +/- - Change movement speed")
        print("   R - Reset camera position")
        print("   ESC - Quit")
        print()
        print("🎲 OBJECTIVE: Navigate to the blue target cube!")
        print("🗺️ Look for coordinate axes (RGB lines) to orient yourself")

        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(window_width, window_height)
        glutInitWindowPosition(100, 100)
        glutCreateWindow(window_title.encode())

        # Set up callbacks
        glutDisplayFunc(display)
        glutReshapeFunc(reshape)
        glutKeyboardFunc(keyboard)
        glutKeyboardUpFunc(keyboard_up)
        glutMouseFunc(mouse_click)
        glutPassiveMotionFunc(mouse_motion)
        glutMotionFunc(mouse_motion)
        glutTimerFunc(16, timer, 0)

        # Initialize OpenGL
        init_gl()

        # Start bridge server if port provided via env var
        import os
        port = os.getenv('PCC_VIEWER_BRIDGE_PORT')
        if port:
            try:
                start_bridge_server(int(port))
                print(f"🔌 Bridge server listening on {port}")
            except Exception as e:
                print(f"⚠️ Bridge server failed: {e}")

        print("✅ Game window opened!")
        print("🖱️ Click the window to start playing")
        
        # Enter main loop
        glutMainLoop()
        
        return True

    except Exception as e:
        print(f"❌ Failed to render game: {e}")
        return False

# Main entry relocated to end of file after bridge defs

# ---------------- Bridge server implementation ----------------

def start_bridge_server(port: int):
    global bridge_enabled, bridge_port, server_thread
    bridge_enabled = True
    bridge_port = port

    def serve():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen(1)
            while True:
                conn, _ = s.accept()
                with conn:
                    while True:
                        hdr = conn.recv(4)
                        if not hdr:
                            break
                        (length,) = struct.unpack("!I", hdr)
                        payload = b""
                        while len(payload) < length:
                            chunk = conn.recv(length - len(payload))
                            if not chunk:
                                break
                            payload += chunk
                        if not payload:
                            break
                        try:
                            msg = json.loads(payload.decode("utf-8"))
                        except Exception:
                            continue
                        handle_bridge_message(conn, msg)

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()


def _send_lenpref_json(conn, obj):
    data = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!I", len(data)) + data)


def send_ack(conn, extra: dict):
    ack = {
        "ack": True,
        "schema_version": 1,
        "applied_step_id": step_id,
        "sim_time_ms": sim_time_ms,
        "frame_id": frame_id,
    }
    if extra:
        ack.update(extra)
    _send_lenpref_json(conn, ack)


def handle_bridge_message(conn, msg: dict):
    global last_step_dt_ms, camera_yaw, camera_pitch
    t = msg.get("type")
    if t == "key_hold":
        code = _keycode_from_name(msg.get("key", ""))
        hold_ms = int(msg.get("hold_ms", 0))
        if code:
            keys_pressed.add(code)
            release_step = step_id + max(1, hold_ms // max(1, last_step_dt_ms))
            scheduled_key_releases.append((release_step, code))
        send_ack(conn, {"latency_steps": 0, "latency_ms": 0})
    elif t == "key_down":
        code = _keycode_from_name(msg.get("key", ""))
        if code:
            keys_pressed.add(code)
        send_ack(conn, {"latency_steps": 0, "latency_ms": 0})
    elif t == "key_up":
        code = _keycode_from_name(msg.get("key", ""))
        if code:
            keys_pressed.discard(code)
        send_ack(conn, {"latency_steps": 0, "latency_ms": 0})
    elif t == "mouse_move":
        mode = msg.get("mode", "relative")
        dx = float(msg.get("dx", 0))
        dy = float(msg.get("dy", 0))
        if mode == "relative":
            camera_yaw -= dx * mouse_sensitivity
            camera_pitch -= dy * mouse_sensitivity
            camera_pitch = max(-85, min(85, camera_pitch))
        send_ack(conn, {"mouse_mode": mode})
    elif t == "step":
        last_step_dt_ms = int(msg.get("dt", dt_ms_default))
        send_ack(conn, {"latency_steps": 1, "latency_ms": last_step_dt_ms})
    elif t == "telemetry_request":
        # Compute coverage visited_pct against a bounded 40x40 grid
        denom = max(1, coverage_grid_span * coverage_grid_span)
        visited_pct = min(1.0, len(visited_cells) / denom)
        # Simple camera clip ratio placeholder (no real clipping implemented yet)
        clip_ratio = 0.0
        telem = {
            "schema_version": 1,
            "sim_time_ms": sim_time_ms,
            "step_id": step_id,
            "fps": 1000.0 / max(1, last_step_dt_ms),
            "exceptions": [],
            "player": {"pos": [camera_x, camera_y, camera_z], "vel": [0, velocity_y, 0]},
            "coverage": {"cell_size": coverage_cell_size, "origin": [0, 0, 0], "quantization": "floor", "visited_pct": visited_pct},
            "stall_secs": stall_secs,
            "camera_clip_ratio": clip_ratio,
            "clip_method": "nearplane_hits",
            "width": window_width,
            "height": window_height
        }
        _send_lenpref_json(conn, telem)
    elif t == "screenshot_request":
        # Return latest captured base64 PNG from GL thread
        _send_lenpref_json(conn, {"schema_version": 1, "png_base64": last_screenshot_b64, "sim_time_ms": sim_time_ms, "step_id": step_id})
    elif t == "frame_request":
        # Optional trigger to flush current ring buffer to an mp4 clip
        if _estimate_captures_dir_bytes() >= per_run_budget_bytes:
            send_ack(conn, {"note": "capture_budget_exceeded"})
        else:
            out = _flush_ring_to_mp4_async(window_width, window_height, fps=10)
            if out:
                send_ack(conn, {"saved_to": out})
            else:
                send_ack(conn, {"note": "no_frames_or_ffmpeg_missing"})
    else:
        send_ack(conn, {"note": "unknown_message"})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_pcc_game_interactive(sys.argv[1])
    else:
        print("Usage: python3 pcc_game_viewer.py <scene_file>")
