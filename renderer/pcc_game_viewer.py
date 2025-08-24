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
from typing import Tuple, Optional, Dict, List

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import numpy as np
except ImportError:
    print("❌ OpenGL not available")
    sys.exit(1)

# Import T08 runtime LOD system
sys.path.append(str(Path(__file__).parent.parent / "agents" / "agent_d" / "mesh"))
try:
    from runtime_lod import RuntimeLODManager, create_view_matrix, create_projection_matrix
    from chunk_streamer import ChunkStreamer
    RUNTIME_LOD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Runtime LOD system not available: {e}")
    RUNTIME_LOD_AVAILABLE = False

# Global variables
scene_data = None
window_title = ""

# Camera and movement variables
camera_x, camera_y, camera_z = 0.0, 5.0, 120.0
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

# Mesh rendering state
mesh_cache = {}  # Cache for loaded meshes
chunk_cache = {}  # Cache for chunked planet data
debug_wireframe = False
debug_aabb = False
debug_normals = False
debug_chunk_aabb = False
debug_crack_lines = False
debug_runtime_lod = False
debug_show_hud = True

# T08 Runtime LOD system
runtime_lod_manager = None
chunk_streamer = None
lod_stats = {
    'frame_time_ms': 0.0,
    'active_chunks': 0,
    'culled_chunks': 0,
    'lod_levels': {},
    'gpu_memory_mb': 0.0
}

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

def LoadMeshFromManifest(json_path: str) -> dict:
    """Load mesh data from manifest into VAO/VBO/EBO"""
    try:
        manifest_path = Path(json_path)
        manifest_dir = manifest_path.parent
        
        with open(json_path, 'r') as f:
            manifest = json.load(f)
        
        mesh_data = manifest.get("mesh", {})
        if not mesh_data:
            return {}
            
        # Load binary buffers
        positions = _load_buffer(mesh_data.get("positions", ""), str(manifest_dir))
        normals = _load_buffer(mesh_data.get("normals", ""), str(manifest_dir))
        tangents = _load_buffer(mesh_data.get("tangents", ""), str(manifest_dir))
        uv0 = _load_buffer(mesh_data.get("uv0", ""), str(manifest_dir))
        indices = _load_buffer(mesh_data.get("indices", ""), str(manifest_dir), dtype=np.uint32)
        
        if positions is None or indices is None:
            return {}
            
        # Create VAO
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        
        # Create VBOs
        position_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, position_vbo)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        normal_vbo = 0
        if normals is not None:
            normal_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
            glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)
            
        tangent_vbo = 0
        if tangents is not None:
            tangent_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, tangent_vbo)
            glBufferData(GL_ARRAY_BUFFER, tangents.nbytes, tangents, GL_STATIC_DRAW)
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(2)
            
        uv_vbo = 0
        if uv0 is not None:
            uv_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, uv_vbo)
            glBufferData(GL_ARRAY_BUFFER, uv0.nbytes, uv0, GL_STATIC_DRAW)
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(3)
        
        # Create EBO
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        
        # Calculate bounds
        bounds = mesh_data.get("bounds", {})
        if not bounds and positions is not None:
            bounds = _calculate_bounds(positions)
            
        return {
            "vao": vao,
            "index_count": len(indices),
            "bounds": bounds,
            "buffers": [position_vbo, normal_vbo, tangent_vbo, uv_vbo, ebo],
            "vertex_count": len(positions) // 3,
            "has_normals": normals is not None,
            "has_tangents": tangents is not None
        }
        
    except Exception as e:
        print(f"❌ Failed to load mesh from {json_path}: {e}")
        return {}

def _load_buffer(buffer_path: str, manifest_dir: str, dtype=np.float32):
    """Load binary buffer from file path"""
    if not buffer_path or not buffer_path.startswith("buffer://"):
        return None
        
    filename = buffer_path.replace("buffer://", "")
    # Resolve relative to manifest directory
    file_path = Path(manifest_dir) / filename
    try:
        return np.fromfile(str(file_path), dtype=dtype)
    except Exception as e:
        print(f"❌ Failed to load buffer {file_path}: {e}")
        return None

def _calculate_bounds(positions):
    """Calculate AABB bounds from position data"""
    if len(positions) < 3:
        return {"center": [0, 0, 0], "radius": 1}
        
    pos_3d = positions.reshape(-1, 3)
    min_vals = np.min(pos_3d, axis=0)
    max_vals = np.max(pos_3d, axis=0)
    center = (min_vals + max_vals) * 0.5
    radius = np.linalg.norm(max_vals - min_vals) * 0.5
    
    return {
        "center": center.tolist(),
        "radius": float(radius)
    }

def RenderMeshVAO(vao: int, index_count: int):
    """Render mesh using VAO with glDrawElements"""
    if vao == 0 or index_count == 0:
        return
        
    glBindVertexArray(vao)
    
    if debug_wireframe:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)
    
    glBindVertexArray(0)

def DrawAABB(bounds: dict):
    """Draw AABB wireframe for debugging"""
    if not bounds:
        return
        
    center = bounds.get("center", [0, 0, 0])
    radius = bounds.get("radius", 1)
    
    # Simple sphere approximation for now
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glColor3f(1, 1, 0)  # Yellow wireframe
    glutWireSphere(radius, 8, 8)
    glPopMatrix()

def DrawNormals(manifest_path: str, scale: float = 0.1):
    """Draw normal vectors for debugging"""
    if not manifest_path:
        return
        
    try:
        manifest_path_obj = Path(manifest_path)
        manifest_dir = manifest_path_obj.parent
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        mesh_data = manifest.get("mesh", {})
        if not mesh_data:
            return
            
        # Load positions and normals
        positions = _load_buffer(mesh_data.get("positions", ""), str(manifest_dir))
        normals = _load_buffer(mesh_data.get("normals", ""), str(manifest_dir))
        
        if positions is None or normals is None:
            return
            
        # Reshape to vertex arrays
        pos_vertices = positions.reshape(-1, 3)
        norm_vertices = normals.reshape(-1, 3)
        
        # Draw normal lines
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0, 1, 1)  # Cyan for normals
        
        glBegin(GL_LINES)
        for i in range(min(len(pos_vertices), 500)):  # Limit to avoid clutter
            pos = pos_vertices[i]
            norm = norm_vertices[i]
            end_pos = pos + norm * scale
            
            glVertex3f(pos[0], pos[1], pos[2])
            glVertex3f(end_pos[0], end_pos[1], end_pos[2])
        glEnd()
        
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)
        
    except Exception as e:
        print(f"❌ Failed to draw normals: {e}")

def DrawChunkEdges(chunk: dict, color: Tuple[float, float, float] = (1.0, 0.0, 1.0)):
    """Draw chunk edge lines for crack detection visualization"""
    try:
        chunk_info = chunk.get("chunk_info", {})
        resolution = chunk_info.get("resolution", 16)
        positions = chunk.get("positions", np.array([]))
        
        if positions.size == 0:
            return
        
        # Reshape positions to grid for edge extraction
        pos_grid = positions.reshape(resolution, resolution, 3)
        
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glColor3f(*color)
        
        glBegin(GL_LINES)
        
        # Draw north edge (top)
        for i in range(resolution - 1):
            p1 = pos_grid[resolution-1, i, :]
            p2 = pos_grid[resolution-1, i+1, :]
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
        
        # Draw south edge (bottom)
        for i in range(resolution - 1):
            p1 = pos_grid[0, i, :]
            p2 = pos_grid[0, i+1, :]
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
        
        # Draw east edge (right)
        for j in range(resolution - 1):
            p1 = pos_grid[j, resolution-1, :]
            p2 = pos_grid[j+1, resolution-1, :]
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
        
        # Draw west edge (left)
        for j in range(resolution - 1):
            p1 = pos_grid[j, 0, :]
            p2 = pos_grid[j+1, 0, :]
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
        
        glEnd()
        
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)
        
    except Exception as e:
        print(f"❌ Failed to draw chunk edges: {e}")

def DetectAndDrawCrackLines(chunked_planet: dict):
    """Detect and visualize potential crack lines between chunks"""
    if not chunked_planet or chunked_planet.get("type") != "chunked_planet":
        return
    
    try:
        # Import crack prevention module
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents', 'agent_d', 'mesh'))
        from crack_prevention import LODCrackPrevention
        
        chunks = chunked_planet.get("chunks", [])
        if not chunks:
            return
        
        # Initialize crack detection
        crack_detector = LODCrackPrevention()
        
        # Detect cracks
        crack_detections = crack_detector.detect_cracks(chunks)
        
        # Visualize crack lines
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(1.0, 0.0, 0.0)  # Red for potential crack lines
        
        # For each potential crack, draw a line between chunk centers
        glBegin(GL_LINES)
        
        drawn_lines = set()  # Avoid duplicate lines
        
        for crack_info in crack_detections:
            chunk_id = crack_info.get("chunk_id", "")
            neighbor_id = crack_info.get("neighbor_id", "")
            
            # Create unique line identifier
            line_id = tuple(sorted([chunk_id, neighbor_id]))
            if line_id in drawn_lines:
                continue
            drawn_lines.add(line_id)
            
            # Find the chunks
            chunk1 = None
            chunk2 = None
            
            for chunk in chunks:
                chunk_info = chunk.get("chunk_info", {})
                if chunk_info.get("chunk_id") == chunk_id:
                    chunk1 = chunk
                elif chunk_info.get("chunk_id") == neighbor_id:
                    chunk2 = chunk
            
            if chunk1 and chunk2:
                # Get chunk centers from AABB
                aabb1 = chunk1.get("chunk_info", {}).get("aabb", {})
                aabb2 = chunk2.get("chunk_info", {}).get("aabb", {})
                
                center1 = aabb1.get("center", [0, 0, 0])
                center2 = aabb2.get("center", [0, 0, 0])
                
                # Draw line between centers
                glVertex3f(center1[0], center1[1], center1[2])
                glVertex3f(center2[0], center2[1], center2[2])
        
        glEnd()
        
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)
        
        # Print crack detection summary
        if crack_detections:
            transition_counts = {}
            for crack in crack_detections:
                transition = crack.get("transition_type", "unknown")
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
            
            print(f"🔍 Crack Detection: {len(crack_detections)} potential cracks")
            for transition, count in transition_counts.items():
                print(f"   {transition}: {count}")
        
    except ImportError:
        print("⚠️ Crack detection module not available")
    except Exception as e:
        print(f"❌ Failed to detect crack lines: {e}")

def LoadChunkedPlanet(planet_manifest_path: str) -> dict:
    """Load chunked planet from manifest - supports both file-based and unified formats"""
    try:
        with open(planet_manifest_path, 'r') as f:
            planet_manifest = json.load(f)
        
        planet_data = planet_manifest.get("planet_info", planet_manifest.get("planet", {}))
        if planet_data.get("type") != "chunked_quadtree":
            print(f"❌ Not a chunked planet: {planet_manifest_path}")
            return {}
        
        chunks_data = planet_manifest.get("chunks", [])
        
        # Check if this is a unified format (chunks are dicts) or file-based format (chunks are filenames)
        if chunks_data and isinstance(chunks_data[0], dict):
            # Unified format: chunks are stored directly in the JSON
            print(f"🔧 Loading {len(chunks_data)} chunks from unified format...")
            
            loaded_chunks = []
            for chunk_data in chunks_data:
                # Convert unified chunk format to viewer format
                chunk = {
                    "chunk_id": chunk_data.get("chunk_id", "unknown"),
                    "positions": np.array(chunk_data.get("positions", [])),
                    "indices": np.array(chunk_data.get("indices", []), dtype=np.uint32),
                    "normals": np.array(chunk_data.get("normals", [])),
                    "vertex_count": chunk_data.get("vertex_count", 0),
                    "index_count": chunk_data.get("index_count", 0),
                    "aabb": chunk_data.get("aabb", {"min": [0,0,0], "max": [0,0,0]}),
                    "material": chunk_data.get("material", "terrain"),
                    "chunk_info": {
                        "bounds": chunk_data.get("chunk_bounds", {}),
                        "triangle_count": chunk_data.get("triangle_count", 0)
                    }
                }
                loaded_chunks.append(chunk)
            
            print(f"✅ Loaded {len(loaded_chunks)} unified chunks successfully")
            
        else:
            # File-based format: chunks are stored in separate files
            manifest_path = Path(planet_manifest_path)
            manifest_dir = manifest_path.parent
            loaded_chunks = []
            
            print(f"🔧 Loading {len(chunks_data)} chunks from file format...")
            
            for chunk_file in chunks_data:
                chunk_path = manifest_dir / chunk_file
                chunk_data = LoadMeshFromManifest(str(chunk_path))
                
                if chunk_data:
                    # Load chunk metadata
                    with open(chunk_path, 'r') as f:
                        chunk_manifest = json.load(f)
                    
                    chunk_info = chunk_manifest.get("chunk", {})
                    chunk_data["chunk_info"] = chunk_info
                    chunk_data["manifest_path"] = str(chunk_path)
                    loaded_chunks.append(chunk_data)
            
            print(f"✅ Loaded {len(loaded_chunks)} file-based chunks successfully")
        
        return {
            "type": "chunked_planet",
            "planet_info": planet_data,
            "terrain_params": planet_manifest.get("terrain_params", {}),
            "chunks": loaded_chunks,
            "statistics": planet_manifest.get("statistics", {}),
            "total_vertices": sum(chunk.get("vertex_count", 0) for chunk in loaded_chunks),
            "total_triangles": sum(chunk.get("index_count", 0) // 3 for chunk in loaded_chunks)
        }
        
    except Exception as e:
        print(f"❌ Failed to load chunked planet from {planet_manifest_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}

def InitializeRuntimeLOD():
    """Initialize runtime LOD system"""
    global runtime_lod_manager, chunk_streamer
    
    if not RUNTIME_LOD_AVAILABLE:
        return False
    
    try:
        # Initialize LOD manager
        runtime_lod_manager = RuntimeLODManager(
            lod_distance_bands=[3.0, 8.0, 20.0, 50.0],
            screen_error_thresholds=[0.5, 2.0, 8.0, 32.0],
            max_chunks_per_frame=80
        )
        
        # Initialize chunk streamer
        chunk_streamer = ChunkStreamer(
            max_memory_mb=128.0,
            max_active_chunks=100,
            load_budget_ms=3.0
        )
        
        print("✅ Runtime LOD system initialized")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize runtime LOD system: {e}")
        return False


def RenderChunkedPlanetWithLOD(chunked_planet: dict):
    """Render chunked planet with runtime LOD selection"""
    global lod_stats
    
    import time
    frame_start = time.time()
    
    # Fall back to basic rendering if LOD system not available
    if not runtime_lod_manager or not chunk_streamer:
        RenderChunkedPlanet(chunked_planet)
        return
    
    try:
        # Get camera parameters
        camera_pos = np.array([camera_x, camera_y, camera_z])
        
        # Calculate camera target
        forward, _, _ = get_camera_vectors()
        camera_target = camera_pos + forward
        
        # Create view and projection matrices  
        view_matrix = create_view_matrix(camera_pos, camera_target, np.array([0, 1, 0]))
        proj_matrix = create_projection_matrix(
            fov_y=math.radians(60), 
            aspect=window_width/window_height, 
            near=0.1, 
            far=100.0
        )
        
        # Get all chunks
        all_chunks = chunked_planet.get("chunks", [])
        
        # Select LOD chunks
        selected_chunks = runtime_lod_manager.select_active_chunks(
            all_chunks, camera_pos, view_matrix, proj_matrix,
            fov_y=math.radians(60), screen_height=window_height
        )
        
        # Update streaming
        load_queue = [chunk.chunk_id for chunk in selected_chunks if chunk.should_load]
        unload_queue = [chunk_id for chunk_id in chunk_streamer.get_loaded_chunk_ids() 
                       if chunk_id not in {c.chunk_id for c in selected_chunks}]
        
        def chunk_data_provider(chunk_id):
            # Find chunk data by ID - handle both formats
            for chunk in all_chunks:
                # Unified format: chunk_id directly in chunk
                if chunk.get("chunk_id") == chunk_id:
                    return chunk
                # Legacy format: chunk_id nested in chunk_info
                if chunk.get("chunk_info", {}).get("chunk_id") == chunk_id:
                    return chunk
            return None
        
        # Track loaded chunks before and after streaming update for LOD manager sync
        pre_loaded = chunk_streamer.get_loaded_chunk_ids()
        chunk_streamer.update_streaming(load_queue, unload_queue, chunk_data_provider)
        post_loaded = chunk_streamer.get_loaded_chunk_ids()
        
        # Sync LOD manager with actual loaded chunks
        newly_loaded = post_loaded - pre_loaded
        newly_unloaded = pre_loaded - post_loaded
        
        for chunk_id in newly_loaded:
            runtime_lod_manager.mark_chunk_loaded(chunk_id)
        for chunk_id in newly_unloaded:
            runtime_lod_manager.mark_chunk_unloaded(chunk_id)
        
        # Render selected chunks
        glEnable(GL_DEPTH_TEST)
        
        if debug_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        chunks_rendered = 0
        # Disable lighting for raw mesh draw to avoid black output if normals/state mismatch
        glDisable(GL_LIGHTING)
        for chunk_lod_info in selected_chunks:
            if not chunk_lod_info.is_visible:
                continue
                
            vao_id = chunk_streamer.get_chunk_vao(chunk_lod_info.chunk_id)
            if vao_id is None:
                continue
            
            # Set LOD-based color tinting
            lod_colors = [
                (1.0, 0.8, 0.8),  # LOD0: Light red (highest detail)
                (0.8, 1.0, 0.8),  # LOD1: Light green
                (0.8, 0.8, 1.0),  # LOD2: Light blue  
                (1.0, 1.0, 0.8)   # LOD3: Light yellow (lowest detail)
            ]
            
            if debug_runtime_lod and chunk_lod_info.lod_level.value < len(lod_colors):
                color = lod_colors[chunk_lod_info.lod_level.value]
                glColor3f(*color)
            else:
                glColor3f(1.0, 1.0, 1.0)  # Default white
            
            # Acquire streamer record (contains raw buffer IDs and counts)
            chunk_rec = None
            try:
                chunk_rec = chunk_streamer.get_chunk_record(chunk_lod_info.chunk_id)
            except Exception:
                chunk_rec = None

            # Fallback path for fixed-function: bind raw buffers with client states
            if chunk_rec is not None:
                # Make sure no VAO is capturing our client states
                glBindVertexArray(0)

                # Positions
                if chunk_rec.pos_buffer > 0:
                    glBindBuffer(GL_ARRAY_BUFFER, chunk_rec.pos_buffer)
                    glEnableClientState(GL_VERTEX_ARRAY)
                    glVertexPointer(3, GL_FLOAT, 0, None)

                # Normals (optional)
                if chunk_rec.norm_buffer > 0:
                    glBindBuffer(GL_ARRAY_BUFFER, chunk_rec.norm_buffer)
                    glEnableClientState(GL_NORMAL_ARRAY)
                    glNormalPointer(GL_FLOAT, 0, None)

                # Colors (per-vertex)
                if chunk_rec.color_buffer > 0:
                    glBindBuffer(GL_ARRAY_BUFFER, chunk_rec.color_buffer)
                    glEnableClientState(GL_COLOR_ARRAY)
                    glColorPointer(3, GL_FLOAT, 0, None)
                else:
                    # Use chunk-level biome color as fallback
                    chunk_data = None
                    for chunk in all_chunks:
                        if chunk.get("chunk_id") == chunk_lod_info.chunk_id or chunk.get("chunk_info", {}).get("chunk_id") == chunk_lod_info.chunk_id:
                            chunk_data = chunk
                            break
                    
                    if chunk_data and "biome_id" in chunk_data:
                        # Get terrain params from cached planet data
                        terrain_params = chunk_cache.get("current_planet", {}).get("terrain_params", {})
                        biomes = terrain_params.get("biomes", [])
                        biome_id = chunk_data["biome_id"]
                        
                        if biome_id < len(biomes):
                            color = biomes[biome_id]["color_rgb"]
                            glColor3f(color[0], color[1], color[2])
                        else:
                            glColor3f(0.431, 0.659, 0.306)  # Default plains color
                    else:
                        glColor3f(1.0, 1.0, 1.0)  # Default white

                # Indices and draw
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, chunk_rec.index_buffer)
                index_count = max(0, int(chunk_rec.triangle_count) * 3)
                if index_count > 0:
                    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)
                    chunks_rendered += 1

                # Cleanup
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_NORMAL_ARRAY)
                glDisableClientState(GL_COLOR_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            else:
                # Existing VAO path (kept for when shader pipeline is added later)
                glBindVertexArray(vao_id)
                
                # Find original chunk for index count
                chunk_data = None
                for chunk in all_chunks:
                    if chunk.get("chunk_id") == chunk_lod_info.chunk_id or chunk.get("chunk_info", {}).get("chunk_id") == chunk_lod_info.chunk_id:
                        chunk_data = chunk
                        break
                        
                if chunk_data:
                    indices = chunk_data.get("indices", np.array([]))
                    if indices.size > 0:
                        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
                        chunks_rendered += 1
                
                glBindVertexArray(0)
        # Re-enable lighting for the rest of the scene
        glEnable(GL_LIGHTING)
        
        # Update statistics
        lod_stats_data = runtime_lod_manager.get_lod_statistics()
        streaming_stats = chunk_streamer.get_streaming_stats()
        
        frame_time = (time.time() - frame_start) * 1000
        lod_stats.update({
            'frame_time_ms': frame_time,
            'active_chunks': chunks_rendered,
            'culled_chunks': lod_stats_data.get('culled_chunks', 0),
            'lod_levels': {f"LOD{k.value if hasattr(k, 'value') else k}": v 
                          for k, v in lod_stats_data.get('active_lod_levels', {}).items()},
            'gpu_memory_mb': chunk_streamer.get_memory_usage_mb()
        })
        
        # Debug visualizations
        if debug_chunk_aabb:
            for chunk_lod_info in selected_chunks[:20]:  # Limit for performance
                # Find original chunk for AABB data
                for chunk in all_chunks:
                    if chunk.get("chunk_info", {}).get("chunk_id") == chunk_lod_info.chunk_id:
                        DrawChunkAABB(chunk)
                        break
        
    except Exception as e:
        print(f"❌ Runtime LOD rendering error: {e}")
        # Fall back to basic rendering
        RenderChunkedPlanet(chunked_planet)


def RenderChunkedPlanet(chunked_planet: dict):
    """Render all chunks of a chunked planet"""
    if not chunked_planet or chunked_planet.get("type") != "chunked_planet":
        return
    
    chunks = chunked_planet.get("chunks", [])
    
    for chunk in chunks:
        if chunk.get("vao"):
            # Render the chunk mesh
            RenderMeshVAO(chunk)
            
            # Debug visualizations
            if debug_chunk_aabb:
                DrawChunkAABB(chunk)
            
            # Draw chunk edges for crack detection
            if debug_crack_lines:
                DrawChunkEdges(chunk, color=(0.0, 1.0, 1.0))  # Cyan for chunk edges
    
    # Draw crack lines between chunks
    if debug_crack_lines:
        DetectAndDrawCrackLines(chunked_planet)


def DrawRuntimeLODHUD():
    """Draw runtime LOD performance HUD"""
    if not debug_show_hud:
        return
    
    # Switch to orthographic projection for HUD
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, window_width, 0, window_height, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    # Disable depth testing for HUD
    glDisable(GL_DEPTH_TEST)
    
    # Semi-transparent background
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # HUD background
    hud_width = 280
    hud_height = 160
    hud_x = 10
    hud_y = window_height - hud_height - 10
    
    glColor4f(0.0, 0.0, 0.0, 0.7)  # Semi-transparent black
    glBegin(GL_QUADS)
    glVertex2f(hud_x, hud_y)
    glVertex2f(hud_x + hud_width, hud_y)
    glVertex2f(hud_x + hud_width, hud_y + hud_height)
    glVertex2f(hud_x, hud_y + hud_height)
    glEnd()
    
    # HUD border
    glColor3f(0.5, 0.8, 1.0)  # Light blue
    glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(hud_x, hud_y)
    glVertex2f(hud_x + hud_width, hud_y)
    glVertex2f(hud_x + hud_width, hud_y + hud_height)
    glVertex2f(hud_x, hud_y + hud_height)
    glEnd()
    
    # HUD text (simplified - just draw colored rectangles for now)
    text_y = hud_y + hud_height - 20
    line_height = 18
    
    # Frame time indicator
    frame_color = (0.0, 1.0, 0.0) if lod_stats['frame_time_ms'] < 16.7 else (1.0, 1.0, 0.0) if lod_stats['frame_time_ms'] < 33.3 else (1.0, 0.0, 0.0)
    frame_bar_width = min(200, lod_stats['frame_time_ms'] * 6)  # Scale for visualization
    
    glColor3f(*frame_color)
    glBegin(GL_QUADS)
    glVertex2f(hud_x + 80, text_y - 5)
    glVertex2f(hud_x + 80 + frame_bar_width, text_y - 5)
    glVertex2f(hud_x + 80 + frame_bar_width, text_y + 10)
    glVertex2f(hud_x + 80, text_y + 10)
    glEnd()
    
    # Active chunks indicator  
    text_y -= line_height
    chunk_color = (0.0, 1.0, 0.0) if lod_stats['active_chunks'] < 50 else (1.0, 1.0, 0.0) if lod_stats['active_chunks'] < 100 else (1.0, 0.0, 0.0)
    chunk_bar_width = min(200, lod_stats['active_chunks'] * 2)  # Scale for visualization
    
    glColor3f(*chunk_color)
    glBegin(GL_QUADS)
    glVertex2f(hud_x + 80, text_y - 5)
    glVertex2f(hud_x + 80 + chunk_bar_width, text_y - 5)
    glVertex2f(hud_x + 80 + chunk_bar_width, text_y + 10)
    glVertex2f(hud_x + 80, text_y + 10)
    glEnd()
    
    # LOD level indicators (colored bars for each level)
    text_y -= line_height
    lod_colors = [(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 1.0)]  # Red, Orange, Green, Blue
    bar_x = hud_x + 80
    
    for i, (lod_level, count) in enumerate(lod_stats.get('lod_levels', {}).items()):
        if i < len(lod_colors):
            bar_width = min(40, count * 2)
            glColor3f(*lod_colors[i])
            glBegin(GL_QUADS)
            glVertex2f(bar_x, text_y - 5)
            glVertex2f(bar_x + bar_width, text_y - 5)
            glVertex2f(bar_x + bar_width, text_y + 10)
            glVertex2f(bar_x, text_y + 10)
            glEnd()
            bar_x += 45
    
    # Memory usage indicator
    text_y -= line_height
    memory_color = (0.0, 1.0, 0.0) if lod_stats['gpu_memory_mb'] < 64 else (1.0, 1.0, 0.0) if lod_stats['gpu_memory_mb'] < 128 else (1.0, 0.0, 0.0)
    memory_bar_width = min(200, lod_stats['gpu_memory_mb'] * 1.5)  # Scale for visualization
    
    glColor3f(*memory_color)
    glBegin(GL_QUADS)
    glVertex2f(hud_x + 80, text_y - 5)
    glVertex2f(hud_x + 80 + memory_bar_width, text_y - 5)
    glVertex2f(hud_x + 80 + memory_bar_width, text_y + 10)
    glVertex2f(hud_x + 80, text_y + 10)
    glEnd()
    
    # Restore OpenGL state
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def DrawChunkAABB(chunk: dict):
    """Draw AABB wireframe for a single chunk"""
    chunk_info = chunk.get("chunk_info", {})
    aabb = chunk_info.get("aabb", {})
    
    if not aabb:
        return
    
    aabb_min = aabb.get("min", [0, 0, 0])
    aabb_max = aabb.get("max", [0, 0, 0])
    
    # Draw wireframe box
    glDisable(GL_LIGHTING)
    glLineWidth(1.0)
    glColor3f(1, 0, 0)  # Red for chunk AABBs
    
    # Draw the 12 edges of the box
    glBegin(GL_LINES)
    
    # Bottom face edges
    glVertex3f(aabb_min[0], aabb_min[1], aabb_min[2])
    glVertex3f(aabb_max[0], aabb_min[1], aabb_min[2])
    
    glVertex3f(aabb_max[0], aabb_min[1], aabb_min[2])
    glVertex3f(aabb_max[0], aabb_min[1], aabb_max[2])
    
    glVertex3f(aabb_max[0], aabb_min[1], aabb_max[2])
    glVertex3f(aabb_min[0], aabb_min[1], aabb_max[2])
    
    glVertex3f(aabb_min[0], aabb_min[1], aabb_max[2])
    glVertex3f(aabb_min[0], aabb_min[1], aabb_min[2])
    
    # Top face edges
    glVertex3f(aabb_min[0], aabb_max[1], aabb_min[2])
    glVertex3f(aabb_max[0], aabb_max[1], aabb_min[2])
    
    glVertex3f(aabb_max[0], aabb_max[1], aabb_min[2])
    glVertex3f(aabb_max[0], aabb_max[1], aabb_max[2])
    
    glVertex3f(aabb_max[0], aabb_max[1], aabb_max[2])
    glVertex3f(aabb_min[0], aabb_max[1], aabb_max[2])
    
    glVertex3f(aabb_min[0], aabb_max[1], aabb_max[2])
    glVertex3f(aabb_min[0], aabb_max[1], aabb_min[2])
    
    # Vertical edges
    glVertex3f(aabb_min[0], aabb_min[1], aabb_min[2])
    glVertex3f(aabb_min[0], aabb_max[1], aabb_min[2])
    
    glVertex3f(aabb_max[0], aabb_min[1], aabb_min[2])
    glVertex3f(aabb_max[0], aabb_max[1], aabb_min[2])
    
    glVertex3f(aabb_max[0], aabb_min[1], aabb_max[2])
    glVertex3f(aabb_max[0], aabb_max[1], aabb_max[2])
    
    glVertex3f(aabb_min[0], aabb_min[1], aabb_max[2])
    glVertex3f(aabb_min[0], aabb_max[1], aabb_max[2])
    
    glEnd()
    
    glEnable(GL_LIGHTING)
    glLineWidth(1.0)

def GenerateTestGridMesh() -> dict:
    """Generate a simple grid mesh for testing"""
    # Create a simple 2x2 grid
    positions = np.array([
        -1.0, 0.0, -1.0,  # 0
         1.0, 0.0, -1.0,  # 1
         1.0, 0.0,  1.0,  # 2
        -1.0, 0.0,  1.0,  # 3
    ], dtype=np.float32)
    
    normals = np.array([
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
    ], dtype=np.float32)
    
    indices = np.array([
        0, 1, 2,  # First triangle
        0, 2, 3,  # Second triangle
    ], dtype=np.uint32)
    
    # Create VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    # Position VBO
    position_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    # Normal VBO
    normal_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)
    
    # EBO
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    glBindVertexArray(0)
    
    return {
        "vao": vao,
        "index_count": len(indices),
        "bounds": {"center": [0, 0, 0], "radius": 1.4},
        "buffers": [position_vbo, normal_vbo, 0, 0, ebo]
    }

def init_gl():
    """Initialize OpenGL settings"""
    global mesh_cache
    
    glClearColor(0.2, 0.4, 0.8, 1.0)  # Nice sky blue background
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    
    # Disable culling to avoid backface issues during debugging
    glDisable(GL_CULL_FACE)
    
    # Better lighting setup
    glLightfv(GL_LIGHT0, GL_POSITION, [5, 10, 5, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 0.8, 1])
    
    # Material properties
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    # Initialize test mesh for replacing spheres
    mesh_cache["test_grid"] = GenerateTestGridMesh()

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

def get_planet_radius():
    """Get planet radius from loaded planet data"""
    if "current_planet" in chunk_cache:
        planet_info = chunk_cache["current_planet"].get("planet_info", {})
        return planet_info.get("radius", 50.0)  # Default radius
    return 50.0  # Fallback

def handle_movement():
    """Handle WASD movement with spherical tangent navigation"""
    global camera_x, camera_y, camera_z, velocity_y

    # Get planet info
    planet_radius = get_planet_radius()
    planet_center = np.array([0.0, 0.0, 0.0])
    
    # Current camera position
    camera_pos = np.array([camera_x, camera_y, camera_z])
    
    # Calculate spherical navigation vectors
    # up = normalize(camera_pos - center)
    up_vec = camera_pos - planet_center
    up_length = np.linalg.norm(up_vec)
    if up_length > 0:
        up = up_vec / up_length
    else:
        up = np.array([0.0, 1.0, 0.0])
    
    # reference_axis = (0,0,1) unless nearly parallel to up, else (1,0,0)
    reference_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, reference_axis)) > 0.9:  # Nearly parallel
        reference_axis = np.array([1.0, 0.0, 0.0])
    
    # right = normalize(cross(up, reference_axis))
    right = np.cross(up, reference_axis)
    right_length = np.linalg.norm(right)
    if right_length > 0:
        right = right / right_length
    else:
        right = np.array([1.0, 0.0, 0.0])
    
    # forward = normalize(cross(right, up))
    forward = np.cross(right, up)
    forward_length = np.linalg.norm(forward)
    if forward_length > 0:
        forward = forward / forward_length
    else:
        forward = np.array([0.0, 0.0, 1.0])

    # Tangential movement
    movement_delta = np.array([0.0, 0.0, 0.0])
    
    if ord('w') in keys_pressed or ord('W') in keys_pressed:
        movement_delta += forward * movement_speed
    if ord('s') in keys_pressed or ord('S') in keys_pressed:
        movement_delta -= forward * movement_speed
    if ord('a') in keys_pressed or ord('A') in keys_pressed:
        movement_delta -= right * movement_speed
    if ord('d') in keys_pressed or ord('D') in keys_pressed:
        movement_delta += right * movement_speed
    
    # Apply movement
    new_camera_pos = camera_pos + movement_delta
    
    # Reproject to maintain altitude (radius + eye_height)
    target_altitude = planet_radius + base_eye_height
    new_distance = np.linalg.norm(new_camera_pos - planet_center)
    if new_distance > 0:
        new_camera_pos = planet_center + (new_camera_pos - planet_center) * (target_altitude / new_distance)
    
    # Handle vertical movement (jump/crouch) - adjust altitude radially
    altitude_adjustment = 0.0
    if ord(' ') in keys_pressed:  # jump
        altitude_adjustment = 2.0  # Increase altitude
    if ord('c') in keys_pressed or ord('C') in keys_pressed:  # crouch
        altitude_adjustment = -1.0  # Decrease altitude
    
    if altitude_adjustment != 0:
        current_altitude = np.linalg.norm(new_camera_pos - planet_center)
        new_altitude = max(planet_radius + base_eye_height * 0.5, current_altitude + altitude_adjustment)
        if current_altitude > 0:
            new_camera_pos = planet_center + (new_camera_pos - planet_center) * (new_altitude / current_altitude)
    
    # Update camera position
    camera_x, camera_y, camera_z = new_camera_pos

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

    # Draw chunked planet if loaded
    if "current_planet" in chunk_cache:
        RenderChunkedPlanetWithLOD(chunk_cache["current_planet"])
    
    # Draw objects from scene data
    elif scene_data:
        for obj in scene_data.get("objects", []):
            obj_type = obj.get("type", "")
            pos = obj.get("pos", [0, 0, 0])
            material = obj.get("material", "")

            glPushMatrix()
            glTranslatef(pos[0], pos[1], pos[2])

            if obj_type == "MESH":
                # Load and render mesh from manifest
                manifest_path = obj.get("manifest", "")
                if manifest_path:
                    mesh_key = f"manifest_{manifest_path}"
                    if mesh_key not in mesh_cache:
                        mesh_cache[mesh_key] = LoadMeshFromManifest(manifest_path)
                    
                    mesh_data = mesh_cache[mesh_key]
                    if mesh_data:
                        if material == "terrain":
                            glColor3f(0.6, 0.4, 0.2)  # Brown terrain
                        elif material == "reference":
                            glColor3f(0.8, 0.8, 0.8)  # Light gray
                        else:
                            glColor3f(1.0, 1.0, 1.0)  # White default
                        
                        RenderMeshVAO(mesh_data["vao"], mesh_data["index_count"])
                        if debug_aabb:
                            DrawAABB(mesh_data["bounds"])
                        if debug_normals:
                            DrawNormals(manifest_path, scale=0.1)
                
            elif obj_type == "CUBE":
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
                
                # Use mesh rendering instead of glutSolidSphere
                glScalef(radius, radius, radius)
                test_mesh = mesh_cache.get("test_grid")
                if test_mesh:
                    RenderMeshVAO(test_mesh["vao"], test_mesh["index_count"])
                    if debug_aabb:
                        DrawAABB(test_mesh["bounds"])
                else:
                    # Fallback to wireframe if mesh not available
                    glutWireSphere(radius, 8, 8)

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

    # Draw runtime LOD HUD
    DrawRuntimeLODHUD()

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
    global movement_speed, mouse_captured, camera_x, camera_y, camera_z, debug_wireframe, debug_aabb, debug_normals, debug_chunk_aabb, debug_crack_lines, debug_runtime_lod, debug_show_hud
    
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
    elif key == b'f' or key == b'F':  # Toggle wireframe
        debug_wireframe = not debug_wireframe
        print(f"🔧 Wireframe mode: {'ON' if debug_wireframe else 'OFF'}")
    elif key == b'b' or key == b'B':  # Toggle AABB debug
        debug_aabb = not debug_aabb
        print(f"🔧 AABB debug: {'ON' if debug_aabb else 'OFF'}")
    elif key == b'n' or key == b'N':  # Toggle normal visualization
        debug_normals = not debug_normals
        print(f"🔧 Normal debug: {'ON' if debug_normals else 'OFF'}")
    elif key == b'x' or key == b'X':  # Toggle chunk AABB debug (changed from C to avoid conflict)
        debug_chunk_aabb = not debug_chunk_aabb
        print(f"🔧 Chunk AABB debug: {'ON' if debug_chunk_aabb else 'OFF'}")
    elif key == b'z' or key == b'Z':  # Toggle crack line visualization
        debug_crack_lines = not debug_crack_lines
        print(f"🔧 Crack line debug: {'ON' if debug_crack_lines else 'OFF'}")
    elif key == b'l' or key == b'L':  # Toggle runtime LOD debug
        debug_runtime_lod = not debug_runtime_lod
        print(f"🔧 Runtime LOD debug: {'ON' if debug_runtime_lod else 'OFF'}")
    elif key == b'h' or key == b'H':  # Toggle HUD
        debug_show_hud = not debug_show_hud
        print(f"🔧 HUD display: {'ON' if debug_show_hud else 'OFF'}")

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
        # Load scene data - check if it's a chunked planet
        with open(scene_file, 'r') as f:
            scene_data = json.load(f)

        # Check if this is a chunked planet manifest
        if scene_data.get("planet_info", {}).get("type") == "chunked_quadtree" or scene_data.get("planet", {}).get("type") == "chunked_quadtree":
            print(f"🪐 Loading chunked planet: {scene_file}")
            chunked_planet = LoadChunkedPlanet(scene_file)
            if chunked_planet:
                chunk_cache["current_planet"] = chunked_planet
                window_title = f"Chunked Planet - {Path(scene_file).name}"
                
                # Initialize spherical spawn position
                planet_radius = chunked_planet.get("planet_info", {}).get("radius", 50.0)
                global camera_x, camera_y, camera_z
                camera_x, camera_y, camera_z = 0.0, planet_radius + base_eye_height, 0.0
                print(f"🌍 Spawned on sphere: radius={planet_radius}, position=({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})")
                
                # Initialize runtime LOD system for chunked planets
                InitializeRuntimeLOD()
            else:
                print(f"❌ Failed to load chunked planet")
                return False
        else:
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
        print("   F - Toggle wireframe mode")
        print("   B - Toggle AABB debug")
        print("   N - Toggle normal visualization")
        print("   X - Toggle chunk AABB debug")
        print("   Z - Toggle crack line debug")
        print("   L - Toggle runtime LOD debug coloring")
        print("   H - Toggle performance HUD")
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
