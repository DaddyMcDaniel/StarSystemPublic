# T16 - Viewer Tools: Diagnostics HUD & Toggles

## Overview

T16 implements comprehensive viewer tools for debugging PCC terrain content and LOD behavior. The system provides real-time diagnostics, visual debug modes, advanced camera navigation, and automated screenshot workflows designed to make terrain debugging efficient and developer-friendly.

## Implementation

### Diagnostics HUD System

**File:** `hud/diagnostics_hud.py` (lines 1-540)

Real-time performance and system diagnostics with comprehensive metrics display:

```python
class DiagnosticsHUD:
    def update_frame_stats(self, fps: float, frame_time_ms: float, draw_calls: int,
                          triangles: int, vertices: int, active_chunks: int,
                          visible_chunks: int, memory_usage_mb: float, vram_usage_mb: float):
        # Updates all frame statistics with performance tracking
        
    def update_lod_stats(self, lod_histogram: Dict[int, int], lod_switches_per_second: float):
        # Updates LOD level distribution and transition statistics
        
    def generate_hud_text(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        # Generates colored HUD text lines with performance-based coloring
```

#### HUD Components

**Core Performance Metrics:**
- **FPS & Frame Time**: Real-time frame rate with color-coded performance indicators
- **Draw Calls**: Rendering pipeline efficiency tracking  
- **Active/Visible Chunks**: Chunk management and culling statistics
- **VRAM/Mesh Memory**: Memory usage with pressure monitoring
- **LOD Histogram**: Visual distribution of Level-of-Detail across terrain

**Advanced Metrics (Detailed View):**
```python
def get_performance_summary(self) -> Dict[str, Any]:
    return {
        'avg_fps': statistics.mean(fps_values),
        'min_fps': min(fps_values),
        'max_fps': max(fps_values),
        'frame_time_variance': statistics.variance(frame_times),
        'frame_count': len(recent_frames)
    }
```

**LOD Histogram Visualization:**
```python
def generate_lod_histogram_text(self) -> List[str]:
    for lod in sorted(self.lod_stats.lod_histogram.keys()):
        count = self.lod_stats.lod_histogram[lod]
        bar_length = int((count / max_count) * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        lines.append(f"LOD {lod:2d}: {bar} {count:3d}")
```

**Performance Color Coding:**
- 🟢 **Green**: Good performance (FPS ≥ 55, Frame Time ≤ 16.67ms)
- 🟡 **Yellow**: Warning (FPS ≥ 30, Frame Time ≤ 33.33ms)  
- 🔴 **Red**: Critical (FPS < 30, Frame Time > 33.33ms)

### Debug Toggles System

**File:** `debug_ui/debug_toggles.py` (lines 1-650)

Comprehensive debug visualization system with mutual exclusivity and state management:

```python
class DebugToggles:
    def toggle(self, toggle_type: ToggleType) -> bool:
        # Handles mutual exclusivity and state transitions
        
    def get_render_state(self) -> RenderState:
        # Returns complete render state for graphics pipeline
        
    def handle_hotkey(self, key: str) -> Optional[ToggleType]:
        # Maps F-key shortcuts to toggle actions
```

#### Available Debug Modes

**Rendering Modes:**
- `F1` **Wireframe**: Shows mesh topology and triangle structure
- `F2` **Normals**: Visualizes vertex normal vectors for lighting debug
- `F8` **Texture Debug**: Shows UV coordinates and texture mapping
- `F9` **Material Debug**: Displays material properties and assignments

**Chunk Debugging:**
- `F3` **Chunk IDs**: Overlays chunk identifiers for spatial debugging
- `F4` **Chunk Boundaries**: Shows chunk edges and seam detection
- Visual chunk grid with color-coded chunk states

**LOD Visualization:**
- `F5` **LOD Heatmap**: Color-codes terrain by level-of-detail
  - Blue: High detail (LOD 0-2)
  - Green: Medium detail (LOD 3-5)
  - Yellow: Low detail (LOD 6-8)
  - Red: Very low detail (LOD 9+)

**Content Filtering:**
- `F6` **Cave Only**: Hide surface geometry, show cave systems only
- `F7` **Surface Only**: Hide cave geometry, show surface only
- Mutually exclusive filtering for focused inspection

**Performance Debugging:**
- `F10` **Lighting Debug**: Shows lighting calculations and shadow maps
- `F11` **Overdraw Debug**: Visualizes pixel overdraw for optimization
- `F12` **Distance Debug**: Color-codes geometry by camera distance

#### Toggle State Management

**Mutual Exclusivity System:**
```python
@dataclass
class ToggleState:
    enabled: bool = False
    hotkey: Optional[str] = None
    mutually_exclusive: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
```

**Render State Generation:**
```python
@dataclass
class RenderState:
    render_mode: RenderMode = RenderMode.NORMAL
    show_wireframe: bool = False
    show_normals: bool = False
    show_chunk_boundaries: bool = False
    show_chunk_ids: bool = False
    show_lod_heatmap: bool = False
    filter_caves_only: bool = False
    filter_surface_only: bool = False
```

### Debug Camera System

**File:** `camera_tools/debug_camera.py` (lines 1-650)

Advanced camera navigation system for efficient terrain inspection:

```python
class DebugCamera:
    def jump_to_face(self, face: Union[CubeFace, str], immediate: bool = False) -> bool:
        # Quick navigation to cube sphere faces
        
    def jump_to_quadtree_node(self, depth: int, x: int, z: int, immediate: bool = False) -> bool:
        # Direct navigation to specific quadtree nodes
        
    def frame_bounds(self, min_bounds: Tuple[float, float, float], 
                    max_bounds: Tuple[float, float, float]) -> bool:
        # Auto-frame specific world regions
```

#### Camera Navigation Features

**Cube Face Navigation:**
- **Faces**: Front (+Z), Back (-Z), Left (-X), Right (+X), Top (+Y), Bottom (-Y)
- **Poles**: North Pole, South Pole with optimal viewing angles
- **Hotkeys**: Number keys 1-6 for instant face switching

**Quadtree Node Inspection:**
```python
def jump_to_quadtree_node(self, depth: int, x: int, z: int) -> bool:
    node = QuadtreeNode(depth=depth, x=x, z=z)
    min_bounds, max_bounds = node.get_world_bounds(self.chunk_size, self.planet_radius)
    
    # Calculate optimal viewing position
    camera_distance = max(self.min_distance, node_size * 2.0)
    camera_height = camera_distance * 0.6
```

**Bookmark System:**
```python
def save_bookmark(self, name: str, description: str = "", tags: List[str] = None) -> bool:
    bookmark = CameraBookmark(
        name=name,
        state=self.current_state,
        description=description,
        tags=tags or []
    )
    self.bookmarks[name] = bookmark
```

**Camera History:**
- Automatic position history (50 positions)
- `Backspace` to return to previous position
- Smooth transitions between positions

**Path Recording:**
```python
def start_path_recording(self):
    self.recording_path = True
    self.current_path = [self.current_state]
    
def record_path_point(self):
    if self.recording_path:
        self.current_path.append(self.current_state)
```

### Screenshot Tool System

**File:** `viewer_tools/screenshot_tool.py` (lines 1-750)

Advanced screenshot capture with automatic PCC+seed filename stamping:

```python
class ScreenshotTool:
    def capture_screenshot(self, pcc_name: str, seed: int, camera_position: Tuple[float, float, float],
                          capture_mode: CaptureMode, debug_toggles: List[str]) -> Optional[str]:
        # Captures screenshot with complete metadata stamping
        
    def _generate_filename(self, metadata: ScreenshotMetadata) -> str:
        # Generates descriptive filenames with PCC info and debug state
```

#### Automatic Filename Generation

**Filename Template:**
```
{pcc_name}_{seed}_{timestamp}_{camera_hash}_{mode}_{debug_hash}.png
```

**Example Filenames:**
```
hero_planet_seed12345_20240819_143052_cam4a2b3c_lod_heatmap_dbg8f9e.png
minimal_sphere_seed54321_20240819_143100_camdef456_wireframe.png
test_terrain_seed98765_20240819_143120_cam789abc_cave_only_dbg1234.png
```

**Metadata Embedding:**
```python
@dataclass
class ScreenshotMetadata:
    pcc_name: str
    seed: int
    camera_position: Tuple[float, float, float]
    camera_target: Tuple[float, float, float]
    capture_mode: CaptureMode
    debug_toggles: List[str]
    performance_stats: Dict[str, Any]
    timestamp: float
```

#### Screenshot Workflows

**LOD Sequence Capture:**
```python
def capture_lod_sequence(self, pcc_name: str, seed: int, camera_position: Tuple[float, float, float],
                        lod_levels: List[int]) -> List[str]:
    series_name = f"{pcc_name}_lod_sequence"
    self.start_series(series_name)
    
    for lod_level in lod_levels:
        screenshot_path = self.capture_screenshot(
            capture_mode=CaptureMode.LOD_HEATMAP,
            debug_toggles=[f"lod_level_{lod_level}"],
            tags=["lod_sequence", f"lod_{lod_level}"]
        )
```

**Face Sequence Capture:**
```python
def capture_face_sequence(self, pcc_name: str, seed: int, faces: List[str]) -> List[str]:
    # Captures screenshots from each cube face for complete coverage
    for face in faces:
        camera_pos = face_positions.get(face)
        screenshot_path = self.capture_screenshot(camera_position=camera_pos)
```

**Debug Comparison Capture:**
```python
def capture_debug_comparison(self, pcc_name: str, seed: int, camera_position: Tuple[float, float, float],
                           debug_modes: List[CaptureMode]) -> List[str]:
    # Captures same view with different debug visualizations
    for mode in debug_modes:
        screenshot_path = self.capture_screenshot(capture_mode=mode)
```

### Integrated Developer Viewer

**File:** `viewer_tools/developer_viewer.py` (lines 1-850)

Unified interface combining all T16 components with workflow automation:

```python
class DeveloperViewer:
    def __init__(self):
        self.hud = DiagnosticsHUD()
        self.toggles = DebugToggles()
        self.camera = DebugCamera()
        self.screenshot_tool = ScreenshotTool()
        
    def handle_key_input(self, key: str, modifiers: List[str] = None) -> bool:
        # Unified keyboard shortcut handling
```

#### Key Binding System

**Camera Navigation:**
- `1-6`: Jump to cube faces (Front, Back, Left, Right, Top, Bottom)
- `G`: Go to quadtree node (interactive input)
- `B`: Bookmark menu
- `Backspace`: Go back to previous position

**Debug Toggles:**
- `F1-F12`: Debug visualization modes
- `H`: Toggle HUD visibility
- `Tab`: Cycle viewer modes

**Screenshots:**
- `F12`: Take screenshot
- `Ctrl+F12`: Start screenshot series
- `Shift+F12`: Quick comparison shots

**Workflow Automation:**
- `Ctrl+L`: LOD debug workflow
- `Ctrl+P`: Performance profile workflow
- `Ctrl+C`: Comparison workflow

**Session Management:**
- `Ctrl+S`: Save session
- `Ctrl+O`: Load session  
- `Ctrl+N`: New session

#### Automated Workflows

**LOD Debug Workflow:**
```python
def _start_lod_debug_workflow(self) -> bool:
    # 1. Enable LOD heatmap
    self.toggles.set_toggle(ToggleType.LOD_HEATMAP, True)
    
    # 2. Jump to overview position
    self.camera.jump_to_face(CubeFace.TOP, immediate=True)
    
    # 3. Take screenshot
    self._take_screenshot()
    
    # 4. Start screenshot series
    self.screenshot_tool.start_series(f"{pcc_name}_lod_debug")
```

**Performance Profile Workflow:**
```python
def _start_performance_workflow(self) -> bool:
    # Enable detailed HUD
    self.hud.toggle_detailed_view()
    
    # Disable resource-intensive debug modes
    self.toggles.disable_all()
    
    # Begin performance monitoring
    self.current_workflow = "performance"
```

**Comparison Workflow:**
```python
def _start_comparison_workflow(self) -> bool:
    # Take baseline screenshot
    self.toggles.disable_all()
    self._take_screenshot()
    
    # Prepare for debug mode comparisons
    self.screenshot_tool.start_series(f"{pcc_name}_comparison")
```

#### Session Persistence

**Session Data:**
```python
@dataclass
class ViewerSession:
    pcc_file: str
    pcc_name: str
    seed: int
    mode: ViewerMode
    camera_state: Dict[str, Any]
    debug_state: Dict[str, Any]
    hud_state: Dict[str, Any]
    screenshots_taken: List[str]
    session_notes: str
```

**Session Export:**
```python
def _save_session(self) -> bool:
    session_data = {
        'pcc_file': self.current_session.pcc_file,
        'pcc_name': self.current_session.pcc_name,
        'seed': self.current_session.seed,
        'camera_state': {
            'position': self.camera.current_state.position,
            'target': self.camera.current_state.target,
            'fov': self.camera.current_state.fov_degrees
        },
        'debug_state': {
            'active_toggles': [t.value for t in self.toggles.get_active_toggles()],
            'render_mode': self.toggles.get_render_state().render_mode.value
        },
        'screenshots_taken': self.current_session.screenshots_taken
    }
```

## Testing and Validation

### Core Component Tests

**Diagnostics HUD Test Results:**
```
🚀 T16 Diagnostics HUD System
============================================================
📊 Testing HUD functionality...
   ✅ HUD data generated: 12 text lines
   ✅ Performance summary: Avg FPS 58.3
   ✅ LOD histogram: 5 lines
   ✅ Detailed view: True

📈 Sample HUD Text Lines:
   FPS: 58.3
   Frame Time: 17.15ms
   Draw Calls: 120
   Triangles: 95,847
   Active Chunks: 32
   Visible Chunks: 24
   VRAM: 456.8MB
   Mesh Memory: 234.5MB

📊 Sample LOD Histogram:
   LOD Histogram:
   LOD  0: ████████████████████  15
   LOD  1: ████████████████░░░░  12
   LOD  2: ██████░░░░░░░░░░░░░░   6
   LOD  3: ███░░░░░░░░░░░░░░░░░   3
```

**Debug Toggles Test Results:**
```
🚀 T16 Debug Toggles System
============================================================
📊 Testing toggle functionality...
   Initial wireframe state: False
   After toggle wireframe: True
   After toggle normals - wireframe: False
   After toggle normals - normals: True
   Render mode: normals_only
   Show normals: True
   F3 hotkey toggles: chunk_ids
   Toggle categories: ['rendering', 'geometry', 'chunking', 'lod', 'filtering', 'materials', 'performance']
   Active toggles: ['normals', 'chunk_ids']

🔧 Testing content filtering...
   Cave only filter: True
   Surface only filter: False
   After surface toggle - cave only: False
   After surface toggle - surface only: True

🎨 Testing LOD heatmap...
   LOD heatmap: True
   Render mode: heatmap
```

**Debug Camera Test Results:**
```
🚀 T16 Debug Camera System
============================================================
📊 Testing camera navigation...
   Initial position: (0.0, 0.0, 100.0)
   Front face position: (0.0, 0.0, 1500.0)
   Top face position: (0.0, 1500.0, 0.0)
   Quadtree node [2](1,2) position: (128.0, 600.0, 256.0)
   Bookmarks: ['test_view']
   After bookmark load: (0.0, 1500.0, 0.0)
   After go back: (0.0, 1500.0, 0.0)
   After frame bounds: (75.0, 75.0, 75.0)
   Orbit position (45°): (141.42, 60.0, 141.42)
   Available faces: ['front', 'back', 'left', 'right', 'top', 'bottom', 'north_pole', 'south_pole']
   Quick distances: ['close', 'medium', 'far', 'overview']
   Recorded path with 3 points
```

**Screenshot Tool Test Results:**
```
🚀 T16 Screenshot Tool System
============================================================
📊 Testing screenshot functionality...
📸 Screenshot saved: hero_planet_seed12345_20240819_143052_cam4a2b3c_standard.png
   PCC: hero_planet, Seed: 12345
   Camera: (100.0, 200.0, 150.0)
   Mode: standard

📸 Screenshot saved: minimal_sphere_seed54321_20240819_143053_cam789def_wireframe_dbg8f9e.png
   PCC: minimal_sphere, Seed: 54321
   Camera: (0.0, 300.0, 500.0)
   Mode: wireframe

📸 Testing LOD sequence...
📸 Started screenshot series: test_terrain_lod_sequence
📸 Ended screenshot series: test_terrain_lod_sequence (4 screenshots)
   ✅ LOD sequence: 4 screenshots

🔄 Testing face sequence...
📸 Started screenshot series: cube_planet_face_sequence
📸 Ended screenshot series: cube_planet_face_sequence (3 screenshots)
   ✅ Face sequence: 3 screenshots

🔧 Testing debug comparison...
📸 Started screenshot series: debug_terrain_debug_comparison
📸 Ended screenshot series: debug_terrain_debug_comparison (3 screenshots)
   ✅ Debug comparison: 3 screenshots

📋 Recent screenshots: 5
📸 Screenshot index exported to test_screenshots/screenshot_index.json
```

**Developer Viewer Integration Test:**
```
🚀 T16 Developer Viewer System
============================================================
📊 Testing viewer functionality...
   Testing key bindings...
📷 Jumped to front face at (0.0, 0.0, 1500.0)
🔧 Toggled wireframe
📸 Screenshot saved: test_terrain_seed12345_20240819_143054_cam123abc_wireframe.png

🔧 Testing workflows...
🔧 Starting LOD debug workflow...
   ✅ LOD debug workflow active
   📍 Use number keys (1-6) to check different faces
   📸 Screenshots will be automatically captured

💾 Testing session management...
💾 Session saved: debug_sessions/session_test_terrain_1692454234.json

📊 UI Data structure:
   HUD visible: True
   Current mode: content_review
   Screenshots taken: 2

❓ Help categories: ['camera', 'debug', 'screenshot', 'ui', 'workflow', 'session']

✅ Developer viewer system functional
   Session: test_terrain
   Components integrated: HUD, Toggles, Camera, Screenshots
   Key bindings: 25
   Workflows available: LOD debug, Performance, Comparison
```

## Usage Examples

### Basic Debug Session

```python
from viewer_tools.developer_viewer import DeveloperViewer

# Initialize viewer
viewer = DeveloperViewer()

# Load PCC file
viewer.load_pcc_file("examples/hero_planet.pcc")

# Quick camera navigation
viewer.handle_key_input("1")    # Jump to front face
viewer.handle_key_input("F5")   # Enable LOD heatmap
viewer.handle_key_input("F12")  # Take screenshot

# Start LOD debug workflow
viewer.handle_key_input("Ctrl+L")
```

### Advanced Debugging Workflow

```python
# Initialize components separately for custom workflow
from hud.diagnostics_hud import DiagnosticsHUD
from debug_ui.debug_toggles import DebugToggles, ToggleType
from camera_tools.debug_camera import DebugCamera, CubeFace
from viewer_tools.screenshot_tool import ScreenshotTool

# Set up debugging session
hud = DiagnosticsHUD()
toggles = DebugToggles()
camera = DebugCamera()
screenshots = ScreenshotTool()

# Configure for cave system debugging
toggles.set_toggle(ToggleType.CAVE_ONLY, True)
toggles.set_toggle(ToggleType.CHUNK_BOUNDARIES, True)

# Navigate to interior view
camera.jump_to_face(CubeFace.FRONT)
camera.orbit_around_point((0, 0, 0), radius=500.0, angle_degrees=45.0)

# Capture documentation
screenshots.start_series("cave_system_analysis")
screenshots.capture_screenshot(
    pcc_name="complex_terrain", 
    seed=12345,
    camera_position=camera.current_state.position,
    capture_mode=CaptureMode.CAVE_ONLY,
    debug_toggles=["cave_only", "chunk_boundaries"],
    description="Cave system with chunk boundary overlay"
)
```

### Performance Analysis Session

```python
# Set up performance monitoring
viewer = DeveloperViewer()
viewer.load_pcc_file("performance_test.pcc")

# Start performance workflow
viewer._start_performance_workflow()

# Simulate movement for LOD testing
test_positions = [
    (100, 100, 100),   # Close view
    (500, 200, 500),   # Medium distance
    (2000, 500, 2000), # Far view
    (5000, 1000, 5000) # Overview
]

for pos in test_positions:
    # Update camera position
    camera.current_state.position = pos
    
    # Update performance metrics
    viewer.update_frame(fps=45.0, draw_calls=150, triangles=95000, active_chunks=32)
    
    # Take performance screenshot
    viewer._take_screenshot()

# Export performance data
viewer.hud.export_performance_data("performance_analysis.json")
```

### Screenshot Documentation Workflow

```python
# Comprehensive documentation capture
tool = ScreenshotTool()

# Document all major views of a terrain
pcc_name = "documented_planet"
seed = 98765

# 1. Face sequence for complete coverage
face_screenshots = tool.capture_face_sequence(
    pcc_name, seed, 
    faces=["front", "back", "left", "right", "top", "bottom"]
)

# 2. LOD progression analysis
lod_screenshots = tool.capture_lod_sequence(
    pcc_name, seed,
    camera_position=(1000, 500, 1000),
    lod_levels=[0, 1, 2, 3, 4, 5]
)

# 3. Debug mode comparison
debug_screenshots = tool.capture_debug_comparison(
    pcc_name, seed,
    camera_position=(800, 400, 800),
    debug_modes=[
        CaptureMode.STANDARD,
        CaptureMode.WIREFRAME, 
        CaptureMode.LOD_HEATMAP,
        CaptureMode.CAVE_ONLY
    ]
)

# Export complete documentation index
tool.export_screenshot_index("terrain_documentation.json")
```

## Verification Status

✅ **T16 Complete**: Viewer tools with diagnostics HUD and toggles successfully implemented

### Diagnostics HUD
- ✅ Real-time FPS, frame time, and performance metrics
- ✅ Draw calls and rendering pipeline statistics
- ✅ Active/visible chunk count with memory tracking
- ✅ VRAM and mesh memory usage monitoring
- ✅ LOD histogram with visual distribution bars
- ✅ Performance color coding and threshold alerts
- ✅ Detailed view with advanced statistics

### Debug Toggles
- ✅ Wireframe rendering mode (`F1`)
- ✅ Normal vector visualization (`F2`)
- ✅ Chunk ID overlay and boundaries (`F3`, `F4`)
- ✅ LOD level heatmap visualization (`F5`)
- ✅ Cave-only and surface-only filtering (`F6`, `F7`)
- ✅ Material and texture debug modes (`F8`, `F9`)
- ✅ Performance debug visualizations (`F10-F12`)
- ✅ Mutual exclusivity and state management

### Camera Navigation Tools
- ✅ Quick jump to cube sphere faces (1-6 keys)
- ✅ Direct navigation to quadtree nodes by coordinates
- ✅ Bookmark system with save/load/delete
- ✅ Camera history with back navigation
- ✅ Auto-framing of specific world regions
- ✅ Path recording and playback functionality

### Screenshot Tool
- ✅ Automatic PCC+seed filename stamping
- ✅ Debug state and camera position embedding
- ✅ Screenshot series management with metadata
- ✅ LOD sequence and face sequence capture
- ✅ Debug mode comparison workflows
- ✅ Performance and content documentation

### Developer-Friendly UX
- ✅ Unified interface integrating all components
- ✅ Comprehensive keyboard shortcut system
- ✅ Automated workflows for common debug tasks
- ✅ Session persistence and restoration
- ✅ Context-sensitive help system
- ✅ Performance monitoring and analysis tools

The T16 implementation successfully makes it **easy to debug content and LOD** as requested, providing comprehensive tools for efficient terrain inspection, performance analysis, and documentation generation with developer-friendly workflows and automation.

## Integration with T13-T15 Pipeline

**T13 Determinism Integration:**
- Screenshot filenames include deterministic seed information
- Debug sessions can be reproduced with identical PCC+seed combinations
- Camera bookmarks preserve exact viewing conditions for consistency

**T14 Performance Integration:**  
- HUD displays T14 performance metrics (parallel efficiency, memory usage)
- Debug toggles optimize for T14 streaming and LOD systems
- Screenshot workflows document T14 performance optimizations

**T15 Schema Integration:**
- Viewer automatically extracts PCC metadata using T15 schema validation
- Debug modes respect T15 node specifications and parameter ranges
- Session files store T15-compliant PCC file references

The viewer tools provide essential debugging capabilities for the complete T13-T15 terrain pipeline while maintaining full integration and workflow efficiency.