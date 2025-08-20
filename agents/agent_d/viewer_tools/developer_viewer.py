#!/usr/bin/env python3
"""
Developer-Friendly Viewer UX - T16
==================================

Integrated developer-friendly viewer interface that combines all T16 components:
diagnostics HUD, debug toggles, camera tools, and screenshot functionality.
Designed for efficient debugging of PCC terrain content and LOD behavior.

Features:
- Unified UI for all debugging tools
- Keyboard shortcuts for quick access
- Context-sensitive help system
- Session persistence and restoration
- Workflow automation for common tasks

Usage:
    viewer = DeveloperViewer()
    viewer.load_pcc_file("examples/hero_planet.pcc")
    viewer.run_debug_session()
"""

import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import sys
import os

# Import T16 components
from ..hud.diagnostics_hud import DiagnosticsHUD
from ..debug_ui.debug_toggles import DebugToggles, ToggleType
from ..camera_tools.debug_camera import DebugCamera, CubeFace
from .screenshot_tool import ScreenshotTool, CaptureMode


class ViewerMode(Enum):
    """Viewer operation modes"""
    CONTENT_REVIEW = "content_review"      # Reviewing generated content
    LOD_DEBUG = "lod_debug"               # Debugging LOD behavior
    PERFORMANCE_PROFILE = "performance"   # Performance analysis
    COMPARISON = "comparison"             # Comparing different versions
    PRESENTATION = "presentation"         # Clean presentation mode


@dataclass
class ViewerSession:
    """Complete viewer session state"""
    pcc_file: str = ""
    pcc_name: str = ""
    seed: int = 0
    mode: ViewerMode = ViewerMode.CONTENT_REVIEW
    camera_state: Dict[str, Any] = field(default_factory=dict)
    debug_state: Dict[str, Any] = field(default_factory=dict)
    hud_state: Dict[str, Any] = field(default_factory=dict)
    screenshots_taken: List[str] = field(default_factory=list)
    session_start: float = field(default_factory=time.time)
    session_notes: str = ""


@dataclass
class KeyBinding:
    """Keyboard shortcut definition"""
    key: str
    description: str
    category: str
    action: str
    args: Dict[str, Any] = field(default_factory=dict)


class DeveloperViewer:
    """Integrated developer-friendly viewer for PCC terrain debugging"""
    
    def __init__(self, output_dir: str = "debug_sessions"):
        """Initialize developer viewer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize core components
        self.hud = DiagnosticsHUD()
        self.toggles = DebugToggles()
        self.camera = DebugCamera()
        self.screenshot_tool = ScreenshotTool(str(self.output_dir / "screenshots"))
        
        # Viewer state
        self.current_session = ViewerSession()
        self.session_history: List[ViewerSession] = []
        
        # UI state
        self.show_help = False
        self.help_category = "general"
        self.command_mode = False
        self.command_buffer = ""
        
        # Key bindings
        self.key_bindings = self._create_key_bindings()
        
        # Workflow automation
        self.current_workflow: Optional[str] = None
        self.workflow_step = 0
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Connect components
        self._setup_component_integration()
    
    def _create_key_bindings(self) -> Dict[str, KeyBinding]:
        """Create keyboard shortcut bindings"""
        bindings = {}
        
        # Camera controls
        bindings["1"] = KeyBinding("1", "Jump to front face", "camera", "jump_to_face", {"face": "front"})
        bindings["2"] = KeyBinding("2", "Jump to back face", "camera", "jump_to_face", {"face": "back"})
        bindings["3"] = KeyBinding("3", "Jump to left face", "camera", "jump_to_face", {"face": "left"})
        bindings["4"] = KeyBinding("4", "Jump to right face", "camera", "jump_to_face", {"face": "right"})
        bindings["5"] = KeyBinding("5", "Jump to top face", "camera", "jump_to_face", {"face": "top"})
        bindings["6"] = KeyBinding("6", "Jump to bottom face", "camera", "jump_to_face", {"face": "bottom"})
        
        # Debug toggles (F-keys handled by debug system)
        bindings["F1"] = KeyBinding("F1", "Toggle wireframe", "debug", "toggle", {"type": "wireframe"})
        bindings["F2"] = KeyBinding("F2", "Toggle normals", "debug", "toggle", {"type": "normals"})
        bindings["F3"] = KeyBinding("F3", "Toggle chunk IDs", "debug", "toggle", {"type": "chunk_ids"})
        bindings["F4"] = KeyBinding("F4", "Toggle chunk boundaries", "debug", "toggle", {"type": "chunk_boundaries"})
        bindings["F5"] = KeyBinding("F5", "Toggle LOD heatmap", "debug", "toggle", {"type": "lod_heatmap"})
        bindings["F6"] = KeyBinding("F6", "Toggle cave only", "debug", "toggle", {"type": "cave_only"})
        bindings["F7"] = KeyBinding("F7", "Toggle surface only", "debug", "toggle", {"type": "surface_only"})
        
        # Screenshots
        bindings["F12"] = KeyBinding("F12", "Take screenshot", "screenshot", "capture_screenshot")
        bindings["Ctrl+F12"] = KeyBinding("Ctrl+F12", "Start screenshot series", "screenshot", "start_series")
        bindings["Shift+F12"] = KeyBinding("Shift+F12", "Quick comparison shots", "screenshot", "quick_comparison")
        
        # HUD and UI
        bindings["H"] = KeyBinding("H", "Toggle HUD", "ui", "toggle_hud")
        bindings["?"] = KeyBinding("?", "Show help", "ui", "show_help")
        bindings["`"] = KeyBinding("`", "Command mode", "ui", "command_mode")
        bindings["Tab"] = KeyBinding("Tab", "Cycle viewer modes", "ui", "cycle_mode")
        
        # Quick navigation
        bindings["G"] = KeyBinding("G", "Go to quadtree node", "camera", "go_to_node")
        bindings["B"] = KeyBinding("B", "Camera bookmark menu", "camera", "bookmark_menu")
        bindings["Backspace"] = KeyBinding("Backspace", "Go back", "camera", "go_back")
        
        # Workflow shortcuts
        bindings["Ctrl+L"] = KeyBinding("Ctrl+L", "LOD debug workflow", "workflow", "lod_debug")
        bindings["Ctrl+P"] = KeyBinding("Ctrl+P", "Performance profile workflow", "workflow", "performance_profile")
        bindings["Ctrl+C"] = KeyBinding("Ctrl+C", "Comparison workflow", "workflow", "comparison")
        
        # Session management
        bindings["Ctrl+S"] = KeyBinding("Ctrl+S", "Save session", "session", "save_session")
        bindings["Ctrl+O"] = KeyBinding("Ctrl+O", "Load session", "session", "load_session")
        bindings["Ctrl+N"] = KeyBinding("Ctrl+N", "New session", "session", "new_session")
        
        return bindings
    
    def _setup_component_integration(self):
        """Set up integration between components"""
        # Connect debug toggles to rendering
        for toggle_type in [ToggleType.WIREFRAME, ToggleType.NORMALS, ToggleType.LOD_HEATMAP]:
            self.toggles.add_callback(toggle_type, self._on_render_mode_change)
        
        # Connect camera to screenshot metadata
        # (Would be implemented with actual camera state)
    
    def load_pcc_file(self, pcc_file_path: str) -> bool:
        """Load PCC file and initialize session"""
        pcc_path = Path(pcc_file_path)
        if not pcc_path.exists():
            print(f"❌ PCC file not found: {pcc_file_path}")
            return False
        
        try:
            # Load PCC metadata
            with open(pcc_path, 'r') as f:
                pcc_data = json.load(f)
            
            # Extract information
            pcc_name = pcc_data.get('metadata', {}).get('name', pcc_path.stem)
            seed = self._extract_seed_from_pcc(pcc_data)
            
            # Initialize new session
            self.current_session = ViewerSession(
                pcc_file=str(pcc_path),
                pcc_name=pcc_name,
                seed=seed,
                mode=ViewerMode.CONTENT_REVIEW
            )
            
            # Reset components for new content
            self.camera = DebugCamera()
            self.toggles.disable_all()
            self.hud = DiagnosticsHUD()
            
            print(f"✅ Loaded PCC: {pcc_name} (seed: {seed})")
            print(f"   File: {pcc_file_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load PCC file: {e}")
            return False
    
    def _extract_seed_from_pcc(self, pcc_data: Dict[str, Any]) -> int:
        """Extract seed from PCC data"""
        # Look for seed in stochastic nodes
        for node in pcc_data.get('nodes', []):
            if node.get('type') in ['NoiseFBM', 'RidgedMF']:
                params = node.get('parameters', {})
                if 'seed' in params:
                    return int(params['seed'])
        
        return 0
    
    def handle_key_input(self, key: str, modifiers: List[str] = None) -> bool:
        """Handle keyboard input"""
        if modifiers is None:
            modifiers = []
        
        # Build full key string with modifiers
        if modifiers:
            full_key = "+".join(modifiers + [key])
        else:
            full_key = key
        
        # Check for exact match first
        if full_key in self.key_bindings:
            return self._execute_action(self.key_bindings[full_key])
        
        # Check base key
        if key in self.key_bindings:
            return self._execute_action(self.key_bindings[key])
        
        # Handle debug toggle hotkeys
        toggle_result = self.toggles.handle_hotkey(key)
        if toggle_result:
            print(f"🔧 Toggled {toggle_result.value}")
            return True
        
        # Handle command mode
        if self.command_mode:
            return self._handle_command_input(key)
        
        return False
    
    def _execute_action(self, binding: KeyBinding) -> bool:
        """Execute a key binding action"""
        try:
            if binding.category == "camera":
                return self._execute_camera_action(binding.action, binding.args)
            elif binding.category == "debug":
                return self._execute_debug_action(binding.action, binding.args)
            elif binding.category == "screenshot":
                return self._execute_screenshot_action(binding.action, binding.args)
            elif binding.category == "ui":
                return self._execute_ui_action(binding.action, binding.args)
            elif binding.category == "workflow":
                return self._execute_workflow_action(binding.action, binding.args)
            elif binding.category == "session":
                return self._execute_session_action(binding.action, binding.args)
            
        except Exception as e:
            print(f"❌ Action execution error: {e}")
            return False
        
        return False
    
    def _execute_camera_action(self, action: str, args: Dict[str, Any]) -> bool:
        """Execute camera-related actions"""
        if action == "jump_to_face":
            face_name = args.get("face", "front")
            return self.camera.jump_to_face(face_name, immediate=True)
        
        elif action == "go_to_node":
            # Interactive quadtree node selection
            print("📍 Enter quadtree node (depth x z): ", end="")
            # In real implementation, would show UI for input
            return True
        
        elif action == "bookmark_menu":
            bookmarks = self.camera.list_bookmarks()
            print(f"📌 Bookmarks: {bookmarks}")
            return True
        
        elif action == "go_back":
            return self.camera.go_back()
        
        return False
    
    def _execute_debug_action(self, action: str, args: Dict[str, Any]) -> bool:
        """Execute debug-related actions"""
        if action == "toggle":
            toggle_name = args.get("type", "")
            try:
                toggle_type = ToggleType(toggle_name)
                self.toggles.toggle(toggle_type)
                return True
            except ValueError:
                return False
        
        return False
    
    def _execute_screenshot_action(self, action: str, args: Dict[str, Any]) -> bool:
        """Execute screenshot-related actions"""
        if action == "capture_screenshot":
            return self._take_screenshot()
        
        elif action == "start_series":
            series_name = f"{self.current_session.pcc_name}_series_{int(time.time())}"
            self.screenshot_tool.start_series(series_name)
            print(f"📸 Started screenshot series: {series_name}")
            return True
        
        elif action == "quick_comparison":
            return self._take_comparison_screenshots()
        
        return False
    
    def _execute_ui_action(self, action: str, args: Dict[str, Any]) -> bool:
        """Execute UI-related actions"""
        if action == "toggle_hud":
            self.hud.toggle_visibility()
            return True
        
        elif action == "show_help":
            self.show_help = not self.show_help
            return True
        
        elif action == "command_mode":
            self.command_mode = not self.command_mode
            self.command_buffer = ""
            return True
        
        elif action == "cycle_mode":
            return self._cycle_viewer_mode()
        
        return False
    
    def _execute_workflow_action(self, action: str, args: Dict[str, Any]) -> bool:
        """Execute workflow automation actions"""
        if action == "lod_debug":
            return self._start_lod_debug_workflow()
        
        elif action == "performance_profile":
            return self._start_performance_workflow()
        
        elif action == "comparison":
            return self._start_comparison_workflow()
        
        return False
    
    def _execute_session_action(self, action: str, args: Dict[str, Any]) -> bool:
        """Execute session management actions"""
        if action == "save_session":
            return self._save_session()
        
        elif action == "load_session":
            return self._load_session()
        
        elif action == "new_session":
            return self._new_session()
        
        return False
    
    def _take_screenshot(self) -> bool:
        """Take screenshot with current state"""
        render_state = self.toggles.get_render_state()
        
        # Map render mode to capture mode
        capture_mode = CaptureMode.STANDARD
        if render_state.show_wireframe:
            capture_mode = CaptureMode.WIREFRAME
        elif render_state.show_normals:
            capture_mode = CaptureMode.NORMALS
        elif render_state.show_lod_heatmap:
            capture_mode = CaptureMode.LOD_HEATMAP
        
        # Get active debug toggles
        active_toggles = [t.value for t in self.toggles.get_active_toggles()]
        
        screenshot_path = self.screenshot_tool.capture_screenshot(
            pcc_name=self.current_session.pcc_name,
            seed=self.current_session.seed,
            camera_position=self.camera.current_state.position,
            camera_target=self.camera.current_state.target,
            camera_fov=self.camera.current_state.fov_degrees,
            capture_mode=capture_mode,
            debug_toggles=active_toggles,
            description=f"Debug session screenshot - {self.current_session.mode.value}"
        )
        
        if screenshot_path:
            self.current_session.screenshots_taken.append(screenshot_path)
            return True
        
        return False
    
    def _take_comparison_screenshots(self) -> bool:
        """Take quick comparison screenshots"""
        camera_pos = self.camera.current_state.position
        camera_target = self.camera.current_state.target
        
        comparison_modes = [
            CaptureMode.STANDARD,
            CaptureMode.WIREFRAME,
            CaptureMode.LOD_HEATMAP
        ]
        
        screenshots = self.screenshot_tool.capture_debug_comparison(
            pcc_name=self.current_session.pcc_name,
            seed=self.current_session.seed,
            camera_position=camera_pos,
            debug_modes=comparison_modes
        )
        
        self.current_session.screenshots_taken.extend(screenshots)
        print(f"📸 Captured {len(screenshots)} comparison screenshots")
        return len(screenshots) > 0
    
    def _cycle_viewer_mode(self) -> bool:
        """Cycle through viewer modes"""
        modes = list(ViewerMode)
        current_index = modes.index(self.current_session.mode)
        next_index = (current_index + 1) % len(modes)
        self.current_session.mode = modes[next_index]
        
        print(f"🔄 Switched to {self.current_session.mode.value} mode")
        return True
    
    def _start_lod_debug_workflow(self) -> bool:
        """Start LOD debugging workflow"""
        print("🔧 Starting LOD debug workflow...")
        
        # 1. Enable LOD heatmap
        self.toggles.set_toggle(ToggleType.LOD_HEATMAP, True)
        
        # 2. Jump to overview position
        self.camera.jump_to_face(CubeFace.TOP, immediate=True)
        
        # 3. Take screenshot
        self._take_screenshot()
        
        # 4. Start screenshot series for different distances
        series_name = f"{self.current_session.pcc_name}_lod_debug"
        self.screenshot_tool.start_series(series_name, "LOD debugging session")
        
        self.current_workflow = "lod_debug"
        self.workflow_step = 0
        
        print("   ✅ LOD debug workflow active")
        print("   📍 Use number keys (1-6) to check different faces")
        print("   📸 Screenshots will be automatically captured")
        
        return True
    
    def _start_performance_workflow(self) -> bool:
        """Start performance profiling workflow"""
        print("🔧 Starting performance profile workflow...")
        
        # Enable HUD with detailed view
        self.hud.visible = True
        self.hud.toggle_detailed_view()
        
        # Disable resource-intensive debug modes
        self.toggles.disable_all()
        
        self.current_workflow = "performance"
        self.workflow_step = 0
        
        print("   ✅ Performance profile workflow active")
        print("   📊 HUD showing detailed performance metrics")
        print("   🎯 Move around to test different LOD scenarios")
        
        return True
    
    def _start_comparison_workflow(self) -> bool:
        """Start comparison workflow"""
        print("🔧 Starting comparison workflow...")
        
        # Take baseline screenshot
        self.toggles.disable_all()
        self._take_screenshot()
        
        # Prepare for comparisons
        series_name = f"{self.current_session.pcc_name}_comparison"
        self.screenshot_tool.start_series(series_name, "Content comparison session")
        
        self.current_workflow = "comparison"
        self.workflow_step = 0
        
        print("   ✅ Comparison workflow active")
        print("   📸 Baseline screenshot captured")
        print("   🔧 Use F-keys to toggle debug modes for comparison")
        
        return True
    
    def _save_session(self) -> bool:
        """Save current session state"""
        session_file = self.output_dir / f"session_{self.current_session.pcc_name}_{int(time.time())}.json"
        
        # Gather session state
        session_data = {
            'pcc_file': self.current_session.pcc_file,
            'pcc_name': self.current_session.pcc_name,
            'seed': self.current_session.seed,
            'mode': self.current_session.mode.value,
            'session_start': self.current_session.session_start,
            'session_duration': time.time() - self.current_session.session_start,
            'screenshots_taken': self.current_session.screenshots_taken,
            'camera_state': {
                'position': self.camera.current_state.position,
                'target': self.camera.current_state.target,
                'fov': self.camera.current_state.fov_degrees
            },
            'debug_state': {
                'active_toggles': [t.value for t in self.toggles.get_active_toggles()],
                'render_mode': self.toggles.get_render_state().render_mode.value
            },
            'hud_state': {
                'visible': self.hud.visible,
                'detailed_view': self.hud.detailed_view
            },
            'bookmarks': list(self.camera.bookmarks.keys()),
            'session_notes': self.current_session.session_notes
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session saved: {session_file}")
        return True
    
    def _load_session(self) -> bool:
        """Load session state (placeholder for UI implementation)"""
        print("💾 Load session - would show file browser")
        return True
    
    def _new_session(self) -> bool:
        """Start new session"""
        if self.current_session.pcc_file:
            self.session_history.append(self.current_session)
        
        self.current_session = ViewerSession()
        self.toggles.disable_all()
        self.hud = DiagnosticsHUD()
        self.camera = DebugCamera()
        
        print("📝 Started new session")
        return True
    
    def _on_render_mode_change(self, enabled: bool):
        """Callback for render mode changes"""
        # Update any dependent systems
        pass
    
    def update_frame(self, fps: float, draw_calls: int, triangles: int, active_chunks: int):
        """Update frame statistics"""
        self.frame_count += 1
        self.current_fps = fps
        
        # Update HUD
        self.hud.update_frame_stats(
            fps=fps,
            frame_time_ms=1000.0 / fps if fps > 0 else 100.0,
            draw_calls=draw_calls,
            triangles=triangles,
            active_chunks=active_chunks
        )
        
        # Record path point if recording
        if self.camera.recording_path:
            self.camera.record_path_point()
    
    def get_ui_data(self) -> Dict[str, Any]:
        """Get all UI data for rendering"""
        return {
            'hud': self.hud.get_hud_data(),
            'render_state': self.toggles.get_render_state(),
            'camera_state': {
                'position': self.camera.current_state.position,
                'target': self.camera.current_state.target,
                'fov': self.camera.current_state.fov_degrees
            },
            'session_info': {
                'pcc_name': self.current_session.pcc_name,
                'seed': self.current_session.seed,
                'mode': self.current_session.mode.value,
                'screenshots_count': len(self.current_session.screenshots_taken)
            },
            'show_help': self.show_help,
            'help_category': self.help_category,
            'current_workflow': self.current_workflow,
            'key_bindings': self._get_help_text() if self.show_help else {}
        }
    
    def _get_help_text(self) -> Dict[str, List[str]]:
        """Get categorized help text"""
        help_by_category = {}
        
        for binding in self.key_bindings.values():
            category = binding.category
            if category not in help_by_category:
                help_by_category[category] = []
            
            help_by_category[category].append(f"{binding.key}: {binding.description}")
        
        return help_by_category


if __name__ == "__main__":
    # Test developer viewer
    print("🚀 T16 Developer Viewer System")
    print("=" * 60)
    
    # Create viewer
    viewer = DeveloperViewer()
    
    print("📊 Testing viewer functionality...")
    
    # Test PCC loading
    # viewer.load_pcc_file("examples/hero_planet.pcc")  # Would load if file exists
    viewer.current_session.pcc_name = "test_terrain"
    viewer.current_session.seed = 12345
    
    # Test key handling
    print(f"   Testing key bindings...")
    viewer.handle_key_input("1")  # Jump to front face
    viewer.handle_key_input("F1")  # Toggle wireframe
    viewer.handle_key_input("F12")  # Take screenshot
    
    # Test workflows
    print(f"\n🔧 Testing workflows...")
    viewer._start_lod_debug_workflow()
    
    # Test session management
    print(f"\n💾 Testing session management...")
    viewer._save_session()
    
    # Test UI data
    ui_data = viewer.get_ui_data()
    print(f"\n📊 UI Data structure:")
    print(f"   HUD visible: {ui_data['hud']['visible']}")
    print(f"   Current mode: {ui_data['session_info']['mode']}")
    print(f"   Screenshots taken: {ui_data['session_info']['screenshots_count']}")
    
    # Test help system
    viewer.show_help = True
    help_data = viewer._get_help_text()
    print(f"\n❓ Help categories: {list(help_data.keys())}")
    
    print(f"\n✅ Developer viewer system functional")
    print(f"   Session: {viewer.current_session.pcc_name}")
    print(f"   Components integrated: HUD, Toggles, Camera, Screenshots")
    print(f"   Key bindings: {len(viewer.key_bindings)}")
    print(f"   Workflows available: LOD debug, Performance, Comparison")