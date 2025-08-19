#!/usr/bin/env python3
"""
Debug Toggles System - T16
==========================

Comprehensive debug visualization toggles for PCC terrain viewer including
wireframe mode, normal visualization, chunk boundaries, LOD heatmaps, and
content filtering for debugging terrain generation and rendering.

Features:
- Wireframe rendering toggle
- Normal vector visualization
- Chunk ID overlay and boundaries
- LOD level heatmap visualization
- Cave-only and surface-only filtering
- Material and texture debug modes
- Real-time toggle state management

Usage:
    from debug_ui.debug_toggles import DebugToggles
    
    toggles = DebugToggles()
    toggles.set_wireframe(True)
    render_state = toggles.get_render_state()
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class ToggleType(Enum):
    """Types of debug toggles available"""
    WIREFRAME = "wireframe"
    NORMALS = "normals"
    CHUNK_IDS = "chunk_ids"
    CHUNK_BOUNDARIES = "chunk_boundaries"
    LOD_HEATMAP = "lod_heatmap"
    CAVE_ONLY = "cave_only"
    SURFACE_ONLY = "surface_only"
    TEXTURE_DEBUG = "texture_debug"
    MATERIAL_DEBUG = "material_debug"
    LIGHTING_DEBUG = "lighting_debug"
    OVERDRAW_DEBUG = "overdraw_debug"
    DISTANCE_DEBUG = "distance_debug"


class RenderMode(Enum):
    """Different rendering modes"""
    NORMAL = "normal"
    WIREFRAME = "wireframe"
    NORMALS_ONLY = "normals_only"
    DEBUG_COLORS = "debug_colors"
    HEATMAP = "heatmap"


@dataclass
class ToggleState:
    """State of a single debug toggle"""
    enabled: bool = False
    hotkey: Optional[str] = None
    description: str = ""
    category: str = "general"
    mutually_exclusive: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    last_toggled: float = 0.0


@dataclass
class RenderState:
    """Complete render state based on active toggles"""
    render_mode: RenderMode = RenderMode.NORMAL
    show_wireframe: bool = False
    show_normals: bool = False
    show_chunk_boundaries: bool = False
    show_chunk_ids: bool = False
    show_lod_heatmap: bool = False
    filter_caves_only: bool = False
    filter_surface_only: bool = False
    debug_textures: bool = False
    debug_materials: bool = False
    debug_lighting: bool = False
    debug_overdraw: bool = False
    debug_distance: bool = False
    wireframe_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    normal_length: float = 1.0
    chunk_boundary_color: Tuple[float, float, float] = (1.0, 0.0, 1.0)
    heatmap_range: Tuple[float, float] = (0.0, 10.0)


class DebugToggles:
    """Debug toggles manager for terrain viewer"""
    
    def __init__(self):
        """Initialize debug toggles system"""
        self.toggles: Dict[ToggleType, ToggleState] = {}
        self.callbacks: Dict[ToggleType, List[Callable]] = {}
        self.hotkey_map: Dict[str, ToggleType] = {}
        
        # Initialize all toggle states
        self._initialize_toggles()
        
        # Current render state
        self.current_render_state = RenderState()
        
        # Toggle history for debugging
        self.toggle_history: List[Tuple[float, ToggleType, bool]] = []
        
        # Categories for UI organization
        self.categories = {
            "rendering": "Rendering Modes",
            "geometry": "Geometry Debug", 
            "chunking": "Chunk Management",
            "lod": "Level of Detail",
            "filtering": "Content Filtering",
            "materials": "Materials & Textures",
            "performance": "Performance Debug"
        }
    
    def _initialize_toggles(self):
        """Initialize all debug toggle states"""
        
        # Rendering toggles
        self.toggles[ToggleType.WIREFRAME] = ToggleState(
            enabled=False,
            hotkey="F1",
            description="Show wireframe overlay",
            category="rendering",
            mutually_exclusive=[ToggleType.NORMALS.value]
        )
        
        self.toggles[ToggleType.NORMALS] = ToggleState(
            enabled=False,
            hotkey="F2", 
            description="Show vertex normal vectors",
            category="geometry",
            mutually_exclusive=[ToggleType.WIREFRAME.value]
        )
        
        # Chunk debugging
        self.toggles[ToggleType.CHUNK_IDS] = ToggleState(
            enabled=False,
            hotkey="F3",
            description="Show chunk ID labels",
            category="chunking"
        )
        
        self.toggles[ToggleType.CHUNK_BOUNDARIES] = ToggleState(
            enabled=False,
            hotkey="F4",
            description="Show chunk boundary lines",
            category="chunking"
        )
        
        # LOD debugging
        self.toggles[ToggleType.LOD_HEATMAP] = ToggleState(
            enabled=False,
            hotkey="F5",
            description="Show LOD level heatmap",
            category="lod",
            mutually_exclusive=[ToggleType.CAVE_ONLY.value, ToggleType.SURFACE_ONLY.value]
        )
        
        # Content filtering
        self.toggles[ToggleType.CAVE_ONLY] = ToggleState(
            enabled=False,
            hotkey="F6",
            description="Show caves only (hide surface)",
            category="filtering",
            mutually_exclusive=[ToggleType.SURFACE_ONLY.value, ToggleType.LOD_HEATMAP.value]
        )
        
        self.toggles[ToggleType.SURFACE_ONLY] = ToggleState(
            enabled=False,
            hotkey="F7",
            description="Show surface only (hide caves)",
            category="filtering",
            mutually_exclusive=[ToggleType.CAVE_ONLY.value, ToggleType.LOD_HEATMAP.value]
        )
        
        # Material and texture debugging
        self.toggles[ToggleType.TEXTURE_DEBUG] = ToggleState(
            enabled=False,
            hotkey="F8",
            description="Show texture coordinate debug",
            category="materials"
        )
        
        self.toggles[ToggleType.MATERIAL_DEBUG] = ToggleState(
            enabled=False,
            hotkey="F9",
            description="Show material property debug",
            category="materials"
        )
        
        # Advanced debugging
        self.toggles[ToggleType.LIGHTING_DEBUG] = ToggleState(
            enabled=False,
            hotkey="F10",
            description="Show lighting debug information",
            category="performance"
        )
        
        self.toggles[ToggleType.OVERDRAW_DEBUG] = ToggleState(
            enabled=False,
            hotkey="F11",
            description="Show overdraw visualization",
            category="performance"
        )
        
        self.toggles[ToggleType.DISTANCE_DEBUG] = ToggleState(
            enabled=False,
            hotkey="F12",
            description="Show distance-based debug colors",
            category="performance"
        )
        
        # Build hotkey map
        for toggle_type, state in self.toggles.items():
            if state.hotkey:
                self.hotkey_map[state.hotkey] = toggle_type
    
    def toggle(self, toggle_type: ToggleType) -> bool:
        """Toggle a debug mode on/off"""
        if toggle_type not in self.toggles:
            return False
        
        current_state = self.toggles[toggle_type].enabled
        new_state = not current_state
        
        # Check mutual exclusivity
        if new_state:
            state = self.toggles[toggle_type]
            for exclusive_toggle in state.mutually_exclusive:
                try:
                    exclusive_type = ToggleType(exclusive_toggle)
                    if self.toggles[exclusive_type].enabled:
                        self._set_toggle_state(exclusive_type, False)
                except ValueError:
                    continue
        
        # Set new state
        self._set_toggle_state(toggle_type, new_state)
        
        # Update render state
        self._update_render_state()
        
        # Record in history
        self.toggle_history.append((time.time(), toggle_type, new_state))
        
        # Call callbacks
        self._call_callbacks(toggle_type, new_state)
        
        return new_state
    
    def _set_toggle_state(self, toggle_type: ToggleType, enabled: bool):
        """Set toggle state without side effects"""
        self.toggles[toggle_type].enabled = enabled
        self.toggles[toggle_type].last_toggled = time.time()
    
    def set_toggle(self, toggle_type: ToggleType, enabled: bool):
        """Explicitly set a toggle state"""
        if toggle_type not in self.toggles:
            return
        
        if self.toggles[toggle_type].enabled == enabled:
            return  # No change needed
        
        # Use toggle to handle mutual exclusivity
        if enabled != self.toggles[toggle_type].enabled:
            self.toggle(toggle_type)
    
    def is_enabled(self, toggle_type: ToggleType) -> bool:
        """Check if a toggle is enabled"""
        return self.toggles.get(toggle_type, ToggleState()).enabled
    
    def handle_hotkey(self, key: str) -> Optional[ToggleType]:
        """Handle hotkey press and toggle corresponding mode"""
        if key in self.hotkey_map:
            toggle_type = self.hotkey_map[key]
            self.toggle(toggle_type)
            return toggle_type
        return None
    
    def _update_render_state(self):
        """Update render state based on active toggles"""
        state = RenderState()
        
        # Basic render mode determination
        if self.is_enabled(ToggleType.WIREFRAME):
            state.render_mode = RenderMode.WIREFRAME
            state.show_wireframe = True
        elif self.is_enabled(ToggleType.NORMALS):
            state.render_mode = RenderMode.NORMALS_ONLY
            state.show_normals = True
        elif self.is_enabled(ToggleType.LOD_HEATMAP):
            state.render_mode = RenderMode.HEATMAP
            state.show_lod_heatmap = True
        else:
            state.render_mode = RenderMode.NORMAL
        
        # Individual toggles
        state.show_chunk_boundaries = self.is_enabled(ToggleType.CHUNK_BOUNDARIES)
        state.show_chunk_ids = self.is_enabled(ToggleType.CHUNK_IDS)
        state.filter_caves_only = self.is_enabled(ToggleType.CAVE_ONLY)
        state.filter_surface_only = self.is_enabled(ToggleType.SURFACE_ONLY)
        state.debug_textures = self.is_enabled(ToggleType.TEXTURE_DEBUG)
        state.debug_materials = self.is_enabled(ToggleType.MATERIAL_DEBUG)
        state.debug_lighting = self.is_enabled(ToggleType.LIGHTING_DEBUG)
        state.debug_overdraw = self.is_enabled(ToggleType.OVERDRAW_DEBUG)
        state.debug_distance = self.is_enabled(ToggleType.DISTANCE_DEBUG)
        
        self.current_render_state = state
    
    def get_render_state(self) -> RenderState:
        """Get current render state"""
        return self.current_render_state
    
    def add_callback(self, toggle_type: ToggleType, callback: Callable[[bool], None]):
        """Add callback for toggle state changes"""
        if toggle_type not in self.callbacks:
            self.callbacks[toggle_type] = []
        self.callbacks[toggle_type].append(callback)
    
    def _call_callbacks(self, toggle_type: ToggleType, enabled: bool):
        """Call all callbacks for a toggle type"""
        if toggle_type in self.callbacks:
            for callback in self.callbacks[toggle_type]:
                try:
                    callback(enabled)
                except Exception as e:
                    print(f"Callback error for {toggle_type.value}: {e}")
    
    def get_toggle_groups(self) -> Dict[str, List[Tuple[ToggleType, ToggleState]]]:
        """Get toggles organized by category"""
        groups = {}
        
        for toggle_type, state in self.toggles.items():
            category = state.category
            if category not in groups:
                groups[category] = []
            groups[category].append((toggle_type, state))
        
        # Sort within each group
        for category in groups:
            groups[category].sort(key=lambda x: x[0].value)
        
        return groups
    
    def get_active_toggles(self) -> List[ToggleType]:
        """Get list of currently active toggles"""
        return [toggle_type for toggle_type, state in self.toggles.items() if state.enabled]
    
    def disable_all(self):
        """Disable all debug toggles"""
        for toggle_type in self.toggles:
            self._set_toggle_state(toggle_type, False)
        self._update_render_state()
    
    def get_toggle_summary(self) -> Dict[str, Any]:
        """Get summary of toggle states for UI display"""
        active_toggles = self.get_active_toggles()
        
        return {
            'active_count': len(active_toggles),
            'active_toggles': [t.value for t in active_toggles],
            'render_mode': self.current_render_state.render_mode.value,
            'categories': self.categories,
            'hotkey_map': {k: v.value for k, v in self.hotkey_map.items()},
            'last_toggled': max([s.last_toggled for s in self.toggles.values()]) if self.toggles else 0.0
        }
    
    def export_state(self, filename: str):
        """Export current toggle state to file"""
        export_data = {
            'timestamp': time.time(),
            'toggles': {
                toggle_type.value: {
                    'enabled': state.enabled,
                    'last_toggled': state.last_toggled
                }
                for toggle_type, state in self.toggles.items()
            },
            'render_state': {
                'render_mode': self.current_render_state.render_mode.value,
                'show_wireframe': self.current_render_state.show_wireframe,
                'show_normals': self.current_render_state.show_normals,
                'show_chunk_boundaries': self.current_render_state.show_chunk_boundaries,
                'show_chunk_ids': self.current_render_state.show_chunk_ids,
                'filter_caves_only': self.current_render_state.filter_caves_only,
                'filter_surface_only': self.current_render_state.filter_surface_only
            },
            'toggle_history': [
                {
                    'timestamp': timestamp,
                    'toggle': toggle_type.value,
                    'enabled': enabled
                }
                for timestamp, toggle_type, enabled in self.toggle_history[-50:]  # Last 50 toggles
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"🔧 Debug toggle state exported to {filename}")
    
    def load_state(self, filename: str) -> bool:
        """Load toggle state from file"""
        try:
            with open(filename, 'r') as f:
                import_data = json.load(f)
            
            # Restore toggle states
            for toggle_name, toggle_data in import_data.get('toggles', {}).items():
                try:
                    toggle_type = ToggleType(toggle_name)
                    self._set_toggle_state(toggle_type, toggle_data['enabled'])
                except ValueError:
                    continue
            
            # Update render state
            self._update_render_state()
            
            print(f"🔧 Debug toggle state loaded from {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load toggle state: {e}")
            return False


if __name__ == "__main__":
    # Test debug toggles system
    print("🚀 T16 Debug Toggles System")
    print("=" * 60)
    
    # Create toggles manager
    toggles = DebugToggles()
    
    print("📊 Testing toggle functionality...")
    
    # Test basic toggles
    print(f"   Initial wireframe state: {toggles.is_enabled(ToggleType.WIREFRAME)}")
    
    toggles.toggle(ToggleType.WIREFRAME)
    print(f"   After toggle wireframe: {toggles.is_enabled(ToggleType.WIREFRAME)}")
    
    # Test mutual exclusivity
    toggles.toggle(ToggleType.NORMALS)  # Should disable wireframe
    print(f"   After toggle normals - wireframe: {toggles.is_enabled(ToggleType.WIREFRAME)}")
    print(f"   After toggle normals - normals: {toggles.is_enabled(ToggleType.NORMALS)}")
    
    # Test render state
    render_state = toggles.get_render_state()
    print(f"   Render mode: {render_state.render_mode.value}")
    print(f"   Show normals: {render_state.show_normals}")
    
    # Test hotkey handling
    hotkey_result = toggles.handle_hotkey("F3")
    print(f"   F3 hotkey toggles: {hotkey_result.value if hotkey_result else 'None'}")
    
    # Test toggle groups
    groups = toggles.get_toggle_groups()
    print(f"   Toggle categories: {list(groups.keys())}")
    
    # Test active toggles
    active = toggles.get_active_toggles()
    print(f"   Active toggles: {[t.value for t in active]}")
    
    # Test toggle summary
    summary = toggles.get_toggle_summary()
    print(f"   Active count: {summary['active_count']}")
    print(f"   Current render mode: {summary['render_mode']}")
    
    # Test multiple toggles
    print(f"\n🔧 Testing content filtering...")
    toggles.set_toggle(ToggleType.CAVE_ONLY, True)
    state = toggles.get_render_state()
    print(f"   Cave only filter: {state.filter_caves_only}")
    print(f"   Surface only filter: {state.filter_surface_only}")
    
    toggles.set_toggle(ToggleType.SURFACE_ONLY, True)  # Should disable cave_only
    state = toggles.get_render_state()
    print(f"   After surface toggle - cave only: {state.filter_caves_only}")
    print(f"   After surface toggle - surface only: {state.filter_surface_only}")
    
    # Test LOD heatmap
    print(f"\n🎨 Testing LOD heatmap...")
    toggles.set_toggle(ToggleType.LOD_HEATMAP, True)
    state = toggles.get_render_state()
    print(f"   LOD heatmap: {state.show_lod_heatmap}")
    print(f"   Render mode: {state.render_mode.value}")
    
    # Test chunk debugging
    print(f"\n📦 Testing chunk debugging...")
    toggles.set_toggle(ToggleType.CHUNK_IDS, True)
    toggles.set_toggle(ToggleType.CHUNK_BOUNDARIES, True)
    state = toggles.get_render_state()
    print(f"   Chunk IDs: {state.show_chunk_ids}")
    print(f"   Chunk boundaries: {state.show_chunk_boundaries}")
    
    # Export state
    toggles.export_state("test_debug_toggles.json")
    
    # Test disable all
    toggles.disable_all()
    final_active = toggles.get_active_toggles()
    print(f"\n🔄 After disable all: {len(final_active)} active toggles")
    
    print(f"\n✅ Debug toggles system functional")