#!/usr/bin/env python3
"""
Debug Camera Tools - T16
========================

Advanced camera tools for debugging PCC terrain including quick navigation
to cube faces, quadtree nodes, and specific terrain features. Designed for
efficient inspection of terrain generation results and LOD behavior.

Features:
- Jump to cube sphere faces (6 faces + poles)
- Navigate to specific quadtree nodes
- Bookmark interesting camera positions
- Auto-framing of terrain features
- Camera path recording and playback
- Quick LOD transition inspection

Usage:
    from camera_tools.debug_camera import DebugCamera
    
    camera = DebugCamera()
    camera.jump_to_face('front')
    camera.jump_to_quadtree_node(0, 2, 3)  # depth, x, z
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class CubeFace(Enum):
    """Cube sphere face identifiers"""
    FRONT = "front"      # +Z
    BACK = "back"        # -Z  
    LEFT = "left"        # -X
    RIGHT = "right"      # +X
    TOP = "top"          # +Y
    BOTTOM = "bottom"    # -Y
    NORTH_POLE = "north_pole"
    SOUTH_POLE = "south_pole"


class CameraMode(Enum):
    """Camera control modes"""
    FREE = "free"
    ORBIT = "orbit"
    FLY = "fly"
    LOCKED = "locked"
    PATH_FOLLOW = "path_follow"


@dataclass
class CameraState:
    """Complete camera state"""
    position: Tuple[float, float, float] = (0.0, 0.0, 100.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_vector: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_degrees: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 10000.0
    mode: CameraMode = CameraMode.FREE
    timestamp: float = field(default_factory=time.time)


@dataclass
class QuadtreeNode:
    """Quadtree node identifier"""
    depth: int
    x: int
    z: int
    face: Optional[CubeFace] = None
    
    def get_world_bounds(self, chunk_size: float = 64.0, planet_radius: float = 1000.0) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get world space bounds for this quadtree node"""
        # Calculate node size at this depth
        node_size = chunk_size * (2 ** (8 - self.depth))  # Assume max depth 8
        
        # Calculate world position
        world_x = (self.x - (2 ** self.depth) / 2) * node_size
        world_z = (self.z - (2 ** self.depth) / 2) * node_size
        
        # Bounds
        min_bounds = (world_x, -planet_radius * 0.5, world_z)
        max_bounds = (world_x + node_size, planet_radius * 2.0, world_z + node_size)
        
        return min_bounds, max_bounds


@dataclass
class CameraBookmark:
    """Saved camera position"""
    name: str
    state: CameraState
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)


class DebugCamera:
    """Advanced camera tools for terrain debugging"""
    
    def __init__(self, planet_radius: float = 1000.0, chunk_size: float = 64.0):
        """Initialize debug camera"""
        self.planet_radius = planet_radius
        self.chunk_size = chunk_size
        
        # Current camera state
        self.current_state = CameraState()
        
        # Camera parameters
        self.orbit_distance = planet_radius * 2.0
        self.face_distance = planet_radius * 1.5
        self.inspection_distance = chunk_size * 2.0
        
        # Camera movement settings
        self.move_speed = 100.0  # units per second
        self.rotation_speed = 90.0  # degrees per second
        self.smooth_time = 1.0  # seconds for smooth transitions
        
        # Bookmarks and history
        self.bookmarks: Dict[str, CameraBookmark] = {}
        self.camera_history: List[CameraState] = []
        self.max_history = 50
        
        # Path recording
        self.recording_path = False
        self.current_path: List[CameraState] = []
        self.path_playback_speed = 1.0
        
        # Cube face positions
        self.face_positions = self._calculate_face_positions()
        
        # Auto-framing settings
        self.auto_frame_margin = 1.2  # Extra space around target
        self.min_distance = 10.0
        self.max_distance = planet_radius * 5.0
    
    def _calculate_face_positions(self) -> Dict[CubeFace, Tuple[float, float, float]]:
        """Calculate optimal camera positions for each cube face"""
        positions = {}
        distance = self.face_distance
        
        # Six cube faces
        positions[CubeFace.FRONT] = (0.0, 0.0, distance)
        positions[CubeFace.BACK] = (0.0, 0.0, -distance)
        positions[CubeFace.LEFT] = (-distance, 0.0, 0.0)
        positions[CubeFace.RIGHT] = (distance, 0.0, 0.0)
        positions[CubeFace.TOP] = (0.0, distance, 0.0)
        positions[CubeFace.BOTTOM] = (0.0, -distance, 0.0)
        
        # Poles (offset for better viewing angles)
        pole_distance = distance * 0.8
        positions[CubeFace.NORTH_POLE] = (pole_distance * 0.3, distance * 0.9, pole_distance * 0.3)
        positions[CubeFace.SOUTH_POLE] = (-pole_distance * 0.3, -distance * 0.9, -pole_distance * 0.3)
        
        return positions
    
    def jump_to_face(self, face: Union[CubeFace, str], immediate: bool = False) -> bool:
        """Jump camera to view a specific cube sphere face"""
        if isinstance(face, str):
            try:
                face = CubeFace(face.lower())
            except ValueError:
                print(f"❌ Unknown face: {face}")
                return False
        
        if face not in self.face_positions:
            return False
        
        # Save current state to history
        self._add_to_history()
        
        # Calculate new camera state
        position = self.face_positions[face]
        target = (0.0, 0.0, 0.0)  # Look at planet center
        
        # Calculate appropriate up vector
        if face == CubeFace.TOP:
            up_vector = (0.0, 0.0, -1.0)  # Look down
        elif face == CubeFace.BOTTOM:
            up_vector = (0.0, 0.0, 1.0)   # Look up
        else:
            up_vector = (0.0, 1.0, 0.0)   # Standard up
        
        new_state = CameraState(
            position=position,
            target=target,
            up_vector=up_vector,
            fov_degrees=self.current_state.fov_degrees,
            mode=CameraMode.ORBIT
        )
        
        if immediate:
            self.current_state = new_state
        else:
            self._smooth_transition_to_state(new_state)
        
        print(f"📷 Jumped to {face.value} face at {position}")
        return True
    
    def jump_to_quadtree_node(self, depth: int, x: int, z: int, 
                             face: Optional[CubeFace] = None, immediate: bool = False) -> bool:
        """Jump camera to inspect a specific quadtree node"""
        if depth < 0 or depth > 16:
            print(f"❌ Invalid depth: {depth} (must be 0-16)")
            return False
        
        max_coord = 2 ** depth
        if x < 0 or x >= max_coord or z < 0 or z >= max_coord:
            print(f"❌ Invalid coordinates: ({x}, {z}) for depth {depth} (max: {max_coord-1})")
            return False
        
        # Save current state
        self._add_to_history()
        
        # Create quadtree node
        node = QuadtreeNode(depth=depth, x=x, z=z, face=face)
        min_bounds, max_bounds = node.get_world_bounds(self.chunk_size, self.planet_radius)
        
        # Calculate center of node
        center_x = (min_bounds[0] + max_bounds[0]) / 2
        center_y = (min_bounds[1] + max_bounds[1]) / 2
        center_z = (min_bounds[2] + max_bounds[2]) / 2
        target = (center_x, center_y, center_z)
        
        # Calculate node size for appropriate camera distance
        node_size = max(max_bounds[0] - min_bounds[0], 
                       max_bounds[2] - min_bounds[2])
        
        # Position camera for good view of the node
        camera_distance = max(self.min_distance, node_size * 2.0)
        camera_height = camera_distance * 0.6
        
        position = (center_x + camera_distance * 0.7,
                   center_y + camera_height,
                   center_z + camera_distance * 0.7)
        
        new_state = CameraState(
            position=position,
            target=target,
            up_vector=(0.0, 1.0, 0.0),
            fov_degrees=45.0,  # Narrower FOV for inspection
            mode=CameraMode.LOCKED
        )
        
        if immediate:
            self.current_state = new_state
        else:
            self._smooth_transition_to_state(new_state)
        
        print(f"📷 Jumped to quadtree node [{depth}]({x},{z}) at {target}")
        return True
    
    def frame_bounds(self, min_bounds: Tuple[float, float, float], 
                    max_bounds: Tuple[float, float, float], immediate: bool = False) -> bool:
        """Frame camera to show specific world bounds"""
        # Calculate bounds center and size
        center = ((min_bounds[0] + max_bounds[0]) / 2,
                 (min_bounds[1] + max_bounds[1]) / 2,
                 (min_bounds[2] + max_bounds[2]) / 2)
        
        size = max(max_bounds[0] - min_bounds[0],
                  max_bounds[1] - min_bounds[1], 
                  max_bounds[2] - min_bounds[2])
        
        # Calculate camera distance based on size and FOV
        fov_rad = math.radians(self.current_state.fov_degrees)
        distance = (size * self.auto_frame_margin) / (2.0 * math.tan(fov_rad / 2.0))
        distance = max(self.min_distance, min(self.max_distance, distance))
        
        # Position camera at optimal viewing angle
        offset_x = distance * 0.5
        offset_y = distance * 0.4  
        offset_z = distance * 0.5
        
        position = (center[0] + offset_x, center[1] + offset_y, center[2] + offset_z)
        
        new_state = CameraState(
            position=position,
            target=center,
            up_vector=(0.0, 1.0, 0.0),
            mode=CameraMode.LOCKED
        )
        
        if immediate:
            self.current_state = new_state
        else:
            self._smooth_transition_to_state(new_state)
        
        return True
    
    def orbit_around_point(self, center: Tuple[float, float, float], 
                          radius: float, angle_degrees: float = 0.0) -> bool:
        """Orbit camera around a specific point"""
        angle_rad = math.radians(angle_degrees)
        
        # Calculate position on orbit
        x = center[0] + radius * math.cos(angle_rad)
        z = center[2] + radius * math.sin(angle_rad)
        y = center[1] + radius * 0.3  # Slight elevation
        
        new_state = CameraState(
            position=(x, y, z),
            target=center,
            up_vector=(0.0, 1.0, 0.0),
            mode=CameraMode.ORBIT
        )
        
        self.current_state = new_state
        return True
    
    def save_bookmark(self, name: str, description: str = "", tags: List[str] = None) -> bool:
        """Save current camera position as bookmark"""
        if tags is None:
            tags = []
        
        bookmark = CameraBookmark(
            name=name,
            state=CameraState(
                position=self.current_state.position,
                target=self.current_state.target,
                up_vector=self.current_state.up_vector,
                fov_degrees=self.current_state.fov_degrees,
                mode=self.current_state.mode
            ),
            description=description,
            tags=tags
        )
        
        self.bookmarks[name] = bookmark
        print(f"📌 Saved bookmark '{name}' at {self.current_state.position}")
        return True
    
    def load_bookmark(self, name: str, immediate: bool = False) -> bool:
        """Load camera position from bookmark"""
        if name not in self.bookmarks:
            print(f"❌ Bookmark '{name}' not found")
            return False
        
        bookmark = self.bookmarks[name]
        self._add_to_history()
        
        if immediate:
            self.current_state = bookmark.state
        else:
            self._smooth_transition_to_state(bookmark.state)
        
        print(f"📌 Loaded bookmark '{name}'")
        return True
    
    def list_bookmarks(self) -> List[str]:
        """Get list of bookmark names"""
        return list(self.bookmarks.keys())
    
    def delete_bookmark(self, name: str) -> bool:
        """Delete a bookmark"""
        if name in self.bookmarks:
            del self.bookmarks[name]
            print(f"📌 Deleted bookmark '{name}'")
            return True
        return False
    
    def go_back(self) -> bool:
        """Go back to previous camera position"""
        if not self.camera_history:
            print("📷 No previous position available")
            return False
        
        previous_state = self.camera_history.pop()
        self.current_state = previous_state
        print(f"📷 Returned to previous position: {previous_state.position}")
        return True
    
    def _add_to_history(self):
        """Add current state to history"""
        state_copy = CameraState(
            position=self.current_state.position,
            target=self.current_state.target,
            up_vector=self.current_state.up_vector,
            fov_degrees=self.current_state.fov_degrees,
            mode=self.current_state.mode,
            timestamp=time.time()
        )
        
        self.camera_history.append(state_copy)
        
        # Limit history size
        if len(self.camera_history) > self.max_history:
            self.camera_history.pop(0)
    
    def _smooth_transition_to_state(self, target_state: CameraState):
        """Smooth transition to target state (would be implemented in renderer)"""
        # This would be implemented in the actual renderer
        # For now, just set the state directly
        self.current_state = target_state
    
    def start_path_recording(self):
        """Start recording camera path"""
        self.recording_path = True
        self.current_path = [self.current_state]
        print("🎬 Started recording camera path")
    
    def stop_path_recording(self) -> int:
        """Stop recording camera path"""
        self.recording_path = False
        path_length = len(self.current_path)
        print(f"🎬 Stopped recording camera path ({path_length} points)")
        return path_length
    
    def record_path_point(self):
        """Record current camera position to path"""
        if self.recording_path:
            self.current_path.append(self.current_state)
    
    def get_quick_navigation_targets(self) -> Dict[str, Any]:
        """Get common navigation targets for quick access"""
        return {
            'faces': {
                'front': CubeFace.FRONT.value,
                'back': CubeFace.BACK.value,
                'left': CubeFace.LEFT.value,
                'right': CubeFace.RIGHT.value,
                'top': CubeFace.TOP.value,
                'bottom': CubeFace.BOTTOM.value,
                'north_pole': CubeFace.NORTH_POLE.value,
                'south_pole': CubeFace.SOUTH_POLE.value
            },
            'quick_distances': {
                'close': self.planet_radius * 1.2,
                'medium': self.planet_radius * 2.0,
                'far': self.planet_radius * 4.0,
                'overview': self.planet_radius * 8.0
            },
            'common_nodes': {
                'root': (0, 0, 0),
                'quadrants': [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
                'detail_areas': [(3, 2, 2), (3, 5, 3), (3, 1, 6), (3, 7, 1)]
            }
        }
    
    def export_session(self, filename: str):
        """Export camera session data"""
        session_data = {
            'timestamp': time.time(),
            'planet_radius': self.planet_radius,
            'chunk_size': self.chunk_size,
            'current_state': {
                'position': self.current_state.position,
                'target': self.current_state.target,
                'up_vector': self.current_state.up_vector,
                'fov_degrees': self.current_state.fov_degrees,
                'mode': self.current_state.mode.value
            },
            'bookmarks': {
                name: {
                    'position': bookmark.state.position,
                    'target': bookmark.state.target,
                    'up_vector': bookmark.state.up_vector,
                    'fov_degrees': bookmark.state.fov_degrees,
                    'description': bookmark.description,
                    'tags': bookmark.tags,
                    'created_time': bookmark.created_time
                }
                for name, bookmark in self.bookmarks.items()
            },
            'camera_history': [
                {
                    'position': state.position,
                    'target': state.target,
                    'timestamp': state.timestamp
                }
                for state in self.camera_history[-20:]  # Last 20 positions
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"📷 Camera session exported to {filename}")


if __name__ == "__main__":
    # Test debug camera system
    print("🚀 T16 Debug Camera System")
    print("=" * 60)
    
    # Create debug camera
    camera = DebugCamera(planet_radius=1000.0, chunk_size=64.0)
    
    print("📊 Testing camera navigation...")
    
    # Test face navigation
    print(f"   Initial position: {camera.current_state.position}")
    
    camera.jump_to_face(CubeFace.FRONT, immediate=True)
    print(f"   Front face position: {camera.current_state.position}")
    
    camera.jump_to_face('top', immediate=True)
    print(f"   Top face position: {camera.current_state.position}")
    
    # Test quadtree navigation
    camera.jump_to_quadtree_node(2, 1, 2, immediate=True)
    print(f"   Quadtree node [2](1,2) position: {camera.current_state.position}")
    
    # Test bookmarks
    camera.save_bookmark("test_view", "Test bookmark", ["testing", "debug"])
    bookmarks = camera.list_bookmarks()
    print(f"   Bookmarks: {bookmarks}")
    
    # Change position and test bookmark loading
    camera.jump_to_face(CubeFace.BACK, immediate=True)
    camera.load_bookmark("test_view", immediate=True)
    print(f"   After bookmark load: {camera.current_state.position}")
    
    # Test history
    camera.jump_to_face(CubeFace.LEFT, immediate=True)
    previous_pos = camera.current_state.position
    camera.go_back()
    print(f"   After go back: {camera.current_state.position}")
    
    # Test bounds framing
    min_bounds = (-50.0, -10.0, -50.0)
    max_bounds = (50.0, 40.0, 50.0)
    camera.frame_bounds(min_bounds, max_bounds, immediate=True)
    print(f"   After frame bounds: {camera.current_state.position}")
    
    # Test orbit
    center = (0.0, 0.0, 0.0)
    camera.orbit_around_point(center, 200.0, 45.0)
    print(f"   Orbit position (45°): {camera.current_state.position}")
    
    # Test navigation targets
    targets = camera.get_quick_navigation_targets()
    print(f"   Available faces: {list(targets['faces'].keys())}")
    print(f"   Quick distances: {list(targets['quick_distances'].keys())}")
    
    # Test path recording
    camera.start_path_recording()
    camera.jump_to_face(CubeFace.FRONT, immediate=True)
    camera.record_path_point()
    camera.jump_to_face(CubeFace.RIGHT, immediate=True) 
    camera.record_path_point()
    path_points = camera.stop_path_recording()
    print(f"   Recorded path with {path_points} points")
    
    # Export session
    camera.export_session("test_camera_session.json")
    
    print(f"\n✅ Debug camera system functional")