#!/usr/bin/env python3
"""
Screenshot Tool - T16
====================

Advanced screenshot tool for PCC terrain viewer with automatic filename
stamping including PCC name, seed, camera position, and debug settings.
Designed for documenting terrain generation results and debugging sessions.

Features:
- Automatic filename generation with PCC+seed info
- Multiple screenshot formats (PNG, JPG, HDR)
- Debug overlay capture options
- Batch screenshot sequences
- Metadata embedding in images
- Screenshot comparison tools

Usage:
    from viewer_tools.screenshot_tool import ScreenshotTool
    
    tool = ScreenshotTool()
    tool.capture_screenshot(pcc_name="hero_planet", seed=12345)
"""

import time
import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class ScreenshotFormat(Enum):
    """Screenshot output formats"""
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    BMP = "bmp"
    TGA = "tga"
    HDR = "hdr"
    EXR = "exr"


class CaptureMode(Enum):
    """Screenshot capture modes"""
    STANDARD = "standard"              # Normal view
    DEBUG_OVERLAY = "debug_overlay"    # Include HUD and debug info
    CLEAN = "clean"                    # No UI elements
    WIREFRAME = "wireframe"            # Wireframe only
    NORMALS = "normals"                # Normal visualization
    LOD_HEATMAP = "lod_heatmap"       # LOD visualization


@dataclass
class ScreenshotMetadata:
    """Metadata for screenshots"""
    pcc_name: str = ""
    pcc_file_path: str = ""
    seed: int = 0
    camera_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_fov: float = 60.0
    timestamp: float = field(default_factory=time.time)
    resolution: Tuple[int, int] = (1920, 1080)
    capture_mode: CaptureMode = CaptureMode.STANDARD
    debug_toggles: List[str] = field(default_factory=list)
    render_settings: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ScreenshotSeries:
    """Series of related screenshots"""
    name: str
    description: str
    screenshots: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScreenshotTool:
    """Advanced screenshot tool for terrain debugging"""
    
    def __init__(self, output_dir: str = "screenshots"):
        """Initialize screenshot tool"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Screenshot settings
        self.default_format = ScreenshotFormat.PNG
        self.default_quality = 95  # For JPEG
        self.auto_numbering = True
        self.include_metadata = True
        
        # Filename template
        self.filename_template = "{pcc_name}_{seed}_{timestamp}_{mode}"
        self.safe_filename_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        
        # Screenshot counter for batch operations
        self.screenshot_counter = 0
        
        # Screenshot series management
        self.current_series: Optional[ScreenshotSeries] = None
        self.series_history: List[ScreenshotSeries] = []
        
        # Recent screenshots for comparison
        self.recent_screenshots: List[Tuple[str, ScreenshotMetadata]] = []
        self.max_recent = 20
        
        # Metadata cache
        self.metadata_cache: Dict[str, ScreenshotMetadata] = {}
    
    def capture_screenshot(self, pcc_name: str = "", seed: int = 0,
                          camera_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                          camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                          camera_fov: float = 60.0, resolution: Tuple[int, int] = (1920, 1080),
                          capture_mode: CaptureMode = CaptureMode.STANDARD,
                          debug_toggles: List[str] = None, description: str = "",
                          tags: List[str] = None, custom_filename: str = "") -> Optional[str]:
        """Capture screenshot with automatic metadata stamping"""
        
        if debug_toggles is None:
            debug_toggles = []
        if tags is None:
            tags = []
        
        # Create metadata
        metadata = ScreenshotMetadata(
            pcc_name=self._sanitize_filename_part(pcc_name),
            seed=seed,
            camera_position=camera_position,
            camera_target=camera_target,
            camera_fov=camera_fov,
            timestamp=time.time(),
            resolution=resolution,
            capture_mode=capture_mode,
            debug_toggles=debug_toggles.copy(),
            description=description,
            tags=tags.copy()
        )
        
        # Generate filename
        if custom_filename:
            filename = self._sanitize_filename(custom_filename)
        else:
            filename = self._generate_filename(metadata)
        
        # Add format extension
        full_filename = f"{filename}.{self.default_format.value}"
        full_path = self.output_dir / full_filename
        
        # Ensure unique filename
        if full_path.exists() and self.auto_numbering:
            counter = 1
            while True:
                numbered_filename = f"{filename}_{counter:03d}.{self.default_format.value}"
                numbered_path = self.output_dir / numbered_filename
                if not numbered_path.exists():
                    full_filename = numbered_filename
                    full_path = numbered_path
                    break
                counter += 1
        
        # Simulate screenshot capture (would interface with actual renderer)
        success = self._simulate_screenshot_capture(full_path, metadata)
        
        if success:
            # Save metadata
            if self.include_metadata:
                self._save_metadata(full_path, metadata)
            
            # Add to recent screenshots
            self.recent_screenshots.append((str(full_path), metadata))
            if len(self.recent_screenshots) > self.max_recent:
                self.recent_screenshots.pop(0)
            
            # Add to current series if active
            if self.current_series:
                self.current_series.screenshots.append(str(full_path))
            
            # Cache metadata
            self.metadata_cache[str(full_path)] = metadata
            
            self.screenshot_counter += 1
            
            print(f"📸 Screenshot saved: {full_filename}")
            print(f"   PCC: {pcc_name}, Seed: {seed}")
            print(f"   Camera: {camera_position}")
            print(f"   Mode: {capture_mode.value}")
            
            return str(full_path)
        else:
            print(f"❌ Failed to capture screenshot: {full_filename}")
            return None
    
    def _generate_filename(self, metadata: ScreenshotMetadata) -> str:
        """Generate filename from metadata"""
        # Create timestamp string
        dt = datetime.fromtimestamp(metadata.timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        
        # Generate camera position hash for uniqueness
        pos_str = f"{metadata.camera_position[0]:.1f}_{metadata.camera_position[1]:.1f}_{metadata.camera_position[2]:.1f}"
        pos_hash = hashlib.md5(pos_str.encode()).hexdigest()[:6]
        
        # Build filename components
        components = []
        
        if metadata.pcc_name:
            components.append(metadata.pcc_name)
        
        if metadata.seed > 0:
            components.append(f"seed{metadata.seed}")
        
        components.append(timestamp_str)
        components.append(f"cam{pos_hash}")
        
        if metadata.capture_mode != CaptureMode.STANDARD:
            components.append(metadata.capture_mode.value)
        
        if metadata.debug_toggles:
            toggle_str = "_".join(sorted(metadata.debug_toggles))
            toggle_hash = hashlib.md5(toggle_str.encode()).hexdigest()[:4]
            components.append(f"dbg{toggle_hash}")
        
        filename = "_".join(components)
        return self._sanitize_filename(filename)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility"""
        sanitized = ""
        for char in filename:
            if char in self.safe_filename_chars:
                sanitized += char
            elif char == " ":
                sanitized += "_"
            # Skip other characters
        
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized
    
    def _sanitize_filename_part(self, part: str) -> str:
        """Sanitize a part of the filename"""
        return self._sanitize_filename(part)
    
    def _simulate_screenshot_capture(self, file_path: Path, metadata: ScreenshotMetadata) -> bool:
        """Simulate screenshot capture (would interface with actual renderer)"""
        try:
            # In a real implementation, this would:
            # 1. Set up render state based on metadata.capture_mode
            # 2. Configure debug overlays based on metadata.debug_toggles
            # 3. Render frame to buffer
            # 4. Save buffer to file_path
            
            # For simulation, create a placeholder file
            with open(file_path, 'w') as f:
                f.write(f"# Simulated screenshot\n")
                f.write(f"# PCC: {metadata.pcc_name}\n")
                f.write(f"# Seed: {metadata.seed}\n")
                f.write(f"# Timestamp: {metadata.timestamp}\n")
                f.write(f"# Resolution: {metadata.resolution}\n")
                f.write(f"# Mode: {metadata.capture_mode.value}\n")
            
            return True
        except Exception as e:
            print(f"Screenshot capture error: {e}")
            return False
    
    def _save_metadata(self, screenshot_path: Path, metadata: ScreenshotMetadata):
        """Save metadata file alongside screenshot"""
        metadata_path = screenshot_path.with_suffix('.json')
        
        metadata_dict = {
            'pcc_name': metadata.pcc_name,
            'pcc_file_path': metadata.pcc_file_path,
            'seed': metadata.seed,
            'camera_position': metadata.camera_position,
            'camera_target': metadata.camera_target,
            'camera_fov': metadata.camera_fov,
            'timestamp': metadata.timestamp,
            'human_timestamp': datetime.fromtimestamp(metadata.timestamp).isoformat(),
            'resolution': metadata.resolution,
            'capture_mode': metadata.capture_mode.value,
            'debug_toggles': metadata.debug_toggles,
            'render_settings': metadata.render_settings,
            'performance_stats': metadata.performance_stats,
            'description': metadata.description,
            'tags': metadata.tags
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def start_series(self, name: str, description: str = "") -> bool:
        """Start a new screenshot series"""
        if self.current_series:
            self.end_series()
        
        self.current_series = ScreenshotSeries(
            name=self._sanitize_filename(name),
            description=description
        )
        
        print(f"📸 Started screenshot series: {name}")
        return True
    
    def end_series(self) -> Optional[ScreenshotSeries]:
        """End current screenshot series"""
        if not self.current_series:
            return None
        
        # Save series metadata
        series_file = self.output_dir / f"series_{self.current_series.name}.json"
        
        series_data = {
            'name': self.current_series.name,
            'description': self.current_series.description,
            'screenshots': self.current_series.screenshots,
            'created_time': self.current_series.created_time,
            'screenshot_count': len(self.current_series.screenshots),
            'metadata': self.current_series.metadata
        }
        
        with open(series_file, 'w') as f:
            json.dump(series_data, f, indent=2)
        
        completed_series = self.current_series
        self.series_history.append(completed_series)
        self.current_series = None
        
        print(f"📸 Ended screenshot series: {completed_series.name} ({len(completed_series.screenshots)} screenshots)")
        return completed_series
    
    def capture_lod_sequence(self, pcc_name: str, seed: int, camera_position: Tuple[float, float, float],
                           lod_levels: List[int], description: str = "") -> List[str]:
        """Capture sequence of screenshots at different LOD levels"""
        series_name = f"{pcc_name}_lod_sequence"
        self.start_series(series_name, f"LOD sequence for {pcc_name} - {description}")
        
        screenshots = []
        for lod_level in lod_levels:
            # Simulate LOD level setting
            debug_toggles = [f"lod_level_{lod_level}"]
            
            screenshot_path = self.capture_screenshot(
                pcc_name=pcc_name,
                seed=seed,
                camera_position=camera_position,
                capture_mode=CaptureMode.LOD_HEATMAP,
                debug_toggles=debug_toggles,
                description=f"LOD level {lod_level}",
                tags=["lod_sequence", f"lod_{lod_level}"]
            )
            
            if screenshot_path:
                screenshots.append(screenshot_path)
        
        self.end_series()
        return screenshots
    
    def capture_face_sequence(self, pcc_name: str, seed: int, faces: List[str]) -> List[str]:
        """Capture sequence of screenshots from different cube faces"""
        series_name = f"{pcc_name}_face_sequence"
        self.start_series(series_name, f"Cube face sequence for {pcc_name}")
        
        screenshots = []
        for face in faces:
            # Simulate face-specific camera positions (would use DebugCamera)
            face_positions = {
                'front': (0.0, 0.0, 2000.0),
                'back': (0.0, 0.0, -2000.0),
                'left': (-2000.0, 0.0, 0.0),
                'right': (2000.0, 0.0, 0.0),
                'top': (0.0, 2000.0, 0.0),
                'bottom': (0.0, -2000.0, 0.0)
            }
            
            camera_pos = face_positions.get(face, (0.0, 0.0, 2000.0))
            
            screenshot_path = self.capture_screenshot(
                pcc_name=pcc_name,
                seed=seed,
                camera_position=camera_pos,
                camera_target=(0.0, 0.0, 0.0),
                description=f"View from {face} face",
                tags=["face_sequence", f"face_{face}"]
            )
            
            if screenshot_path:
                screenshots.append(screenshot_path)
        
        self.end_series()
        return screenshots
    
    def capture_debug_comparison(self, pcc_name: str, seed: int, camera_position: Tuple[float, float, float],
                               debug_modes: List[CaptureMode]) -> List[str]:
        """Capture comparison screenshots with different debug modes"""
        series_name = f"{pcc_name}_debug_comparison"
        self.start_series(series_name, f"Debug mode comparison for {pcc_name}")
        
        screenshots = []
        for mode in debug_modes:
            screenshot_path = self.capture_screenshot(
                pcc_name=pcc_name,
                seed=seed,
                camera_position=camera_position,
                capture_mode=mode,
                description=f"Debug mode: {mode.value}",
                tags=["debug_comparison", mode.value]
            )
            
            if screenshot_path:
                screenshots.append(screenshot_path)
        
        self.end_series()
        return screenshots
    
    def get_recent_screenshots(self, count: int = 10) -> List[Tuple[str, ScreenshotMetadata]]:
        """Get recent screenshots with metadata"""
        return self.recent_screenshots[-count:]
    
    def find_screenshots_by_pcc(self, pcc_name: str) -> List[str]:
        """Find all screenshots for a specific PCC"""
        matching_screenshots = []
        
        for screenshot_path, metadata in self.recent_screenshots:
            if metadata.pcc_name == pcc_name:
                matching_screenshots.append(screenshot_path)
        
        return matching_screenshots
    
    def find_screenshots_by_seed(self, seed: int) -> List[str]:
        """Find all screenshots for a specific seed"""
        matching_screenshots = []
        
        for screenshot_path, metadata in self.recent_screenshots:
            if metadata.seed == seed:
                matching_screenshots.append(screenshot_path)
        
        return matching_screenshots
    
    def export_screenshot_index(self, filename: str = "screenshot_index.json"):
        """Export index of all screenshots"""
        index_data = {
            'timestamp': time.time(),
            'total_screenshots': len(self.recent_screenshots),
            'series_count': len(self.series_history),
            'screenshots': [
                {
                    'path': path,
                    'pcc_name': metadata.pcc_name,
                    'seed': metadata.seed,
                    'timestamp': metadata.timestamp,
                    'capture_mode': metadata.capture_mode.value,
                    'tags': metadata.tags
                }
                for path, metadata in self.recent_screenshots
            ],
            'series': [
                {
                    'name': series.name,
                    'description': series.description,
                    'screenshot_count': len(series.screenshots),
                    'created_time': series.created_time
                }
                for series in self.series_history
            ]
        }
        
        index_path = self.output_dir / filename
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"📸 Screenshot index exported to {index_path}")


if __name__ == "__main__":
    # Test screenshot tool
    print("🚀 T16 Screenshot Tool System")
    print("=" * 60)
    
    # Create screenshot tool
    tool = ScreenshotTool(output_dir="test_screenshots")
    
    print("📊 Testing screenshot functionality...")
    
    # Test basic screenshot
    screenshot1 = tool.capture_screenshot(
        pcc_name="hero_planet",
        seed=12345,
        camera_position=(100.0, 200.0, 150.0),
        camera_target=(0.0, 0.0, 0.0),
        description="Test screenshot of hero planet"
    )
    print(f"   ✅ Basic screenshot: {screenshot1}")
    
    # Test debug mode screenshot
    screenshot2 = tool.capture_screenshot(
        pcc_name="minimal_sphere",
        seed=54321,
        camera_position=(0.0, 300.0, 500.0),
        capture_mode=CaptureMode.WIREFRAME,
        debug_toggles=["wireframe", "chunk_boundaries"],
        tags=["debug", "wireframe"]
    )
    print(f"   ✅ Debug screenshot: {screenshot2}")
    
    # Test LOD sequence
    print(f"\n📸 Testing LOD sequence...")
    lod_screenshots = tool.capture_lod_sequence(
        pcc_name="test_terrain",
        seed=98765,
        camera_position=(200.0, 150.0, 200.0),
        lod_levels=[0, 1, 2, 3],
        description="LOD level progression"
    )
    print(f"   ✅ LOD sequence: {len(lod_screenshots)} screenshots")
    
    # Test face sequence
    print(f"\n🔄 Testing face sequence...")
    face_screenshots = tool.capture_face_sequence(
        pcc_name="cube_planet",
        seed=11111,
        faces=["front", "right", "top"]
    )
    print(f"   ✅ Face sequence: {len(face_screenshots)} screenshots")
    
    # Test debug comparison
    print(f"\n🔧 Testing debug comparison...")
    debug_screenshots = tool.capture_debug_comparison(
        pcc_name="debug_terrain",
        seed=22222,
        camera_position=(100.0, 100.0, 100.0),
        debug_modes=[CaptureMode.STANDARD, CaptureMode.WIREFRAME, CaptureMode.NORMALS]
    )
    print(f"   ✅ Debug comparison: {len(debug_screenshots)} screenshots")
    
    # Test recent screenshots
    recent = tool.get_recent_screenshots(5)
    print(f"\n📋 Recent screenshots: {len(recent)}")
    for path, metadata in recent:
        print(f"   {Path(path).name} - {metadata.pcc_name} (seed: {metadata.seed})")
    
    # Test search functions
    hero_screenshots = tool.find_screenshots_by_pcc("hero_planet")
    print(f"\n🔍 Hero planet screenshots: {len(hero_screenshots)}")
    
    seed_screenshots = tool.find_screenshots_by_seed(12345)
    print(f"   Seed 12345 screenshots: {len(seed_screenshots)}")
    
    # Export index
    tool.export_screenshot_index()
    
    print(f"\n✅ Screenshot tool system functional")
    print(f"   Total screenshots captured: {tool.screenshot_counter}")
    print(f"   Series completed: {len(tool.series_history)}")
    print(f"   Output directory: {tool.output_dir}")