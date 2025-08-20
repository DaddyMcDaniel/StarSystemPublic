#!/usr/bin/env python3
"""
Hero World Documentation & Screenshots - T17
=============================================

Comprehensive screenshot documentation system for the hero planet showcase.
Uses T16 viewer tools to capture high-quality screenshots demonstrating all
terrain features and debugging capabilities.

Features:
- Automated screenshot sequences for all major features
- Debug mode comparisons (wireframe, LOD heatmap, cave-only, etc.)
- Face-by-face planet coverage for complete documentation
- Performance profiling screenshots with HUD overlay
- Screenshot series with metadata and organization
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Import T16 viewer tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from agents.agent_d.viewer_tools.developer_viewer import DeveloperViewer, ViewerMode
from agents.agent_d.viewer_tools.screenshot_tool import ScreenshotTool, CaptureMode
from agents.agent_d.camera_tools.debug_camera import DebugCamera, CubeFace
from agents.agent_d.debug_ui.debug_toggles import DebugToggles, ToggleType
from agents.agent_d.hud.diagnostics_hud import DiagnosticsHUD


class HeroWorldDocumentation:
    """Comprehensive documentation system for hero world"""
    
    def __init__(self, hero_world_pcc: str, output_dir: str = "hero_world_screenshots"):
        self.hero_world_pcc = hero_world_pcc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load hero world configuration
        with open(hero_world_pcc, 'r') as f:
            self.pcc_config = json.load(f)
        
        self.seed = self.pcc_config['metadata']['seed']
        self.planet_name = self.pcc_config['metadata']['name']
        
        # Initialize T16 viewer tools
        self.viewer = DeveloperViewer(str(self.output_dir))
        self.screenshot_tool = ScreenshotTool(str(self.output_dir))
        self.camera = DebugCamera()
        self.toggles = DebugToggles()
        self.hud = DiagnosticsHUD()
        
        # Documentation sequences
        self.documentation_plan = self._create_documentation_plan()
        
    def _create_documentation_plan(self) -> List[Dict[str, Any]]:
        """Create comprehensive documentation plan"""
        
        return [
            {
                'name': 'overview_sequence',
                'description': 'Overview shots of entire planet from different distances',
                'shots': [
                    {'camera_distance': 5000, 'description': 'Far overview'},
                    {'camera_distance': 3000, 'description': 'Medium overview'}, 
                    {'camera_distance': 1500, 'description': 'Close overview'},
                    {'camera_distance': 800, 'description': 'Surface approach'}
                ]
            },
            {
                'name': 'face_coverage',
                'description': 'Complete coverage of all cube faces',
                'faces': ['front', 'back', 'left', 'right', 'top', 'bottom']
            },
            {
                'name': 'terrain_features',
                'description': 'Showcase of major terrain features',
                'features': [
                    {'name': 'ridged_mountains', 'location': 'face_top', 'description': 'Ridged mountain macro features'},
                    {'name': 'warped_dunes', 'location': 'face_front', 'description': 'Warped dune micro detail'},
                    {'name': 'equatorial_archipelago', 'location': 'equator_band', 'description': 'Equatorial island chains'},
                    {'name': 'polar_regions', 'location': 'face_bottom', 'description': 'Polar terrain variation'}
                ]
            },
            {
                'name': 'cave_system',
                'description': 'Underground cave system documentation',
                'cave_types': [
                    {'name': 'gyroidal_primary', 'description': 'Primary gyroidal tunnel networks'},
                    {'name': 'gyroidal_secondary', 'description': 'Secondary gyroidal chambers'},
                    {'name': 'sphere_large', 'description': 'Large spherical cavities'},
                    {'name': 'sphere_small', 'description': 'Small distributed cavities'},
                    {'name': 'cave_intersections', 'description': 'Complex cave intersections'}
                ]
            },
            {
                'name': 'debug_modes',
                'description': 'Debug visualization modes comparison',
                'modes': [
                    CaptureMode.STANDARD,
                    CaptureMode.WIREFRAME,
                    CaptureMode.NORMALS,
                    CaptureMode.LOD_HEATMAP
                ]
            },
            {
                'name': 'performance_documentation',
                'description': 'Performance and optimization showcase',
                'scenarios': [
                    {'name': 'optimal_performance', 'lod_distance': 1000, 'description': '60+ fps scenario'},
                    {'name': 'stress_test', 'lod_distance': 200, 'description': 'High detail stress test'},
                    {'name': 'long_distance', 'lod_distance': 5000, 'description': 'Long distance performance'}
                ]
            }
        ]
    
    def generate_complete_documentation(self) -> Dict[str, Any]:
        """Generate complete hero world documentation"""
        
        print(f"🚀 Starting Hero World Documentation")
        print(f"   Planet: {self.planet_name}")
        print(f"   Seed: {self.seed}")
        print(f"   Output: {self.output_dir}")
        
        documentation_results = {
            'metadata': {
                'planet_name': self.planet_name,
                'seed': self.seed,
                'generation_time': time.time(),
                'pcc_file': self.hero_world_pcc
            },
            'sequences': {}
        }
        
        # Execute all documentation sequences
        for sequence in self.documentation_plan:
            sequence_name = sequence['name']
            print(f"\n📸 Capturing {sequence_name}...")
            
            if sequence_name == 'overview_sequence':
                results = self._capture_overview_sequence(sequence)
            elif sequence_name == 'face_coverage':
                results = self._capture_face_coverage(sequence)
            elif sequence_name == 'terrain_features':
                results = self._capture_terrain_features(sequence)
            elif sequence_name == 'cave_system':
                results = self._capture_cave_system(sequence)
            elif sequence_name == 'debug_modes':
                results = self._capture_debug_modes(sequence)
            elif sequence_name == 'performance_documentation':
                results = self._capture_performance_documentation(sequence)
            else:
                results = {'screenshots': [], 'notes': f'Unknown sequence: {sequence_name}'}
            
            documentation_results['sequences'][sequence_name] = results
            print(f"   ✅ Captured {len(results['screenshots'])} screenshots")
        
        # Generate summary documentation
        self._generate_summary_documentation(documentation_results)
        
        return documentation_results
    
    def _capture_overview_sequence(self, sequence: Dict[str, Any]) -> Dict[str, Any]:
        """Capture overview sequence at different distances"""
        
        screenshots = []
        series_name = f"{self.planet_name}_overview"
        self.screenshot_tool.start_series(series_name, "Complete planet overview sequence")
        
        # Jump to optimal overview position
        self.camera.jump_to_face(CubeFace.FRONT, immediate=True)
        
        for shot in sequence['shots']:
            distance = shot['camera_distance']
            description = shot['description']
            
            # Position camera at specified distance
            self.camera.orbit_around_point((0, 0, 0), radius=distance, angle_degrees=30.0)
            
            # Update HUD with simulated performance data
            self._update_simulated_hud(distance)
            
            # Capture screenshot
            screenshot_path = self.screenshot_tool.capture_screenshot(
                pcc_name=self.planet_name,
                seed=self.seed,
                camera_position=self.camera.current_state.position,
                camera_target=self.camera.current_state.target,
                camera_fov=self.camera.current_state.fov_degrees,
                capture_mode=CaptureMode.STANDARD,
                description=f"Overview - {description} ({distance}m)",
                tags=["overview", f"distance_{distance}"]
            )
            
            if screenshot_path:
                screenshots.append({
                    'path': screenshot_path,
                    'distance': distance,
                    'description': description,
                    'camera_position': self.camera.current_state.position
                })
        
        self.screenshot_tool.end_series()
        
        return {
            'screenshots': screenshots,
            'series_name': series_name,
            'total_shots': len(screenshots),
            'notes': 'Complete planet overview from multiple distances'
        }
    
    def _capture_face_coverage(self, sequence: Dict[str, Any]) -> Dict[str, Any]:
        """Capture complete face coverage of cube sphere planet"""
        
        screenshots = []
        series_name = f"{self.planet_name}_face_coverage"
        self.screenshot_tool.start_series(series_name, "Complete cube face coverage")
        
        for face_name in sequence['faces']:
            # Jump to face
            self.camera.jump_to_face(face_name, immediate=True)
            
            # Capture standard view
            screenshot_path = self.screenshot_tool.capture_screenshot(
                pcc_name=self.planet_name,
                seed=self.seed,
                camera_position=self.camera.current_state.position,
                camera_target=self.camera.current_state.target,
                camera_fov=self.camera.current_state.fov_degrees,
                capture_mode=CaptureMode.STANDARD,
                description=f"Face coverage - {face_name}",
                tags=["face_coverage", f"face_{face_name}"]
            )
            
            if screenshot_path:
                screenshots.append({
                    'path': screenshot_path,
                    'face': face_name,
                    'camera_position': self.camera.current_state.position
                })
        
        self.screenshot_tool.end_series()
        
        return {
            'screenshots': screenshots,
            'series_name': series_name,
            'total_shots': len(screenshots),
            'notes': 'Complete cube face coverage for planet inspection'
        }
    
    def _capture_terrain_features(self, sequence: Dict[str, Any]) -> Dict[str, Any]:
        """Capture major terrain features in detail"""
        
        screenshots = []
        series_name = f"{self.planet_name}_terrain_features"
        self.screenshot_tool.start_series(series_name, "Major terrain feature showcase")
        
        for feature in sequence['features']:
            feature_name = feature['name']
            location = feature['location']
            description = feature['description']
            
            # Position camera for feature
            if location == 'face_top':
                self.camera.jump_to_face(CubeFace.TOP, immediate=True)
                self.camera.orbit_around_point((0, 0, 0), radius=1200, angle_degrees=45.0)
            elif location == 'face_front':
                self.camera.jump_to_face(CubeFace.FRONT, immediate=True)
                self.camera.orbit_around_point((0, 0, 0), radius=800, angle_degrees=15.0)
            elif location == 'equator_band':
                self.camera.jump_to_face(CubeFace.RIGHT, immediate=True)
                self.camera.orbit_around_point((0, 0, 0), radius=1000, angle_degrees=0.0)
            elif location == 'face_bottom':
                self.camera.jump_to_face(CubeFace.BOTTOM, immediate=True)
                self.camera.orbit_around_point((0, 0, 0), radius=1500, angle_degrees=60.0)
            
            # Capture feature screenshot
            screenshot_path = self.screenshot_tool.capture_screenshot(
                pcc_name=self.planet_name,
                seed=self.seed,
                camera_position=self.camera.current_state.position,
                camera_target=self.camera.current_state.target,
                camera_fov=45.0,  # Tighter FOV for feature detail
                capture_mode=CaptureMode.STANDARD,
                description=f"Terrain feature - {description}",
                tags=["terrain_features", feature_name, location]
            )
            
            if screenshot_path:
                screenshots.append({
                    'path': screenshot_path,
                    'feature': feature_name,
                    'location': location,
                    'description': description,
                    'camera_position': self.camera.current_state.position
                })
        
        self.screenshot_tool.end_series()
        
        return {
            'screenshots': screenshots,
            'series_name': series_name,
            'total_shots': len(screenshots),
            'notes': 'Detailed showcase of major terrain features'
        }
    
    def _capture_cave_system(self, sequence: Dict[str, Any]) -> Dict[str, Any]:
        """Capture cave system documentation"""
        
        screenshots = []
        series_name = f"{self.planet_name}_cave_system"
        self.screenshot_tool.start_series(series_name, "Underground cave system documentation")
        
        # Enable cave-only view
        self.toggles.set_toggle(ToggleType.CAVE_ONLY, True)
        
        # Position camera inside planet for cave views
        cave_positions = [
            (500, 0, 500),     # Interior position 1
            (-300, 200, 400),  # Interior position 2
            (200, -100, -600), # Interior position 3
            (0, 300, 0),       # Central position
            (-400, -200, 300)  # Interior position 4
        ]
        
        for i, position in enumerate(cave_positions):
            self.camera.current_state.position = position
            self.camera.current_state.target = (0, 0, 0)
            
            # Capture cave screenshot
            screenshot_path = self.screenshot_tool.capture_screenshot(
                pcc_name=self.planet_name,
                seed=self.seed,
                camera_position=position,
                camera_target=(0, 0, 0),
                camera_fov=75.0,  # Wide FOV for cave interiors
                capture_mode=CaptureMode.STANDARD,
                debug_toggles=["cave_only"],
                description=f"Cave system interior view {i+1}",
                tags=["cave_system", "interior", f"position_{i+1}"]
            )
            
            if screenshot_path:
                screenshots.append({
                    'path': screenshot_path,
                    'position_index': i + 1,
                    'camera_position': position,
                    'description': f"Cave interior view {i+1}"
                })
        
        # Disable cave-only view
        self.toggles.set_toggle(ToggleType.CAVE_ONLY, False)
        self.screenshot_tool.end_series()
        
        return {
            'screenshots': screenshots,
            'series_name': series_name,
            'total_shots': len(screenshots),
            'notes': 'Underground cave system exploration'
        }
    
    def _capture_debug_modes(self, sequence: Dict[str, Any]) -> Dict[str, Any]:
        """Capture debug mode comparisons"""
        
        screenshots = []
        series_name = f"{self.planet_name}_debug_modes"
        self.screenshot_tool.start_series(series_name, "Debug visualization modes comparison")
        
        # Position camera for good debug view
        self.camera.jump_to_face(CubeFace.FRONT, immediate=True)
        self.camera.orbit_around_point((0, 0, 0), radius=1200, angle_degrees=25.0)
        
        camera_pos = self.camera.current_state.position
        camera_target = self.camera.current_state.target
        
        # Capture each debug mode
        for mode in sequence['modes']:
            # Configure debug toggles for mode
            self.toggles.disable_all()
            
            if mode == CaptureMode.WIREFRAME:
                self.toggles.set_toggle(ToggleType.WIREFRAME, True)
            elif mode == CaptureMode.NORMALS:
                self.toggles.set_toggle(ToggleType.NORMALS, True)
            elif mode == CaptureMode.LOD_HEATMAP:
                self.toggles.set_toggle(ToggleType.LOD_HEATMAP, True)
            
            # Capture screenshot
            screenshot_path = self.screenshot_tool.capture_screenshot(
                pcc_name=self.planet_name,
                seed=self.seed,
                camera_position=camera_pos,
                camera_target=camera_target,
                camera_fov=60.0,
                capture_mode=mode,
                debug_toggles=[t.value for t in self.toggles.get_active_toggles()],
                description=f"Debug mode - {mode.value}",
                tags=["debug_modes", mode.value]
            )
            
            if screenshot_path:
                screenshots.append({
                    'path': screenshot_path,
                    'mode': mode.value,
                    'description': f"Debug visualization - {mode.value}"
                })
        
        # Reset toggles
        self.toggles.disable_all()
        self.screenshot_tool.end_series()
        
        return {
            'screenshots': screenshots,
            'series_name': series_name,
            'total_shots': len(screenshots),
            'notes': 'Debug visualization modes comparison'
        }
    
    def _capture_performance_documentation(self, sequence: Dict[str, Any]) -> Dict[str, Any]:
        """Capture performance scenarios with HUD"""
        
        screenshots = []
        series_name = f"{self.planet_name}_performance"
        self.screenshot_tool.start_series(series_name, "Performance optimization showcase")
        
        # Enable HUD for performance display
        self.hud.visible = True
        self.hud.toggle_detailed_view()
        
        for scenario in sequence['scenarios']:
            scenario_name = scenario['name']
            lod_distance = scenario['lod_distance']
            description = scenario['description']
            
            # Position camera at specified distance
            self.camera.jump_to_face(CubeFace.FRONT, immediate=True)
            self.camera.orbit_around_point((0, 0, 0), radius=lod_distance, angle_degrees=30.0)
            
            # Update HUD with scenario-specific performance data
            self._update_simulated_hud(lod_distance, scenario_name)
            
            # Capture performance screenshot
            screenshot_path = self.screenshot_tool.capture_screenshot(
                pcc_name=self.planet_name,
                seed=self.seed,
                camera_position=self.camera.current_state.position,
                camera_target=self.camera.current_state.target,
                camera_fov=60.0,
                capture_mode=CaptureMode.DEBUG_OVERLAY,
                description=f"Performance - {description}",
                tags=["performance", scenario_name, f"distance_{lod_distance}"]
            )
            
            if screenshot_path:
                screenshots.append({
                    'path': screenshot_path,
                    'scenario': scenario_name,
                    'lod_distance': lod_distance,
                    'description': description,
                    'performance_data': self._get_simulated_performance_data(lod_distance)
                })
        
        self.screenshot_tool.end_series()
        
        return {
            'screenshots': screenshots,
            'series_name': series_name,
            'total_shots': len(screenshots),
            'notes': 'Performance optimization scenarios with HUD overlay'
        }
    
    def _update_simulated_hud(self, camera_distance: float, scenario: str = "standard"):
        """Update HUD with simulated performance data"""
        
        # Simulate performance based on distance and complexity
        base_triangles = 500000
        distance_factor = max(0.1, 1000.0 / camera_distance)
        triangle_count = int(base_triangles * distance_factor)
        
        # Simulate FPS based on triangle count
        if triangle_count < 200000:
            fps = 72.0
        elif triangle_count < 400000:
            fps = 65.0
        elif triangle_count < 600000:
            fps = 58.0
        else:
            fps = 45.0
        
        # Simulate other metrics
        frame_time_ms = 1000.0 / fps
        draw_calls = min(150, triangle_count // 5000)
        active_chunks = min(64, triangle_count // 10000)
        visible_chunks = int(active_chunks * 0.7)
        memory_usage = triangle_count * 0.0008  # MB
        vram_usage = memory_usage * 1.5
        
        # Update HUD
        self.hud.update_frame_stats(
            fps=fps,
            frame_time_ms=frame_time_ms,
            draw_calls=draw_calls,
            triangles=triangle_count,
            vertices=triangle_count * 3,
            active_chunks=active_chunks,
            visible_chunks=visible_chunks,
            memory_usage_mb=memory_usage,
            vram_usage_mb=vram_usage
        )
        
        # Update LOD stats
        lod_histogram = self._generate_lod_histogram(camera_distance)
        self.hud.update_lod_stats(lod_histogram, 2.5)
    
    def _generate_lod_histogram(self, camera_distance: float) -> Dict[int, int]:
        """Generate realistic LOD histogram based on distance"""
        
        if camera_distance < 500:
            # Close view - high detail LODs
            return {0: 15, 1: 12, 2: 8, 3: 4, 4: 2}
        elif camera_distance < 1500:
            # Medium view - mixed LODs
            return {0: 8, 1: 15, 2: 12, 3: 8, 4: 4, 5: 2}
        elif camera_distance < 3000:
            # Far view - lower detail LODs
            return {2: 6, 3: 12, 4: 15, 5: 10, 6: 5}
        else:
            # Very far - lowest detail LODs
            return {4: 4, 5: 8, 6: 12, 7: 8, 8: 3}
    
    def _get_simulated_performance_data(self, lod_distance: float) -> Dict[str, Any]:
        """Get simulated performance data for documentation"""
        
        base_triangles = 500000
        distance_factor = max(0.1, 1000.0 / lod_distance)
        triangle_count = int(base_triangles * distance_factor)
        
        fps = 72.0 if triangle_count < 200000 else 65.0 if triangle_count < 400000 else 58.0 if triangle_count < 600000 else 45.0
        
        return {
            'fps': fps,
            'triangle_count': triangle_count,
            'distance': lod_distance,
            'performance_rating': 'Excellent' if fps >= 60 else 'Good' if fps >= 45 else 'Acceptable'
        }
    
    def _generate_summary_documentation(self, results: Dict[str, Any]):
        """Generate summary documentation file"""
        
        summary = {
            'hero_world_documentation': {
                'generation_info': results['metadata'],
                'total_sequences': len(results['sequences']),
                'total_screenshots': sum(len(seq['screenshots']) for seq in results['sequences'].values()),
                'sequences_summary': {}
            }
        }
        
        # Summarize each sequence
        for seq_name, seq_data in results['sequences'].items():
            summary['hero_world_documentation']['sequences_summary'][seq_name] = {
                'screenshot_count': len(seq_data['screenshots']),
                'series_name': seq_data.get('series_name', ''),
                'notes': seq_data.get('notes', '')
            }
        
        # Export detailed results
        results_file = self.output_dir / "hero_world_documentation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Export summary
        summary_file = self.output_dir / "documentation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Documentation Summary:")
        print(f"   Total sequences: {summary['hero_world_documentation']['total_sequences']}")
        print(f"   Total screenshots: {summary['hero_world_documentation']['total_screenshots']}")
        print(f"   Results file: {results_file}")
        print(f"   Summary file: {summary_file}")


if __name__ == "__main__":
    # Generate hero world documentation
    print("🚀 T17 Hero World Documentation & Screenshots")
    print("=" * 60)
    
    # Configuration
    hero_world_pcc = "/home/colling/PCC-LanguageV2/agents/agent_d/examples/planets/hero_world.pcc.json"
    output_dir = "/home/colling/PCC-LanguageV2/agents/agent_d/examples/planets/hero_world_screenshots"
    
    # Create documentation system
    doc_system = HeroWorldDocumentation(hero_world_pcc, output_dir)
    
    # Generate complete documentation
    results = doc_system.generate_complete_documentation()
    
    print(f"\n🎉 Hero World Documentation Complete!")
    print(f"   Planet: {results['metadata']['planet_name']}")
    print(f"   Screenshots: {sum(len(seq['screenshots']) for seq in results['sequences'].values())}")
    print(f"   Sequences: {len(results['sequences'])}")
    print(f"   Output directory: {output_dir}")
    
    # List generated sequences
    print(f"\n📸 Generated Sequences:")
    for seq_name, seq_data in results['sequences'].items():
        print(f"   {seq_name}: {len(seq_data['screenshots'])} screenshots")
    
    print(f"\n✅ T17 hero world documentation and screenshots complete")