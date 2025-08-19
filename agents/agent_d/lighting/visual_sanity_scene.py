#!/usr/bin/env python3
"""
Visual Sanity Scene - T12
==========================

Creates a test scene with sun/sky and moving light to validate normal mapping
and consistent shading across terrain and cave surfaces.

Features:
- Realistic sun/sky lighting with time-of-day cycle
- Moving spotlight for dynamic normal map validation
- Multi-light setup for comprehensive shading tests
- Integration with terrain normal mapping shaders
- Visual validation of TBN space correctness

Usage:
    from visual_sanity_scene import VisualSanityLighting
    
    lighting = VisualSanityLighting()
    lighting.update_time_of_day(0.5)  # Midday
    lighting.apply_to_shader(shader_manager)
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import shader system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shaders'))
from terrain_normal_mapping import TerrainShaderManager

try:
    from OpenGL.GL import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


@dataclass
class LightSource:
    """Individual light source definition"""
    name: str
    position: np.ndarray  # [x, y, z] world position
    color: np.ndarray     # [r, g, b] color
    intensity: float      # Light intensity
    light_type: str       # "directional", "point", "spot"
    enabled: bool = True
    
    # Spotlight parameters
    direction: np.ndarray = None  # For spotlights
    cone_angle: float = 45.0      # Cone angle in degrees
    falloff: float = 1.0          # Distance falloff exponent


@dataclass 
class SkyboxParameters:
    """Sky rendering parameters"""
    sun_direction: np.ndarray     # Normalized sun direction
    sun_color: np.ndarray        # Sun color
    sky_color: np.ndarray        # Sky color
    horizon_color: np.ndarray    # Horizon color
    ground_color: np.ndarray     # Ground color
    sun_intensity: float         # Sun brightness
    sky_intensity: float         # Overall sky brightness


class TimeOfDay(Enum):
    """Time of day presets"""
    DAWN = 0.2
    MORNING = 0.3
    MIDDAY = 0.5
    AFTERNOON = 0.7
    DUSK = 0.8
    NIGHT = 0.0


class VisualSanityLighting:
    """Manages lighting for visual sanity validation scene"""
    
    def __init__(self):
        """Initialize visual sanity lighting system"""
        self.light_sources: Dict[str, LightSource] = {}
        self.skybox_params = self._create_default_skybox()
        self.time_of_day = 0.5  # 0.0 = midnight, 0.5 = noon, 1.0 = midnight
        
        # Animation state
        self.animation_time = 0.0
        self.animation_speed = 1.0
        self.moving_light_enabled = True
        
        # Create default lighting setup
        self._create_default_lights()
    
    def _create_default_skybox(self) -> SkyboxParameters:
        """Create default skybox parameters for midday"""
        return SkyboxParameters(
            sun_direction=np.array([0.3, 0.8, 0.5]),  # Angled sun
            sun_color=np.array([1.0, 0.95, 0.8]),     # Warm white
            sky_color=np.array([0.4, 0.7, 1.0]),      # Blue sky
            horizon_color=np.array([0.8, 0.9, 1.0]),  # Light blue horizon
            ground_color=np.array([0.3, 0.25, 0.2]),  # Brown ground
            sun_intensity=3.0,
            sky_intensity=0.3
        )
    
    def _create_default_lights(self):
        """Create default light setup for visual validation"""
        
        # Primary sun light (directional)
        self.light_sources["sun"] = LightSource(
            name="sun",
            position=np.array([10.0, 10.0, 10.0]),  # High position
            color=np.array([1.0, 0.95, 0.8]),       # Warm sunlight
            intensity=2.5,
            light_type="directional"
        )
        
        # Moving spotlight for normal map validation
        self.light_sources["moving_spot"] = LightSource(
            name="moving_spot",
            position=np.array([0.0, 3.0, 0.0]),     # Above terrain
            color=np.array([1.0, 0.8, 0.6]),        # Warm spot
            intensity=1.5,
            light_type="spot",
            direction=np.array([0.0, -1.0, 0.0]),   # Pointing down
            cone_angle=30.0
        )
        
        # Fill light for shadow detail
        self.light_sources["fill"] = LightSource(
            name="fill",
            position=np.array([-5.0, 5.0, -5.0]),   # Offset position
            color=np.array([0.7, 0.8, 1.0]),        # Cool fill light
            intensity=0.8,
            light_type="point"
        )
        
        # Ambient rim light for edge definition
        self.light_sources["rim"] = LightSource(
            name="rim",
            position=np.array([0.0, 2.0, -10.0]),   # Behind camera
            color=np.array([0.9, 0.9, 1.0]),        # Neutral rim
            intensity=0.6,
            light_type="directional"
        )
    
    def update_time_of_day(self, time_value: float):
        """Update lighting based on time of day (0.0 = midnight, 0.5 = noon)"""
        self.time_of_day = time_value % 1.0
        
        # Calculate sun angle based on time
        sun_angle = (time_value - 0.5) * math.pi  # -π/2 to π/2 around noon
        sun_elevation = math.sin(sun_angle * 0.5 + math.pi * 0.5)  # 0 to 1
        sun_elevation = max(0.0, sun_elevation)  # Clamp to positive
        
        # Update sun direction
        sun_x = math.sin(sun_angle) * 0.6
        sun_y = sun_elevation * 0.8 + 0.2  # Keep sun above horizon
        sun_z = math.cos(sun_angle) * 0.6
        
        self.skybox_params.sun_direction = np.array([sun_x, sun_y, sun_z])
        self.skybox_params.sun_direction /= np.linalg.norm(self.skybox_params.sun_direction)
        
        # Update sun light position and intensity
        if "sun" in self.light_sources:
            sun_light = self.light_sources["sun"]
            sun_light.position = self.skybox_params.sun_direction * 100.0  # Far away
            sun_light.intensity = 1.5 + sun_elevation * 1.5  # Brighter at noon
        
        # Update sky colors based on time
        self._update_sky_colors(time_value, sun_elevation)
        
        print(f"🌅 Updated time of day: {time_value:.2f}, sun elevation: {sun_elevation:.2f}")
    
    def _update_sky_colors(self, time_value: float, sun_elevation: float):
        """Update sky colors based on time of day"""
        
        if sun_elevation > 0.8:  # Midday
            self.skybox_params.sun_color = np.array([1.0, 0.95, 0.8])
            self.skybox_params.sky_color = np.array([0.4, 0.7, 1.0])
            self.skybox_params.sun_intensity = 3.0
        elif sun_elevation > 0.3:  # Day
            self.skybox_params.sun_color = np.array([1.0, 0.9, 0.7])
            self.skybox_params.sky_color = np.array([0.6, 0.8, 1.0])
            self.skybox_params.sun_intensity = 2.0
        elif sun_elevation > 0.1:  # Dawn/Dusk
            self.skybox_params.sun_color = np.array([1.0, 0.6, 0.3])
            self.skybox_params.sky_color = np.array([0.9, 0.5, 0.3])
            self.skybox_params.sun_intensity = 1.0
        else:  # Night
            self.skybox_params.sun_color = np.array([0.3, 0.3, 0.5])
            self.skybox_params.sky_color = np.array([0.1, 0.1, 0.3])
            self.skybox_params.sun_intensity = 0.2
        
        # Update ambient light for night
        self.skybox_params.sky_intensity = 0.1 + sun_elevation * 0.4
    
    def update_animation(self, delta_time: float):
        """Update animated lights and effects"""
        self.animation_time += delta_time * self.animation_speed
        
        if self.moving_light_enabled and "moving_spot" in self.light_sources:
            spotlight = self.light_sources["moving_spot"]
            
            # Circular movement around origin
            radius = 4.0
            height = 3.0 + math.sin(self.animation_time * 0.5) * 1.0  # Bob up/down
            
            angle = self.animation_time * 0.3  # Rotate around
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            
            spotlight.position = np.array([x, height, z])
            
            # Point spotlight toward origin
            to_origin = -spotlight.position.copy()
            to_origin[1] = -abs(to_origin[1])  # Point downward
            spotlight.direction = to_origin / np.linalg.norm(to_origin)
            
            # Vary spotlight intensity
            spotlight.intensity = 1.2 + 0.3 * math.sin(self.animation_time * 0.8)
    
    def apply_to_shader(self, shader_mgr: TerrainShaderManager, 
                       camera_position: Tuple[float, float, float] = (0.0, 2.0, 5.0)):
        """Apply lighting parameters to terrain shader"""
        if not shader_mgr.current_shader:
            return
        
        # Use primary sun light as main light source
        if "sun" in self.light_sources:
            sun_light = self.light_sources["sun"]
            shader_mgr.set_lighting(
                light_pos=tuple(sun_light.position),
                light_color=tuple(sun_light.color),
                light_intensity=sun_light.intensity,
                ambient_light=tuple(self.skybox_params.sky_color * self.skybox_params.sky_intensity),
                view_pos=camera_position
            )
        
        # TODO: Extend shader system for multiple lights
        # For now, we use the primary sun light
    
    def render_skybox(self, camera_position: Tuple[float, float, float]):
        """Render simple skybox (if OpenGL available)"""
        if not OPENGL_AVAILABLE:
            return
        
        # Simple skybox rendering with OpenGL immediate mode
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Move skybox with camera
        glPushMatrix()
        glTranslatef(*camera_position)
        
        # Large skybox quad
        size = 50.0
        
        # Sky gradient (simple version)
        glBegin(GL_QUADS)
        
        # Top (sky color)
        glColor3f(*self.skybox_params.sky_color)
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        
        # Bottom (ground color)
        glColor3f(*self.skybox_params.ground_color)
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, -size, -size)
        
        glEnd()
        
        # Render sun as bright point
        sun_pos = camera_position + self.skybox_params.sun_direction * 30.0
        glColor3f(*self.skybox_params.sun_color)
        glPointSize(20.0)
        glBegin(GL_POINTS)
        glVertex3f(*sun_pos)
        glEnd()
        
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def render_debug_lights(self, camera_position: Tuple[float, float, float]):
        """Render debug visualization of light sources"""
        if not OPENGL_AVAILABLE:
            return
        
        glDisable(GL_LIGHTING)
        glPointSize(8.0)
        
        for light in self.light_sources.values():
            if not light.enabled:
                continue
            
            glColor3f(*light.color)
            glBegin(GL_POINTS)
            glVertex3f(*light.position)
            glEnd()
            
            # Draw light direction for spotlights
            if light.light_type == "spot" and light.direction is not None:
                glColor3f(*(light.color * 0.5))
                glBegin(GL_LINES)
                glVertex3f(*light.position)
                end_pos = light.position + light.direction * 2.0
                glVertex3f(*end_pos)
                glEnd()
        
        glEnable(GL_LIGHTING)
    
    def set_preset(self, preset: TimeOfDay):
        """Set lighting to predefined time of day preset"""
        self.update_time_of_day(preset.value)
        print(f"🎨 Applied lighting preset: {preset.name}")
    
    def toggle_moving_light(self) -> bool:
        """Toggle moving spotlight animation"""
        self.moving_light_enabled = not self.moving_light_enabled
        print(f"💡 Moving light: {'ON' if self.moving_light_enabled else 'OFF'}")
        return self.moving_light_enabled
    
    def get_lighting_stats(self) -> Dict[str, Any]:
        """Get current lighting statistics"""
        enabled_lights = [light for light in self.light_sources.values() if light.enabled]
        
        return {
            'time_of_day': self.time_of_day,
            'sun_direction': self.skybox_params.sun_direction.tolist(),
            'sun_intensity': self.skybox_params.sun_intensity,
            'total_lights': len(self.light_sources),
            'enabled_lights': len(enabled_lights),
            'moving_light_enabled': self.moving_light_enabled,
            'animation_time': self.animation_time
        }


if __name__ == "__main__":
    # Test visual sanity lighting system
    print("🚀 T12 Visual Sanity Lighting System")
    print("=" * 60)
    
    # Create lighting system
    lighting = VisualSanityLighting()
    
    # Test time of day updates
    print("🌅 Testing time of day cycle:")
    for preset in TimeOfDay:
        lighting.set_preset(preset)
        stats = lighting.get_lighting_stats()
        print(f"   {preset.name}: sun_intensity={stats['sun_intensity']:.2f}")
    
    # Test animation updates
    print("\n🎬 Testing animation updates:")
    for i in range(3):
        lighting.update_animation(0.5)  # 0.5 second steps
        spotlight = lighting.light_sources["moving_spot"]
        print(f"   Frame {i}: spotlight at ({spotlight.position[0]:.2f}, {spotlight.position[1]:.2f}, {spotlight.position[2]:.2f})")
    
    # Test lighting statistics
    final_stats = lighting.get_lighting_stats()
    print(f"\n📊 Final Lighting Stats:")
    for key, value in final_stats.items():
        if isinstance(value, list):
            print(f"   {key}: [{', '.join(f'{v:.2f}' for v in value)}]")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Visual sanity lighting system functional")
    
    if OPENGL_AVAILABLE:
        print("✅ OpenGL rendering ready")
    else:
        print("⚠️ OpenGL not available - lighting defined but not rendered")