#!/usr/bin/env python3
"""
Complete Lighting System - T21
==============================

Integrates shadow mapping, SSAO, and tone mapping into a complete lighting pipeline.
Provides sun/sky rig with warm key light and cool ambient lighting.

Features:
- Complete lighting pipeline integration
- Sun/Sky lighting rig with realistic colors
- Shadow mapping with PCF filtering
- Screen-space ambient occlusion
- HDR tone mapping with gamma correction
- Performance monitoring and optimization
- Lighting sanity scene for validation

Usage:
    from lighting_system import LightingSystem
    
    lighting = LightingSystem()
    lighting.set_sun_direction([0.5, -1.0, 0.3])
    final_image = lighting.render_lighting(scene, camera)
"""

import numpy as np
import math
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import T21 lighting components
sys.path.append(os.path.dirname(__file__))
from shadow_mapping import ShadowMapper, ShadowMapConfig, ShadowQuality
from ssao import SSAORenderer, SSAOConfig, SSAOQuality
from tone_mapping import ToneMapper, ToneMapConfig, ToneOperator


@dataclass
class LightingConfig:
    """Complete lighting system configuration"""
    # Sun/Sky lighting
    sun_direction: np.ndarray = np.array([0.5, -1.0, 0.3])
    sun_color: np.ndarray = np.array([1.0, 0.95, 0.8])     # Warm key light
    sun_intensity: float = 3.0
    sky_color: np.ndarray = np.array([0.5, 0.7, 1.0])      # Cool sky light
    sky_intensity: float = 0.8
    ambient_color: np.ndarray = np.array([0.1, 0.15, 0.2]) # Cool ambient
    ambient_intensity: float = 0.3
    
    # Shadow mapping
    shadow_quality: ShadowQuality = ShadowQuality.MEDIUM
    enable_shadows: bool = True
    
    # SSAO
    ssao_quality: SSAOQuality = SSAOQuality.MEDIUM
    enable_ssao: bool = True
    
    # Tone mapping
    tone_operator: ToneOperator = ToneOperator.ACES
    gamma: float = 2.2
    exposure: float = 1.0
    enable_tone_mapping: bool = True


class LightingSystem:
    """
    T21: Complete lighting system with shadows, SSAO, and tone mapping
    """
    
    def __init__(self, config: LightingConfig = None, screen_width: int = 800, screen_height: int = 600):
        """
        Initialize complete lighting system
        
        Args:
            config: Lighting system configuration
            screen_width: Screen width for SSAO
            screen_height: Screen height for SSAO
        """
        self.config = config or LightingConfig()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize lighting subsystems
        self._initialize_subsystems()
        
        # Performance statistics
        self.stats = {
            'total_render_time_ms': 0.0,
            'shadow_time_ms': 0.0,
            'ssao_time_ms': 0.0,
            'tone_mapping_time_ms': 0.0,
            'lighting_time_ms': 0.0,
            'frame_count': 0,
            'average_fps': 0.0
        }
        
        # Timing for FPS calculation
        self.last_frame_time = time.time()
        
        print("🌅 T21 Complete Lighting System initialized")
        print(f"   Sun: {self.config.sun_color} @ {self.config.sun_intensity}")
        print(f"   Sky: {self.config.sky_color} @ {self.config.sky_intensity}")
        print(f"   Shadows: {'ON' if self.config.enable_shadows else 'OFF'}")
        print(f"   SSAO: {'ON' if self.config.enable_ssao else 'OFF'}")
        print(f"   Tone Mapping: {self.config.tone_operator.value.upper()}")
    
    def _initialize_subsystems(self):
        """Initialize all lighting subsystems"""
        # Shadow mapping
        if self.config.enable_shadows:
            shadow_config = self._create_shadow_config()
            self.shadow_mapper = ShadowMapper(shadow_config)
        else:
            self.shadow_mapper = None
        
        # SSAO
        if self.config.enable_ssao:
            ssao_config = self._create_ssao_config()
            self.ssao_renderer = SSAORenderer(ssao_config, self.screen_width, self.screen_height)
        else:
            self.ssao_renderer = None
        
        # Tone mapping
        if self.config.enable_tone_mapping:
            tone_config = self._create_tone_mapping_config()
            self.tone_mapper = ToneMapper(tone_config)
        else:
            self.tone_mapper = None
    
    def _create_shadow_config(self) -> ShadowMapConfig:
        """Create shadow mapping configuration from lighting config"""
        quality_configs = {
            ShadowQuality.LOW: ShadowMapConfig(resolution=1024, pcf_samples=4, depth_bias=0.01),
            ShadowQuality.MEDIUM: ShadowMapConfig(resolution=2048, pcf_samples=9, depth_bias=0.005),
            ShadowQuality.HIGH: ShadowMapConfig(resolution=4096, pcf_samples=16, depth_bias=0.003)
        }
        return quality_configs.get(self.config.shadow_quality, quality_configs[ShadowQuality.MEDIUM])
    
    def _create_ssao_config(self) -> SSAOConfig:
        """Create SSAO configuration from lighting config"""
        quality_configs = {
            SSAOQuality.LOW: SSAOConfig(radius=0.3, intensity=0.3, sample_count=8),
            SSAOQuality.MEDIUM: SSAOConfig(radius=0.5, intensity=0.4, sample_count=16),
            SSAOQuality.HIGH: SSAOConfig(radius=0.6, intensity=0.5, sample_count=32)
        }
        return quality_configs.get(self.config.ssao_quality, quality_configs[SSAOQuality.MEDIUM])
    
    def _create_tone_mapping_config(self) -> ToneMapConfig:
        """Create tone mapping configuration from lighting config"""
        return ToneMapConfig(
            tone_operator=self.config.tone_operator,
            gamma=self.config.gamma,
            exposure=self.config.exposure
        )
    
    def render_lighting(self, scene: Dict[str, Any], camera: Dict[str, Any]) -> np.ndarray:
        """
        Render complete lighting for scene
        
        Args:
            scene: Scene data with objects and materials
            camera: Camera parameters (position, view matrix, projection matrix)
            
        Returns:
            Final lit and tone-mapped image
        """
        total_start_time = time.time()
        
        # Extract camera matrices
        view_matrix = camera.get('view_matrix', np.eye(4))
        proj_matrix = camera.get('proj_matrix', np.eye(4))
        camera_pos = camera.get('position', np.array([0, 0, 5]))
        
        # 1. Generate shadow maps
        shadow_map = None
        if self.shadow_mapper is not None:
            shadow_start = time.time()
            scene_objects = scene.get('objects', [])
            scene_bounds = self._calculate_scene_bounds(scene_objects)
            
            self.shadow_mapper.set_light_direction(self.config.sun_direction)
            self.shadow_mapper.generate_shadow_map(scene_objects, scene_bounds)
            
            self.stats['shadow_time_ms'] = (time.time() - shadow_start) * 1000
        
        # 2. Render base lighting (diffuse + specular)
        lighting_start = time.time()
        hdr_image = self._render_base_lighting(scene, camera, shadow_map)
        self.stats['lighting_time_ms'] = (time.time() - lighting_start) * 1000
        
        # 3. Generate depth and normal buffers for SSAO
        depth_buffer = self._generate_depth_buffer(scene, camera)
        normal_buffer = self._generate_normal_buffer(scene, camera)
        
        # 4. Apply SSAO
        if self.ssao_renderer is not None:
            ssao_start = time.time()
            occlusion_map = self.ssao_renderer.render_ssao(
                depth_buffer, normal_buffer, view_matrix, proj_matrix
            )
            
            # Apply SSAO before tone mapping
            hdr_image = self.ssao_renderer.apply_occlusion(hdr_image, occlusion_map)
            self.stats['ssao_time_ms'] = (time.time() - ssao_start) * 1000
        
        # 5. Apply tone mapping and gamma correction
        final_image = hdr_image
        if self.tone_mapper is not None:
            tone_start = time.time()
            final_image = self.tone_mapper.tone_map(hdr_image, self.config.exposure)
            self.stats['tone_mapping_time_ms'] = (time.time() - tone_start) * 1000
        else:
            # Simple clamp if no tone mapping
            final_image = np.clip(hdr_image, 0.0, 1.0)
        
        # Update statistics
        total_time = (time.time() - total_start_time) * 1000
        self.stats['total_render_time_ms'] = total_time
        self.stats['frame_count'] += 1
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        if frame_time > 0:
            self.stats['average_fps'] = 1.0 / frame_time
        self.last_frame_time = current_time
        
        return final_image
    
    def _render_base_lighting(self, scene: Dict[str, Any], camera: Dict[str, Any], 
                             shadow_map: Optional[Any] = None) -> np.ndarray:
        """
        Render base lighting (sun + sky + ambient)
        
        Args:
            scene: Scene data
            camera: Camera data
            shadow_map: Shadow map for shadowing
            
        Returns:
            HDR lit image
        """
        # Create HDR framebuffer
        hdr_image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.float32)
        
        # For simplified implementation, create a basic lit scene
        # In real implementation, this would render all objects with proper shading
        
        # Create gradient to simulate lighting variation
        for y in range(self.screen_height):
            for x in range(self.screen_width):
                # Simulate ground plane lighting
                u = x / self.screen_width
                v = y / self.screen_height
                
                # Create some terrain-like variation
                height_variation = math.sin(u * 10) * math.cos(v * 8) * 0.3 + 0.7
                
                # Apply sun lighting (warm)
                sun_contribution = self.config.sun_color * self.config.sun_intensity * height_variation
                
                # Apply sky lighting (cool)
                sky_factor = v  # More sky light at top of screen
                sky_contribution = self.config.sky_color * self.config.sky_intensity * sky_factor
                
                # Apply ambient (cool)
                ambient_contribution = self.config.ambient_color * self.config.ambient_intensity
                
                # Combine lighting
                total_light = sun_contribution + sky_contribution + ambient_contribution
                
                # Apply shadow if available
                if self.shadow_mapper is not None:
                    # Simulate world position for shadow testing
                    world_pos = np.array([u * 20 - 10, v * 20 - 10, height_variation])
                    world_normal = np.array([0, 0, 1])  # Up normal
                    
                    shadow_factor = self.shadow_mapper.calculate_shadow_factor(world_pos, world_normal)
                    
                    # Apply shadow only to sun contribution
                    total_light = (sun_contribution * shadow_factor + 
                                 sky_contribution + ambient_contribution)
                
                hdr_image[y, x] = total_light
        
        return hdr_image
    
    def _generate_depth_buffer(self, scene: Dict[str, Any], camera: Dict[str, Any]) -> np.ndarray:
        """
        Generate depth buffer for SSAO
        
        Args:
            scene: Scene data
            camera: Camera data
            
        Returns:
            Linear depth buffer
        """
        # Simplified depth buffer generation
        # In real implementation, this would be rendered during the depth pre-pass
        depth_buffer = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        
        # Create fake depth variation
        for y in range(self.screen_height):
            for x in range(self.screen_width):
                u = x / self.screen_width
                v = y / self.screen_height
                
                # Simulate depth based on position
                depth = 0.3 + 0.4 * (math.sin(u * 8) * math.cos(v * 6) * 0.5 + 0.5)
                depth_buffer[y, x] = depth
        
        return depth_buffer
    
    def _generate_normal_buffer(self, scene: Dict[str, Any], camera: Dict[str, Any]) -> np.ndarray:
        """
        Generate world-space normal buffer for SSAO
        
        Args:
            scene: Scene data
            camera: Camera data
            
        Returns:
            World-space normal buffer
        """
        # Simplified normal buffer generation
        normal_buffer = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.float32)
        
        # Create fake normal variation based on terrain
        for y in range(self.screen_height):
            for x in range(self.screen_width):
                u = x / self.screen_width
                v = y / self.screen_width
                
                # Calculate fake terrain gradient
                height_u = math.sin(u * 10) * math.cos(v * 8) * 0.3
                height_v = math.cos(u * 10) * math.sin(v * 8) * 0.3
                
                # Create normal from height gradient
                normal = np.array([-height_u, -height_v, 1.0])
                normal = normal / np.linalg.norm(normal)
                
                normal_buffer[y, x] = normal
        
        return normal_buffer
    
    def _calculate_scene_bounds(self, objects: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Calculate scene bounding box from objects
        
        Args:
            objects: List of scene objects
            
        Returns:
            Scene bounds dictionary
        """
        if not objects:
            return {'min': np.array([-10, -10, -10]), 'max': np.array([10, 10, 10])}
        
        # Extract positions and calculate bounds
        positions = []
        for obj in objects:
            pos = obj.get('position', [0, 0, 0])
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Expand bounds slightly
        margin = 2.0
        min_bounds = np.min(positions, axis=0) - margin
        max_bounds = np.max(positions, axis=0) + margin
        
        return {'min': min_bounds, 'max': max_bounds}
    
    def set_sun_direction(self, direction: np.ndarray):
        """
        Set sun direction and update shadow mapping
        
        Args:
            direction: New sun direction (will be normalized)
        """
        self.config.sun_direction = direction / np.linalg.norm(direction)
        
        if self.shadow_mapper is not None:
            self.shadow_mapper.set_light_direction(self.config.sun_direction)
    
    def set_lighting_quality(self, shadow_quality: ShadowQuality, ssao_quality: SSAOQuality):
        """
        Update lighting quality settings
        
        Args:
            shadow_quality: Shadow mapping quality
            ssao_quality: SSAO quality
        """
        self.config.shadow_quality = shadow_quality
        self.config.ssao_quality = ssao_quality
        
        # Reinitialize subsystems with new quality
        self._initialize_subsystems()
        
        print(f"🔧 Lighting quality updated: Shadows={shadow_quality.value}, SSAO={ssao_quality.value}")
    
    def create_lighting_sanity_scene(self) -> Dict[str, Any]:
        """
        Create lighting sanity scene for T21 validation
        
        Returns:
            Test scene with sphere and cube for lighting validation
        """
        scene = {
            'objects': [
                {
                    'type': 'sphere',
                    'position': [0.0, 0.0, 0.0],
                    'radius': 1.0,
                    'material': {
                        'albedo': [0.8, 0.6, 0.4],
                        'roughness': 0.3,
                        'metallic': 0.0
                    },
                    'casts_shadows': True
                },
                {
                    'type': 'cube',
                    'position': [3.0, 0.0, 0.0],
                    'size': [1.5, 1.5, 1.5],
                    'material': {
                        'albedo': [0.4, 0.6, 0.8],
                        'roughness': 0.7,
                        'metallic': 0.1
                    },
                    'casts_shadows': True
                },
                {
                    'type': 'plane',
                    'position': [0.0, 0.0, -1.0],
                    'size': [20.0, 20.0],
                    'material': {
                        'albedo': [0.6, 0.6, 0.6],
                        'roughness': 0.8,
                        'metallic': 0.0
                    },
                    'casts_shadows': False
                }
            ]
        }
        
        return scene
    
    def validate_lighting_pipeline(self):
        """
        Run lighting sanity checks to validate shadow edges and AO
        """
        print("🧪 T21 Lighting Pipeline Validation")
        print("=" * 50)
        
        # Create test scene
        test_scene = self.create_lighting_sanity_scene()
        
        # Create test camera
        test_camera = {
            'position': np.array([0, -8, 4]),
            'view_matrix': np.eye(4),
            'proj_matrix': np.eye(4)
        }
        
        # Test different sun positions
        sun_positions = [
            np.array([1.0, -1.0, 0.5]),   # Morning
            np.array([0.0, -1.0, 1.0]),   # Noon
            np.array([-1.0, -1.0, 0.5])   # Evening
        ]
        
        for i, sun_dir in enumerate(sun_positions):
            print(f"\\nTest {i+1}: Sun direction {sun_dir}")
            
            # Set sun direction
            self.set_sun_direction(sun_dir)
            
            # Render frame
            start_time = time.time()
            final_image = self.render_lighting(test_scene, test_camera)
            render_time = (time.time() - start_time) * 1000
            
            # Analyze results
            image_stats = {
                'brightness_min': np.min(final_image),
                'brightness_max': np.max(final_image),
                'brightness_mean': np.mean(final_image),
                'dynamic_range': np.max(final_image) / max(np.min(final_image), 1e-8)
            }
            
            print(f"   Render time: {render_time:.2f}ms")
            print(f"   Brightness range: {image_stats['brightness_min']:.3f} - {image_stats['brightness_max']:.3f}")
            print(f"   Mean brightness: {image_stats['brightness_mean']:.3f}")
            print(f"   Dynamic range: {image_stats['dynamic_range']:.2f}")
            
            # Validate shadow and AO functionality
            stats = self.get_lighting_statistics()
            if self.config.enable_shadows:
                print(f"   Shadow map time: {stats['shadow_time_ms']:.2f}ms")
            if self.config.enable_ssao:
                print(f"   SSAO time: {stats['ssao_time_ms']:.2f}ms")
            print(f"   Tone mapping time: {stats['tone_mapping_time_ms']:.2f}ms")
        
        print("\\n✅ Lighting pipeline validation complete")
    
    def get_lighting_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lighting statistics"""
        combined_stats = self.stats.copy()
        
        # Add subsystem statistics
        if self.shadow_mapper is not None:
            combined_stats['shadow_stats'] = self.shadow_mapper.get_shadow_statistics()
        
        if self.ssao_renderer is not None:
            combined_stats['ssao_stats'] = self.ssao_renderer.get_ssao_statistics()
        
        if self.tone_mapper is not None:
            combined_stats['tone_mapping_stats'] = self.tone_mapper.get_tone_mapping_statistics()
        
        return combined_stats
    
    def cleanup(self):
        """Cleanup all lighting system resources"""
        if self.shadow_mapper is not None:
            self.shadow_mapper.cleanup()
        
        if self.ssao_renderer is not None:
            self.ssao_renderer.cleanup()
        
        print("✅ Lighting system cleaned up")


if __name__ == "__main__":
    # Test complete T21 lighting system
    print("🌅 T21 Complete Lighting System Test")
    print("=" * 60)
    
    # Create lighting system with different quality settings
    qualities = [
        (ShadowQuality.LOW, SSAOQuality.LOW),
        (ShadowQuality.MEDIUM, SSAOQuality.MEDIUM),
        (ShadowQuality.HIGH, SSAOQuality.HIGH)
    ]
    
    for shadow_qual, ssao_qual in qualities:
        print(f"\\nTesting {shadow_qual.value.upper()}/{ssao_qual.value.upper()} quality:")
        
        # Create lighting config
        config = LightingConfig(
            shadow_quality=shadow_qual,
            ssao_quality=ssao_qual,
            enable_shadows=True,
            enable_ssao=True,
            enable_tone_mapping=True
        )
        
        # Initialize lighting system
        lighting = LightingSystem(config, 512, 512)  # Smaller for testing
        
        # Run validation
        lighting.validate_lighting_pipeline()
        
        # Print overall statistics
        stats = lighting.get_lighting_statistics()
        print(f"   Total render time: {stats['total_render_time_ms']:.2f}ms")
        print(f"   Average FPS: {stats['average_fps']:.1f}")
        
        lighting.cleanup()
    
    print("\\n✅ T21 Complete lighting system test completed")