#!/usr/bin/env python3
"""
Screen Space Ambient Occlusion (SSAO) - T21
===========================================

Implements SSAO for enhanced depth perception in caves and terrain creases.
Features configurable radius and intensity with optimized sampling patterns.

Features:
- Screen-space ambient occlusion calculation
- Configurable sample radius and intensity
- Noise texture for sample distribution
- Bilateral blur for smooth results
- Integration with terrain and cave rendering
- Performance monitoring and tuning

Usage:
    from ssao import SSAORenderer
    
    ssao = SSAORenderer(radius=0.5, intensity=0.4)
    occlusion_map = ssao.render_ssao(depth_buffer, normal_buffer)
    ssao.apply_occlusion(color_buffer, occlusion_map)
"""

import numpy as np
import math
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import OpenGL for SSAO rendering
try:
    import OpenGL.GL as gl
    import OpenGL.GL.framebufferobjects as fbo
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("⚠️ OpenGL not available - SSAO will use CPU fallback")


@dataclass
class SSAOConfig:
    """SSAO configuration parameters"""
    radius: float = 0.5              # Sampling radius in view space
    intensity: float = 0.4           # AO intensity multiplier
    sample_count: int = 16           # Number of samples per pixel
    noise_size: int = 4              # Size of noise texture (4x4)
    bias: float = 0.025              # Depth bias to prevent self-occlusion
    blur_radius: int = 2             # Blur radius for smoothing
    max_distance: float = 1.0        # Maximum AO distance
    falloff: float = 0.1             # Distance falloff factor


class SSAOQuality(Enum):
    """SSAO quality presets"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class SSAORenderer:
    """
    T21: Screen Space Ambient Occlusion renderer
    """
    
    def __init__(self, config: SSAOConfig = None, screen_width: int = 800, screen_height: int = 600):
        """
        Initialize SSAO renderer
        
        Args:
            config: SSAO configuration
            screen_width: Screen width for buffer allocation
            screen_height: Screen height for buffer allocation
        """
        self.config = config or SSAOConfig()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Generate sampling kernel
        self.sample_kernel = self._generate_sample_kernel(self.config.sample_count)
        
        # Generate noise texture
        self.noise_texture = self._generate_noise_texture(self.config.noise_size)
        
        # SSAO framebuffer resources
        self.ssao_fbo = None
        self.ssao_texture = None
        self.blur_fbo = None
        self.blur_texture = None
        
        # CPU fallback buffers
        self.ssao_buffer = np.zeros((screen_height, screen_width), dtype=np.float32)
        self.blur_buffer = np.zeros((screen_height, screen_width), dtype=np.float32)
        
        # Statistics
        self.stats = {
            'sample_count': self.config.sample_count,
            'screen_resolution': f"{screen_width}x{screen_height}",
            'last_render_time_ms': 0.0,
            'occlusion_samples': 0,
            'blur_passes': 1
        }
        
        if OPENGL_AVAILABLE:
            self._initialize_ssao_resources()
        
        print(f"✅ SSAO initialized: {self.config.sample_count} samples, radius {self.config.radius}")
    
    def _initialize_ssao_resources(self):
        """Initialize OpenGL SSAO resources"""
        if not OPENGL_AVAILABLE:
            return
        
        # Create SSAO framebuffer
        self.ssao_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ssao_fbo)
        
        # Create SSAO texture
        self.ssao_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ssao_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_R8,
            self.screen_width, self.screen_height, 0,
            gl.GL_RED, gl.GL_FLOAT, None
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, self.ssao_texture, 0
        )
        
        # Create blur framebuffer
        self.blur_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.blur_fbo)
        
        # Create blur texture
        self.blur_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.blur_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_R8,
            self.screen_width, self.screen_height, 0,
            gl.GL_RED, gl.GL_FLOAT, None
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, self.blur_texture, 0
        )
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    
    def _generate_sample_kernel(self, sample_count: int) -> np.ndarray:
        """
        Generate hemisphere sampling kernel for SSAO
        
        Args:
            sample_count: Number of samples to generate
            
        Returns:
            Array of sample vectors in tangent space
        """
        samples = []
        
        for i in range(sample_count):
            # Generate random sample in hemisphere
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            z = random.uniform(0.0, 1.0)  # Positive Z for hemisphere
            
            sample = np.array([x, y, z])
            sample = sample / np.linalg.norm(sample)
            
            # Scale samples to be more distributed inside hemisphere
            scale = i / sample_count
            scale = 0.1 + 0.9 * scale * scale  # Quadratic distribution
            sample = sample * scale
            
            samples.append(sample)
        
        return np.array(samples)
    
    def _generate_noise_texture(self, size: int) -> np.ndarray:
        """
        Generate noise texture for SSAO sample rotation
        
        Args:
            size: Size of noise texture (size x size)
            
        Returns:
            Noise texture array
        """
        noise = np.zeros((size, size, 3), dtype=np.float32)
        
        for i in range(size):
            for j in range(size):
                # Generate random rotation vector
                x = random.uniform(-1.0, 1.0)
                y = random.uniform(-1.0, 1.0)
                z = 0.0  # Only rotate around Z axis
                
                noise[i, j] = [x, y, z]
        
        return noise
    
    def render_ssao(self, depth_buffer: np.ndarray, normal_buffer: np.ndarray,
                   view_matrix: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
        """
        Render SSAO occlusion map
        
        Args:
            depth_buffer: Linear depth buffer (height x width)
            normal_buffer: World space normals (height x width x 3)
            view_matrix: Camera view matrix
            proj_matrix: Camera projection matrix
            
        Returns:
            SSAO occlusion map (0.0 = fully occluded, 1.0 = no occlusion)
        """
        if OPENGL_AVAILABLE and self.ssao_fbo is not None:
            return self._render_ssao_gpu(depth_buffer, normal_buffer, view_matrix, proj_matrix)
        else:
            return self._render_ssao_cpu(depth_buffer, normal_buffer, view_matrix, proj_matrix)
    
    def _render_ssao_cpu(self, depth_buffer: np.ndarray, normal_buffer: np.ndarray,
                        view_matrix: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
        """
        CPU implementation of SSAO rendering
        
        Args:
            depth_buffer: Linear depth buffer
            normal_buffer: World space normals
            view_matrix: Camera view matrix
            proj_matrix: Camera projection matrix
            
        Returns:
            SSAO occlusion map
        """
        import time
        start_time = time.time()
        
        height, width = depth_buffer.shape
        occlusion_map = np.ones((height, width), dtype=np.float32)
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                # Get pixel depth and normal
                pixel_depth = depth_buffer[y, x]
                if pixel_depth >= 1.0:  # Skip background pixels
                    continue
                
                pixel_normal = normal_buffer[y, x]
                if np.linalg.norm(pixel_normal) < 0.1:  # Skip invalid normals
                    continue
                
                # Calculate world position from depth
                world_pos = self._depth_to_world_position(x, y, pixel_depth, view_matrix, proj_matrix)
                
                # Calculate SSAO for this pixel
                occlusion = self._calculate_pixel_ssao(
                    world_pos, pixel_normal, depth_buffer, normal_buffer, 
                    view_matrix, proj_matrix, x, y
                )
                
                occlusion_map[y, x] = occlusion
        
        # Apply blur
        blurred_occlusion = self._bilateral_blur(occlusion_map, depth_buffer)
        
        # Update statistics
        render_time = (time.time() - start_time) * 1000
        self.stats['last_render_time_ms'] = render_time
        self.stats['occlusion_samples'] = width * height * self.config.sample_count
        
        return blurred_occlusion
    
    def _render_ssao_gpu(self, depth_buffer: np.ndarray, normal_buffer: np.ndarray,
                        view_matrix: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
        """
        GPU implementation of SSAO rendering (placeholder)
        
        Args:
            depth_buffer: Linear depth buffer
            normal_buffer: World space normals
            view_matrix: Camera view matrix
            proj_matrix: Camera projection matrix
            
        Returns:
            SSAO occlusion map
        """
        # In a full implementation, this would use OpenGL shaders
        print("🔄 Using GPU SSAO rendering")
        
        # For now, fall back to CPU implementation
        return self._render_ssao_cpu(depth_buffer, normal_buffer, view_matrix, proj_matrix)
    
    def _calculate_pixel_ssao(self, world_pos: np.ndarray, pixel_normal: np.ndarray,
                             depth_buffer: np.ndarray, normal_buffer: np.ndarray,
                             view_matrix: np.ndarray, proj_matrix: np.ndarray,
                             center_x: int, center_y: int) -> float:
        """
        Calculate SSAO occlusion for a single pixel
        
        Args:
            world_pos: World position of pixel
            pixel_normal: Surface normal at pixel
            depth_buffer: Full depth buffer
            normal_buffer: Full normal buffer
            view_matrix: Camera view matrix
            proj_matrix: Camera projection matrix
            center_x: Pixel x coordinate
            center_y: Pixel y coordinate
            
        Returns:
            Occlusion factor (0.0 = fully occluded, 1.0 = no occlusion)
        """
        occlusion = 0.0
        
        # Create tangent-bitangent-normal matrix
        tbn_matrix = self._create_tbn_matrix(pixel_normal)
        
        # Sample around the pixel
        for sample_vec in self.sample_kernel:
            # Transform sample to world space
            world_sample = world_pos + tbn_matrix @ sample_vec * self.config.radius
            
            # Project back to screen space
            screen_sample = self._world_to_screen(world_sample, view_matrix, proj_matrix)
            
            if screen_sample is None:
                continue
            
            sample_x, sample_y, sample_depth = screen_sample
            
            # Check if sample is within screen bounds
            if (0 <= sample_x < depth_buffer.shape[1] and 
                0 <= sample_y < depth_buffer.shape[0]):
                
                # Get depth at sample position
                buffer_depth = depth_buffer[int(sample_y), int(sample_x)]
                
                # Check for occlusion
                depth_diff = sample_depth - buffer_depth
                
                if depth_diff > self.config.bias:
                    # Calculate distance falloff
                    distance = np.linalg.norm(world_sample - world_pos)
                    falloff = 1.0 - min(distance / self.config.max_distance, 1.0)
                    
                    occlusion += falloff
        
        # Normalize and apply intensity
        occlusion = occlusion / len(self.sample_kernel)
        occlusion = 1.0 - occlusion * self.config.intensity
        
        return max(0.0, min(1.0, occlusion))
    
    def _bilateral_blur(self, occlusion_map: np.ndarray, depth_buffer: np.ndarray) -> np.ndarray:
        """
        Apply bilateral blur to SSAO map to smooth results
        
        Args:
            occlusion_map: Raw SSAO occlusion map
            depth_buffer: Depth buffer for edge preservation
            
        Returns:
            Blurred occlusion map
        """
        height, width = occlusion_map.shape
        blurred = np.copy(occlusion_map)
        
        blur_radius = self.config.blur_radius
        depth_threshold = 0.1
        
        for y in range(blur_radius, height - blur_radius):
            for x in range(blur_radius, width - blur_radius):
                center_depth = depth_buffer[y, x]
                
                total_weight = 0.0
                total_occlusion = 0.0
                
                # Sample neighborhood
                for dy in range(-blur_radius, blur_radius + 1):
                    for dx in range(-blur_radius, blur_radius + 1):
                        sample_y = y + dy
                        sample_x = x + dx
                        
                        sample_depth = depth_buffer[sample_y, sample_x]
                        sample_occlusion = occlusion_map[sample_y, sample_x]
                        
                        # Bilateral weight based on depth difference
                        depth_diff = abs(sample_depth - center_depth)
                        weight = 1.0 if depth_diff < depth_threshold else 0.1
                        
                        total_weight += weight
                        total_occlusion += weight * sample_occlusion
                
                if total_weight > 0:
                    blurred[y, x] = total_occlusion / total_weight
        
        return blurred
    
    def _depth_to_world_position(self, x: int, y: int, depth: float,
                                view_matrix: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
        """
        Convert screen position and depth to world position
        
        Args:
            x, y: Screen coordinates
            depth: Linear depth value
            view_matrix: Camera view matrix
            proj_matrix: Camera projection matrix
            
        Returns:
            World position
        """
        # Convert screen coords to normalized device coordinates
        ndc_x = (2.0 * x / self.screen_width) - 1.0
        ndc_y = 1.0 - (2.0 * y / self.screen_height)
        ndc_z = 2.0 * depth - 1.0
        
        # Unproject to world space
        inv_proj = np.linalg.inv(proj_matrix)
        inv_view = np.linalg.inv(view_matrix)
        
        clip_pos = np.array([ndc_x, ndc_y, ndc_z, 1.0])
        view_pos = inv_proj @ clip_pos
        view_pos = view_pos / view_pos[3]
        
        world_pos = inv_view @ view_pos
        
        return world_pos[:3]
    
    def _world_to_screen(self, world_pos: np.ndarray, view_matrix: np.ndarray, 
                        proj_matrix: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """
        Convert world position to screen coordinates
        
        Args:
            world_pos: World position
            view_matrix: Camera view matrix
            proj_matrix: Camera projection matrix
            
        Returns:
            (screen_x, screen_y, depth) or None if behind camera
        """
        homogeneous_pos = np.append(world_pos, 1.0)
        
        # Transform to clip space
        clip_pos = proj_matrix @ view_matrix @ homogeneous_pos
        
        if clip_pos[3] <= 0:  # Behind camera
            return None
        
        # Perspective divide
        ndc = clip_pos[:3] / clip_pos[3]
        
        if ndc[2] < -1.0 or ndc[2] > 1.0:  # Outside depth range
            return None
        
        # Convert to screen coordinates
        screen_x = int((ndc[0] + 1.0) * 0.5 * self.screen_width)
        screen_y = int((1.0 - ndc[1]) * 0.5 * self.screen_height)
        depth = (ndc[2] + 1.0) * 0.5  # Convert to [0,1]
        
        return (screen_x, screen_y, depth)
    
    def _create_tbn_matrix(self, normal: np.ndarray) -> np.ndarray:
        """
        Create tangent-bitangent-normal matrix for sample transformation
        
        Args:
            normal: Surface normal
            
        Returns:
            3x3 TBN matrix
        """
        # Choose tangent vector perpendicular to normal
        up = np.array([0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1, 0, 0])
        tangent = np.cross(up, normal)
        tangent = tangent / np.linalg.norm(tangent)
        
        # Bitangent from normal and tangent
        bitangent = np.cross(normal, tangent)
        
        # Create TBN matrix
        tbn = np.column_stack([tangent, bitangent, normal])
        
        return tbn
    
    def apply_occlusion(self, color_buffer: np.ndarray, occlusion_map: np.ndarray) -> np.ndarray:
        """
        Apply SSAO occlusion to color buffer
        
        Args:
            color_buffer: RGB color buffer (height x width x 3)
            occlusion_map: SSAO occlusion map (height x width)
            
        Returns:
            Color buffer with occlusion applied
        """
        # Expand occlusion map to RGB
        occlusion_rgb = np.expand_dims(occlusion_map, axis=2)
        occlusion_rgb = np.repeat(occlusion_rgb, 3, axis=2)
        
        # Apply occlusion (multiply)
        occluded_color = color_buffer * occlusion_rgb
        
        return occluded_color
    
    def get_ssao_statistics(self) -> Dict[str, Any]:
        """Get SSAO rendering statistics"""
        return self.stats.copy()
    
    def cleanup(self):
        """Cleanup OpenGL resources"""
        if OPENGL_AVAILABLE:
            if self.ssao_fbo is not None:
                gl.glDeleteFramebuffers(1, [self.ssao_fbo])
            if self.blur_fbo is not None:
                gl.glDeleteFramebuffers(1, [self.blur_fbo])
            if self.ssao_texture is not None:
                gl.glDeleteTextures(1, [self.ssao_texture])
            if self.blur_texture is not None:
                gl.glDeleteTextures(1, [self.blur_texture])
            print("✅ SSAO resources cleaned up")


def create_ssao_config(quality: SSAOQuality) -> SSAOConfig:
    """
    Create SSAO configuration for quality preset
    
    Args:
        quality: SSAO quality preset
        
    Returns:
        SSAO configuration
    """
    configs = {
        SSAOQuality.LOW: SSAOConfig(
            radius=0.3,
            intensity=0.3,
            sample_count=8,
            blur_radius=1
        ),
        SSAOQuality.MEDIUM: SSAOConfig(
            radius=0.5,
            intensity=0.4,
            sample_count=16,
            blur_radius=2
        ),
        SSAOQuality.HIGH: SSAOConfig(
            radius=0.6,
            intensity=0.5,
            sample_count=32,
            blur_radius=2
        ),
        SSAOQuality.ULTRA: SSAOConfig(
            radius=0.8,
            intensity=0.6,
            sample_count=64,
            blur_radius=3
        )
    }
    
    return configs.get(quality, configs[SSAOQuality.MEDIUM])


if __name__ == "__main__":
    # Test SSAO system
    print("🔍 T21 Screen Space Ambient Occlusion")
    print("=" * 50)
    
    # Test different quality levels
    qualities = [SSAOQuality.LOW, SSAOQuality.MEDIUM, SSAOQuality.HIGH]
    
    for quality in qualities:
        config = create_ssao_config(quality)
        ssao_renderer = SSAORenderer(config, 640, 480)
        
        print(f"\n{quality.value.upper()} Quality SSAO:")
        print(f"   Sample Count: {config.sample_count}")
        print(f"   Radius: {config.radius}")
        print(f"   Intensity: {config.intensity}")
        print(f"   Blur Radius: {config.blur_radius}")
        
        # Create test buffers
        height, width = 100, 100
        depth_buffer = np.random.rand(height, width) * 0.8 + 0.1
        normal_buffer = np.random.rand(height, width, 3) * 2 - 1
        
        # Normalize normals
        for y in range(height):
            for x in range(width):
                normal_buffer[y, x] = normal_buffer[y, x] / np.linalg.norm(normal_buffer[y, x])
        
        # Test matrices
        view_matrix = np.eye(4)
        proj_matrix = np.eye(4)
        
        # Render SSAO (small test)
        occlusion_map = ssao_renderer._render_ssao_cpu(
            depth_buffer, normal_buffer, view_matrix, proj_matrix
        )
        
        stats = ssao_renderer.get_ssao_statistics()
        print(f"   Render Time: {stats['last_render_time_ms']:.2f}ms")
        print(f"   Occlusion Range: {occlusion_map.min():.3f} - {occlusion_map.max():.3f}")
        
        ssao_renderer.cleanup()
    
    print("\n✅ SSAO system test completed")