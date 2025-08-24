#!/usr/bin/env python3
"""
Tone Mapping and Gamma Correction - T21
=======================================

Implements tone mapping and gamma correction for proper HDR to LDR conversion.
Features ACES filmic tone mapping and configurable gamma correction.

Features:
- Multiple tone mapping operators (ACES, Reinhard, Filmic)
- Gamma correction with configurable gamma value
- Exposure control for HDR scenes
- Color grading and saturation adjustment
- Performance optimized implementations
- Integration with lighting pipeline

Usage:
    from tone_mapping import ToneMapper
    
    tone_mapper = ToneMapper(gamma=2.2, tone_operator='aces')
    ldr_image = tone_mapper.tone_map(hdr_image, exposure=1.0)
"""

import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import OpenGL for GPU tone mapping
try:
    import OpenGL.GL as gl
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


class ToneOperator(Enum):
    """Tone mapping operator types"""
    REINHARD = "reinhard"
    ACES = "aces"
    FILMIC = "filmic"
    LINEAR = "linear"
    EXPOSURE = "exposure"


@dataclass
class ToneMapConfig:
    """Tone mapping configuration"""
    tone_operator: ToneOperator = ToneOperator.ACES
    gamma: float = 2.2                    # Gamma correction value
    exposure: float = 1.0                 # Exposure adjustment
    white_point: float = 1.0              # White point for tone mapping
    contrast: float = 1.0                 # Contrast adjustment
    saturation: float = 1.0               # Color saturation
    lift: float = 0.0                     # Shadow lift
    gain: float = 1.0                     # Highlight gain
    temperature: float = 6500.0           # Color temperature (K)


class ToneMapper:
    """
    T21: Tone mapping and gamma correction system
    """
    
    def __init__(self, config: ToneMapConfig = None):
        """
        Initialize tone mapper
        
        Args:
            config: Tone mapping configuration
        """
        self.config = config or ToneMapConfig()
        
        # Tone mapping operators
        self.operators: Dict[ToneOperator, Callable] = {
            ToneOperator.REINHARD: self._reinhard_tone_mapping,
            ToneOperator.ACES: self._aces_tone_mapping,
            ToneOperator.FILMIC: self._filmic_tone_mapping,
            ToneOperator.LINEAR: self._linear_tone_mapping,
            ToneOperator.EXPOSURE: self._exposure_tone_mapping
        }
        
        # Color temperature conversion matrices
        self.temp_matrices = self._generate_temperature_matrices()
        
        # Statistics
        self.stats = {
            'tone_operator': self.config.tone_operator.value,
            'gamma': self.config.gamma,
            'last_process_time_ms': 0.0,
            'pixels_processed': 0,
            'dynamic_range_input': 0.0,
            'dynamic_range_output': 0.0
        }
        
        print(f"✅ Tone mapper initialized: {self.config.tone_operator.value}, gamma {self.config.gamma}")
    
    def tone_map(self, hdr_image: np.ndarray, exposure: Optional[float] = None) -> np.ndarray:
        """
        Apply tone mapping to HDR image
        
        Args:
            hdr_image: HDR image array (height x width x 3) with values in [0, inf]
            exposure: Override exposure value (uses config if None)
            
        Returns:
            LDR image array (height x width x 3) with values in [0, 1]
        """
        import time
        start_time = time.time()
        
        # Use provided exposure or config default
        current_exposure = exposure if exposure is not None else self.config.exposure
        
        # Input statistics
        input_min, input_max = np.min(hdr_image), np.max(hdr_image)
        self.stats['dynamic_range_input'] = input_max / max(input_min, 1e-8)
        
        # Apply exposure adjustment
        exposed_image = hdr_image * (2.0 ** current_exposure)
        
        # Apply color temperature adjustment
        temp_adjusted = self._apply_color_temperature(exposed_image)
        
        # Apply tone mapping operator
        tone_operator = self.operators[self.config.tone_operator]
        tone_mapped = tone_operator(temp_adjusted)
        
        # Apply color grading
        graded_image = self._apply_color_grading(tone_mapped)
        
        # Apply gamma correction
        gamma_corrected = self._apply_gamma_correction(graded_image)
        
        # Clamp to valid range
        final_image = np.clip(gamma_corrected, 0.0, 1.0)
        
        # Output statistics
        output_min, output_max = np.min(final_image), np.max(final_image)
        self.stats['dynamic_range_output'] = output_max / max(output_min, 1e-8)
        self.stats['pixels_processed'] = hdr_image.size
        
        # Update timing
        process_time = (time.time() - start_time) * 1000
        self.stats['last_process_time_ms'] = process_time
        
        return final_image
    
    def _reinhard_tone_mapping(self, hdr_image: np.ndarray) -> np.ndarray:
        """
        Reinhard tone mapping operator
        
        Args:
            hdr_image: HDR image
            
        Returns:
            Tone mapped image
        """
        # Simple Reinhard: L_out = L_in / (1 + L_in)
        luminance = self._rgb_to_luminance(hdr_image)
        scale_factor = luminance / (1.0 + luminance / self.config.white_point)
        
        # Avoid division by zero
        safe_luminance = np.maximum(luminance, 1e-8)
        scale_factor = scale_factor / safe_luminance
        
        # Apply scaling to each channel
        tone_mapped = hdr_image * np.expand_dims(scale_factor, axis=2)
        
        return tone_mapped
    
    def _aces_tone_mapping(self, hdr_image: np.ndarray) -> np.ndarray:
        """
        ACES filmic tone mapping operator
        
        Args:
            hdr_image: HDR image
            
        Returns:
            Tone mapped image
        """
        # ACES RRT/ODT approximation
        # https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
        
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        
        # Apply ACES curve to each pixel
        numerator = hdr_image * (a * hdr_image + b)
        denominator = hdr_image * (c * hdr_image + d) + e
        
        # Avoid division by zero
        safe_denominator = np.maximum(denominator, 1e-8)
        tone_mapped = numerator / safe_denominator
        
        return tone_mapped
    
    def _filmic_tone_mapping(self, hdr_image: np.ndarray) -> np.ndarray:
        """
        Filmic tone mapping operator (Hable/Uncharted 2)
        
        Args:
            hdr_image: HDR image
            
        Returns:
            Tone mapped image
        """
        # Uncharted 2 filmic tone mapping
        A = 0.15  # Shoulder strength
        B = 0.50  # Linear strength
        C = 0.10  # Linear angle
        D = 0.20  # Toe strength
        E = 0.02  # Toe numerator
        F = 0.30  # Toe denominator
        
        def filmic_curve(x):
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
        
        # Apply curve
        tone_mapped = filmic_curve(hdr_image)
        
        # Normalize by white point
        white_scale = 1.0 / filmic_curve(self.config.white_point)
        tone_mapped = tone_mapped * white_scale
        
        return tone_mapped
    
    def _linear_tone_mapping(self, hdr_image: np.ndarray) -> np.ndarray:
        """
        Linear tone mapping (simple clamp)
        
        Args:
            hdr_image: HDR image
            
        Returns:
            Tone mapped image
        """
        return np.clip(hdr_image / self.config.white_point, 0.0, 1.0)
    
    def _exposure_tone_mapping(self, hdr_image: np.ndarray) -> np.ndarray:
        """
        Simple exposure-based tone mapping
        
        Args:
            hdr_image: HDR image
            
        Returns:
            Tone mapped image
        """
        # Simple exposure mapping: 1 - exp(-exposure * hdr)
        tone_mapped = 1.0 - np.exp(-self.config.exposure * hdr_image)
        return tone_mapped
    
    def _apply_color_temperature(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color temperature adjustment
        
        Args:
            image: Input image
            
        Returns:
            Temperature-adjusted image
        """
        if abs(self.config.temperature - 6500.0) < 10.0:
            return image  # No adjustment needed for neutral temperature
        
        # Simple color temperature adjustment using RGB scaling
        # This is a simplified version - full implementation would use proper color science
        temp_factor = self.config.temperature / 6500.0
        
        if temp_factor > 1.0:
            # Warmer (more red/yellow)
            temp_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0, temp_factor, 0.0],
                [0.0, 0.0, 1.0 / temp_factor]
            ])
        else:
            # Cooler (more blue)
            temp_matrix = np.array([
                [1.0 / temp_factor, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, temp_factor]
            ])
        
        # Apply temperature matrix
        original_shape = image.shape
        flat_image = image.reshape(-1, 3)
        adjusted_image = (temp_matrix @ flat_image.T).T
        
        return adjusted_image.reshape(original_shape)
    
    def _apply_color_grading(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color grading (contrast, saturation, lift, gain)
        
        Args:
            image: Input image
            
        Returns:
            Color graded image
        """
        graded = np.copy(image)
        
        # Apply lift (raise shadows)
        graded = graded + self.config.lift
        
        # Apply gain (multiply highlights)
        graded = graded * self.config.gain
        
        # Apply contrast
        if self.config.contrast != 1.0:
            # Pivot contrast around 0.5
            graded = (graded - 0.5) * self.config.contrast + 0.5
        
        # Apply saturation
        if self.config.saturation != 1.0:
            # Convert to HSV-like saturation adjustment
            luminance = self._rgb_to_luminance(graded)
            luminance_expanded = np.expand_dims(luminance, axis=2)
            
            # Lerp between desaturated and original based on saturation
            graded = luminance_expanded + (graded - luminance_expanded) * self.config.saturation
        
        return graded
    
    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction
        
        Args:
            image: Linear image
            
        Returns:
            Gamma corrected image
        """
        if abs(self.config.gamma - 1.0) < 0.01:
            return image  # No gamma correction needed
        
        # Apply gamma: output = input^(1/gamma)
        gamma_corrected = np.power(np.maximum(image, 0.0), 1.0 / self.config.gamma)
        
        return gamma_corrected
    
    def _rgb_to_luminance(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB to luminance using Rec.709 weights
        
        Args:
            rgb_image: RGB image
            
        Returns:
            Luminance values
        """
        # Rec.709 luminance weights
        weights = np.array([0.2126, 0.7152, 0.0722])
        luminance = np.dot(rgb_image, weights)
        
        return luminance
    
    def _generate_temperature_matrices(self) -> Dict[float, np.ndarray]:
        """Generate color temperature conversion matrices"""
        # Simplified temperature matrices - full implementation would use proper color science
        return {
            3000.0: np.array([[1.2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.8]]),  # Warm
            6500.0: np.eye(3),  # Neutral
            9000.0: np.array([[0.8, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.2]])   # Cool
        }
    
    def create_lut(self, size: int = 32) -> np.ndarray:
        """
        Create 3D lookup table for fast tone mapping
        
        Args:
            size: LUT size per dimension (size^3 total entries)
            
        Returns:
            3D LUT array (size x size x size x 3)
        """
        lut = np.zeros((size, size, size, 3), dtype=np.float32)
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Convert indices to HDR values (0 to 4 range for HDR)
                    input_rgb = np.array([
                        (r / (size - 1)) * 4.0,
                        (g / (size - 1)) * 4.0,
                        (b / (size - 1)) * 4.0
                    ])
                    
                    # Apply tone mapping
                    output_rgb = self.tone_map(input_rgb.reshape(1, 1, 3)).flatten()
                    
                    lut[r, g, b] = output_rgb
        
        return lut
    
    def apply_lut(self, hdr_image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Apply 3D LUT for fast tone mapping
        
        Args:
            hdr_image: HDR input image
            lut: 3D lookup table
            
        Returns:
            Tone mapped image using LUT
        """
        height, width, channels = hdr_image.shape
        lut_size = lut.shape[0]
        
        # Scale HDR values to LUT range
        scaled_image = np.clip(hdr_image / 4.0 * (lut_size - 1), 0, lut_size - 1)
        
        # Trilinear interpolation through LUT
        output_image = np.zeros_like(hdr_image)
        
        for y in range(height):
            for x in range(width):
                rgb = scaled_image[y, x]
                
                # Get integer and fractional parts
                r0, g0, b0 = int(rgb[0]), int(rgb[1]), int(rgb[2])
                r1 = min(r0 + 1, lut_size - 1)
                g1 = min(g0 + 1, lut_size - 1)
                b1 = min(b0 + 1, lut_size - 1)
                
                fr, fg, fb = rgb[0] - r0, rgb[1] - g0, rgb[2] - b0
                
                # Trilinear interpolation
                c000 = lut[r0, g0, b0]
                c001 = lut[r0, g0, b1]
                c010 = lut[r0, g1, b0]
                c011 = lut[r0, g1, b1]
                c100 = lut[r1, g0, b0]
                c101 = lut[r1, g0, b1]
                c110 = lut[r1, g1, b0]
                c111 = lut[r1, g1, b1]
                
                c00 = c000 * (1 - fb) + c001 * fb
                c01 = c010 * (1 - fb) + c011 * fb
                c10 = c100 * (1 - fb) + c101 * fb
                c11 = c110 * (1 - fb) + c111 * fb
                
                c0 = c00 * (1 - fg) + c01 * fg
                c1 = c10 * (1 - fg) + c11 * fg
                
                final_color = c0 * (1 - fr) + c1 * fr
                
                output_image[y, x] = final_color
        
        return output_image
    
    def get_tone_mapping_statistics(self) -> Dict[str, Any]:
        """Get tone mapping statistics"""
        return self.stats.copy()


def create_lighting_sanity_scene() -> Dict[str, Any]:
    """
    Create lighting sanity scene for T21 validation
    
    Returns:
        Scene dictionary with test objects and lighting setup
    """
    scene = {
        'objects': [
            {
                'type': 'sphere',
                'position': [0.0, 0.0, 0.0],
                'radius': 1.0,
                'material': 'test_material'
            },
            {
                'type': 'cube',
                'position': [2.0, 0.0, 0.0],
                'size': [1.0, 1.0, 1.0],
                'material': 'test_material'
            }
        ],
        'lighting': {
            'sun': {
                'direction': [0.5, -1.0, 0.3],
                'color': [1.0, 0.95, 0.8],
                'intensity': 3.0
            },
            'ambient': {
                'color': [0.1, 0.15, 0.2],
                'intensity': 0.3
            },
            'sky': {
                'color': [0.5, 0.7, 1.0],
                'intensity': 0.5
            }
        },
        'camera': {
            'position': [0.0, -5.0, 2.0],
            'target': [0.0, 0.0, 0.0],
            'fov': 60.0
        }
    }
    
    return scene


def validate_tone_mapping_pipeline():
    """
    Validate the complete T21 tone mapping pipeline
    """
    print("🎨 T21 Tone Mapping Pipeline Validation")
    print("=" * 50)
    
    # Test different tone operators
    operators = [ToneOperator.ACES, ToneOperator.REINHARD, ToneOperator.FILMIC]
    
    for operator in operators:
        config = ToneMapConfig(tone_operator=operator, gamma=2.2, exposure=0.5)
        tone_mapper = ToneMapper(config)
        
        print(f"\nTesting {operator.value.upper()} tone mapping:")
        
        # Create test HDR image
        height, width = 256, 256
        hdr_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Create gradient with high dynamic range
        for y in range(height):
            for x in range(width):
                # Exponential gradient
                intensity = np.exp((x / width) * 4.0)  # 0 to ~55 range
                hdr_image[y, x] = [intensity, intensity * 0.9, intensity * 0.8]
        
        # Apply tone mapping
        ldr_image = tone_mapper.tone_map(hdr_image)
        
        # Print statistics
        stats = tone_mapper.get_tone_mapping_statistics()
        print(f"   Input range: {hdr_image.min():.3f} - {hdr_image.max():.3f}")
        print(f"   Output range: {ldr_image.min():.3f} - {ldr_image.max():.3f}")
        print(f"   Process time: {stats['last_process_time_ms']:.2f}ms")
        print(f"   Dynamic range reduction: {stats['dynamic_range_input']:.1f} -> {stats['dynamic_range_output']:.1f}")
    
    print("\n✅ Tone mapping pipeline validation complete")


if __name__ == "__main__":
    # Run tone mapping validation
    validate_tone_mapping_pipeline()
    
    # Test LUT creation
    print("\n📊 Testing 3D LUT generation...")
    config = ToneMapConfig(tone_operator=ToneOperator.ACES)
    tone_mapper = ToneMapper(config)
    
    lut = tone_mapper.create_lut(size=16)  # Small LUT for testing
    print(f"   LUT shape: {lut.shape}")
    print(f"   LUT range: {lut.min():.3f} - {lut.max():.3f}")
    
    print("\n✅ T21 tone mapping system test completed")