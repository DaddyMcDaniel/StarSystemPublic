#!/usr/bin/env python3
"""
Terrain Quality Validator - T19-T22 Validation
==============================================

Comprehensive validation system for all terrain generation improvements.
Tests mesh density, LOD transitions, cave quality, lighting, and normal/tangent quality.

Features:
- T19: LOD transition quality validation
- T20: Marching Cubes and cave seam validation
- T21: Lighting system validation (shadows, SSAO, tone mapping)
- T22: Normal/tangent quality validation after displacement
- Performance metrics and regression testing
- Visual quality scoring system

Usage:
    from terrain_quality_validator import TerrainQualityValidator
    
    validator = TerrainQualityValidator()
    results = validator.validate_all_improvements()
    validator.generate_quality_report(results)
"""

import numpy as np
import math
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import terrain system components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mesh'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lighting'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'marching_cubes'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sdf'))

from mesh.runtime_lod import RuntimeLODManager, LODLevel
from mesh.quadtree_chunking import QuadtreeChunker, generate_chunked_planet
from marching_cubes.marching_cubes import MarchingCubes
from mesh.shading_basis import (compute_angle_weighted_normals, compute_tangent_basis, 
                                validate_seam_consistency, validate_shading_basis)
from lighting.lighting_system import LightingSystem, LightingConfig
from lighting.shadow_mapping import ShadowQuality
from lighting.ssao import SSAOQuality
from lighting.tone_mapping import ToneOperator
from sdf.sdf_evaluator import ChunkBounds, VoxelGrid


class ValidationResult(Enum):
    """Validation result categories"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class QualityMetrics:
    """Quality metrics for terrain validation"""
    triangle_density: float = 0.0         # Triangles per unit area
    lod_transition_smoothness: float = 0.0 # LOD transition quality (0-1)
    seam_consistency: float = 0.0          # UV seam quality (0-1)
    normal_quality: float = 0.0            # Normal vector quality (0-1)
    tangent_quality: float = 0.0           # Tangent space quality (0-1)
    shadow_quality: float = 0.0            # Shadow mapping quality (0-1)
    ssao_quality: float = 0.0              # SSAO quality (0-1)
    lighting_coherence: float = 0.0        # Overall lighting quality (0-1)
    performance_score: float = 0.0         # Performance metric (0-1)
    overall_score: float = 0.0             # Weighted overall score


class TerrainQualityValidator:
    """
    Comprehensive terrain quality validation system for T19-T22
    """
    
    def __init__(self):
        """Initialize terrain quality validator"""
        self.validation_results: Dict[str, Any] = {}
        
        # Quality thresholds
        self.quality_thresholds = {
            ValidationResult.EXCELLENT: 0.9,
            ValidationResult.GOOD: 0.75,
            ValidationResult.ACCEPTABLE: 0.6,
            ValidationResult.POOR: 0.4
        }
        
        # Metric weights for overall score
        self.metric_weights = {
            'triangle_density': 0.15,
            'lod_transition_smoothness': 0.15,
            'seam_consistency': 0.15,
            'normal_quality': 0.15,
            'tangent_quality': 0.15,
            'shadow_quality': 0.05,
            'ssao_quality': 0.05,
            'lighting_coherence': 0.1,
            'performance_score': 0.05
        }
        
        print("🔍 Terrain Quality Validator initialized")
        print("   Validating improvements: T19, T20, T21, T22")
    
    def validate_all_improvements(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of all terrain improvements
        
        Returns:
            Validation results dictionary
        """
        print("🧪 Running comprehensive terrain quality validation...")
        print("=" * 60)
        
        results = {}
        
        # T19: LOD and mesh density validation
        print("\n📊 T19: Validating LOD transitions and mesh density...")
        results['t19'] = self._validate_lod_improvements()
        
        # T20: Marching Cubes and cave quality validation
        print("\n🕳️ T20: Validating Marching Cubes and cave quality...")
        results['t20'] = self._validate_cave_improvements()
        
        # T21: Lighting system validation
        print("\n🌅 T21: Validating lighting system...")
        results['t21'] = self._validate_lighting_improvements()
        
        # T22: Normal/tangent quality validation
        print("\n🔢 T22: Validating normal/tangent quality...")
        results['t22'] = self._validate_shading_improvements()
        
        # Overall quality assessment
        print("\n📈 Computing overall quality metrics...")
        results['overall'] = self._compute_overall_quality(results)
        
        self.validation_results = results
        return results
    
    def _validate_lod_improvements(self) -> Dict[str, Any]:
        """Validate T19 LOD and mesh density improvements"""
        # Create test LOD manager
        lod_manager = RuntimeLODManager(
            lod_distance_bands=[3.5, 10.0, 24.0, 60.0],  # T19 improved distances
            screen_error_thresholds=[0.3, 1.0, 4.0, 16.0],  # T19 improved thresholds
            chunk_res_per_lod={
                LODLevel.LOD0: 128,  # T19 increased resolution
                LODLevel.LOD1: 128,
                LODLevel.LOD2: 64,
                LODLevel.LOD3: 32
            }
        )
        
        # Test with sample chunks at different distances
        test_chunks = self._create_test_chunks()
        camera_pos = np.array([0, 0, 5])
        view_matrix = np.eye(4)
        proj_matrix = np.eye(4)
        
        # Test LOD selection
        selected_chunks = lod_manager.select_active_chunks(
            test_chunks, camera_pos, view_matrix, proj_matrix
        )
        
        # Calculate metrics
        lod_stats = lod_manager.get_lod_statistics()
        
        # Analyze triangle density
        total_triangles = lod_stats.get('total_triangles', 0)
        triangle_density = total_triangles / max(len(selected_chunks), 1) / 1000  # Normalize
        
        # Analyze LOD transition smoothness
        lod_histogram = lod_stats.get('lod_histogram', {})
        lod_distribution = [lod_histogram.get(lod, 0) for lod in LODLevel]
        smoothness = self._calculate_lod_smoothness(lod_distribution)
        
        # Performance assessment
        lod_time = lod_stats.get('lod_time_ms', 0)
        performance_score = max(0, 1.0 - lod_time / 10.0)  # Target < 10ms
        
        return {
            'triangle_density': min(1.0, triangle_density),
            'lod_transition_smoothness': smoothness,
            'performance_score': performance_score,
            'median_triangles': lod_stats.get('median_triangle_count', 0),
            'total_chunks': len(selected_chunks),
            'validation_result': self._classify_result((triangle_density + smoothness) / 2)
        }
    
    def _validate_cave_improvements(self) -> Dict[str, Any]:
        """Validate T20 Marching Cubes and cave improvements"""
        # Create enhanced marching cubes with T20 improvements
        mc = MarchingCubes(iso_value=0.0, use_sdf_normals=True)  # T20 enhanced normals
        
        # Create test SDF for cave generation
        test_bounds = self._create_test_chunk_bounds()
        test_sdf = self._create_test_cave_sdf()
        
        # Generate test voxel grid with higher resolution (T20)
# VoxelGrid imported at top
        voxel_grid = VoxelGrid(test_bounds, resolution=64)  # T20 increased from 32
        
        # Sample SDF
        from sdf_evaluator import SDFEvaluator
        evaluator = SDFEvaluator()
        scalar_field = evaluator.sample_voxel_grid(voxel_grid, test_sdf)
        
        # Generate cave mesh
        vertices, triangles = mc.polygonize(voxel_grid, scalar_field, test_sdf)
        
        # Analyze quality
        if len(vertices) == 0:
            return {
                'seam_consistency': 0.0,
                'mesh_quality': 0.0,
                'resolution_improvement': 0.0,
                'validation_result': ValidationResult.FAILED
            }
        
        # Check for duplicate vertices (seam quality)
        position_array = np.array([v.position for v in vertices])
        unique_positions = len(np.unique(position_array.round(decimals=6), axis=0))
        deduplication_ratio = unique_positions / len(vertices)
        
        # Calculate mesh quality metrics
        mesh_quality = self._analyze_mesh_quality(vertices, triangles)
        
        # Resolution improvement (64 vs 32)
        resolution_improvement = 64 / 32 - 1.0  # Should be 1.0 for 100% improvement
        
        return {
            'seam_consistency': deduplication_ratio,
            'mesh_quality': mesh_quality,
            'resolution_improvement': min(1.0, resolution_improvement),
            'vertex_count': len(vertices),
            'triangle_count': len(triangles),
            'validation_result': self._classify_result((deduplication_ratio + mesh_quality) / 2)
        }
    
    def _validate_lighting_improvements(self) -> Dict[str, Any]:
        """Validate T21 lighting system improvements"""
        # Create lighting system with T21 improvements
        config = LightingConfig(
            enable_shadows=True,
            enable_ssao=True,
            enable_tone_mapping=True,
            shadow_quality=ShadowQuality.MEDIUM,
            ssao_quality=SSAOQuality.MEDIUM,
            tone_operator=ToneOperator.ACES
        )
        
        lighting_system = LightingSystem(config, 512, 512)
        
        # Create test scene
        test_scene = lighting_system.create_lighting_sanity_scene()
        test_camera = {
            'position': np.array([0, -8, 4]),
            'view_matrix': np.eye(4),
            'proj_matrix': np.eye(4)
        }
        
        # Test lighting render
        start_time = time.time()
        final_image = lighting_system.render_lighting(test_scene, test_camera)
        render_time = (time.time() - start_time) * 1000
        
        # Analyze lighting quality
        lighting_stats = lighting_system.get_lighting_statistics()
        
        # Shadow quality assessment
        shadow_time = lighting_stats.get('shadow_time_ms', 0)
        shadow_quality = max(0, 1.0 - shadow_time / 5.0)  # Target < 5ms
        
        # SSAO quality assessment
        ssao_time = lighting_stats.get('ssao_time_ms', 0)
        ssao_quality = max(0, 1.0 - ssao_time / 10.0)  # Target < 10ms
        
        # Image quality metrics
        brightness_range = np.max(final_image) - np.min(final_image)
        mean_brightness = np.mean(final_image)
        lighting_coherence = min(1.0, brightness_range * 2.0)  # Good dynamic range
        
        # Performance assessment
        total_time = lighting_stats.get('total_render_time_ms', render_time)
        performance_score = max(0, 1.0 - total_time / 50.0)  # Target < 50ms
        
        lighting_system.cleanup()
        
        return {
            'shadow_quality': shadow_quality,
            'ssao_quality': ssao_quality,
            'lighting_coherence': lighting_coherence,
            'performance_score': performance_score,
            'render_time_ms': total_time,
            'brightness_range': brightness_range,
            'validation_result': self._classify_result((shadow_quality + ssao_quality + lighting_coherence) / 3)
        }
    
    def _validate_shading_improvements(self) -> Dict[str, Any]:
        """Validate T22 normal/tangent quality improvements"""
        # Create test terrain mesh
        test_positions, test_indices, test_uvs = self._create_test_terrain_mesh()
        
        # Apply displacement to test post-displacement improvements
        displaced_positions = self._apply_test_displacement(test_positions)
        
        # Compute T22 enhanced normals and tangents
        normals = compute_angle_weighted_normals(displaced_positions, test_indices, post_displacement=True)
        tangents, bitangents = compute_tangent_basis(
            displaced_positions, normals, test_uvs, test_indices, post_displacement=True
        )
        
        # Validate shading basis quality
        basis_valid = validate_shading_basis(normals, tangents, bitangents)
        
        # Check seam consistency
        seam_consistent = validate_seam_consistency(
            normals, tangents, displaced_positions, test_indices, test_uvs
        )
        
        # Analyze normal quality
        normal_quality = self._analyze_normal_quality(normals, displaced_positions, test_indices)
        
        # Analyze tangent quality
        tangent_quality = self._analyze_tangent_quality(tangents, bitangents, normals)
        
        # Overall shading quality
        shading_quality = (normal_quality + tangent_quality) / 2
        if basis_valid:
            shading_quality *= 1.1  # Bonus for valid basis
        if seam_consistent:
            shading_quality *= 1.1  # Bonus for seam consistency
        
        shading_quality = min(1.0, shading_quality)
        
        return {
            'normal_quality': normal_quality,
            'tangent_quality': tangent_quality,
            'basis_valid': basis_valid,
            'seam_consistent': seam_consistent,
            'shading_quality': shading_quality,
            'vertex_count': len(test_positions),
            'validation_result': self._classify_result(shading_quality)
        }
    
    def _compute_overall_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall quality metrics from individual test results"""
        metrics = QualityMetrics()
        
        # Extract metrics from individual tests
        if 't19' in results:
            metrics.triangle_density = results['t19'].get('triangle_density', 0)
            metrics.lod_transition_smoothness = results['t19'].get('lod_transition_smoothness', 0)
        
        if 't20' in results:
            metrics.seam_consistency = results['t20'].get('seam_consistency', 0)
        
        if 't21' in results:
            metrics.shadow_quality = results['t21'].get('shadow_quality', 0)
            metrics.ssao_quality = results['t21'].get('ssao_quality', 0)
            metrics.lighting_coherence = results['t21'].get('lighting_coherence', 0)
        
        if 't22' in results:
            metrics.normal_quality = results['t22'].get('normal_quality', 0)
            metrics.tangent_quality = results['t22'].get('tangent_quality', 0)
        
        # Compute performance score as average
        perf_scores = []
        for test_results in results.values():
            if isinstance(test_results, dict) and 'performance_score' in test_results:
                perf_scores.append(test_results['performance_score'])
        metrics.performance_score = np.mean(perf_scores) if perf_scores else 0.0
        
        # Compute weighted overall score
        metrics.overall_score = (
            metrics.triangle_density * self.metric_weights['triangle_density'] +
            metrics.lod_transition_smoothness * self.metric_weights['lod_transition_smoothness'] +
            metrics.seam_consistency * self.metric_weights['seam_consistency'] +
            metrics.normal_quality * self.metric_weights['normal_quality'] +
            metrics.tangent_quality * self.metric_weights['tangent_quality'] +
            metrics.shadow_quality * self.metric_weights['shadow_quality'] +
            metrics.ssao_quality * self.metric_weights['ssao_quality'] +
            metrics.lighting_coherence * self.metric_weights['lighting_coherence'] +
            metrics.performance_score * self.metric_weights['performance_score']
        )
        
        overall_result = self._classify_result(metrics.overall_score)
        
        return {
            'metrics': metrics,
            'overall_score': metrics.overall_score,
            'validation_result': overall_result,
            'grade': overall_result.value.upper()
        }
    
    def _create_test_chunks(self) -> List[Dict[str, Any]]:
        """Create test chunks for LOD validation"""
        chunks = []
        
        # Create chunks at various distances
        distances = [2.0, 5.0, 12.0, 30.0, 70.0]
        
        for i, distance in enumerate(distances):
            chunk = {
                'chunk_info': {
                    'chunk_id': f'test_chunk_{i}',
                    'distance': distance
                },
                'positions': np.random.rand(100, 3).astype(np.float32) * 2 - 1,
                'aabb': {
                    'min': [-1, -1, -1],
                    'max': [1, 1, 1],
                    'center': [0, 0, 0]
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_test_chunk_bounds(self):
        """Create test chunk bounds for cave validation"""
        return ChunkBounds(
            min_point=np.array([-2, -2, -2]),
            max_point=np.array([2, 2, 2])
        )
    
    def _create_test_cave_sdf(self):
        """Create test SDF for cave validation"""
        from sdf_primitives import SDFSphere
        return SDFSphere(center=[0, 0, 0], radius=1.5, seed=42)
    
    def _create_test_terrain_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create test terrain mesh for shading validation"""
        # Simple heightfield grid
        resolution = 32
        positions = []
        indices = []
        uvs = []
        
        for j in range(resolution):
            for i in range(resolution):
                x = (i / (resolution - 1)) * 4 - 2
                y = (j / (resolution - 1)) * 4 - 2
                z = math.sin(x) * math.cos(y) * 0.5  # Simple height function
                
                positions.append([x, y, z])
                uvs.append([i / (resolution - 1), j / (resolution - 1)])
        
        # Generate indices
        for j in range(resolution - 1):
            for i in range(resolution - 1):
                bl = j * resolution + i
                br = j * resolution + (i + 1)
                tl = (j + 1) * resolution + i
                tr = (j + 1) * resolution + (i + 1)
                
                indices.extend([bl, br, tl])
                indices.extend([br, tr, tl])
        
        return (np.array(positions, dtype=np.float32),
                np.array(indices, dtype=np.uint32),
                np.array(uvs, dtype=np.float32))
    
    def _apply_test_displacement(self, positions: np.ndarray) -> np.ndarray:
        """Apply test displacement to terrain mesh"""
        displaced = np.copy(positions)
        
        for i, pos in enumerate(positions):
            # Apply some displacement based on position
            displacement = math.sin(pos[0] * 2) * math.cos(pos[1] * 2) * 0.2
            displaced[i][2] += displacement
        
        return displaced
    
    def _calculate_lod_smoothness(self, lod_distribution: List[int]) -> float:
        """Calculate LOD transition smoothness metric"""
        if sum(lod_distribution) == 0:
            return 0.0
        
        # Good LOD distribution should have gradual falloff
        total = sum(lod_distribution)
        normalized = [count / total for count in lod_distribution]
        
        # Penalize abrupt changes
        smoothness = 1.0
        for i in range(len(normalized) - 1):
            diff = abs(normalized[i] - normalized[i + 1])
            smoothness *= (1.0 - diff * 0.5)
        
        return max(0.0, smoothness)
    
    def _analyze_mesh_quality(self, vertices: List, triangles: np.ndarray) -> float:
        """Analyze mesh quality for cave validation"""
        if len(vertices) == 0 or len(triangles) == 0:
            return 0.0
        
        # Check for degenerate triangles
        valid_triangles = 0
        
        for tri in triangles:
            if len(tri) >= 3:
                v0 = vertices[tri[0]].position
                v1 = vertices[tri[1]].position
                v2 = vertices[tri[2]].position
                
                # Check triangle area
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = np.linalg.norm(cross) * 0.5
                
                if area > 1e-8:
                    valid_triangles += 1
        
        return valid_triangles / len(triangles) if len(triangles) > 0 else 0.0
    
    def _analyze_normal_quality(self, normals: np.ndarray, positions: np.ndarray, indices: np.ndarray) -> float:
        """Analyze normal vector quality"""
        if len(normals) == 0:
            return 0.0
        
        # Check if normals are normalized
        lengths = np.linalg.norm(normals, axis=1)
        normalized_count = np.sum(np.abs(lengths - 1.0) < 0.01)
        normalization_quality = normalized_count / len(normals)
        
        # Check normal consistency (neighboring normals shouldn't vary too much)
        consistency_quality = self._check_normal_consistency(normals, positions, indices)
        
        return (normalization_quality + consistency_quality) / 2
    
    def _analyze_tangent_quality(self, tangents: np.ndarray, bitangents: np.ndarray, normals: np.ndarray) -> float:
        """Analyze tangent space quality"""
        if len(tangents) == 0:
            return 0.0
        
        quality_sum = 0.0
        
        for i in range(len(tangents)):
            t = tangents[i][:3]  # Extract XYZ
            b = bitangents[i]
            n = normals[i]
            
            # Check normalization
            t_len = np.linalg.norm(t)
            b_len = np.linalg.norm(b)
            n_len = np.linalg.norm(n)
            
            norm_quality = min(1.0, (t_len + b_len + n_len) / 3.0)
            
            # Check orthogonality
            dot_tn = abs(np.dot(t, n))
            dot_bn = abs(np.dot(b, n))
            dot_tb = abs(np.dot(t, b))
            
            ortho_quality = 1.0 - (dot_tn + dot_bn + dot_tb) / 3.0
            
            quality_sum += (norm_quality + ortho_quality) / 2
        
        return quality_sum / len(tangents)
    
    def _check_normal_consistency(self, normals: np.ndarray, positions: np.ndarray, indices: np.ndarray) -> float:
        """Check consistency of normals across mesh"""
        if len(normals) < 2:
            return 1.0
        
        # Build adjacency
        adjacency = [[] for _ in range(len(normals))]
        triangles = indices.reshape(-1, 3)
        
        for tri in triangles:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adjacency[tri[i]].append(tri[j])
        
        # Check normal variation among adjacent vertices
        consistent_count = 0
        total_checks = 0
        
        for i in range(len(normals)):
            if len(adjacency[i]) == 0:
                continue
            
            current_normal = normals[i]
            
            for adj_idx in adjacency[i]:
                adj_normal = normals[adj_idx]
                dot_product = np.dot(current_normal, adj_normal)
                
                # Good consistency if dot product > 0.8 (< ~37 degrees)
                if dot_product > 0.8:
                    consistent_count += 1
                total_checks += 1
        
        return consistent_count / max(total_checks, 1)
    
    def _classify_result(self, score: float) -> ValidationResult:
        """Classify validation result based on score"""
        if score >= self.quality_thresholds[ValidationResult.EXCELLENT]:
            return ValidationResult.EXCELLENT
        elif score >= self.quality_thresholds[ValidationResult.GOOD]:
            return ValidationResult.GOOD
        elif score >= self.quality_thresholds[ValidationResult.ACCEPTABLE]:
            return ValidationResult.ACCEPTABLE
        elif score >= self.quality_thresholds[ValidationResult.POOR]:
            return ValidationResult.POOR
        else:
            return ValidationResult.FAILED
    
    def generate_quality_report(self, results: Dict[str, Any]):
        """Generate comprehensive quality report"""
        print("\n" + "="*80)
        print("🏆 TERRAIN GENERATION QUALITY REPORT")
        print("="*80)
        
        overall = results.get('overall', {})
        overall_score = overall.get('overall_score', 0)
        grade = overall.get('grade', 'UNKNOWN')
        
        print(f"\n📊 OVERALL QUALITY SCORE: {overall_score:.3f} ({grade})")
        
        # Individual test results
        test_sections = {
            't19': '📈 T19: LOD & Mesh Density Improvements',
            't20': '🕳️  T20: Marching Cubes & Cave Quality',
            't21': '🌅 T21: Lighting System (Shadows/SSAO/Tone Mapping)',
            't22': '🔢 T22: Normal/Tangent Quality After Displacement'
        }
        
        for test_key, title in test_sections.items():
            if test_key in results:
                print(f"\n{title}")
                print("-" * len(title))
                
                test_result = results[test_key]
                validation_result = test_result.get('validation_result', ValidationResult.FAILED)
                
                print(f"Result: {validation_result.value.upper()}")
                
                # Print key metrics
                for key, value in test_result.items():
                    if key != 'validation_result' and isinstance(value, (int, float)):
                        if isinstance(value, float):
                            print(f"  {key}: {value:.3f}")
                        else:
                            print(f"  {key}: {value}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_score >= 0.9:
            print("  ✅ Excellent terrain quality achieved! All improvements working well.")
        elif overall_score >= 0.75:
            print("  ✅ Good terrain quality. Minor optimizations could improve performance.")
        elif overall_score >= 0.6:
            print("  ⚠️  Acceptable quality. Consider reviewing lower-scoring components.")
        else:
            print("  ❌ Quality issues detected. Review failed components and re-test.")
        
        # Next steps
        print(f"\n🔧 NEXT STEPS:")
        
        if 't19' in results and results['t19'].get('lod_transition_smoothness', 0) < 0.7:
            print("  - Fine-tune LOD distance bands for smoother transitions")
        
        if 't20' in results and results['t20'].get('seam_consistency', 0) < 0.8:
            print("  - Improve chunk overlap and vertex deduplication")
        
        if 't21' in results and results['t21'].get('performance_score', 0) < 0.7:
            print("  - Optimize lighting system performance")
        
        if 't22' in results and not results['t22'].get('seam_consistent', True):
            print("  - Address normal/tangent seam inconsistencies")
        
        print("\n" + "="*80)
        print("🎯 Terrain generation validation complete!")
        print("="*80)


if __name__ == "__main__":
    # Run comprehensive terrain quality validation
    print("🚀 Running Comprehensive Terrain Quality Validation")
    print("   Testing T19, T20, T21, T22 improvements")
    
    validator = TerrainQualityValidator()
    
    # Run full validation suite
    validation_results = validator.validate_all_improvements()
    
    # Generate quality report
    validator.generate_quality_report(validation_results)
    
    print("\n✅ Terrain quality validation complete!")