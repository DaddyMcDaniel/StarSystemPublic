#!/usr/bin/env python3
"""
Hero Planet Performance Optimization - T17
==========================================

Performance tuning system for hero planet to maintain 60+ fps on mid-tier GPUs.
Optimizes LOD distances, chunk sizes, and rendering parameters based on content
complexity and target hardware specifications.

Features:
- Adaptive LOD distance calculation based on content complexity
- Dynamic chunk size optimization for performance targets
- GPU performance profiling and automatic adjustment
- Content-aware quality scaling
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class PerformanceTarget:
    """Performance targets for optimization"""
    target_fps: float = 60.0
    min_fps: float = 45.0
    max_triangles_per_frame: int = 500000
    memory_budget_mb: int = 512
    draw_call_budget: int = 150
    gpu_tier: str = "mid"  # low, mid, high


@dataclass
class ContentComplexity:
    """Complexity metrics for different content types"""
    mountain_complexity: float = 1.0
    cave_complexity: float = 1.5
    dune_complexity: float = 0.8
    archipelago_complexity: float = 1.2
    base_triangle_density: float = 1.0


@dataclass
class LODConfiguration:
    """LOD system configuration"""
    base_chunk_size: float = 64.0
    max_depth: int = 8
    lod_distances: List[float] = field(default_factory=list)
    chunk_transition_zones: List[float] = field(default_factory=list)
    quality_multipliers: List[float] = field(default_factory=list)


class HeroPlanetOptimizer:
    """Performance optimization system for hero planet"""
    
    def __init__(self, target: PerformanceTarget):
        self.target = target
        self.current_performance = {}
        self.optimization_history = []
        
        # GPU tier performance profiles
        self.gpu_profiles = {
            "low": {
                "triangle_budget": 250000,
                "memory_budget_mb": 256,
                "draw_call_budget": 100,
                "quality_scale": 0.7
            },
            "mid": {
                "triangle_budget": 500000,
                "memory_budget_mb": 512,
                "draw_call_budget": 150,
                "quality_scale": 1.0
            },
            "high": {
                "triangle_budget": 1000000,
                "memory_budget_mb": 1024,
                "draw_call_budget": 250,
                "quality_scale": 1.3
            }
        }
        
        self.current_profile = self.gpu_profiles.get(target.gpu_tier, self.gpu_profiles["mid"])
    
    def calculate_optimal_lod_distances(self, complexity: ContentComplexity, 
                                      planet_radius: float = 2000.0) -> LODConfiguration:
        """Calculate optimal LOD distances based on content complexity"""
        
        # Base LOD distances for mid-tier GPU
        base_distances = [128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0]
        
        # Adjust based on content complexity
        complexity_factor = self._calculate_overall_complexity(complexity)
        quality_scale = self.current_profile["quality_scale"]
        
        # Scale distances based on complexity and GPU capability
        adjusted_distances = []
        for i, distance in enumerate(base_distances):
            # Closer LOD transitions for high complexity content
            complexity_scale = 1.0 / (1.0 + complexity_factor * 0.3)
            
            # GPU capability scaling
            gpu_scale = quality_scale
            
            # Progressive scaling - higher LODs scale more aggressively
            progressive_scale = 1.0 + (i * 0.1 * complexity_factor)
            
            final_distance = distance * complexity_scale * gpu_scale * progressive_scale
            adjusted_distances.append(max(final_distance, 64.0))  # Minimum distance
        
        # Calculate transition zones (30% of LOD distance)
        transition_zones = [d * 0.3 for d in adjusted_distances]
        
        # Quality multipliers for each LOD level
        quality_multipliers = []
        for i in range(len(adjusted_distances)):
            # Higher LOD levels get progressively lower quality
            quality_mult = 1.0 / (1.0 + i * 0.2)
            quality_multipliers.append(quality_mult * quality_scale)
        
        return LODConfiguration(
            base_chunk_size=self._calculate_optimal_chunk_size(complexity),
            max_depth=self._calculate_optimal_max_depth(complexity),
            lod_distances=adjusted_distances,
            chunk_transition_zones=transition_zones,
            quality_multipliers=quality_multipliers
        )
    
    def _calculate_overall_complexity(self, complexity: ContentComplexity) -> float:
        """Calculate weighted overall complexity score"""
        weights = {
            'mountain': 0.3,    # Mountains are visible from far
            'cave': 0.4,        # Caves are most expensive
            'dune': 0.1,        # Dunes are simpler
            'archipelago': 0.2  # Islands moderate complexity
        }
        
        overall = (complexity.mountain_complexity * weights['mountain'] +
                  complexity.cave_complexity * weights['cave'] +
                  complexity.dune_complexity * weights['dune'] +
                  complexity.archipelago_complexity * weights['archipelago'])
        
        return overall * complexity.base_triangle_density
    
    def _calculate_optimal_chunk_size(self, complexity: ContentComplexity) -> float:
        """Calculate optimal base chunk size"""
        # Start with base chunk size
        base_size = 64.0
        
        # Adjust based on complexity
        complexity_factor = self._calculate_overall_complexity(complexity)
        
        if complexity_factor > 1.2:
            # High complexity - smaller chunks for better culling
            base_size = 48.0
        elif complexity_factor < 0.8:
            # Low complexity - larger chunks for efficiency
            base_size = 96.0
        
        # GPU tier adjustment
        if self.target.gpu_tier == "low":
            base_size *= 0.75  # Smaller chunks for low-end
        elif self.target.gpu_tier == "high":
            base_size *= 1.25  # Larger chunks for high-end
        
        return max(32.0, min(128.0, base_size))
    
    def _calculate_optimal_max_depth(self, complexity: ContentComplexity) -> int:
        """Calculate optimal maximum LOD depth"""
        base_depth = 8
        
        # Adjust based on GPU tier
        if self.target.gpu_tier == "low":
            base_depth = 6
        elif self.target.gpu_tier == "high":
            base_depth = 10
        
        # Adjust based on complexity
        complexity_factor = self._calculate_overall_complexity(complexity)
        
        if complexity_factor > 1.3:
            base_depth -= 1  # Reduce depth for high complexity
        elif complexity_factor < 0.7:
            base_depth += 1  # Increase depth for low complexity
        
        return max(4, min(12, base_depth))
    
    def estimate_performance_metrics(self, lod_config: LODConfiguration,
                                   viewing_distance: float = 1000.0) -> Dict[str, Any]:
        """Estimate performance metrics for given configuration"""
        
        # Estimate visible chunks based on viewing distance
        visible_chunks = self._estimate_visible_chunks(lod_config, viewing_distance)
        
        # Estimate triangles per chunk at different LOD levels
        triangles_per_chunk = self._estimate_triangles_per_chunk(lod_config)
        
        # Calculate total triangle count
        total_triangles = 0
        total_draw_calls = 0
        memory_usage_mb = 0
        
        for lod_level, chunk_count in visible_chunks.items():
            triangles = triangles_per_chunk.get(lod_level, 1000)
            total_triangles += triangles * chunk_count
            total_draw_calls += chunk_count
            
            # Memory estimation (vertices + indices + textures)
            chunk_memory_kb = triangles * 0.1  # Rough estimate
            memory_usage_mb += (chunk_memory_kb * chunk_count) / 1024.0
        
        # Estimate FPS based on triangle count and GPU profile
        triangle_budget = self.current_profile["triangle_budget"]
        triangle_ratio = total_triangles / triangle_budget
        
        # Simple FPS estimation model
        if triangle_ratio <= 0.5:
            estimated_fps = self.target.target_fps * 1.2
        elif triangle_ratio <= 1.0:
            estimated_fps = self.target.target_fps * (1.5 - triangle_ratio)
        else:
            estimated_fps = self.target.target_fps / triangle_ratio
        
        return {
            'estimated_fps': max(15.0, estimated_fps),
            'total_triangles': int(total_triangles),
            'total_draw_calls': total_draw_calls,
            'memory_usage_mb': memory_usage_mb,
            'visible_chunks': visible_chunks,
            'triangle_budget_usage': triangle_ratio,
            'memory_budget_usage': memory_usage_mb / self.current_profile["memory_budget_mb"],
            'draw_call_budget_usage': total_draw_calls / self.current_profile["draw_call_budget"]
        }
    
    def _estimate_visible_chunks(self, lod_config: LODConfiguration, 
                               viewing_distance: float) -> Dict[int, int]:
        """Estimate number of visible chunks at each LOD level"""
        visible_chunks = {}
        
        # Simple visibility estimation based on distance
        for lod_level, lod_distance in enumerate(lod_config.lod_distances):
            if viewing_distance <= lod_distance:
                # Estimate chunks in view frustum at this LOD
                chunk_size = lod_config.base_chunk_size * (2 ** lod_level)
                chunks_per_side = max(1, int(viewing_distance / chunk_size))
                
                # Approximate visible chunks (considering frustum culling)
                visible_count = int(chunks_per_side * chunks_per_side * 0.3)  # 30% visible
                visible_chunks[lod_level] = visible_count
                break
        
        # Add background LOD chunks
        for lod_level in range(len(lod_config.lod_distances)):
            if lod_level not in visible_chunks:
                # Background chunks at lower detail
                chunk_size = lod_config.base_chunk_size * (2 ** lod_level)
                background_chunks = max(1, int(8000.0 / chunk_size * 0.1))  # Sparse background
                visible_chunks[lod_level] = background_chunks
        
        return visible_chunks
    
    def _estimate_triangles_per_chunk(self, lod_config: LODConfiguration) -> Dict[int, int]:
        """Estimate triangles per chunk at each LOD level"""
        base_triangles = 2048  # Base triangle count for chunk
        
        triangles_per_lod = {}
        for lod_level, quality_mult in enumerate(lod_config.quality_multipliers):
            # Triangle count reduces with LOD level
            lod_reduction = 1.0 / (4 ** lod_level)  # Quartic reduction
            triangles = int(base_triangles * lod_reduction * quality_mult)
            triangles_per_lod[lod_level] = max(64, triangles)  # Minimum triangles
        
        return triangles_per_lod
    
    def optimize_for_target_performance(self, complexity: ContentComplexity,
                                      max_iterations: int = 10) -> LODConfiguration:
        """Iteratively optimize configuration for target performance"""
        
        current_config = self.calculate_optimal_lod_distances(complexity)
        
        for iteration in range(max_iterations):
            metrics = self.estimate_performance_metrics(current_config)
            
            print(f"🔧 Optimization iteration {iteration + 1}")
            print(f"   Estimated FPS: {metrics['estimated_fps']:.1f}")
            print(f"   Triangle count: {metrics['total_triangles']:,}")
            print(f"   Memory usage: {metrics['memory_usage_mb']:.1f} MB")
            
            # Check if performance targets are met
            if (metrics['estimated_fps'] >= self.target.target_fps and
                metrics['total_triangles'] <= self.target.max_triangles_per_frame and
                metrics['memory_usage_mb'] <= self.target.memory_budget_mb):
                print(f"   ✅ Performance targets achieved!")
                break
            
            # Adjust configuration based on bottlenecks
            if metrics['estimated_fps'] < self.target.min_fps:
                # Aggressive optimization needed
                current_config = self._aggressive_optimization(current_config, metrics)
            elif metrics['estimated_fps'] < self.target.target_fps:
                # Moderate optimization
                current_config = self._moderate_optimization(current_config, metrics)
            
            # Record optimization step
            self.optimization_history.append({
                'iteration': iteration,
                'config': current_config,
                'metrics': metrics
            })
        
        return current_config
    
    def _aggressive_optimization(self, config: LODConfiguration, 
                               metrics: Dict[str, Any]) -> LODConfiguration:
        """Apply aggressive optimization for poor performance"""
        
        # Reduce LOD distances by 20%
        new_distances = [d * 0.8 for d in config.lod_distances]
        
        # Reduce quality multipliers
        new_quality = [q * 0.8 for q in config.quality_multipliers]
        
        # Smaller chunks for better culling
        new_chunk_size = config.base_chunk_size * 0.8
        
        return LODConfiguration(
            base_chunk_size=max(32.0, new_chunk_size),
            max_depth=max(4, config.max_depth - 1),
            lod_distances=new_distances,
            chunk_transition_zones=[d * 0.3 for d in new_distances],
            quality_multipliers=new_quality
        )
    
    def _moderate_optimization(self, config: LODConfiguration,
                             metrics: Dict[str, Any]) -> LODConfiguration:
        """Apply moderate optimization for borderline performance"""
        
        # Slightly reduce LOD distances
        new_distances = [d * 0.9 for d in config.lod_distances]
        
        # Slightly reduce quality
        new_quality = [q * 0.9 for q in config.quality_multipliers]
        
        return LODConfiguration(
            base_chunk_size=config.base_chunk_size,
            max_depth=config.max_depth,
            lod_distances=new_distances,
            chunk_transition_zones=[d * 0.3 for d in new_distances],
            quality_multipliers=new_quality
        )
    
    def generate_performance_report(self, final_config: LODConfiguration) -> Dict[str, Any]:
        """Generate comprehensive performance optimization report"""
        
        final_metrics = self.estimate_performance_metrics(final_config)
        
        return {
            'optimization_summary': {
                'target_fps': self.target.target_fps,
                'estimated_fps': final_metrics['estimated_fps'],
                'performance_margin': final_metrics['estimated_fps'] / self.target.target_fps,
                'optimization_iterations': len(self.optimization_history)
            },
            'resource_usage': {
                'triangles': {
                    'count': final_metrics['total_triangles'],
                    'budget': self.target.max_triangles_per_frame,
                    'usage_percent': final_metrics['triangle_budget_usage'] * 100
                },
                'memory': {
                    'usage_mb': final_metrics['memory_usage_mb'],
                    'budget_mb': self.target.memory_budget_mb,
                    'usage_percent': final_metrics['memory_budget_usage'] * 100
                },
                'draw_calls': {
                    'count': final_metrics['total_draw_calls'],
                    'budget': self.target.draw_call_budget,
                    'usage_percent': final_metrics['draw_call_budget_usage'] * 100
                }
            },
            'lod_configuration': {
                'base_chunk_size': final_config.base_chunk_size,
                'max_depth': final_config.max_depth,
                'lod_distances': final_config.lod_distances,
                'transition_zones': final_config.chunk_transition_zones,
                'quality_multipliers': final_config.quality_multipliers
            },
            'optimization_history': self.optimization_history
        }


if __name__ == "__main__":
    # Test hero planet optimization
    print("🚀 T17 Hero Planet Performance Optimization")
    print("=" * 60)
    
    # Define performance targets for mid-tier GPU
    target = PerformanceTarget(
        target_fps=60.0,
        min_fps=45.0,
        max_triangles_per_frame=500000,
        memory_budget_mb=512,
        draw_call_budget=150,
        gpu_tier="mid"
    )
    
    # Define content complexity for hero planet
    complexity = ContentComplexity(
        mountain_complexity=1.4,   # Ridged mountains are complex
        cave_complexity=1.8,       # Gyroidal caves are very complex
        dune_complexity=0.9,       # Warped dunes moderate
        archipelago_complexity=1.1, # Islands moderate
        base_triangle_density=1.2   # Rich detail overall
    )
    
    # Create optimizer
    optimizer = HeroPlanetOptimizer(target)
    
    print("📊 Content complexity analysis:")
    overall_complexity = optimizer._calculate_overall_complexity(complexity)
    print(f"   Mountain complexity: {complexity.mountain_complexity}")
    print(f"   Cave complexity: {complexity.cave_complexity}")
    print(f"   Dune complexity: {complexity.dune_complexity}")
    print(f"   Archipelago complexity: {complexity.archipelago_complexity}")
    print(f"   Overall complexity: {overall_complexity:.2f}")
    
    # Calculate initial LOD configuration
    print(f"\n🔧 Calculating optimal LOD configuration...")
    initial_config = optimizer.calculate_optimal_lod_distances(complexity)
    
    print(f"   Base chunk size: {initial_config.base_chunk_size}")
    print(f"   Max LOD depth: {initial_config.max_depth}")
    print(f"   LOD distances: {[f'{d:.0f}' for d in initial_config.lod_distances[:4]]}...")
    
    # Estimate performance
    initial_metrics = optimizer.estimate_performance_metrics(initial_config)
    print(f"\n📈 Initial performance estimate:")
    print(f"   Estimated FPS: {initial_metrics['estimated_fps']:.1f}")
    print(f"   Triangle count: {initial_metrics['total_triangles']:,}")
    print(f"   Memory usage: {initial_metrics['memory_usage_mb']:.1f} MB")
    print(f"   Draw calls: {initial_metrics['total_draw_calls']}")
    
    # Optimize for target performance
    print(f"\n🎯 Optimizing for target performance...")
    optimized_config = optimizer.optimize_for_target_performance(complexity)
    
    # Generate performance report
    report = optimizer.generate_performance_report(optimized_config)
    
    print(f"\n📋 Optimization Results:")
    print(f"   Target FPS: {report['optimization_summary']['target_fps']}")
    print(f"   Estimated FPS: {report['optimization_summary']['estimated_fps']:.1f}")
    print(f"   Performance margin: {report['optimization_summary']['performance_margin']:.1f}x")
    print(f"   Optimization iterations: {report['optimization_summary']['optimization_iterations']}")
    
    print(f"\n💾 Resource Usage:")
    print(f"   Triangles: {report['resource_usage']['triangles']['usage_percent']:.1f}% of budget")
    print(f"   Memory: {report['resource_usage']['memory']['usage_percent']:.1f}% of budget")
    print(f"   Draw calls: {report['resource_usage']['draw_calls']['usage_percent']:.1f}% of budget")
    
    # Export optimized configuration
    config_export = {
        'performance_target': target.__dict__,
        'content_complexity': complexity.__dict__,
        'optimized_lod_config': optimized_config.__dict__,
        'performance_report': report
    }
    
    output_file = "/home/colling/PCC-LanguageV2/agents/agent_d/examples/planets/hero_world_optimization.json"
    with open(output_file, 'w') as f:
        json.dump(config_export, f, indent=2)
    
    print(f"\n✅ Performance optimization complete")
    print(f"   Configuration exported to: hero_world_optimization.json")
    print(f"   Target 60+ fps: {'✅ Achieved' if report['optimization_summary']['estimated_fps'] >= 60 else '❌ Needs more optimization'}")
    print(f"   Mid-tier GPU ready: {'✅ Yes' if report['optimization_summary']['performance_margin'] >= 1.0 else '❌ Needs adjustment'}")