#!/usr/bin/env python3
"""
Profiling System - T14
======================

Comprehensive profiling system for bake, load, and draw operations with
performance targets and bottleneck identification for big world optimization.

Features:
- Detailed timing for bake, load, draw operations
- Performance targets and threshold monitoring
- Frame rate analysis and smoothness metrics
- Memory usage and allocation tracking
- Bottleneck identification and recommendations

Usage:
    from profiling_system import PerformanceProfiler
    
    profiler = PerformanceProfiler()
    with profiler.profile("bake_operation"):
        # Your baking code here
        pass
"""

import time
import threading
import psutil
import statistics
from typing import Dict, List, Tuple, Optional, Any, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import sys
import os


class OperationType(Enum):
    """Types of operations to profile"""
    BAKE_TERRAIN = "bake_terrain"
    BAKE_CAVES = "bake_caves"
    BAKE_FUSION = "bake_fusion"
    LOAD_MANIFEST = "load_manifest"
    LOAD_BUFFER = "load_buffer"
    STREAM_CHUNK = "stream_chunk"
    DRAW_TERRAIN = "draw_terrain"
    DRAW_CAVES = "draw_caves"
    DRAW_FRAME = "draw_frame"
    UPDATE_LOD = "update_lod"
    MEMORY_ALLOC = "memory_alloc"


@dataclass
class PerformanceTarget:
    """Performance target definition"""
    operation: OperationType
    target_time_ms: float
    warning_time_ms: float
    critical_time_ms: float
    description: str


@dataclass
class ProfiledOperation:
    """Single profiled operation record"""
    operation: OperationType
    start_time: float
    end_time: float
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    thread_id: int
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def exceeded_target(self) -> bool:
        """Check if operation exceeded performance target"""
        # This would be filled by the profiler
        return self.duration_ms > 100  # Placeholder


@dataclass 
class FrameMetrics:
    """Frame rendering metrics"""
    frame_number: int
    frame_time_ms: float
    draw_time_ms: float
    update_time_ms: float
    memory_usage_mb: float
    chunk_count: int
    triangle_count: int
    timestamp: float


class PerformanceProfiler:
    """Main performance profiler for terrain system"""
    
    def __init__(self):
        """Initialize performance profiler"""
        self.active_operations: Dict[str, float] = {}
        self.completed_operations: List[ProfiledOperation] = []
        self.frame_metrics: deque = deque(maxlen=1000)  # Keep last 1000 frames
        
        self._lock = threading.RLock()
        self.profiling_enabled = True
        
        # Performance targets
        self.performance_targets = self._create_performance_targets()
        
        # Frame rate tracking
        self.target_fps = 60.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Memory tracking
        self.process = psutil.Process()
        self.baseline_memory_mb = self.process.memory_info().rss / 1024 / 1024
    
    def _create_performance_targets(self) -> Dict[OperationType, PerformanceTarget]:
        """Create performance targets for different operations"""
        targets = {}
        
        # Baking targets
        targets[OperationType.BAKE_TERRAIN] = PerformanceTarget(
            OperationType.BAKE_TERRAIN,
            target_time_ms=50.0,
            warning_time_ms=100.0,
            critical_time_ms=200.0,
            description="Terrain chunk baking per chunk"
        )
        
        targets[OperationType.BAKE_CAVES] = PerformanceTarget(
            OperationType.BAKE_CAVES,
            target_time_ms=80.0,
            warning_time_ms=150.0,
            critical_time_ms=300.0,
            description="Cave SDF generation per chunk"
        )
        
        targets[OperationType.BAKE_FUSION] = PerformanceTarget(
            OperationType.BAKE_FUSION,
            target_time_ms=30.0,
            warning_time_ms=60.0,
            critical_time_ms=120.0,
            description="Terrain-cave fusion per chunk"
        )
        
        # Loading targets
        targets[OperationType.LOAD_MANIFEST] = PerformanceTarget(
            OperationType.LOAD_MANIFEST,
            target_time_ms=5.0,
            warning_time_ms=10.0,
            critical_time_ms=20.0,
            description="Manifest loading per chunk"
        )
        
        targets[OperationType.LOAD_BUFFER] = PerformanceTarget(
            OperationType.LOAD_BUFFER,
            target_time_ms=20.0,
            warning_time_ms=50.0,
            critical_time_ms=100.0,
            description="Buffer loading and decompression"
        )
        
        targets[OperationType.STREAM_CHUNK] = PerformanceTarget(
            OperationType.STREAM_CHUNK,
            target_time_ms=100.0,
            warning_time_ms=200.0,
            critical_time_ms=500.0,
            description="Complete chunk streaming"
        )
        
        # Rendering targets
        targets[OperationType.DRAW_FRAME] = PerformanceTarget(
            OperationType.DRAW_FRAME,
            target_time_ms=16.67,  # 60 FPS
            warning_time_ms=33.33,  # 30 FPS
            critical_time_ms=50.0,   # 20 FPS
            description="Complete frame rendering"
        )
        
        targets[OperationType.DRAW_TERRAIN] = PerformanceTarget(
            OperationType.DRAW_TERRAIN,
            target_time_ms=8.0,
            warning_time_ms=15.0,
            critical_time_ms=25.0,
            description="Terrain rendering per frame"
        )
        
        targets[OperationType.UPDATE_LOD] = PerformanceTarget(
            OperationType.UPDATE_LOD,
            target_time_ms=2.0,
            warning_time_ms=5.0,
            critical_time_ms=10.0,
            description="LOD system update"
        )
        
        return targets
    
    def profile(self, operation: OperationType, context: Dict[str, Any] = None) -> ContextManager:
        """Context manager for profiling operations"""
        return ProfileContext(self, operation, context or {})
    
    def start_operation(self, operation: OperationType, context: Dict[str, Any] = None) -> str:
        """Start profiling an operation"""
        if not self.profiling_enabled:
            return ""
        
        operation_id = f"{operation.value}_{threading.get_ident()}_{time.time()}"
        
        with self._lock:
            self.active_operations[operation_id] = {
                'operation': operation,
                'start_time': time.time(),
                'memory_before': self.process.memory_info().rss / 1024 / 1024,
                'context': context or {},
                'thread_id': threading.get_ident()
            }
        
        return operation_id
    
    def end_operation(self, operation_id: str):
        """End profiling an operation"""
        if not self.profiling_enabled or not operation_id:
            return
        
        end_time = time.time()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        
        with self._lock:
            if operation_id not in self.active_operations:
                return
            
            op_data = self.active_operations.pop(operation_id)
            
            # Create profiled operation record
            duration_ms = (end_time - op_data['start_time']) * 1000
            
            profiled_op = ProfiledOperation(
                operation=op_data['operation'],
                start_time=op_data['start_time'],
                end_time=end_time,
                duration_ms=duration_ms,
                memory_before_mb=op_data['memory_before'],
                memory_after_mb=memory_after,
                memory_delta_mb=memory_after - op_data['memory_before'],
                thread_id=op_data['thread_id'],
                context_data=op_data['context']
            )
            
            self.completed_operations.append(profiled_op)
    
    def record_frame_metrics(self, frame_time_ms: float, draw_time_ms: float, 
                           update_time_ms: float, chunk_count: int, triangle_count: int):
        """Record frame rendering metrics"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        frame_metrics = FrameMetrics(
            frame_number=self.frame_count,
            frame_time_ms=frame_time_ms,
            draw_time_ms=draw_time_ms,
            update_time_ms=update_time_ms,
            memory_usage_mb=current_memory,
            chunk_count=chunk_count,
            triangle_count=triangle_count,
            timestamp=time.time()
        )
        
        with self._lock:
            self.frame_metrics.append(frame_metrics)
            self.frame_count += 1
            
            # Update FPS calculation
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps_frames = sum(1 for f in self.frame_metrics 
                               if current_time - f.timestamp <= 1.0)
                self.current_fps = fps_frames
                self.last_fps_time = current_time
    
    def get_operation_stats(self, operation: OperationType) -> Dict[str, Any]:
        """Get statistics for specific operation type"""
        with self._lock:
            ops = [op for op in self.completed_operations if op.operation == operation]
        
        if not ops:
            return {'count': 0}
        
        durations = [op.duration_ms for op in ops]
        memory_deltas = [op.memory_delta_mb for op in ops]
        
        target = self.performance_targets.get(operation)
        exceeded_target_count = 0
        if target:
            exceeded_target_count = sum(1 for d in durations if d > target.warning_time_ms)
        
        return {
            'count': len(ops),
            'total_time_ms': sum(durations),
            'avg_time_ms': statistics.mean(durations),
            'min_time_ms': min(durations),
            'max_time_ms': max(durations),
            'median_time_ms': statistics.median(durations),
            'std_dev_ms': statistics.stdev(durations) if len(durations) > 1 else 0.0,
            'avg_memory_delta_mb': statistics.mean(memory_deltas),
            'exceeded_target_count': exceeded_target_count,
            'target_adherence_rate': 1.0 - (exceeded_target_count / len(ops))
        }
    
    def get_frame_stats(self, last_n_frames: int = 300) -> Dict[str, Any]:
        """Get frame rate and rendering statistics"""
        with self._lock:
            recent_frames = list(self.frame_metrics)[-last_n_frames:]
        
        if not recent_frames:
            return {'frame_count': 0}
        
        frame_times = [f.frame_time_ms for f in recent_frames]
        draw_times = [f.draw_time_ms for f in recent_frames]
        memory_usage = [f.memory_usage_mb for f in recent_frames]
        chunk_counts = [f.chunk_count for f in recent_frames]
        
        # Calculate frame rate statistics
        avg_frame_time = statistics.mean(frame_times)
        avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Frame time consistency (smoothness)
        frame_time_variance = statistics.variance(frame_times) if len(frame_times) > 1 else 0
        smoothness_score = max(0, 1.0 - (frame_time_variance / (avg_frame_time ** 2)))
        
        # Performance classification
        if avg_fps >= 55:
            performance_tier = "Excellent"
        elif avg_fps >= 45:
            performance_tier = "Good"  
        elif avg_fps >= 30:
            performance_tier = "Acceptable"
        else:
            performance_tier = "Poor"
        
        return {
            'frame_count': len(recent_frames),
            'avg_frame_time_ms': avg_frame_time,
            'avg_fps': avg_fps,
            'min_fps': 1000.0 / max(frame_times) if frame_times else 0,
            'max_fps': 1000.0 / min(frame_times) if frame_times else 0,
            'avg_draw_time_ms': statistics.mean(draw_times),
            'avg_memory_mb': statistics.mean(memory_usage),
            'avg_chunk_count': statistics.mean(chunk_counts),
            'smoothness_score': smoothness_score,
            'performance_tier': performance_tier,
            'frame_time_variance': frame_time_variance
        }
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks and provide recommendations"""
        analysis = {
            'bottlenecks': [],
            'recommendations': [],
            'critical_operations': [],
            'memory_issues': []
        }
        
        # Analyze each operation type
        for operation in OperationType:
            stats = self.get_operation_stats(operation)
            target = self.performance_targets.get(operation)
            
            if stats['count'] == 0 or not target:
                continue
            
            # Check if operation exceeds targets
            if stats['avg_time_ms'] > target.critical_time_ms:
                analysis['critical_operations'].append({
                    'operation': operation.value,
                    'avg_time_ms': stats['avg_time_ms'],
                    'target_ms': target.target_time_ms,
                    'severity': 'Critical'
                })
                
                analysis['bottlenecks'].append(f"{operation.value}: {stats['avg_time_ms']:.1f}ms (target: {target.target_time_ms:.1f}ms)")
            
            elif stats['avg_time_ms'] > target.warning_time_ms:
                analysis['bottlenecks'].append(f"{operation.value}: {stats['avg_time_ms']:.1f}ms (warning)")
            
            # Memory analysis
            if stats.get('avg_memory_delta_mb', 0) > 50:  # More than 50MB average allocation
                analysis['memory_issues'].append({
                    'operation': operation.value,
                    'avg_memory_delta_mb': stats['avg_memory_delta_mb'],
                    'issue': 'High memory allocation'
                })
        
        # Generate recommendations
        frame_stats = self.get_frame_stats()
        
        if frame_stats.get('avg_fps', 0) < 30:
            analysis['recommendations'].append("Frame rate below 30 FPS - consider reducing terrain detail or chunk count")
        
        if frame_stats.get('smoothness_score', 0) < 0.7:
            analysis['recommendations'].append("Frame time variance high - enable buffer pooling and async streaming")
        
        if any('bake' in b for b in analysis['bottlenecks']):
            analysis['recommendations'].append("Baking operations slow - enable parallel baking with more worker threads")
        
        if any('load' in b for b in analysis['bottlenecks']):
            analysis['recommendations'].append("Loading operations slow - increase I/O worker threads and cache size")
        
        if len(analysis['memory_issues']) > 0:
            analysis['recommendations'].append("High memory usage detected - enable buffer pooling and LRU cache")
        
        return analysis
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': time.time(),
            'profiling_duration_sec': time.time() - (self.completed_operations[0].start_time if self.completed_operations else time.time()),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'baseline_memory_mb': self.baseline_memory_mb,
                'current_memory_mb': self.process.memory_info().rss / 1024 / 1024
            },
            'frame_performance': self.get_frame_stats(),
            'operation_performance': {},
            'bottleneck_analysis': self.analyze_bottlenecks(),
            'performance_targets': {}
        }
        
        # Operation performance breakdown
        for operation in OperationType:
            stats = self.get_operation_stats(operation)
            if stats['count'] > 0:
                report['operation_performance'][operation.value] = stats
        
        # Performance targets
        for operation, target in self.performance_targets.items():
            report['performance_targets'][operation.value] = {
                'target_ms': target.target_time_ms,
                'warning_ms': target.warning_time_ms,
                'critical_ms': target.critical_time_ms,
                'description': target.description
            }
        
        return report
    
    def export_report(self, output_path: str):
        """Export performance report to JSON file"""
        report = self.generate_performance_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report exported to {output_path}")


class ProfileContext:
    """Context manager for profiling operations"""
    
    def __init__(self, profiler: PerformanceProfiler, operation: OperationType, context: Dict[str, Any]):
        self.profiler = profiler
        self.operation = operation
        self.context = context
        self.operation_id = ""
    
    def __enter__(self):
        self.operation_id = self.profiler.start_operation(self.operation, self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_operation(self.operation_id)


if __name__ == "__main__":
    # Test profiling system
    print("🚀 T14 Performance Profiling System")
    print("=" * 60)
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Test operation profiling
    print("📊 Testing operation profiling...")
    
    # Simulate some operations
    with profiler.profile(OperationType.BAKE_TERRAIN, {'chunk_id': '0_0'}):
        time.sleep(0.05)  # 50ms
    
    with profiler.profile(OperationType.BAKE_CAVES, {'chunk_id': '0_0'}):
        time.sleep(0.12)  # 120ms (exceeds warning)
    
    with profiler.profile(OperationType.LOAD_BUFFER, {'buffer_id': 'heightfield'}):
        time.sleep(0.03)  # 30ms
    
    with profiler.profile(OperationType.DRAW_FRAME):
        time.sleep(0.02)  # 20ms
    
    # Simulate frame metrics
    print("\n🎬 Simulating frame metrics...")
    for i in range(10):
        frame_time = 16 + (i % 3) * 2  # Varying frame times
        profiler.record_frame_metrics(
            frame_time_ms=frame_time,
            draw_time_ms=frame_time * 0.6,
            update_time_ms=frame_time * 0.2,
            chunk_count=16,
            triangle_count=50000
        )
    
    # Get statistics
    terrain_stats = profiler.get_operation_stats(OperationType.BAKE_TERRAIN)
    cave_stats = profiler.get_operation_stats(OperationType.BAKE_CAVES)
    frame_stats = profiler.get_frame_stats()
    
    print(f"\n📈 Operation Statistics:")
    print(f"   Terrain Baking: {terrain_stats['avg_time_ms']:.1f}ms avg")
    print(f"   Cave Baking: {cave_stats['avg_time_ms']:.1f}ms avg (target adherence: {cave_stats['target_adherence_rate']:.1%})")
    print(f"   Frame Rate: {frame_stats['avg_fps']:.1f} FPS ({frame_stats['performance_tier']})")
    print(f"   Frame Smoothness: {frame_stats['smoothness_score']:.2f}")
    
    # Bottleneck analysis
    bottlenecks = profiler.analyze_bottlenecks()
    print(f"\n🔍 Bottleneck Analysis:")
    print(f"   Bottlenecks found: {len(bottlenecks['bottlenecks'])}")
    for bottleneck in bottlenecks['bottlenecks']:
        print(f"      - {bottleneck}")
    
    print(f"   Recommendations: {len(bottlenecks['recommendations'])}")
    for recommendation in bottlenecks['recommendations']:
        print(f"      • {recommendation}")
    
    # Export report
    profiler.export_report("performance_report.json")
    
    print("\n✅ Performance profiling system functional")