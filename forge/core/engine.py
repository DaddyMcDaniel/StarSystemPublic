#!/usr/bin/env python3
"""
Forge Engine Core
The main engine architecture with modular systems and Claude collective intelligence
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import Claude collective intelligence
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.collective_subconscious import consult_collective_wisdom, store_agent_experience

# Engine configuration
@dataclass
class EngineConfig:
    """Engine configuration settings"""
    name: str = "Forge"
    version: str = "1.0.0"
    target_fps: int = 60
    enable_claude_intelligence: bool = True
    module_hot_swap: bool = True
    procedural_optimization: bool = True
    pcc_runtime_enabled: bool = True
    debug_mode: bool = False
    max_concurrent_processes: int = 8

# Base interfaces for engine modules
class IEngineModule(ABC):
    """Base interface for all engine modules"""
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """Unique module identifier"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Module version"""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of required module IDs"""
        pass
    
    @abstractmethod
    async def initialize(self, engine_context: 'ForgeEngine') -> bool:
        """Initialize module with engine context"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of module"""
        pass
    
    @abstractmethod
    async def update(self, delta_time: float) -> None:
        """Update module per frame"""
        pass
    
    @abstractmethod
    async def handle_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle engine events"""
        pass

class IClaudeIntegrated(ABC):
    """Interface for modules that use Claude intelligence"""
    
    @abstractmethod
    def consult_claude(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Consult Claude for module-specific intelligence"""
        pass
    
    @abstractmethod
    async def store_module_experience(self, experience: str, exp_type: str = "experience",
                                    context: Dict[str, Any] = None, tags: List[str] = None):
        """Store module experience in collective memory"""
        pass

@dataclass
class EngineEvent:
    """Engine event structure"""
    event_type: str
    source_module: str
    target_module: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 0 = highest priority

class ModuleRegistry:
    """Registry for managing engine modules"""
    
    def __init__(self):
        self.modules: Dict[str, IEngineModule] = {}
        self.module_types: Dict[str, Type[IEngineModule]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.initialization_order: List[str] = []
    
    def register_module_type(self, module_type: Type[IEngineModule]):
        """Register a module type for dynamic loading"""
        # Get module ID from a temporary instance
        temp_instance = module_type()
        self.module_types[temp_instance.module_id] = module_type
        print(f"🔧 Registered module type: {temp_instance.module_id}")
    
    def add_module(self, module: IEngineModule):
        """Add module to registry"""
        self.modules[module.module_id] = module
        self.dependency_graph[module.module_id] = module.dependencies
        self._update_initialization_order()
        print(f"📦 Added module: {module.module_id}")
    
    def remove_module(self, module_id: str):
        """Remove module from registry"""
        if module_id in self.modules:
            del self.modules[module_id]
            if module_id in self.dependency_graph:
                del self.dependency_graph[module_id]
            self._update_initialization_order()
            print(f"🗑️ Removed module: {module_id}")
    
    def get_module(self, module_id: str) -> Optional[IEngineModule]:
        """Get module by ID"""
        return self.modules.get(module_id)
    
    def _update_initialization_order(self):
        """Update module initialization order based on dependencies"""
        # Topological sort for dependency resolution
        visited = set()
        temp_mark = set()
        self.initialization_order = []
        
        def visit(module_id: str):
            if module_id in temp_mark:
                raise ValueError(f"Circular dependency detected involving {module_id}")
            if module_id in visited:
                return
            
            temp_mark.add(module_id)
            for dep in self.dependency_graph.get(module_id, []):
                if dep in self.dependency_graph:  # Only visit if dependency is registered
                    visit(dep)
            temp_mark.remove(module_id)
            visited.add(module_id)
            self.initialization_order.append(module_id)
        
        for module_id in self.modules.keys():
            if module_id not in visited:
                visit(module_id)

class EventBus:
    """Engine event bus for module communication"""
    
    def __init__(self):
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Dict[str, List[IEngineModule]] = {}
        self.event_history: List[EngineEvent] = []
        self.max_history = 1000
        self.running = False
    
    async def start(self):
        """Start event processing"""
        self.running = True
        asyncio.create_task(self._process_events())
        print("🔄 Event bus started")
    
    async def stop(self):
        """Stop event processing"""
        self.running = False
        # Give some time for cleanup but don't wait indefinitely
        try:
            await asyncio.wait_for(self.event_queue.join(), timeout=2.0)
        except asyncio.TimeoutError:
            print("⚠️ Event queue cleanup timed out, forcing stop")
        print("⏹️ Event bus stopped")
    
    def subscribe(self, event_type: str, module: IEngineModule):
        """Subscribe module to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(module)
    
    def unsubscribe(self, event_type: str, module: IEngineModule):
        """Unsubscribe module from event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type] = [m for m in self.subscribers[event_type] if m != module]
    
    async def emit(self, event: EngineEvent):
        """Emit event to bus"""
        await self.event_queue.put(event)
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
    
    async def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)
                await self._dispatch_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Error processing event: {e}")
    
    async def _dispatch_event(self, event: EngineEvent):
        """Dispatch event to subscribers"""
        subscribers = self.subscribers.get(event.event_type, [])
        
        # Add specific target if specified
        if event.target_module:
            # Implementation would need module registry access
            pass
        
        for module in subscribers:
            try:
                await module.handle_event({
                    "type": event.event_type,
                    "source": event.source_module,
                    "data": event.data,
                    "timestamp": event.timestamp
                })
            except Exception as e:
                print(f"❌ Error in module {getattr(module, 'module_id', 'unknown')}: {e}")

class ForgeEngine:
    """
    Main Forge Engine
    
    A modular, PCC-powered game engine with Claude collective intelligence.
    Designed for maximum efficiency, procedural generation, and AI-driven development.
    """
    
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self.module_registry = ModuleRegistry()
        self.event_bus = EventBus()
        
        # Engine state
        self.running = False
        self.start_time = 0.0
        self.frame_count = 0
        self.delta_time = 0.0
        self.last_frame_time = 0.0
        
        # Claude integration
        self.claude_enabled = self.config.enable_claude_intelligence
        self.engine_id = f"forge_engine_{int(time.time())}"
        
        # Performance metrics
        self.performance_metrics = {
            "fps": 0.0,
            "frame_time_ms": 0.0,
            "module_times": {},
            "memory_usage_mb": 0.0,
            "active_modules": 0
        }
        
        print(f"🔥 Forge Engine {self.config.version} initialized")
        if self.claude_enabled:
            print("🧠 Claude collective intelligence enabled")
    
    async def initialize(self):
        """Initialize the engine and all modules"""
        print("🚀 Initializing Forge Engine...")
        
        # Start event bus
        await self.event_bus.start()
        
        # Initialize modules in dependency order
        await self._initialize_modules()
        
        # Store initialization in Claude if enabled
        if self.claude_enabled:
            try:
                await self._store_engine_experience(
                    f"Forge Engine initialized with {len(self.module_registry.modules)} modules",
                    "initialization",
                    context={
                        "config": self.config.__dict__,
                        "modules": list(self.module_registry.modules.keys()),
                        "version": self.config.version
                    },
                    tags=["initialization", "engine", "forge"]
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not store initialization experience: {e}")
        
        # Emit initialization complete event
        await self.event_bus.emit(EngineEvent(
            event_type="engine.initialized",
            source_module="forge_engine",
            target_module=None,
            data={"config": self.config.__dict__}
        ))
        
        print("✅ Forge Engine initialization complete")
    
    async def start(self):
        """Start the engine main loop"""
        if self.running:
            print("⚠️ Engine already running")
            return
        
        self.running = True
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        print("🎬 Starting Forge Engine main loop...")
        
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            print("🛑 Engine interrupted by user")
        except Exception as e:
            print(f"❌ Engine error: {e}")
            if self.claude_enabled:
                try:
                    await self._store_engine_experience(
                        f"Engine error: {e}",
                        "error",
                        context={"error_type": type(e).__name__, "stack_trace": str(e)},
                        tags=["error", "engine", "critical"]
                    )
                except Exception as store_error:
                    print(f"⚠️ Could not store error experience: {store_error}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the engine and all modules"""
        print("🛑 Shutting down Forge Engine...")
        
        self.running = False
        
        # Shutdown modules in reverse order
        shutdown_order = list(reversed(self.module_registry.initialization_order))
        for module_id in shutdown_order:
            module = self.module_registry.get_module(module_id)
            if module:
                try:
                    await module.shutdown()
                    print(f"✅ Shutdown module: {module_id}")
                except Exception as e:
                    print(f"❌ Error shutting down {module_id}: {e}")
        
        # Stop event bus
        await self.event_bus.stop()
        
        # Store shutdown experience
        if self.claude_enabled:
            try:
                await self._store_engine_experience(
                    f"Forge Engine shutdown after {self.frame_count} frames",
                    "shutdown",
                    context={
                        "runtime_seconds": time.time() - self.start_time,
                        "total_frames": self.frame_count,
                        "final_performance": self.performance_metrics
                    },
                    tags=["shutdown", "engine", "metrics"]
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not store shutdown experience: {e}")
        
        print("🏁 Forge Engine shutdown complete")
    
    async def _main_loop(self):
        """Main engine loop"""
        target_frame_time = 1.0 / self.config.target_fps
        
        while self.running:
            frame_start = time.time()
            
            # Calculate delta time
            self.delta_time = frame_start - self.last_frame_time
            self.last_frame_time = frame_start
            
            # Update all modules
            await self._update_modules()
            
            # Update performance metrics
            self._update_performance_metrics(frame_start)
            
            # Frame rate limiting
            frame_time = time.time() - frame_start
            if frame_time < target_frame_time:
                await asyncio.sleep(target_frame_time - frame_time)
            
            self.frame_count += 1
            
            # Periodic Claude consultation for optimization
            if (self.claude_enabled and 
                self.frame_count % (self.config.target_fps * 10) == 0):  # Every 10 seconds
                await self._periodic_optimization()
    
    async def _initialize_modules(self):
        """Initialize all modules in dependency order"""
        for module_id in self.module_registry.initialization_order:
            module = self.module_registry.get_module(module_id)
            if module:
                try:
                    success = await module.initialize(self)
                    if success:
                        print(f"✅ Initialized module: {module_id}")
                    else:
                        print(f"❌ Failed to initialize module: {module_id}")
                except Exception as e:
                    print(f"❌ Error initializing {module_id}: {e}")
    
    async def _update_modules(self):
        """Update all modules"""
        module_start_times = {}
        
        for module_id in self.module_registry.initialization_order:
            module = self.module_registry.get_module(module_id)
            if module:
                try:
                    start_time = time.time()
                    await module.update(self.delta_time)
                    module_start_times[module_id] = (time.time() - start_time) * 1000  # ms
                except Exception as e:
                    print(f"❌ Error updating {module_id}: {e}")
        
        self.performance_metrics["module_times"] = module_start_times
    
    def _update_performance_metrics(self, frame_start: float):
        """Update engine performance metrics"""
        current_time = time.time()
        frame_time = current_time - frame_start
        
        self.performance_metrics.update({
            "fps": 1.0 / max(self.delta_time, 0.001),
            "frame_time_ms": frame_time * 1000,
            "active_modules": len(self.module_registry.modules),
            "total_runtime": current_time - self.start_time
        })
    
    async def _periodic_optimization(self):
        """Periodic optimization using Claude intelligence"""
        if not self.claude_enabled:
            return
        
        try:
            optimization_query = f"""
            Forge Engine Performance Analysis - Frame {self.frame_count}
            
            Current Metrics:
            - FPS: {self.performance_metrics['fps']:.1f}
            - Frame Time: {self.performance_metrics['frame_time_ms']:.2f}ms
            - Active Modules: {self.performance_metrics['active_modules']}
            - Module Times: {self.performance_metrics['module_times']}
            
            Engine Config:
            - Target FPS: {self.config.target_fps}
            - Max Concurrent Processes: {self.config.max_concurrent_processes}
            - Procedural Optimization: {self.config.procedural_optimization}
            
            Analyze performance and suggest optimizations:
            1. Module performance bottlenecks
            2. Memory usage optimization opportunities  
            3. Concurrent processing improvements
            4. PCC runtime optimization suggestions
            5. Overall engine architecture improvements
            
            Provide specific, actionable optimization recommendations.
            """
            
            optimization_result = await consult_collective_wisdom(
                optimization_query,
                self.engine_id,
                context={
                    "performance_metrics": self.performance_metrics,
                    "engine_config": self.config.__dict__,
                    "frame_count": self.frame_count
                }
            )
            
            # Apply optimizations if suggested
            await self._apply_claude_optimizations(optimization_result)
            
        except Exception as e:
            print(f"⚠️ Error in periodic optimization: {e}")
    
    async def _apply_claude_optimizations(self, optimization_result: Dict[str, Any]):
        """Apply optimizations suggested by Claude"""
        insights = optimization_result.get("insights", "")
        
        # Parse and apply optimizations (simplified implementation)
        if "increase max_concurrent_processes" in insights.lower():
            self.config.max_concurrent_processes = min(
                self.config.max_concurrent_processes + 1, 16
            )
            print(f"🔧 Increased max concurrent processes to {self.config.max_concurrent_processes}")
        
        if "enable procedural_optimization" in insights.lower():
            self.config.procedural_optimization = True
            print("🔧 Enabled procedural optimization")
        
        # Store optimization application
        await self._store_engine_experience(
            f"Applied Claude optimizations: {insights[:200]}...",
            "optimization",
            context={
                "optimization_insights": insights,
                "performance_before": self.performance_metrics.copy(),
                "config_changes": "automatic"
            },
            tags=["optimization", "claude", "performance"]
        )
    
    async def _store_engine_experience(self, experience: str, exp_type: str = "experience",
                                     context: Dict[str, Any] = None, tags: List[str] = None):
        """Store engine experience in Claude collective memory"""
        if not self.claude_enabled:
            return
        
        try:
            # store_agent_experience is not async, so don't await it
            store_agent_experience(
                experience,
                self.engine_id,
                exp_type,
                context,
                tags
            )
        except Exception as e:
            print(f"⚠️ Error storing engine experience: {e}")
    
    # Module management methods
    def register_module(self, module: IEngineModule):
        """Register a module with the engine"""
        self.module_registry.add_module(module)
        
        # Subscribe module to relevant events
        if hasattr(module, 'subscribed_events'):
            for event_type in module.subscribed_events:
                self.event_bus.subscribe(event_type, module)
    
    def unregister_module(self, module_id: str):
        """Unregister a module from the engine"""
        module = self.module_registry.get_module(module_id)
        if module:
            # Unsubscribe from all events
            for event_type, subscribers in self.event_bus.subscribers.items():
                self.event_bus.unsubscribe(event_type, module)
        
        self.module_registry.remove_module(module_id)
    
    def get_module(self, module_id: str) -> Optional[IEngineModule]:
        """Get a module by ID"""
        return self.module_registry.get_module(module_id)
    
    async def emit_event(self, event_type: str, source_module: str, 
                        data: Dict[str, Any], target_module: str = None):
        """Emit an event through the engine"""
        event = EngineEvent(
            event_type=event_type,
            source_module=source_module,
            target_module=target_module,
            data=data
        )
        await self.event_bus.emit(event)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current engine performance metrics"""
        return self.performance_metrics.copy()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            "running": self.running,
            "config": self.config.__dict__,
            "modules": list(self.module_registry.modules.keys()),
            "performance": self.performance_metrics,
            "frame_count": self.frame_count,
            "runtime_seconds": time.time() - self.start_time if self.start_time > 0 else 0,
            "claude_enabled": self.claude_enabled
        }

# Utility functions for creating and managing engine instances
_engine_instance: Optional[ForgeEngine] = None

def create_engine(config: EngineConfig = None) -> ForgeEngine:
    """Create a new Forge engine instance"""
    global _engine_instance
    _engine_instance = ForgeEngine(config)
    return _engine_instance

def get_engine() -> Optional[ForgeEngine]:
    """Get the current engine instance"""
    return _engine_instance

async def run_engine(config: EngineConfig = None) -> ForgeEngine:
    """Create and run a Forge engine"""
    engine = create_engine(config)
    await engine.initialize()
    await engine.start()
    return engine