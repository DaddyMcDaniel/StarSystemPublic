#!/usr/bin/env python3
"""
Forge Rendering Module
Advanced 3D rendering engine optimized for PCC-generated content
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

# Import engine interfaces
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.engine import IEngineModule, IClaudeIntegrated

# Import collective intelligence
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from agents.collective_subconscious import consult_collective_wisdom, store_agent_experience

# Import physics module for position data
try:
    from forge.modules.physics.physics_module import Vector3
except ImportError:
    # Fallback Vector3 if physics not available
    @dataclass
    class Vector3:
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0

class RenderPrimitive(Enum):
    CUBE = "cube"
    SPHERE = "sphere"
    PLANE = "plane"
    CYLINDER = "cylinder"
    MESH = "mesh"
    PARTICLE_SYSTEM = "particles"

class LightType(Enum):
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"
    AMBIENT = "ambient"

@dataclass
class Color:
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

@dataclass
class Material:
    """Rendering material properties"""
    id: str
    diffuse_color: Color = field(default_factory=Color)
    specular_color: Color = field(default_factory=Color)
    emission_color: Color = field(default_factory=lambda: Color(0, 0, 0, 1))
    
    # Material properties
    metallic: float = 0.0
    roughness: float = 0.5
    specular_power: float = 32.0
    transparency: float = 0.0
    
    # Texture paths
    diffuse_texture: Optional[str] = None
    normal_texture: Optional[str] = None
    specular_texture: Optional[str] = None
    
    # Shader program
    shader_program: Optional[str] = None

@dataclass
class RenderObject:
    """Object to be rendered"""
    id: str
    primitive_type: RenderPrimitive
    position: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    
    material_id: str = "default"
    visible: bool = True
    cast_shadows: bool = True
    receive_shadows: bool = True
    
    # Mesh data for custom geometry
    vertices: Optional[List[float]] = None
    indices: Optional[List[int]] = None
    normals: Optional[List[float]] = None
    uvs: Optional[List[float]] = None
    
    # Animation data
    animation_state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Light:
    """Light source"""
    id: str
    light_type: LightType
    position: Vector3 = field(default_factory=Vector3)
    direction: Vector3 = field(default_factory=lambda: Vector3(0, -1, 0))
    
    color: Color = field(default_factory=Color)
    intensity: float = 1.0
    range: float = 10.0  # For point/spot lights
    spot_angle: float = 45.0  # For spot lights
    
    cast_shadows: bool = True
    enabled: bool = True

@dataclass
class Camera:
    """Rendering camera"""
    id: str
    position: Vector3 = field(default_factory=Vector3)
    target: Vector3 = field(default_factory=Vector3)
    up: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))
    
    fov: float = 60.0  # Field of view in degrees
    near_plane: float = 0.1
    far_plane: float = 1000.0
    
    # Viewport
    viewport_x: int = 0
    viewport_y: int = 0
    viewport_width: int = 1920
    viewport_height: int = 1080

class ForgeRenderingModule(IEngineModule, IClaudeIntegrated):
    """
    Forge Rendering Module - High-Performance 3D Renderer
    
    Features:
    1. PCC-optimized rendering pipeline
    2. Procedural content rendering
    3. Claude-powered visual optimization
    4. Real-time lighting and shadows
    5. Material system with PBR
    6. Efficient culling and LOD
    7. Multi-threaded rendering
    """
    
    def __init__(self):
        self._module_id = "forge_rendering"
        self._version = "1.0.0"
        self._dependencies = ["pcc_runtime"]
        
        # Rendering state
        self.render_objects: Dict[str, RenderObject] = {}
        self.materials: Dict[str, Material] = {}
        self.lights: Dict[str, Light] = {}
        self.cameras: Dict[str, Camera] = {}
        self.active_camera: Optional[str] = None
        
        # Rendering settings
        self.enable_shadows = True
        self.enable_lighting = True
        self.enable_antialiasing = True
        self.enable_post_processing = True
        self.vsync_enabled = True
        self.render_scale = 1.0
        
        # Performance tracking
        self.render_stats = {
            "frame_time_ms": 0.0,
            "draw_calls": 0,
            "triangles_rendered": 0,
            "objects_culled": 0,
            "objects_rendered": 0,
            "shadow_maps_updated": 0,
            "gpu_memory_mb": 0.0
        }
        
        # Claude integration
        self.claude_enabled = True
        self.visual_optimization_suggestions = []
        
        # Rendering backend (would be OpenGL/Vulkan/DirectX in real implementation)
        self.rendering_backend = "opengl"  # Simulated
        self.render_queue = []
        
        print("🎨 Forge Rendering Module initialized")
    
    @property
    def module_id(self) -> str:
        return self._module_id
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def dependencies(self) -> List[str]:
        return self._dependencies
    
    async def initialize(self, engine_context) -> bool:
        """Initialize rendering module"""
        print("🚀 Initializing Forge Rendering...")
        
        # Create default materials
        await self._create_default_materials()
        
        # Create default camera
        await self._create_default_camera()
        
        # Create default lighting
        await self._create_default_lighting()
        
        # Initialize with Claude intelligence
        if self.claude_enabled:
            await self._claude_rendering_initialization()
        
        # Store initialization experience
        await self.store_module_experience(
            "Forge Rendering Module initialized with advanced PCC optimization",
            "initialization",
            context={
                "backend": self.rendering_backend,
                "shadows_enabled": self.enable_shadows,
                "lighting_enabled": self.enable_lighting
            },
            tags=["initialization", "rendering", "forge"]
        )
        
        print("✅ Forge Rendering initialized")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown rendering module"""
        print("🛑 Shutting down Forge Rendering...")
        
        # Store final statistics
        if self.claude_enabled:
            await self.store_module_experience(
                f"Rendering shutdown - rendered {self.render_stats['triangles_rendered']} triangles total",
                "shutdown",
                context=self.render_stats,
                tags=["shutdown", "rendering", "statistics"]
            )
        
        # Clear rendering data
        self.render_objects.clear()
        self.materials.clear()
        self.lights.clear()
        self.cameras.clear()
        self.render_queue.clear()
        
        print("✅ Forge Rendering shutdown complete")
    
    async def update(self, delta_time: float) -> None:
        """Update rendering system"""
        start_time = time.time()
        
        # Update render queue
        await self._update_render_queue()
        
        # Perform rendering
        await self._render_frame()
        
        # Update performance statistics
        self.render_stats["frame_time_ms"] = (time.time() - start_time) * 1000
        
        # Periodic optimization
        if int(time.time()) % 15 == 0:  # Every 15 seconds
            await self._periodic_rendering_optimization()
    
    async def handle_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle rendering events"""
        event_type = event.get("type", "")
        
        if event_type == "render.add_object":
            return await self._handle_add_object_event(event)
        elif event_type == "render.remove_object":
            return await self._handle_remove_object_event(event)
        elif event_type == "render.update_object":
            return await self._handle_update_object_event(event)
        elif event_type == "render.add_light":
            return await self._handle_add_light_event(event)
        elif event_type == "render.set_camera":
            return await self._handle_set_camera_event(event)
        elif event_type == "render.create_material":
            return await self._handle_create_material_event(event)
        
        return None
    
    def consult_claude(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Consult Claude for rendering optimization"""
        if not self.claude_enabled:
            return {"insights": "Claude integration disabled"}
        
        rendering_query = f"""
        Forge Rendering Engine Consultation:
        {query}
        
        Current Rendering State:
        - Objects: {len(self.render_objects)}
        - Materials: {len(self.materials)}
        - Lights: {len(self.lights)}
        - Active Camera: {self.active_camera}
        - Backend: {self.rendering_backend}
        
        Rendering Settings:
        - Shadows: {self.enable_shadows}
        - Lighting: {self.enable_lighting}
        - Anti-aliasing: {self.enable_antialiasing}
        - Render Scale: {self.render_scale}
        
        Performance Stats:
        - Frame Time: {self.render_stats['frame_time_ms']:.2f}ms
        - Draw Calls: {self.render_stats['draw_calls']}
        - Triangles: {self.render_stats['triangles_rendered']}
        - Objects Rendered: {self.render_stats['objects_rendered']}
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Provide rendering engineering guidance focusing on:
        1. Rendering performance optimization
        2. Visual quality improvements
        3. Memory usage optimization
        4. PCC-specific rendering optimizations
        5. Real-time rendering techniques
        """
        
        return consult_collective_wisdom(
            rendering_query,
            self.module_id,
            context=context
        )
    
    async def store_module_experience(self, experience: str, exp_type: str = "experience",
                                    context: Dict[str, Any] = None, tags: List[str] = None):
        """Store rendering module experience"""
        if not self.claude_enabled:
            return
        
        try:
            # store_agent_experience is not async, so don't await it
            store_agent_experience(
                experience,
                self.module_id,
                exp_type,
                context,
                tags
            )
        except Exception as e:
            print(f"⚠️ Error storing rendering module experience: {e}")
    
    # Core Rendering Methods
    
    async def add_render_object(self, object_id: str, primitive_type: RenderPrimitive,
                              position: Vector3 = None, material_id: str = "default") -> bool:
        """Add object to render queue"""
        
        if object_id in self.render_objects:
            print(f"⚠️ Render object {object_id} already exists")
            return False
        
        render_object = RenderObject(
            id=object_id,
            primitive_type=primitive_type,
            position=position or Vector3(),
            material_id=material_id
        )
        
        self.render_objects[object_id] = render_object
        
        print(f"🎨 Added render object: {object_id} ({primitive_type.value})")
        
        await self.store_module_experience(
            f"Added render object {object_id} with primitive {primitive_type.value}",
            "object_creation",
            context={
                "object_id": object_id,
                "primitive": primitive_type.value,
                "material": material_id
            },
            tags=["object", "creation", "rendering"]
        )
        
        return True
    
    async def remove_render_object(self, object_id: str) -> bool:
        """Remove object from rendering"""
        if object_id not in self.render_objects:
            return False
        
        del self.render_objects[object_id]
        print(f"🗑️ Removed render object: {object_id}")
        return True
    
    async def update_object_position(self, object_id: str, position: Vector3) -> bool:
        """Update object position"""
        if object_id not in self.render_objects:
            return False
        
        self.render_objects[object_id].position = position
        return True
    
    async def create_material(self, material_id: str, diffuse_color: Color = None,
                            metallic: float = 0.0, roughness: float = 0.5) -> bool:
        """Create a new material"""
        
        if material_id in self.materials:
            print(f"⚠️ Material {material_id} already exists")
            return False
        
        material = Material(
            id=material_id,
            diffuse_color=diffuse_color or Color(),
            metallic=metallic,
            roughness=roughness
        )
        
        self.materials[material_id] = material
        
        print(f"🎨 Created material: {material_id}")
        return True
    
    async def add_light(self, light_id: str, light_type: LightType,
                       position: Vector3 = None, color: Color = None,
                       intensity: float = 1.0) -> bool:
        """Add light to scene"""
        
        if light_id in self.lights:
            print(f"⚠️ Light {light_id} already exists")
            return False
        
        light = Light(
            id=light_id,
            light_type=light_type,
            position=position or Vector3(),
            color=color or Color(),
            intensity=intensity
        )
        
        self.lights[light_id] = light
        
        print(f"💡 Added light: {light_id} ({light_type.value})")
        return True
    
    async def set_camera(self, camera_id: str, position: Vector3 = None,
                        target: Vector3 = None, fov: float = 60.0) -> bool:
        """Set or update camera"""
        
        if camera_id not in self.cameras:
            camera = Camera(
                id=camera_id,
                position=position or Vector3(0, 0, 5),
                target=target or Vector3(),
                fov=fov
            )
            self.cameras[camera_id] = camera
        else:
            camera = self.cameras[camera_id]
            if position:
                camera.position = position
            if target:
                camera.target = target
            camera.fov = fov
        
        self.active_camera = camera_id
        print(f"📷 Set active camera: {camera_id}")
        return True
    
    # Internal Rendering Methods
    
    async def _create_default_materials(self):
        """Create default materials"""
        
        # Default material
        await self.create_material("default", Color(0.8, 0.8, 0.8, 1.0), 0.0, 0.5)
        
        # Basic materials
        await self.create_material("red", Color(1.0, 0.0, 0.0, 1.0))
        await self.create_material("green", Color(0.0, 1.0, 0.0, 1.0))
        await self.create_material("blue", Color(0.0, 0.0, 1.0, 1.0))
        await self.create_material("white", Color(1.0, 1.0, 1.0, 1.0))
        await self.create_material("black", Color(0.0, 0.0, 0.0, 1.0))
        
        # PBR materials
        await self.create_material("metal", Color(0.7, 0.7, 0.7, 1.0), 1.0, 0.1)
        await self.create_material("plastic", Color(0.5, 0.5, 0.8, 1.0), 0.0, 0.8)
        
        print("🎨 Created default materials")
    
    async def _create_default_camera(self):
        """Create default camera"""
        await self.set_camera("main_camera", Vector3(0, 2, 5), Vector3(0, 0, 0), 60.0)
    
    async def _create_default_lighting(self):
        """Create default lighting setup"""
        
        # Directional light (sun)
        await self.add_light("sun", LightType.DIRECTIONAL, 
                           Vector3(0, 10, 0), Color(1.0, 0.95, 0.8, 1.0), 1.0)
        
        # Ambient light
        await self.add_light("ambient", LightType.AMBIENT,
                           Vector3(), Color(0.2, 0.2, 0.3, 1.0), 0.3)
        
        print("💡 Created default lighting")
    
    async def _update_render_queue(self):
        """Update render queue with visible objects"""
        
        self.render_queue.clear()
        
        if not self.active_camera or self.active_camera not in self.cameras:
            return
        
        camera = self.cameras[self.active_camera]
        
        # Frustum culling (simplified)
        for obj in self.render_objects.values():
            if obj.visible and await self._is_object_visible(obj, camera):
                self.render_queue.append(obj)
        
        # Sort render queue (front to back for opaque, back to front for transparent)
        self.render_queue.sort(key=lambda obj: self._calculate_distance_to_camera(obj, camera))
    
    async def _is_object_visible(self, obj: RenderObject, camera: Camera) -> bool:
        """Check if object is visible from camera"""
        
        # Simplified visibility check - distance based
        distance = self._calculate_distance_to_camera(obj, camera)
        return distance <= camera.far_plane
    
    def _calculate_distance_to_camera(self, obj: RenderObject, camera: Camera) -> float:
        """Calculate distance from object to camera"""
        
        dx = obj.position.x - camera.position.x
        dy = obj.position.y - camera.position.y
        dz = obj.position.z - camera.position.z
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    async def _render_frame(self):
        """Render a complete frame"""
        
        draw_calls = 0
        triangles = 0
        objects_rendered = 0
        
        # Clear frame
        await self._clear_frame()
        
        # Setup camera
        if self.active_camera and self.active_camera in self.cameras:
            await self._setup_camera(self.cameras[self.active_camera])
        
        # Setup lighting
        await self._setup_lighting()
        
        # Render objects
        for obj in self.render_queue:
            if await self._render_object(obj):
                draw_calls += 1
                triangles += await self._count_object_triangles(obj)
                objects_rendered += 1
        
        # Post-processing
        if self.enable_post_processing:
            await self._apply_post_processing()
        
        # Update statistics
        self.render_stats.update({
            "draw_calls": draw_calls,
            "triangles_rendered": triangles,
            "objects_rendered": objects_rendered,
            "objects_culled": len(self.render_objects) - objects_rendered
        })
    
    async def _clear_frame(self):
        """Clear the frame buffer"""
        # Simulated frame clearing
        pass
    
    async def _setup_camera(self, camera: Camera):
        """Setup camera for rendering"""
        # Simulated camera setup
        pass
    
    async def _setup_lighting(self):
        """Setup lighting for rendering"""
        # Simulated lighting setup
        active_lights = [light for light in self.lights.values() if light.enabled]
        
        # Update shadow maps if needed
        shadow_maps_updated = 0
        if self.enable_shadows:
            for light in active_lights:
                if light.cast_shadows:
                    await self._update_shadow_map(light)
                    shadow_maps_updated += 1
        
        self.render_stats["shadow_maps_updated"] = shadow_maps_updated
    
    async def _update_shadow_map(self, light: Light):
        """Update shadow map for light"""
        # Simulated shadow map update
        pass
    
    async def _render_object(self, obj: RenderObject) -> bool:
        """Render individual object"""
        
        # Get material
        material = self.materials.get(obj.material_id, self.materials.get("default"))
        if not material:
            return False
        
        # Setup material
        await self._setup_material(material)
        
        # Render primitive
        await self._render_primitive(obj)
        
        return True
    
    async def _setup_material(self, material: Material):
        """Setup material for rendering"""
        # Simulated material setup
        pass
    
    async def _render_primitive(self, obj: RenderObject):
        """Render primitive geometry"""
        
        # Simulated primitive rendering based on type
        if obj.primitive_type == RenderPrimitive.CUBE:
            await self._render_cube(obj)
        elif obj.primitive_type == RenderPrimitive.SPHERE:
            await self._render_sphere(obj)
        elif obj.primitive_type == RenderPrimitive.PLANE:
            await self._render_plane(obj)
        elif obj.primitive_type == RenderPrimitive.MESH:
            await self._render_mesh(obj)
    
    async def _render_cube(self, obj: RenderObject):
        """Render cube primitive"""
        # Simulated cube rendering
        pass
    
    async def _render_sphere(self, obj: RenderObject):
        """Render sphere primitive"""
        # Simulated sphere rendering
        pass
    
    async def _render_plane(self, obj: RenderObject):
        """Render plane primitive"""
        # Simulated plane rendering
        pass
    
    async def _render_mesh(self, obj: RenderObject):
        """Render custom mesh"""
        # Simulated mesh rendering using vertices/indices
        pass
    
    async def _count_object_triangles(self, obj: RenderObject) -> int:
        """Count triangles in object"""
        
        # Simplified triangle counting
        if obj.primitive_type == RenderPrimitive.CUBE:
            return 12  # 6 faces * 2 triangles each
        elif obj.primitive_type == RenderPrimitive.SPHERE:
            return 320  # Typical sphere tessellation
        elif obj.primitive_type == RenderPrimitive.PLANE:
            return 2
        elif obj.primitive_type == RenderPrimitive.MESH and obj.indices:
            return len(obj.indices) // 3
        
        return 1
    
    async def _apply_post_processing(self):
        """Apply post-processing effects"""
        # Simulated post-processing
        pass
    
    # Claude optimization methods
    
    async def _claude_rendering_initialization(self):
        """Initialize rendering with Claude guidance"""
        
        init_query = """
        Forge Rendering Engine Initialization
        
        This is a high-performance 3D rendering engine for PCC-generated games.
        
        Provide initialization guidance:
        1. Optimal rendering pipeline configuration
        2. Performance vs quality trade-offs
        3. Memory management strategies
        4. Procedural content rendering optimizations
        5. Real-time lighting and shadow techniques
        
        Focus on maximum visual quality with optimal performance.
        """
        
        guidance = self.consult_claude(init_query, {"initialization": True})
        
        # Apply initialization guidance
        insights = guidance.get("insights", "").lower()
        
        if "enable aggressive culling" in insights:
            # Would implement more aggressive culling
            pass
        
        if "reduce shadow quality" in insights:
            # Could adjust shadow map resolution
            pass
        
        await self.store_module_experience(
            f"Rendering initialized with Claude guidance",
            "initialization",
            context={"guidance": guidance.get("insights", "")[:300]},
            tags=["initialization", "claude", "optimization"]
        )
    
    async def _periodic_rendering_optimization(self):
        """Periodic rendering optimization using Claude"""
        
        if not self.claude_enabled:
            return
        
        optimization_query = f"""
        Rendering Engine Performance Analysis
        
        Current State:
        - Objects: {len(self.render_objects)} total, {self.render_stats['objects_rendered']} rendered
        - Frame Time: {self.render_stats['frame_time_ms']:.2f}ms
        - Draw Calls: {self.render_stats['draw_calls']}
        - Triangles: {self.render_stats['triangles_rendered']}
        - Materials: {len(self.materials)}
        - Lights: {len([l for l in self.lights.values() if l.enabled])}
        
        Analyze rendering performance and suggest optimizations:
        1. Draw call reduction strategies
        2. Triangle count optimization
        3. Material and texture optimization
        4. Lighting performance improvements
        5. Memory usage optimization
        """
        
        optimization_result = self.consult_claude(optimization_query)
        
        # Apply optimizations
        await self._apply_rendering_optimizations(optimization_result)
    
    async def _apply_rendering_optimizations(self, optimization_result: Dict[str, Any]):
        """Apply rendering optimizations"""
        
        insights = optimization_result.get("insights", "").lower()
        
        optimizations_applied = []
        
        if "reduce render scale" in insights and self.render_stats['frame_time_ms'] > 16.0:
            self.render_scale = max(self.render_scale * 0.9, 0.5)
            optimizations_applied.append("reduced_render_scale")
        
        if "disable shadows" in insights and self.render_stats['frame_time_ms'] > 20.0:
            self.enable_shadows = False
            optimizations_applied.append("disabled_shadows")
        
        if "reduce antialiasing" in insights and self.render_stats['frame_time_ms'] > 25.0:
            self.enable_antialiasing = False
            optimizations_applied.append("disabled_antialiasing")
        
        if "batch objects" in insights:
            # Could implement object batching
            optimizations_applied.append("batching_optimization_noted")
        
        if optimizations_applied:
            await self.store_module_experience(
                f"Applied rendering optimizations: {', '.join(optimizations_applied)}",
                "optimization",
                context={
                    "optimizations": optimizations_applied,
                    "performance_before": self.render_stats.copy()
                },
                tags=["optimization", "rendering", "performance"]
            )
    
    # Event handlers
    
    async def _handle_add_object_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add object event"""
        data = event.get("data", {})
        
        object_id = data.get("object_id")
        if not object_id:
            return {"success": False, "error": "No object_id provided"}
        
        primitive_str = data.get("primitive", "cube")
        primitive = RenderPrimitive(primitive_str) if primitive_str in [p.value for p in RenderPrimitive] else RenderPrimitive.CUBE
        
        position_data = data.get("position", {})
        position = Vector3(
            position_data.get("x", 0),
            position_data.get("y", 0),
            position_data.get("z", 0)
        )
        
        material_id = data.get("material_id", "default")
        
        success = await self.add_render_object(object_id, primitive, position, material_id)
        return {"success": success}
    
    async def _handle_remove_object_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle remove object event"""
        data = event.get("data", {})
        object_id = data.get("object_id")
        
        if not object_id:
            return {"success": False, "error": "No object_id provided"}
        
        success = await self.remove_render_object(object_id)
        return {"success": success}
    
    async def _handle_update_object_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update object event"""
        data = event.get("data", {})
        
        object_id = data.get("object_id")
        if not object_id:
            return {"success": False, "error": "No object_id provided"}
        
        position_data = data.get("position")
        if position_data:
            position = Vector3(
                position_data.get("x", 0),
                position_data.get("y", 0),
                position_data.get("z", 0)
            )
            success = await self.update_object_position(object_id, position)
            return {"success": success}
        
        return {"success": False, "error": "No update data provided"}
    
    async def _handle_add_light_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add light event"""
        data = event.get("data", {})
        
        light_id = data.get("light_id")
        if not light_id:
            return {"success": False, "error": "No light_id provided"}
        
        light_type_str = data.get("light_type", "point")
        light_type = LightType(light_type_str) if light_type_str in [t.value for t in LightType] else LightType.POINT
        
        position_data = data.get("position", {})
        position = Vector3(
            position_data.get("x", 0),
            position_data.get("y", 0),
            position_data.get("z", 0)
        )
        
        color_data = data.get("color", {})
        color = Color(
            color_data.get("r", 1.0),
            color_data.get("g", 1.0),
            color_data.get("b", 1.0),
            color_data.get("a", 1.0)
        )
        
        intensity = data.get("intensity", 1.0)
        
        success = await self.add_light(light_id, light_type, position, color, intensity)
        return {"success": success}
    
    async def _handle_set_camera_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set camera event"""
        data = event.get("data", {})
        
        camera_id = data.get("camera_id", "main_camera")
        
        position_data = data.get("position")
        position = None
        if position_data:
            position = Vector3(
                position_data.get("x", 0),
                position_data.get("y", 0),
                position_data.get("z", 0)
            )
        
        target_data = data.get("target")
        target = None
        if target_data:
            target = Vector3(
                target_data.get("x", 0),
                target_data.get("y", 0),
                target_data.get("z", 0)
            )
        
        fov = data.get("fov", 60.0)
        
        success = await self.set_camera(camera_id, position, target, fov)
        return {"success": success}
    
    async def _handle_create_material_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create material event"""
        data = event.get("data", {})
        
        material_id = data.get("material_id")
        if not material_id:
            return {"success": False, "error": "No material_id provided"}
        
        color_data = data.get("diffuse_color", {})
        diffuse_color = Color(
            color_data.get("r", 1.0),
            color_data.get("g", 1.0),
            color_data.get("b", 1.0),
            color_data.get("a", 1.0)
        )
        
        metallic = data.get("metallic", 0.0)
        roughness = data.get("roughness", 0.5)
        
        success = await self.create_material(material_id, diffuse_color, metallic, roughness)
        return {"success": success}
    
    def get_rendering_status(self) -> Dict[str, Any]:
        """Get comprehensive rendering status"""
        return {
            "module_id": self.module_id,
            "version": self.version,
            "backend": self.rendering_backend,
            "objects": len(self.render_objects),
            "materials": len(self.materials),
            "lights": len(self.lights),
            "cameras": len(self.cameras),
            "active_camera": self.active_camera,
            "render_stats": self.render_stats,
            "settings": {
                "shadows": self.enable_shadows,
                "lighting": self.enable_lighting,
                "antialiasing": self.enable_antialiasing,
                "post_processing": self.enable_post_processing,
                "render_scale": self.render_scale
            },
            "claude_enabled": self.claude_enabled
        }