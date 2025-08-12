#!/usr/bin/env python3
"""
Forge Physics Module
Advanced physics engine designed for PCC-generated games with Claude optimization
"""

import asyncio
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import engine interfaces
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.engine import IEngineModule, IClaudeIntegrated

# Import collective intelligence
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from agents.collective_subconscious import consult_collective_wisdom, store_agent_experience

class CollisionShape(Enum):
    BOX = "box"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    MESH = "mesh"
    HEIGHTFIELD = "heightfield"

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> 'Vector3':
        mag = self.magnitude()
        if mag == 0:
            return Vector3()
        return Vector3(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

@dataclass
class PhysicsBody:
    """Represents a physics body in the simulation"""
    id: str
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.5  # Bounciness
    
    collision_shape: CollisionShape = CollisionShape.BOX
    collision_size: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    
    is_static: bool = False
    is_kinematic: bool = False
    is_trigger: bool = False
    
    # Runtime state
    forces: List[Vector3] = field(default_factory=list)
    collisions: List[str] = field(default_factory=list)
    last_update: float = 0.0

@dataclass
class CollisionInfo:
    """Information about a collision"""
    body_a: str
    body_b: str
    contact_point: Vector3
    contact_normal: Vector3
    penetration_depth: float
    collision_time: float

@dataclass
class PhysicsConstraint:
    """Physics constraint between bodies"""
    id: str
    body_a: str
    body_b: str
    constraint_type: str  # "spring", "distance", "hinge", etc.
    parameters: Dict[str, Any]

class ForgePhysicsModule(IEngineModule, IClaudeIntegrated):
    """
    Forge Physics Module - PCC-Optimized Physics Engine
    
    Features:
    1. High-performance 3D physics simulation
    2. PCC-optimized collision detection
    3. Claude-powered optimization suggestions
    4. Procedural physics parameter tuning
    5. Advanced constraint systems
    6. Real-time performance monitoring
    """
    
    def __init__(self):
        self._module_id = "forge_physics"
        self._version = "1.0.0"
        self._dependencies = ["pcc_runtime"]  # Depends on PCC runtime
        
        # Physics world state
        self.bodies: Dict[str, PhysicsBody] = {}
        self.constraints: Dict[str, PhysicsConstraint] = {}
        self.collision_pairs: List[CollisionInfo] = []
        
        # Physics settings
        self.gravity = Vector3(0, -9.81, 0)
        self.time_step = 1.0 / 120.0  # 120Hz physics
        self.max_substeps = 8
        self.solver_iterations = 10
        
        # Performance tracking
        self.physics_stats = {
            "total_bodies": 0,
            "active_bodies": 0,
            "collision_checks": 0,
            "collisions_detected": 0,
            "physics_time_ms": 0.0,
            "solver_time_ms": 0.0,
            "broadphase_time_ms": 0.0
        }
        
        # Claude integration
        self.claude_enabled = True
        self.optimization_suggestions = []
        self.last_optimization_check = 0.0
        
        # Spatial partitioning for optimization
        self.spatial_grid_size = 10.0
        self.spatial_grid: Dict[Tuple[int, int, int], List[str]] = {}
        
        print("🔬 Forge Physics Module initialized")
    
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
        """Initialize physics module"""
        print("🚀 Initializing Forge Physics...")
        
        # Initialize with Claude intelligence
        if self.claude_enabled:
            await self._claude_physics_initialization()
        
        # Store initialization experience
        await self.store_module_experience(
            "Forge Physics Module initialized with advanced PCC optimization",
            "initialization",
            context={
                "gravity": {"x": self.gravity.x, "y": self.gravity.y, "z": self.gravity.z},
                "time_step": self.time_step,
                "solver_iterations": self.solver_iterations
            },
            tags=["initialization", "physics", "forge"]
        )
        
        print("✅ Forge Physics initialized")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown physics module"""
        print("🛑 Shutting down Forge Physics...")
        
        # Store final statistics
        if self.claude_enabled:
            await self.store_module_experience(
                f"Physics shutdown - simulated {self.physics_stats['total_bodies']} bodies total",
                "shutdown",
                context=self.physics_stats,
                tags=["shutdown", "physics", "statistics"]
            )
        
        # Clear physics world
        self.bodies.clear()
        self.constraints.clear()
        self.collision_pairs.clear()
        self.spatial_grid.clear()
        
        print("✅ Forge Physics shutdown complete")
    
    async def update(self, delta_time: float) -> None:
        """Update physics simulation"""
        start_time = time.time()
        
        # Step physics simulation
        await self._step_physics(delta_time)
        
        # Update performance statistics
        self.physics_stats["physics_time_ms"] = (time.time() - start_time) * 1000
        self.physics_stats["total_bodies"] = len(self.bodies)
        self.physics_stats["active_bodies"] = len([b for b in self.bodies.values() if not b.is_static])
        
        # Periodic optimization check
        if time.time() - self.last_optimization_check > 10.0:  # Every 10 seconds
            await self._periodic_physics_optimization()
            self.last_optimization_check = time.time()
    
    async def handle_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle physics events"""
        event_type = event.get("type", "")
        
        if event_type == "physics.add_body":
            return await self._handle_add_body_event(event)
        elif event_type == "physics.remove_body":
            return await self._handle_remove_body_event(event)
        elif event_type == "physics.apply_force":
            return await self._handle_apply_force_event(event)
        elif event_type == "physics.set_gravity":
            return await self._handle_set_gravity_event(event)
        elif event_type == "physics.query_collision":
            return await self._handle_collision_query_event(event)
        
        return None
    
    def consult_claude(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Consult Claude for physics optimization"""
        if not self.claude_enabled:
            return {"insights": "Claude integration disabled"}
        
        physics_query = f"""
        Forge Physics Engine Consultation:
        {query}
        
        Current Physics State:
        - Bodies: {len(self.bodies)} ({self.physics_stats['active_bodies']} active)
        - Constraints: {len(self.constraints)}
        - Gravity: {self.gravity.x}, {self.gravity.y}, {self.gravity.z}
        - Time Step: {self.time_step}s
        - Solver Iterations: {self.solver_iterations}
        
        Performance Stats:
        - Physics Time: {self.physics_stats['physics_time_ms']:.2f}ms
        - Collision Checks: {self.physics_stats['collision_checks']}
        - Collisions Detected: {self.physics_stats['collisions_detected']}
        - Solver Time: {self.physics_stats['solver_time_ms']:.2f}ms
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Provide physics engineering guidance focusing on:
        1. Performance optimization strategies
        2. Collision detection improvements
        3. Solver stability enhancements
        4. Memory usage optimizations
        5. PCC-specific physics optimizations
        """
        
        return consult_collective_wisdom(
            physics_query,
            self.module_id,
            context=context
        )
    
    async def store_module_experience(self, experience: str, exp_type: str = "experience",
                                    context: Dict[str, Any] = None, tags: List[str] = None):
        """Store physics module experience"""
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
            print(f"⚠️ Error storing physics module experience: {e}")
    
    # Core Physics Methods
    
    async def add_physics_body(self, body_id: str, position: Vector3 = None,
                             shape: CollisionShape = CollisionShape.BOX,
                             size: Vector3 = None, mass: float = 1.0,
                             is_static: bool = False) -> bool:
        """Add a physics body to the simulation"""
        
        if body_id in self.bodies:
            print(f"⚠️ Body {body_id} already exists")
            return False
        
        body = PhysicsBody(
            id=body_id,
            position=position or Vector3(),
            collision_shape=shape,
            collision_size=size or Vector3(1, 1, 1),
            mass=mass,
            is_static=is_static,
            last_update=time.time()
        )
        
        self.bodies[body_id] = body
        await self._update_spatial_grid(body)
        
        print(f"📦 Added physics body: {body_id}")
        
        await self.store_module_experience(
            f"Added physics body {body_id} with shape {shape.value}",
            "body_creation",
            context={
                "body_id": body_id,
                "shape": shape.value,
                "mass": mass,
                "is_static": is_static
            },
            tags=["body", "creation", "physics"]
        )
        
        return True
    
    async def remove_physics_body(self, body_id: str) -> bool:
        """Remove a physics body from the simulation"""
        if body_id not in self.bodies:
            return False
        
        body = self.bodies[body_id]
        await self._remove_from_spatial_grid(body)
        del self.bodies[body_id]
        
        # Remove any constraints involving this body
        constraints_to_remove = [
            c_id for c_id, constraint in self.constraints.items()
            if constraint.body_a == body_id or constraint.body_b == body_id
        ]
        
        for constraint_id in constraints_to_remove:
            del self.constraints[constraint_id]
        
        print(f"🗑️ Removed physics body: {body_id}")
        return True
    
    async def apply_force(self, body_id: str, force: Vector3) -> bool:
        """Apply force to a physics body"""
        if body_id not in self.bodies:
            return False
        
        body = self.bodies[body_id]
        if body.is_static:
            return False
        
        body.forces.append(force)
        return True
    
    async def set_body_position(self, body_id: str, position: Vector3) -> bool:
        """Set physics body position"""
        if body_id not in self.bodies:
            return False
        
        body = self.bodies[body_id]
        old_position = body.position
        body.position = position
        
        # Update spatial grid
        await self._update_spatial_grid(body)
        
        return True
    
    async def get_body_position(self, body_id: str) -> Optional[Vector3]:
        """Get physics body position"""
        if body_id not in self.bodies:
            return None
        
        return self.bodies[body_id].position
    
    async def add_constraint(self, constraint_id: str, body_a: str, body_b: str,
                           constraint_type: str, parameters: Dict[str, Any] = None) -> bool:
        """Add a physics constraint"""
        
        if constraint_id in self.constraints:
            return False
        
        if body_a not in self.bodies or body_b not in self.bodies:
            return False
        
        constraint = PhysicsConstraint(
            id=constraint_id,
            body_a=body_a,
            body_b=body_b,
            constraint_type=constraint_type,
            parameters=parameters or {}
        )
        
        self.constraints[constraint_id] = constraint
        
        print(f"🔗 Added constraint: {constraint_id} ({constraint_type})")
        return True
    
    async def query_collisions(self, body_id: str) -> List[CollisionInfo]:
        """Query collisions for a specific body"""
        return [collision for collision in self.collision_pairs 
                if collision.body_a == body_id or collision.body_b == body_id]
    
    # Internal Physics Engine Methods
    
    async def _step_physics(self, delta_time: float):
        """Step the physics simulation"""
        
        # Use fixed timestep with substeps for stability
        remaining_time = delta_time
        substeps = 0
        
        while remaining_time > 0 and substeps < self.max_substeps:
            step_time = min(self.time_step, remaining_time)
            
            # Integration step
            await self._integrate_bodies(step_time)
            
            # Collision detection
            collision_start = time.time()
            await self._detect_collisions()
            self.physics_stats["broadphase_time_ms"] = (time.time() - collision_start) * 1000
            
            # Constraint solving
            solver_start = time.time()
            await self._solve_constraints(step_time)
            self.physics_stats["solver_time_ms"] = (time.time() - solver_start) * 1000
            
            # Collision response
            await self._resolve_collisions(step_time)
            
            remaining_time -= step_time
            substeps += 1
    
    async def _integrate_bodies(self, delta_time: float):
        """Integrate physics bodies forward in time"""
        
        for body in self.bodies.values():
            if body.is_static:
                continue
            
            # Apply gravity
            if not body.is_kinematic:
                body.acceleration = self.gravity
                
                # Apply forces
                for force in body.forces:
                    body.acceleration = body.acceleration + (force * (1.0 / body.mass))
            
            # Integrate velocity
            body.velocity = body.velocity + (body.acceleration * delta_time)
            
            # Apply damping
            damping = 0.99
            body.velocity = body.velocity * damping
            
            # Integrate position
            body.position = body.position + (body.velocity * delta_time)
            
            # Clear forces
            body.forces.clear()
            
            # Update spatial grid
            await self._update_spatial_grid(body)
    
    async def _detect_collisions(self):
        """Detect collisions between physics bodies"""
        
        self.collision_pairs.clear()
        collision_checks = 0
        
        # Broad phase - spatial grid optimization
        potential_pairs = await self._get_potential_collision_pairs()
        
        # Narrow phase - detailed collision detection
        for body_a_id, body_b_id in potential_pairs:
            body_a = self.bodies[body_a_id]
            body_b = self.bodies[body_b_id]
            
            collision_checks += 1
            
            # Skip if both are static
            if body_a.is_static and body_b.is_static:
                continue
            
            # Perform collision test
            collision_info = await self._test_collision(body_a, body_b)
            if collision_info:
                self.collision_pairs.append(collision_info)
        
        self.physics_stats["collision_checks"] = collision_checks
        self.physics_stats["collisions_detected"] = len(self.collision_pairs)
    
    async def _get_potential_collision_pairs(self) -> List[Tuple[str, str]]:
        """Get potential collision pairs using spatial partitioning"""
        
        pairs = []
        processed_pairs = set()
        
        # Check spatial grid cells
        for cell_bodies in self.spatial_grid.values():
            if len(cell_bodies) > 1:
                # All pairs within this cell
                for i in range(len(cell_bodies)):
                    for j in range(i + 1, len(cell_bodies)):
                        body_a_id = cell_bodies[i]
                        body_b_id = cell_bodies[j]
                        
                        pair_key = tuple(sorted([body_a_id, body_b_id]))
                        if pair_key not in processed_pairs:
                            pairs.append((body_a_id, body_b_id))
                            processed_pairs.add(pair_key)
        
        return pairs
    
    async def _test_collision(self, body_a: PhysicsBody, body_b: PhysicsBody) -> Optional[CollisionInfo]:
        """Test collision between two bodies"""
        
        # Simple box-box collision detection
        if (body_a.collision_shape == CollisionShape.BOX and 
            body_b.collision_shape == CollisionShape.BOX):
            return await self._test_box_box_collision(body_a, body_b)
        
        # Simple sphere-sphere collision detection
        elif (body_a.collision_shape == CollisionShape.SPHERE and 
              body_b.collision_shape == CollisionShape.SPHERE):
            return await self._test_sphere_sphere_collision(body_a, body_b)
        
        # For now, use simple distance-based collision
        return await self._test_simple_collision(body_a, body_b)
    
    async def _test_box_box_collision(self, body_a: PhysicsBody, body_b: PhysicsBody) -> Optional[CollisionInfo]:
        """Test box-box collision"""
        
        # AABB collision test
        a_min = Vector3(
            body_a.position.x - body_a.collision_size.x / 2,
            body_a.position.y - body_a.collision_size.y / 2,
            body_a.position.z - body_a.collision_size.z / 2
        )
        a_max = Vector3(
            body_a.position.x + body_a.collision_size.x / 2,
            body_a.position.y + body_a.collision_size.y / 2,
            body_a.position.z + body_a.collision_size.z / 2
        )
        
        b_min = Vector3(
            body_b.position.x - body_b.collision_size.x / 2,
            body_b.position.y - body_b.collision_size.y / 2,
            body_b.position.z - body_b.collision_size.z / 2
        )
        b_max = Vector3(
            body_b.position.x + body_b.collision_size.x / 2,
            body_b.position.y + body_b.collision_size.y / 2,
            body_b.position.z + body_b.collision_size.z / 2
        )
        
        # Check overlap
        if (a_max.x >= b_min.x and a_min.x <= b_max.x and
            a_max.y >= b_min.y and a_min.y <= b_max.y and
            a_max.z >= b_min.z and a_min.z <= b_max.z):
            
            # Calculate collision normal and penetration
            overlap_x = min(a_max.x - b_min.x, b_max.x - a_min.x)
            overlap_y = min(a_max.y - b_min.y, b_max.y - a_min.y)
            overlap_z = min(a_max.z - b_min.z, b_max.z - a_min.z)
            
            # Find minimum overlap axis
            min_overlap = min(overlap_x, overlap_y, overlap_z)
            
            if min_overlap == overlap_x:
                normal = Vector3(1 if body_a.position.x > body_b.position.x else -1, 0, 0)
            elif min_overlap == overlap_y:
                normal = Vector3(0, 1 if body_a.position.y > body_b.position.y else -1, 0)
            else:
                normal = Vector3(0, 0, 1 if body_a.position.z > body_b.position.z else -1)
            
            contact_point = Vector3(
                (body_a.position.x + body_b.position.x) / 2,
                (body_a.position.y + body_b.position.y) / 2,
                (body_a.position.z + body_b.position.z) / 2
            )
            
            return CollisionInfo(
                body_a=body_a.id,
                body_b=body_b.id,
                contact_point=contact_point,
                contact_normal=normal,
                penetration_depth=min_overlap,
                collision_time=time.time()
            )
        
        return None
    
    async def _test_sphere_sphere_collision(self, body_a: PhysicsBody, body_b: PhysicsBody) -> Optional[CollisionInfo]:
        """Test sphere-sphere collision"""
        
        radius_a = body_a.collision_size.x  # Use x as radius
        radius_b = body_b.collision_size.x
        
        distance_vector = body_b.position - body_a.position
        distance = distance_vector.magnitude()
        
        if distance < radius_a + radius_b:
            # Collision detected
            if distance > 0:
                normal = distance_vector.normalized()
            else:
                normal = Vector3(1, 0, 0)  # Default normal
            
            penetration = (radius_a + radius_b) - distance
            contact_point = body_a.position + (normal * radius_a)
            
            return CollisionInfo(
                body_a=body_a.id,
                body_b=body_b.id,
                contact_point=contact_point,
                contact_normal=normal,
                penetration_depth=penetration,
                collision_time=time.time()
            )
        
        return None
    
    async def _test_simple_collision(self, body_a: PhysicsBody, body_b: PhysicsBody) -> Optional[CollisionInfo]:
        """Simple distance-based collision test"""
        
        distance = (body_b.position - body_a.position).magnitude()
        collision_distance = (body_a.collision_size.magnitude() + body_b.collision_size.magnitude()) / 2
        
        if distance < collision_distance:
            direction = (body_b.position - body_a.position).normalized()
            
            return CollisionInfo(
                body_a=body_a.id,
                body_b=body_b.id,
                contact_point=body_a.position + (direction * (collision_distance / 2)),
                contact_normal=direction,
                penetration_depth=collision_distance - distance,
                collision_time=time.time()
            )
        
        return None
    
    async def _solve_constraints(self, delta_time: float):
        """Solve physics constraints"""
        
        # Iterative constraint solver
        for iteration in range(self.solver_iterations):
            for constraint in self.constraints.values():
                await self._solve_constraint(constraint, delta_time)
    
    async def _solve_constraint(self, constraint: PhysicsConstraint, delta_time: float):
        """Solve individual constraint"""
        
        body_a = self.bodies.get(constraint.body_a)
        body_b = self.bodies.get(constraint.body_b)
        
        if not body_a or not body_b:
            return
        
        if constraint.constraint_type == "distance":
            await self._solve_distance_constraint(body_a, body_b, constraint.parameters)
        elif constraint.constraint_type == "spring":
            await self._solve_spring_constraint(body_a, body_b, constraint.parameters, delta_time)
    
    async def _solve_distance_constraint(self, body_a: PhysicsBody, body_b: PhysicsBody, params: Dict[str, Any]):
        """Solve distance constraint"""
        
        target_distance = params.get("distance", 1.0)
        stiffness = params.get("stiffness", 1.0)
        
        direction = body_b.position - body_a.position
        current_distance = direction.magnitude()
        
        if current_distance == 0:
            return
        
        error = current_distance - target_distance
        correction = direction.normalized() * (error * stiffness * 0.5)
        
        if not body_a.is_static:
            body_a.position = body_a.position + correction
        if not body_b.is_static:
            body_b.position = body_b.position - correction
    
    async def _solve_spring_constraint(self, body_a: PhysicsBody, body_b: PhysicsBody, 
                                     params: Dict[str, Any], delta_time: float):
        """Solve spring constraint"""
        
        spring_constant = params.get("spring_constant", 100.0)
        damping = params.get("damping", 10.0)
        rest_length = params.get("rest_length", 1.0)
        
        direction = body_b.position - body_a.position
        current_length = direction.magnitude()
        
        if current_length == 0:
            return
        
        unit_direction = direction.normalized()
        
        # Spring force
        spring_force = unit_direction * ((current_length - rest_length) * spring_constant)
        
        # Damping force
        relative_velocity = body_b.velocity - body_a.velocity
        damping_force = unit_direction * (relative_velocity.dot(unit_direction) * damping)
        
        total_force = spring_force + damping_force
        
        if not body_a.is_static:
            body_a.forces.append(total_force)
        if not body_b.is_static:
            body_b.forces.append(total_force * -1)
    
    async def _resolve_collisions(self, delta_time: float):
        """Resolve detected collisions"""
        
        for collision in self.collision_pairs:
            body_a = self.bodies.get(collision.body_a)
            body_b = self.bodies.get(collision.body_b)
            
            if not body_a or not body_b:
                continue
            
            await self._resolve_collision(body_a, body_b, collision, delta_time)
    
    async def _resolve_collision(self, body_a: PhysicsBody, body_b: PhysicsBody,
                               collision: CollisionInfo, delta_time: float):
        """Resolve individual collision"""
        
        # Position correction
        correction_factor = 0.8
        correction = collision.contact_normal * (collision.penetration_depth * correction_factor)
        
        if not body_a.is_static and not body_b.is_static:
            body_a.position = body_a.position - (correction * 0.5)
            body_b.position = body_b.position + (correction * 0.5)
        elif not body_a.is_static:
            body_a.position = body_a.position - correction
        elif not body_b.is_static:
            body_b.position = body_b.position + correction
        
        # Velocity response
        relative_velocity = body_b.velocity - body_a.velocity
        velocity_along_normal = relative_velocity.dot(collision.contact_normal)
        
        if velocity_along_normal > 0:
            return  # Objects separating
        
        # Calculate restitution
        restitution = min(body_a.restitution, body_b.restitution)
        
        # Calculate impulse
        impulse_magnitude = -(1 + restitution) * velocity_along_normal
        
        if not body_a.is_static and not body_b.is_static:
            impulse_magnitude /= (1.0 / body_a.mass + 1.0 / body_b.mass)
        elif not body_a.is_static:
            impulse_magnitude /= (1.0 / body_a.mass)
        elif not body_b.is_static:
            impulse_magnitude /= (1.0 / body_b.mass)
        
        impulse = collision.contact_normal * impulse_magnitude
        
        # Apply impulse
        if not body_a.is_static:
            body_a.velocity = body_a.velocity - (impulse * (1.0 / body_a.mass))
        if not body_b.is_static:
            body_b.velocity = body_b.velocity + (impulse * (1.0 / body_b.mass))
    
    # Spatial partitioning methods
    
    async def _update_spatial_grid(self, body: PhysicsBody):
        """Update spatial grid for body"""
        
        # Remove from old cells
        await self._remove_from_spatial_grid(body)
        
        # Add to new cells
        grid_x = int(body.position.x // self.spatial_grid_size)
        grid_y = int(body.position.y // self.spatial_grid_size)
        grid_z = int(body.position.z // self.spatial_grid_size)
        
        cell_key = (grid_x, grid_y, grid_z)
        
        if cell_key not in self.spatial_grid:
            self.spatial_grid[cell_key] = []
        
        self.spatial_grid[cell_key].append(body.id)
    
    async def _remove_from_spatial_grid(self, body: PhysicsBody):
        """Remove body from spatial grid"""
        
        for cell_bodies in self.spatial_grid.values():
            if body.id in cell_bodies:
                cell_bodies.remove(body.id)
        
        # Clean up empty cells
        empty_cells = [key for key, bodies in self.spatial_grid.items() if not bodies]
        for key in empty_cells:
            del self.spatial_grid[key]
    
    # Claude optimization methods
    
    async def _claude_physics_initialization(self):
        """Initialize physics with Claude guidance"""
        
        init_query = """
        Forge Physics Engine Initialization
        
        This is a high-performance 3D physics engine designed for PCC-generated games.
        
        Provide initialization guidance:
        1. Optimal physics simulation parameters
        2. Performance tuning recommendations
        3. Collision detection optimization strategies
        4. Memory management approaches
        5. Real-time constraint solving techniques
        
        Focus on maximum performance for procedurally generated content.
        """
        
        guidance = self.consult_claude(init_query, {"initialization": True})
        
        # Apply initialization guidance
        insights = guidance.get("insights", "").lower()
        
        if "increase solver iterations" in insights:
            self.solver_iterations = min(self.solver_iterations + 2, 20)
        
        if "reduce time step" in insights:
            self.time_step = max(self.time_step * 0.8, 1.0 / 240.0)
        
        await self.store_module_experience(
            f"Physics initialized with Claude guidance",
            "initialization",
            context={"guidance": guidance.get("insights", "")[:300]},
            tags=["initialization", "claude", "optimization"]
        )
    
    async def _periodic_physics_optimization(self):
        """Periodic physics optimization using Claude"""
        
        if not self.claude_enabled:
            return
        
        optimization_query = f"""
        Physics Engine Performance Analysis
        
        Current State:
        - Bodies: {len(self.bodies)} total, {self.physics_stats['active_bodies']} active
        - Performance: {self.physics_stats['physics_time_ms']:.2f}ms per frame
        - Collision Checks: {self.physics_stats['collision_checks']} per frame
        - Solver Time: {self.physics_stats['solver_time_ms']:.2f}ms
        - Constraints: {len(self.constraints)}
        
        Analyze performance and suggest optimizations:
        1. Collision detection bottlenecks
        2. Solver performance improvements
        3. Spatial partitioning optimization
        4. Memory usage reduction
        5. Overall architecture improvements
        """
        
        optimization_result = self.consult_claude(optimization_query)
        
        # Apply optimizations
        await self._apply_physics_optimizations(optimization_result)
    
    async def _apply_physics_optimizations(self, optimization_result: Dict[str, Any]):
        """Apply physics optimizations"""
        
        insights = optimization_result.get("insights", "").lower()
        
        optimizations_applied = []
        
        if "increase spatial grid size" in insights:
            self.spatial_grid_size *= 1.2
            optimizations_applied.append("increased_spatial_grid_size")
        
        if "reduce solver iterations" in insights and self.physics_stats['solver_time_ms'] > 5.0:
            self.solver_iterations = max(self.solver_iterations - 1, 3)
            optimizations_applied.append("reduced_solver_iterations")
        
        if "optimize broadphase" in insights:
            # Could implement more advanced broadphase
            optimizations_applied.append("broadphase_optimization_noted")
        
        if optimizations_applied:
            await self.store_module_experience(
                f"Applied physics optimizations: {', '.join(optimizations_applied)}",
                "optimization",
                context={
                    "optimizations": optimizations_applied,
                    "performance_before": self.physics_stats.copy()
                },
                tags=["optimization", "physics", "performance"]
            )
    
    # Event handlers
    
    async def _handle_add_body_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add body event"""
        data = event.get("data", {})
        
        body_id = data.get("body_id")
        if not body_id:
            return {"success": False, "error": "No body_id provided"}
        
        position_data = data.get("position", {})
        position = Vector3(
            position_data.get("x", 0),
            position_data.get("y", 0),
            position_data.get("z", 0)
        )
        
        shape_str = data.get("shape", "box")
        shape = CollisionShape(shape_str) if shape_str in [s.value for s in CollisionShape] else CollisionShape.BOX
        
        size_data = data.get("size", {})
        size = Vector3(
            size_data.get("x", 1),
            size_data.get("y", 1),
            size_data.get("z", 1)
        )
        
        mass = data.get("mass", 1.0)
        is_static = data.get("is_static", False)
        
        success = await self.add_physics_body(body_id, position, shape, size, mass, is_static)
        return {"success": success}
    
    async def _handle_remove_body_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle remove body event"""
        data = event.get("data", {})
        body_id = data.get("body_id")
        
        if not body_id:
            return {"success": False, "error": "No body_id provided"}
        
        success = await self.remove_physics_body(body_id)
        return {"success": success}
    
    async def _handle_apply_force_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle apply force event"""
        data = event.get("data", {})
        
        body_id = data.get("body_id")
        if not body_id:
            return {"success": False, "error": "No body_id provided"}
        
        force_data = data.get("force", {})
        force = Vector3(
            force_data.get("x", 0),
            force_data.get("y", 0),
            force_data.get("z", 0)
        )
        
        success = await self.apply_force(body_id, force)
        return {"success": success}
    
    async def _handle_set_gravity_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set gravity event"""
        data = event.get("data", {})
        
        gravity_data = data.get("gravity", {})
        self.gravity = Vector3(
            gravity_data.get("x", 0),
            gravity_data.get("y", -9.81),
            gravity_data.get("z", 0)
        )
        
        return {"success": True, "gravity": {"x": self.gravity.x, "y": self.gravity.y, "z": self.gravity.z}}
    
    async def _handle_collision_query_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collision query event"""
        data = event.get("data", {})
        body_id = data.get("body_id")
        
        if not body_id:
            return {"success": False, "error": "No body_id provided"}
        
        collisions = await self.query_collisions(body_id)
        
        collision_data = []
        for collision in collisions:
            collision_data.append({
                "other_body": collision.body_b if collision.body_a == body_id else collision.body_a,
                "contact_point": {"x": collision.contact_point.x, "y": collision.contact_point.y, "z": collision.contact_point.z},
                "normal": {"x": collision.contact_normal.x, "y": collision.contact_normal.y, "z": collision.contact_normal.z},
                "penetration": collision.penetration_depth
            })
        
        return {"success": True, "collisions": collision_data}
    
    def get_physics_status(self) -> Dict[str, Any]:
        """Get comprehensive physics status"""
        return {
            "module_id": self.module_id,
            "version": self.version,
            "bodies": len(self.bodies),
            "active_bodies": self.physics_stats["active_bodies"],
            "constraints": len(self.constraints),
            "gravity": {"x": self.gravity.x, "y": self.gravity.y, "z": self.gravity.z},
            "physics_stats": self.physics_stats,
            "spatial_grid_cells": len(self.spatial_grid),
            "claude_enabled": self.claude_enabled
        }