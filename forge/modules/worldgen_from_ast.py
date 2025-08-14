#!/usr/bin/env python3
"""
Worldgen from PCC AST
Deterministic spherical-world generator that converts Agent A's PCC AST into a
scene JSON used by the OpenGL viewers. Ensures objects are grounded to the
planet surface and oriented so their local +Y points along the outward surface
normal, matching player orientation.

Inputs (subset expected from Agent A):
- ast: {
    type: 'SPHERICAL_PLANET_GAME',
    nodes: [
      { type: 'SPHERICAL_WORLD', terrain: { type: 'sphere', radius, center, material }, features: [...], building_zones: [...] },
      ...
    ],
    planet_seed: int
  }

Outputs:
- scene: {
    metadata: { scene_type: 'miniplanet', seed, layer: 'surface' },
    terrain: { type: 'sphere', radius, center, material },
    objects: [ { type: 'CUBE'|'SPHERE'|'MESH', pos, size|radius, material, grounded: true, align_to_surface: true, up: [nx,ny,nz] }, ... ]
  }
"""

from __future__ import annotations

import math
import random
from typing import Dict, Any, List, Tuple


def _unit_normal(x: float, y: float, z: float) -> Tuple[float, float, float]:
    l = max(1e-9, math.sqrt(x * x + y * y + z * z))
    return x / l, y / l, z / l


def _surface_point(radius: float, theta: float, phi: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    sx = radius * math.sin(phi) * math.cos(theta)
    sy = radius * math.cos(phi)
    sz = radius * math.sin(phi) * math.sin(theta)
    nx, ny, nz = _unit_normal(sx, sy, sz)
    return (sx, sy, sz), (nx, ny, nz)


def _place_cube_on_surface(center: Tuple[float, float, float], normal: Tuple[float, float, float], height: float) -> List[float]:
    nx, ny, nz = normal
    cx, cy, cz = center
    # Seat bottom face on surface by moving center half-height along normal
    return [cx + nx * (height * 0.5), cy + ny * (height * 0.5), cz + nz * (height * 0.5)]


def _place_sphere_on_surface(center: Tuple[float, float, float], normal: Tuple[float, float, float], radius: float) -> List[float]:
    nx, ny, nz = normal
    cx, cy, cz = center
    return [cx + nx * radius, cy + ny * radius, cz + nz * radius]


def _feature_to_objects(rng: random.Random, feature: Dict[str, Any], radius: float, count_hint: int = 8) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    ftype = feature.get("type", "")

    def scatter_spheres(material: str, density: int):
        for _ in range(density):
            theta = rng.uniform(0, 2 * math.pi)
            phi = rng.uniform(0.2, math.pi - 0.2)
            center, normal = _surface_point(radius, theta, phi)
            r = rng.uniform(0.5, 1.2)
            pos = _place_sphere_on_surface(center, normal, r)
            objects.append({
                "type": "SPHERE",
                "pos": pos,
                "radius": r,
                "material": material,
                "grounded": True,
                "align_to_surface": True,
                "up": [normal[0], normal[1], normal[2]]
            })

    def scatter_cubes(material: str, density: int, height_range=(1.0, 3.0)):
        for _ in range(density):
            theta = rng.uniform(0, 2 * math.pi)
            phi = rng.uniform(0.2, math.pi - 0.2)
            center, normal = _surface_point(radius, theta, phi)
            h = rng.uniform(*height_range)
            pos = _place_cube_on_surface(center, normal, h)
            objects.append({
                "type": "CUBE",
                "pos": pos,
                "size": [rng.uniform(0.6, 1.2), h, rng.uniform(0.6, 1.2)],
                "material": material,
                "grounded": True,
                "align_to_surface": True,
                "up": [normal[0], normal[1], normal[2]]
            })

    if ftype == "CRYSTAL_FORMATION":
        scatter_spheres("resource_crystal", density=max(4, count_hint // 2))
    elif ftype == "ANCIENT_STRUCTURE":
        style = feature.get("style", "temple")
        mat = "structure_temple" if style == "temple" else "structure_monument"
        scatter_cubes(mat, density=max(3, count_hint // 3), height_range=(2.5, 5.0))
    elif ftype == "RESOURCE_NODES":
        mat = feature.get("material", "resource_ore")
        scatter_spheres(mat, density=max(5, count_hint // 2))
    elif ftype == "CAVE_SYSTEMS":
        # Place near-surface dark spheres as entrances
        for _ in range(max(3, count_hint // 3)):
            theta = rng.uniform(0, 2 * math.pi)
            phi = rng.uniform(0.3, math.pi - 0.3)
            center, normal = _surface_point(radius * 0.95, theta, phi)
            r = 1.5
            pos = [center[0], center[1], center[2]]
            objects.append({
                "type": "SPHERE",
                "pos": pos,
                "radius": r,
                "material": "cave_entrance",
                "grounded": True,
                "align_to_surface": True,
                "up": [normal[0], normal[1], normal[2]]
            })
    elif ftype == "ATMOSPHERIC_EFFECT":
        # Viewer already renders atmosphere; skip adding objects
        pass
    else:
        # Default sprinkle of rocks/grass
        scatter_cubes("terrain_rock", density=max(3, count_hint // 4))
        scatter_cubes("terrain_grass", density=max(3, count_hint // 4), height_range=(0.8, 1.6))

    return objects


def generate_scene_from_pcc(pcc_ast: Dict[str, Any], seed: int | None = None) -> Dict[str, Any]:
    if seed is None:
        seed = int(pcc_ast.get("planet_seed", 42))
    rng = random.Random(seed)

    # Extract world node
    nodes = pcc_ast.get("nodes", [])
    world = next((n for n in nodes if n.get("type") == "SPHERICAL_WORLD"), {})
    terrain = world.get("terrain", {"type": "sphere", "radius": 25.0, "center": [0, 0, 0], "material": "rock"})
    radius = float(terrain.get("radius", 25.0))

    scene = {
        "metadata": {
            "scene_type": "miniplanet",
            "seed": seed,
            "generated_at": "2025-01-01T00:00:00Z",
            "layer": "surface"
        },
        "terrain": {
            "type": "sphere",
            "radius": radius,
            "center": terrain.get("center", [0, 0, 0]),
            "material": terrain.get("material", "rock")
        },
        "objects": []
    }

    # North pole beacon
    scene["objects"].append({
        "type": "SPHERE",
        "pos": [0, radius + 2.0, 0],
        "radius": 1.8,
        "material": "beacon_major",
        "grounded": True,
        "align_to_surface": True,
        "up": [0.0, 1.0, 0.0]
    })

    # Features → objects
    features = world.get("features", [])
    for feat in features:
        scene["objects"].extend(_feature_to_objects(rng, feat, radius, count_hint=8))

    # Simple equator structures
    for ang_deg in [0, 90, 180, 270]:
        ang = math.radians(ang_deg)
        ex = radius * math.cos(ang)
        ez = radius * math.sin(ang)
        up = _unit_normal(ex, 0.0, ez)
        h = 4.0
        pos = [ex + up[0] * (h * 0.5 + 1.0), up[1] * (h * 0.5 + 1.0), ez + up[2] * (h * 0.5 + 1.0)]
        scene["objects"].append({
            "type": "CUBE",
            "pos": pos,
            "size": [2.0, h, 2.0],
            "material": "structure_temple",
            "grounded": True,
            "align_to_surface": True,
            "up": [up[0], up[1], up[2]]
        })

    return scene


