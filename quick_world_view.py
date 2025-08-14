#!/usr/bin/env python3
"""
Quick world viewer - shows what's in the generated world without OpenGL
"""
import json
import sys
from pathlib import Path

def show_world(world_file):
    """Display world contents in text format"""
    
    if not Path(world_file).exists():
        print(f"❌ World file not found: {world_file}")
        return
        
    with open(world_file) as f:
        world = json.load(f)
    
    print(f"🌍 WORLD: {world_file}")
    print("=" * 60)
    
    # Show metadata
    meta = world.get('metadata', {})
    print(f"📊 Type: {meta.get('scene_type', 'unknown')}")
    print(f"🌱 Seed: {meta.get('seed', 'unknown')}")
    print(f"⏰ Generated: {meta.get('generated_at', 'unknown')}")
    print()
    
    # Show terrain
    terrain = world.get('terrain', {})
    print(f"🏔️  TERRAIN: {terrain.get('type', 'unknown')}")
    if terrain.get('type') == 'sphere':
        print(f"   📏 Radius: {terrain.get('radius', 0)}")
        center = terrain.get('center', [0, 0, 0])
        print(f"   📍 Center: ({center[0]}, {center[1]}, {center[2]})")
    print(f"   🏗️  Material: {terrain.get('material', 'unknown')}")
    print()
    
    # Categorize objects
    objects = world.get('objects', [])
    categories = {}
    
    for obj in objects:
        material = obj.get('material', 'unknown')
        category = material.split('_')[0] if '_' in material else material
        if category not in categories:
            categories[category] = []
        categories[category].append(obj)
    
    print(f"🎯 OBJECTS: {len(objects)} total")
    print("-" * 40)
    
    for category, items in sorted(categories.items()):
        print(f"\n{category.upper()}: {len(items)} items")
        
        # Show a few examples
        for i, obj in enumerate(items[:3]):
            pos = obj.get('pos', [0, 0, 0])
            obj_type = obj.get('type', 'unknown')
            material = obj.get('material', 'unknown')
            
            if obj_type == 'SPHERE':
                size_info = f"r={obj.get('radius', 0):.1f}"
            else:
                size = obj.get('size', [1, 1, 1])
                size_info = f"size=({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f})"
            
            print(f"   {i+1}. {material} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) {size_info}")
        
        if len(items) > 3:
            print(f"   ... and {len(items) - 3} more")
    
    print()
    print("🎮 This world is ready for exploration!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_world_view.py <world_file.json>")
        sys.exit(1)
    
    show_world(sys.argv[1])