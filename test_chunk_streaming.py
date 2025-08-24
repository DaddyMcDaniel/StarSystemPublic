#!/usr/bin/env python3
"""
Test script to demonstrate the chunk streaming system
"""

import json
import numpy as np
from create_mini_planet import MiniPlanetGenerator

def test_streaming_system():
    """Test the chunk streaming system with different camera positions"""
    
    # Generate planet with streaming
    generator = MiniPlanetGenerator()
    planet_data = generator.generate()
    
    if planet_data.get('type') != 'streaming_chunks':
        print("❌ Planet data is not in streaming format")
        return
    
    streaming_manager = planet_data['manager']
    total_chunks = len(planet_data['chunks'])
    
    print(f"🌍 Planet with {total_chunks} total chunks created")
    print("🎥 Testing different camera positions:")
    
    # Test camera positions
    test_positions = [
        ([60, 0, 0], "East surface"),
        ([0, 60, 0], "North surface"), 
        ([0, 0, 60], "Top surface"),
        ([100, 100, 100], "Far distance"),
        ([20, 10, 5], "Close to surface")
    ]
    
    for camera_pos, description in test_positions:
        camera_dir = [0, 0, -1]  # Looking down
        
        # Get visible chunks
        visible_chunks = streaming_manager.get_visible_chunks(
            camera_pos, camera_dir, max_distance=80.0, max_chunks=20
        )
        
        print(f"  📍 {description} ({camera_pos}): {len(visible_chunks)} chunks loaded")
        
        # Show LOD distribution
        lod_counts = {}
        for chunk in visible_chunks:
            lod = chunk['lod_level']
            lod_counts[lod] = lod_counts.get(lod, 0) + 1
        
        print(f"     LOD distribution: {dict(sorted(lod_counts.items()))}")
    
    print(f"\n✅ Streaming system test complete!")
    print(f"   Cache contains {len(streaming_manager.loaded_chunks)} chunks")
    print(f"   Memory efficient: Only loads {len(streaming_manager.loaded_chunks)}/{total_chunks} chunks ({100*len(streaming_manager.loaded_chunks)/total_chunks:.1f}%)")

if __name__ == "__main__":
    test_streaming_system()