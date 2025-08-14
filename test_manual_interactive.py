#!/usr/bin/env python3
"""
Interactive manual testing for human feedback collection
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path

def main():
    print("🎮 Interactive Manual Testing")
    print("=" * 40)
    
    # Use existing planet
    planet_file = "runs/miniplanet_seed_42.json"
    print(f"🌍 Testing planet: {planet_file}")
    
    if not Path(planet_file).exists():
        print(f"❌ Planet file not found: {planet_file}")
        print("💡 Run: python scripts/run_gl.py --seed 42 --generate-only")
        return 1
    
    print()
    print("🚀 Launching spherical planet viewer...")
    print("Instructions:")
    print("  - WASD: Walk on planet surface")
    print("  - Mouse: Look around")
    print("  - SPACE: Jump")
    print("  - R: Reset position")
    print("  - ESC: Exit when done testing")
    print()
    
    input("Press ENTER to launch viewer...")
    
    # Launch viewer
    viewer_cmd = [
        "python", "renderer/pcc_spherical_viewer.py", 
        planet_file
    ]
    
    try:
        subprocess.run(viewer_cmd)
        print("\n✅ Viewer session completed")
    except KeyboardInterrupt:
        print("\n⚠️ Viewer interrupted")
    except Exception as e:
        print(f"\n❌ Viewer failed: {e}")
        return 1
    
    # Collect feedback
    print("\n📝 Feedback Collection")
    print("=" * 25)
    
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "tester": "human_interactive",
        "planet_file": planet_file
    }
    
    # Simple feedback questions
    print("\n🌍 SPHERICAL PLANET ASSESSMENT:")
    
    feedback["surface_walking"] = input("1. Surface walking quality (excellent/good/fair/poor): ")
    feedback["horizon_curvature"] = input("2. Can you see curved horizon? (yes/no): ")
    feedback["circumnavigation"] = input("3. Can you walk around the planet? (yes/no): ")
    feedback["planet_size"] = input("4. Planet size feels (too_small/perfect/too_large): ")
    feedback["terrain_variety"] = input("5. Terrain variety (excellent/good/basic/poor): ")
    feedback["building_zones"] = input("6. Suitable building areas (many/some/few/none): ")
    
    print("\n📊 OVERALL:")
    feedback["rating"] = int(input("7. Overall rating (1-10): "))
    feedback["biggest_issue"] = input("8. Biggest issue to fix: ")
    feedback["best_feature"] = input("9. What works best: ")
    feedback["next_priority"] = input("10. Next improvement priority: ")
    
    # Save feedback
    feedback_file = Path("agents/communication/human_feedback_interactive.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(feedback_file, 'w') as f:
        json.dump(feedback, f, indent=2)
    
    print(f"\n💾 Feedback saved to: {feedback_file}")
    
    # Summary
    print("\n📊 Feedback Summary:")
    print(f"   Rating: {feedback['rating']}/10")
    print(f"   Top Issue: {feedback['biggest_issue']}")
    print(f"   Best Feature: {feedback['best_feature']}")
    print(f"   Next Priority: {feedback['next_priority']}")
    
    if feedback["rating"] >= 8:
        print("✅ Excellent feedback - Planet ready for building!")
    elif feedback["rating"] >= 6:
        print("🔄 Good progress - Minor improvements needed")
    else:
        print("🚨 Significant improvements required")
    
    print("\n🎉 Interactive testing completed!")
    return 0

if __name__ == "__main__":
    exit(main())