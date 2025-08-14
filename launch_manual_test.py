#!/usr/bin/env python3
"""
Direct launcher for manual testing - no interactive prompts
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🎮 Manual Planet Testing Launcher")
    print("=" * 40)
    
    # Use existing planet
    planet_file = "runs/miniplanet_seed_42.json"
    print(f"🌍 Testing planet: {planet_file}")
    
    if not Path(planet_file).exists():
        print(f"❌ Planet file not found: {planet_file}")
        print("💡 Generating planet...")
        
        # Generate planet first
        result = subprocess.run([
            sys.executable, "scripts/run_gl.py",
            "--seed", "42",
            "--generate-only"
        ])
        
        if result.returncode != 0:
            print("❌ Failed to generate planet")
            return 1
    
    print()
    print("🚀 Launching spherical planet viewer...")
    print("=" * 40)
    print("CONTROLS:")
    print("  WASD - Walk on planet surface")
    print("  Mouse - Look around (first-person)")
    print("  SPACE - Jump")
    print("  R - Reset position to equator")
    print("  +/- - Change movement speed")
    print("  TAB - Toggle mouse capture")
    print("  ESC - Exit when done testing")
    print()
    print("TESTING OBJECTIVES:")
    print("  1. Test spherical surface walking")
    print("  2. Check if you can see curved horizon")
    print("  3. Try walking around the entire planet")
    print("  4. Look for suitable building areas")
    print("  5. Assess visual quality and terrain variety")
    print()
    print("The viewer will launch now...")
    print("Close the viewer window when you're done testing.")
    print()
    
    # Launch viewer directly
    viewer_cmd = [
        sys.executable, "renderer/pcc_spherical_viewer.py",
        planet_file
    ]
    
    print(f"🚀 Running: {' '.join(viewer_cmd)}")
    
    try:
        result = subprocess.run(viewer_cmd)
        
        if result.returncode == 0:
            print("\n✅ Manual testing completed successfully")
            print("\n📝 To provide feedback, please run:")
            print("   python collect_feedback.py")
        else:
            print(f"\n❌ Viewer exited with code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error launching viewer: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())