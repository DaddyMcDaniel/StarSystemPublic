#!/usr/bin/env python3
"""
Quick launcher for manual evolution loop with human testing
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run manual evolution loop"""
    script_path = Path(__file__).parent / "agents" / "run_evolution_manual.py"
    
    # Pass through all arguments
    cmd = ["python", str(script_path)] + sys.argv[1:]
    
    print("🎮 Starting Manual Evolution Loop")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⚠️ Manual loop interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Manual loop failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())