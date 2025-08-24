#!/usr/bin/env python3
"""
Simple Mini-Planet Generator
Usage: python generate_miniplanet.py [--seed SEED] [--view]
"""

import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate Mini-Planet")
    parser.add_argument("--seed", type=int, help="Planet seed (random if not specified)")
    parser.add_argument("--view", action="store_true", help="Launch viewer after generation")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    
    args = parser.parse_args()
    
    # Build command
    cmd = [sys.executable, "unified_miniplanet.py"]
    
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    else:
        cmd.append("--random-seed")
    
    if args.view:
        cmd.append("--view")
    
    if args.debug:
        cmd.append("--debug")
    
    # Run the unified system
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️ Generation cancelled")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())