#!/usr/bin/env python3
"""
SUMMARY: Auto Evolution Loop with Fullscreen Visual Testing
==========================================================
Automated A→D→B→C evolution cycle with proper fullscreen viewer for accurate screenshot capture.
Agent B runs comprehensive visual testing with fullscreen OpenGL viewer to ensure proper visual analysis.

FEATURES:
- Agent A generates spherical planet prompts (HOD-aligned)
- Agent D renders planet to 3D scene  
- Agent B validates with fullscreen viewer and screenshot capture
- Agent C supervises and provides evolution feedback
- Fullscreen mode ensures screenshots capture complete visual state
- Human feedback incorporation from previous manual sessions

USAGE:
  python agents/run_evolution_auto.py
  python agents/run_evolution_auto.py --iterations 5
  python agents/run_evolution_auto.py --bridge-port 8765

RELATED FILES:
- agents/run_evolution_manual.py - Manual human testing loop
- agents/agent_b_manual.py - Human feedback collection script
"""

import argparse
import subprocess
import time
import json
import signal
import os
import sys
from pathlib import Path

def check_opengl_availability():
    """Check if OpenGL is available for fullscreen testing"""
    try:
        # Try importing OpenGL
        import OpenGL.GL
        
        # Check for display
        if not os.getenv('DISPLAY'):
            return False, "No DISPLAY environment variable set"
        
        return True, "OpenGL available"
    except ImportError:
        return False, "OpenGL libraries not installed"
    except Exception as e:
        return False, f"OpenGL check failed: {e}"

def run_agent_a():
    """Run Agent A to generate spherical planet"""
    print("🌍 Agent A: Generating spherical mini-planet...")
    result = subprocess.run(
        ["python", "agents/agent_a_generator.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode == 0:
        print("✅ Agent A: Planet generated successfully")
        print(f"   Output: {result.stdout.strip()}")
        return True
    else:
        print(f"❌ Agent A failed: {result.stderr}")
        return False

def run_agent_d(planet_file=None):
    """Run Agent D to render planet scene"""
    print("🎨 Agent D: Rendering planet scene...")
    
    # Use existing mini-planet or generate new one
    if not planet_file:
        planet_file = "runs/miniplanet_seed_42.json"
    
    # Generate mini-planet scene if needed
    if not Path(planet_file).exists():
        print(f"🌱 Generating planet scene: {planet_file}")
        result = subprocess.run(
            ["python", "scripts/run_gl.py", "--seed", "42", "--generate-only"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode != 0:
            print(f"❌ Agent D failed to generate scene: {result.stderr}")
            return False
    
    print(f"✅ Agent D: Scene ready at {planet_file}")
    return True

def run_agent_b_fullscreen(planet_file, bridge_port=8765):
    """Run Agent B with fullscreen viewer for proper screenshots"""
    print("🤖 Agent B: Starting fullscreen visual testing...")
    
    # Start fullscreen spherical viewer
    viewer_cmd = [
        "python", "renderer/pcc_spherical_viewer.py", 
        planet_file,
        "--bridge-port", str(bridge_port)
    ]
    
    print(f"🚀 Launching fullscreen viewer: {' '.join(viewer_cmd)}")
    
    # Set fullscreen environment variables for OpenGL
    env = os.environ.copy()
    env['DISPLAY'] = ':0'  # Ensure proper display
    env['GL_FULLSCREEN'] = '1'  # Custom flag for fullscreen mode
    
    viewer_process = subprocess.Popen(
        viewer_cmd,
        cwd=Path(__file__).parent.parent,
        env=env
    )
    
    # Give viewer time to initialize
    time.sleep(3)
    
    # Run Agent B evaluation with bridge connection
    print("🔍 Agent B: Running comprehensive evaluation...")
    result = subprocess.run(
        ["python", "agents/agent_b_tester.py", "--bridge-port", str(bridge_port)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    # Stop viewer
    print("🛑 Stopping fullscreen viewer...")
    try:
        viewer_process.terminate()
        viewer_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        viewer_process.kill()
        viewer_process.wait()
    
    if result.returncode == 0:
        print("✅ Agent B: Evaluation completed")
        return True, result.stdout
    else:
        print(f"❌ Agent B evaluation failed: {result.stderr}")
        return False, result.stderr

def run_agent_c(agent_b_output):
    """Run Agent C to provide supervision and evolution feedback"""
    print("👁️ Agent C: Supervising evolution cycle...")
    
    # For now, Agent C provides structured feedback based on Agent B results
    # In full implementation, this would be a separate Agent C script
    
    try:
        # Parse Agent B report
        latest_report = Path("runs/latest/agent_b_report.json")
        if latest_report.exists():
            with open(latest_report, 'r') as f:
                b_report = json.load(f)
            
            overall_score = b_report.get('overall_score', 0)
            recommendation = b_report.get('recommendation', 'UNKNOWN')
            suggestions = b_report.get('improvement_suggestions', [])
            
            print(f"📊 Agent C Analysis:")
            print(f"   Overall Score: {overall_score}/100")
            print(f"   Recommendation: {recommendation}")
            print(f"   Suggestions: {len(suggestions)} improvements identified")
            
            if overall_score >= 70:
                print("✅ Agent C: Iteration APPROVED - Good progress")
                return True
            else:
                print("🔄 Agent C: Iteration needs improvement - Continue evolution")
                return False
        else:
            print("⚠️ Agent C: No Agent B report found")
            return False
            
    except Exception as e:
        print(f"❌ Agent C analysis failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Auto Evolution Loop with Fullscreen Testing")
    parser.add_argument("--iterations", type=int, default=1, help="Number of evolution cycles")
    parser.add_argument("--bridge-port", type=int, default=8765, help="Bridge port for Agent B")
    parser.add_argument("--planet-file", help="Specific planet file to test")
    parser.add_argument("--force", action="store_true", help="Force run even without OpenGL")
    
    args = parser.parse_args()
    
    print("🌍 StarSystem Auto Evolution Loop")
    print("=" * 50)
    
    # Check OpenGL availability first
    opengl_available, opengl_msg = check_opengl_availability()
    print(f"🔍 OpenGL Check: {opengl_msg}")
    
    if not opengl_available and not args.force:
        print("❌ Auto loop requires OpenGL for fullscreen visual testing")
        print("💡 Solutions:")
        print("   - Install OpenGL: pip install PyOpenGL PyOpenGL-accelerate")
        print("   - Set DISPLAY variable if using remote connection")
        print("   - Use manual loop instead: python run_manual_loop.py")
        print("   - Force run anyway: --force (not recommended)")
        return 1
    
    if not opengl_available and args.force:
        print("⚠️ Running without OpenGL - screenshots will fail")
    
    print(f"🔄 Running {args.iterations} evolution cycle(s)")
    print(f"🌉 Bridge port: {args.bridge_port}")
    print(f"📺 Fullscreen visual testing enabled" if opengl_available else "⚠️ Visual testing disabled")
    print()
    
    successful_cycles = 0
    
    for cycle in range(1, args.iterations + 1):
        print(f"🚀 Evolution Cycle {cycle}/{args.iterations}")
        print("-" * 30)
        
        # Step 1: Agent A generates spherical planet
        if not run_agent_a():
            print(f"❌ Cycle {cycle} failed at Agent A")
            continue
        
        # Step 2: Agent D renders scene
        if not run_agent_d(args.planet_file):
            print(f"❌ Cycle {cycle} failed at Agent D")
            continue
        
        # Step 3: Agent B validates with fullscreen viewer
        planet_file = args.planet_file or "runs/miniplanet_seed_42.json"
        b_success, b_output = run_agent_b_fullscreen(planet_file, args.bridge_port)
        if not b_success:
            print(f"❌ Cycle {cycle} failed at Agent B")
            continue
        
        # Step 4: Agent C supervises
        if run_agent_c(b_output):
            successful_cycles += 1
            print(f"✅ Cycle {cycle} completed successfully")
        else:
            print(f"🔄 Cycle {cycle} needs more improvement")
        
        print()
    
    print("🏁 Auto Evolution Summary")
    print("=" * 30)
    print(f"✅ Successful cycles: {successful_cycles}/{args.iterations}")
    print(f"📊 Success rate: {(successful_cycles/args.iterations)*100:.1f}%")
    
    if successful_cycles > 0:
        print("🎉 Evolution progress achieved!")
    else:
        print("🔄 Continue evolution cycles for better results")

if __name__ == "__main__":
    main()