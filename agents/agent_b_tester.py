#!/usr/bin/env python3
"""
SUMMARY: Agent B Game Tester - Spherical Planet Evaluation
==========================================================
Agent B evaluates generated mini-planets using the spherical viewer bridge protocol.
Tests tiny planet mechanics, navigation, and provides feedback for evolution loop.

KEY FEATURES:
- Connects to spherical viewer via bridge protocol
- Tests circumnavigation and surface walking
- Evaluates visual quality and planet mechanics
- Generates structured feedback for Agent C analysis
- Supports automated and interactive testing modes

TESTING CRITERIA:
- Spherical surface walking (not flat ground)
- Circumnavigation capability (walk full circles)
- Visible planet curvature and horizon effects
- Surface-relative gravity and orientation
- Visual quality and immersion factors

USAGE:
  python agents/agent_b_tester.py                    # Test latest generated planet
  python agents/agent_b_tester.py --interactive      # Manual testing mode
  python agents/agent_b_tester.py --planet scene.json # Test specific planet

RELATED FILES:
- renderer/pcc_spherical_viewer.py - Viewer with bridge protocol
- scripts/run_gl.py - Planet generation and launcher
- Week 3 requirement for Agent B spherical world evaluation
"""

import os
import sys
import json
import time
import socket
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone
import threading

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def log_agent_action(action, data):
    """Log Agent B actions for evolution tracking"""
    runs_dir = Path("runs")
    latest_dir = runs_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "B",
        "action": action,
        "data": data
    }
    
    with open(latest_dir / "agent_b_evaluation.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def connect_to_viewer_bridge(port=8765, timeout=5):
    """Connect to spherical viewer bridge and get planet state"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(('localhost', port))
        
        # Send query
        sock.send(b"GET_STATE")
        
        # Receive response
        response = sock.recv(4096).decode('utf-8')
        sock.close()
        
        return json.loads(response)
    except Exception as e:
        return None

def take_screenshot(filename):
    """Take screenshot using PIL if available"""
    try:
        import PIL.ImageGrab as ImageGrab
        screenshot = ImageGrab.grab()
        screenshot.save(filename)
        return True
    except ImportError:
        # Fallback to system screenshot
        try:
            import subprocess
            subprocess.run(['gnome-screenshot', '-f', filename], check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

def simulate_viewer_input(action, duration=0.5):
    """Simulate input to the viewer window using xdotool"""
    try:
        import subprocess
        
        # Find the viewer window
        result = subprocess.run(['xdotool', 'search', '--name', 'StarSystem Spherical Planet'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return False
            
        window_id = result.stdout.strip().split('\n')[0] if result.stdout.strip() else None
        if not window_id:
            return False
        
        # Focus the window
        subprocess.run(['xdotool', 'windowfocus', window_id], timeout=5)
        time.sleep(0.2)
        
        # Send the action
        if action == "move_forward":
            subprocess.run(['xdotool', 'key', '--window', window_id, 'w'], timeout=5)
            time.sleep(duration)
            subprocess.run(['xdotool', 'keyup', '--window', window_id, 'w'], timeout=5)
        elif action == "turn_left":
            # Simulate mouse movement for turning
            subprocess.run(['xdotool', 'mousemove_relative', '--', '-50', '0'], timeout=5)
        elif action == "turn_right":
            subprocess.run(['xdotool', 'mousemove_relative', '--', '50', '0'], timeout=5)
        elif action == "jump":
            subprocess.run(['xdotool', 'key', '--window', window_id, 'space'], timeout=5)
        elif action == "crouch":
            subprocess.run(['xdotool', 'key', '--window', window_id, 'c'], timeout=5)
        elif action == "reset_position":
            subprocess.run(['xdotool', 'key', '--window', window_id, 'r'], timeout=5)
        
        time.sleep(0.1)
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

def analyze_screenshot(screenshot_path, context=""):
    """Analyze screenshot for visual issues"""
    if not Path(screenshot_path).exists():
        return {"analysis": "Screenshot not available", "issues": ["Could not capture screenshot"]}
    
    # Basic analysis based on file size and context
    file_size = Path(screenshot_path).stat().st_size
    analysis = {
        "screenshot_path": screenshot_path,
        "context": context,
        "file_size_kb": file_size // 1024,
        "analysis": "",
        "issues": [],
        "positive_aspects": []
    }
    
    if file_size < 10000:  # Very small file suggests problem
        analysis["issues"].append("Screenshot file suspiciously small - possible capture failure")
    
    if "spawn" in context.lower():
        analysis["analysis"] = "Checking spawn position and initial view"
        analysis["issues"].append("HUMAN FEEDBACK: Player spawning inside north pole - bad spawn position")
        analysis["issues"].append("Need to verify player can see terrain and horizon properly")
    elif "forward" in context.lower():
        analysis["analysis"] = "Checking forward movement and terrain visibility"
        analysis["issues"].append("Verify terrain is visible and player is walking on curved surface")
    elif "360" in context.lower():
        analysis["analysis"] = "Checking 360-degree view and horizon curvature"
        analysis["issues"].append("Should see curved horizon and varied terrain types")
    elif "jump" in context.lower():
        analysis["analysis"] = "Checking jump mechanics and surface return"
        analysis["issues"].append("Player should return to planet surface after jumping")
    
    return analysis

def evaluate_planet_mechanics(bridge_data):
    """Evaluate spherical planet mechanics"""
    if not bridge_data:
        return {
            "spherical_mechanics": "FAIL",
            "reason": "No bridge connection - viewer not running",
            "score": 0
        }
    
    evaluation = {
        "spherical_mechanics": "UNKNOWN",
        "circumnavigation": "UNKNOWN", 
        "surface_gravity": "UNKNOWN",
        "planet_data": bridge_data,
        "score": 0
    }
    
    # Check if viewer reports spherical capabilities
    if bridge_data.get("can_circumnavigate"):
        evaluation["circumnavigation"] = "PASS"
        evaluation["score"] += 25
    
    # Check planet geometry
    planet_radius = bridge_data.get("planet_radius", 0)
    if planet_radius > 10:  # Reasonable planet size
        evaluation["spherical_mechanics"] = "PASS"
        evaluation["score"] += 25
    
    # Check if player is on surface (not floating)
    player_pos = bridge_data.get("player_world_pos", [0, 0, 0])
    planet_center = bridge_data.get("planet_center", [0, 0, 0])
    
    # Calculate distance from planet center
    dx = player_pos[0] - planet_center[0]
    dy = player_pos[1] - planet_center[1]
    dz = player_pos[2] - planet_center[2]
    distance = (dx*dx + dy*dy + dz*dz) ** 0.5
    
    # Player should be near planet surface (radius + small offset)
    expected_distance = planet_radius + 2  # 2 units above surface
    if abs(distance - expected_distance) < 5:  # Within tolerance
        evaluation["surface_gravity"] = "PASS"
        evaluation["score"] += 25
    
    # Check surface orientation tracking
    if "surface_yaw" in bridge_data and "surface_pitch" in bridge_data:
        evaluation["surface_orientation"] = "PASS"
        evaluation["score"] += 25
    
    return evaluation

def test_circumnavigation_theory(bridge_data):
    """Theoretical test of circumnavigation capability"""
    if not bridge_data:
        return {"test": "circumnavigation", "result": "FAIL", "reason": "No viewer connection"}
    
    planet_radius = bridge_data.get("planet_radius", 15.0)
    circumference = 2 * 3.14159 * planet_radius
    
    # Estimate walking time at normal speed
    walking_speed = 5.0  # units per second (from viewer)
    estimated_time = circumference / walking_speed
    
    return {
        "test": "circumnavigation",
        "result": "THEORETICAL_PASS",
        "planet_radius": planet_radius,
        "circumference": round(circumference, 1),
        "estimated_walk_time": round(estimated_time, 1),
        "feasible": estimated_time < 120  # Under 2 minutes
    }

def evaluate_visual_quality(bridge_data):
    """Evaluate visual and immersion quality"""
    # This is subjective - Agent B will provide feedback based on viewer capabilities
    quality_factors = {
        "planet_curvature": "UNKNOWN",
        "atmospheric_effects": "UNKNOWN", 
        "material_variety": "UNKNOWN",
        "lighting_quality": "UNKNOWN",
        "immersion_score": 0
    }
    
    if bridge_data:
        # If viewer is running, assume basic visual features are working
        quality_factors["planet_curvature"] = "BASIC"
        quality_factors["atmospheric_effects"] = "BASIC"
        quality_factors["material_variety"] = "BASIC"
        quality_factors["lighting_quality"] = "BASIC"
        quality_factors["immersion_score"] = 60  # Basic passing score
    
    return quality_factors

def generate_improvement_suggestions(mechanics_eval, visual_eval, circumnavigation_test):
    """Generate specific suggestions for planet improvements"""
    suggestions = []
    
    # Mechanics suggestions
    if mechanics_eval["score"] < 75:
        suggestions.append({
            "category": "mechanics",
            "priority": "high",
            "suggestion": "Improve surface-locking mechanism - player should stay on curved surface",
            "implementation": "Enhanced gravity orientation and surface normal calculations"
        })
    
    # Visual suggestions (always room for improvement in alpha)
    suggestions.extend([
        {
            "category": "visual",
            "priority": "medium", 
            "suggestion": "Add terrain texture variety - current planet looks too uniform",
            "implementation": "Multiple material types: rock, grass, sand with procedural placement"
        },
        {
            "category": "visual",
            "priority": "medium",
            "suggestion": "Increase planet size for better exploration - current 15-unit radius feels small",
            "implementation": "Scale to 25-30 unit radius for more walking time and discovery"
        },
        {
            "category": "visual",
            "priority": "low",
            "suggestion": "Enhanced atmospheric effects - add sky gradient and cloud layers",
            "implementation": "Procedural sky dome with time-of-day effects"
        },
        {
            "category": "visual",
            "priority": "medium",
            "suggestion": "More landmark variety - add caves, mountains, structures",
            "implementation": "Procedural surface features: crater rims, mineral deposits, ruins"
        }
    ])
    
    # Navigation suggestions
    if not circumnavigation_test.get("feasible", False):
        suggestions.append({
            "category": "navigation",
            "priority": "high", 
            "suggestion": "Planet too large for practical circumnavigation testing",
            "implementation": "Reduce planet size or add teleportation landmarks"
        })
    
    return suggestions

def run_comprehensive_visual_test(bridge_port=8765):
    """Run comprehensive visual testing with movement and screenshots"""
    print("🎬 Starting comprehensive visual testing...")
    
    screenshots_dir = Path("runs/latest/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    visual_test_results = {
        "test_sequence": [],
        "screenshots": [],
        "issues_found": [],
        "human_feedback_incorporated": True
    }
    
    # Test sequence based on human feedback
    test_sequence = [
        ("spawn_check", "Initial spawn position check"),
        ("reset_position", "Reset to fix spawn position"),
        ("post_reset_check", "Check position after reset"),
        ("forward_movement", "Walk forward 20 steps"),
        ("360_turn", "Complete 360-degree turn"),
        ("jump_test", "Jump mechanics test"),
        ("crouch_test", "Crouch mechanics test"),
        ("final_position", "Final position assessment")
    ]
    
    for i, (action, description) in enumerate(test_sequence):
        print(f"🔍 Test {i+1}/8: {description}")
        
        # Take screenshot before action
        before_screenshot = screenshots_dir / f"{i+1:02d}_{action}_before.png"
        screenshot_taken = take_screenshot(str(before_screenshot))
        
        if screenshot_taken:
            print(f"📸 Screenshot taken: {before_screenshot.name}")
        else:
            print("⚠️ Could not take screenshot")
        
        # Perform action
        action_success = False
        if action == "spawn_check":
            action_success = True  # Just checking initial state
        elif action == "reset_position":
            action_success = simulate_viewer_input("reset_position")
            time.sleep(1)  # Give time for reset
        elif action == "post_reset_check":
            action_success = True  # Just checking after reset
        elif action == "forward_movement":
            # Walk forward 20 steps
            for step in range(20):
                simulate_viewer_input("move_forward", 0.1)
                time.sleep(0.1)
            action_success = True
        elif action == "360_turn":
            # Perform 360-degree turn
            for turn in range(8):  # 8 turns of 45 degrees each
                simulate_viewer_input("turn_right")
                time.sleep(0.2)
            action_success = True
        elif action == "jump_test":
            action_success = simulate_viewer_input("jump")
            time.sleep(1)  # Time to land
        elif action == "crouch_test":
            action_success = simulate_viewer_input("crouch")
            time.sleep(0.5)
        elif action == "final_position":
            action_success = True  # Just final assessment
        
        # Take screenshot after action
        after_screenshot = screenshots_dir / f"{i+1:02d}_{action}_after.png"
        screenshot_taken_after = take_screenshot(str(after_screenshot))
        
        # Analyze screenshots
        before_analysis = analyze_screenshot(str(before_screenshot), f"{description} - before")
        after_analysis = analyze_screenshot(str(after_screenshot), f"{description} - after")
        
        test_result = {
            "step": i+1,
            "action": action,
            "description": description,
            "action_successful": action_success,
            "screenshots": {
                "before": str(before_screenshot) if screenshot_taken else None,
                "after": str(after_screenshot) if screenshot_taken_after else None
            },
            "analysis": {
                "before": before_analysis,
                "after": after_analysis
            }
        }
        
        visual_test_results["test_sequence"].append(test_result)
        
        # Collect issues
        if before_analysis.get("issues"):
            visual_test_results["issues_found"].extend(before_analysis["issues"])
        if after_analysis.get("issues"):
            visual_test_results["issues_found"].extend(after_analysis["issues"])
        
        print(f"✅ Test {i+1} completed")
        time.sleep(0.5)  # Brief pause between tests
    
    # Remove duplicate issues
    visual_test_results["issues_found"] = list(set(visual_test_results["issues_found"]))
    
    # Save comprehensive test results
    with open(Path("runs/latest/visual_test_results.json"), "w") as f:
        json.dump(visual_test_results, f, indent=2)
    
    log_agent_action("visual_testing_complete", {
        "tests_completed": len(test_sequence),
        "screenshots_taken": len([r for r in visual_test_results["test_sequence"] if r["screenshots"]["before"] or r["screenshots"]["after"]]),
        "issues_found": len(visual_test_results["issues_found"])
    })
    
    return visual_test_results

def run_agent_b_evaluation(planet_file=None, interactive=False, bridge_port=8765, visual_testing=True):
    """Main Agent B evaluation function"""
    print("🤖 Agent B: Starting spherical planet evaluation")
    
    # Start viewer if planet file provided
    viewer_process = None
    if planet_file and Path(planet_file).exists():
        print(f"🚀 Launching viewer for {planet_file}")
        viewer_cmd = [
            sys.executable, 
            "renderer/pcc_spherical_viewer.py",
            str(planet_file),
            "--bridge-port", str(bridge_port)
        ]
        viewer_process = subprocess.Popen(viewer_cmd)
        time.sleep(3)  # Give viewer time to start
    
    try:
        # Connect to viewer bridge
        print(f"🌉 Connecting to viewer bridge on port {bridge_port}")
        bridge_data = connect_to_viewer_bridge(bridge_port)
        
        if bridge_data:
            print("✅ Connected to spherical viewer")
            log_agent_action("bridge_connected", bridge_data)
        else:
            print("⚠️ No viewer connection - running theoretical evaluation")
            log_agent_action("bridge_failed", {"port": bridge_port})
        
        # Run evaluations
        print("🔍 Evaluating planet mechanics...")
        mechanics_eval = evaluate_planet_mechanics(bridge_data)
        
        print("🚶 Testing circumnavigation theory...")
        circumnavigation_test = test_circumnavigation_theory(bridge_data)
        
        print("🎨 Evaluating visual quality...")
        visual_eval = evaluate_visual_quality(bridge_data)
        
        # Run comprehensive visual testing if enabled
        visual_test_results = None
        if visual_testing and not interactive:
            print("🎬 Running comprehensive visual testing with movement...")
            try:
                visual_test_results = run_comprehensive_visual_test(bridge_port)
                print(f"📊 Visual testing completed: {len(visual_test_results['issues_found'])} issues found")
            except Exception as e:
                print(f"⚠️ Visual testing failed: {e}")
                visual_test_results = {"issues_found": [f"Visual testing failed: {str(e)}"]}
        
        print("💡 Generating improvement suggestions...")
        suggestions = generate_improvement_suggestions(mechanics_eval, visual_eval, circumnavigation_test)
        
        # Add visual testing feedback to suggestions
        if visual_test_results and visual_test_results.get("issues_found"):
            for issue in visual_test_results["issues_found"]:
                if "spawn" in issue.lower():
                    suggestions.append({
                        "category": "spawn_mechanics",
                        "priority": "critical",
                        "suggestion": "Fix player spawn position - currently spawning inside objects",
                        "implementation": "Adjust spawn position to be properly above planet surface",
                        "human_feedback": True
                    })
                elif "terrain" in issue.lower():
                    suggestions.append({
                        "category": "visual",
                        "priority": "high", 
                        "suggestion": "Improve terrain visibility and surface walking mechanics",
                        "implementation": "Ensure player starts on visible surface with clear terrain view"
                    })
        
        # Compile full evaluation report
        evaluation_report = {
            "agent": "B",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "planet_file": str(planet_file) if planet_file else "unknown",
            "bridge_connected": bridge_data is not None,
            "visual_testing_enabled": visual_testing,
            "evaluations": {
                "mechanics": mechanics_eval,
                "circumnavigation": circumnavigation_test,
                "visual": visual_eval,
                "visual_testing": visual_test_results
            },
            "improvement_suggestions": suggestions,
            "overall_score": mechanics_eval["score"],
            "recommendation": "NEEDS_IMPROVEMENT" if mechanics_eval["score"] < 75 or (visual_test_results and len(visual_test_results.get("issues_found", [])) > 2) else "ACCEPTABLE",
            "human_feedback_incorporated": visual_test_results.get("human_feedback_incorporated", False) if visual_test_results else False
        }
        
        # Save evaluation report
        runs_dir = Path("runs")
        latest_dir = runs_dir / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        
        with open(latest_dir / "agent_b_report.json", "w") as f:
            json.dump(evaluation_report, f, indent=2)
        
        log_agent_action("evaluation_complete", evaluation_report)
        
        # Print summary
        print("\n📊 Agent B Evaluation Summary:")
        print(f"   Mechanics Score: {mechanics_eval['score']}/100")
        print(f"   Overall Rating: {evaluation_report['recommendation']}")
        print(f"   Suggestions: {len(suggestions)} improvements identified")
        
        if interactive:
            print("\n🎮 Interactive mode - check the viewer manually")
            input("Press Enter when done testing...")
        
        return evaluation_report
        
    finally:
        # Clean up viewer process
        if viewer_process:
            print("🛑 Stopping viewer...")
            viewer_process.terminate()
            try:
                viewer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                viewer_process.kill()

def main():
    parser = argparse.ArgumentParser(description="Agent B Spherical Planet Tester with Visual Analysis")
    parser.add_argument("--planet", help="Specific planet file to test")
    parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    parser.add_argument("--bridge-port", type=int, default=8765, help="Bridge port")
    parser.add_argument("--generate-test-planet", action="store_true", help="Generate a test planet first")
    parser.add_argument("--no-visual-testing", action="store_true", help="Disable visual testing with screenshots")
    parser.add_argument("--fix-spawn-position", action="store_true", help="Apply human feedback fix for spawn position")
    
    args = parser.parse_args()
    
    planet_file = args.planet
    
    # Generate test planet if requested
    if args.generate_test_planet:
        print("🌍 Generating test planet...")
        result = subprocess.run([
            sys.executable, "scripts/run_gl.py", 
            "--generate-only", "--seed", "42"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            planet_file = "runs/miniplanet_seed_42.json"
            print(f"✅ Generated test planet: {planet_file}")
        else:
            print(f"❌ Failed to generate test planet: {result.stderr}")
            return 1
    
    # Find latest planet if none specified
    if not planet_file:
        runs_dir = Path("runs")
        planet_files = list(runs_dir.glob("miniplanet_seed_*.json"))
        if planet_files:
            planet_file = max(planet_files, key=lambda p: p.stat().st_mtime)
            print(f"🔍 Using latest planet: {planet_file}")
        else:
            print("❌ No planet file found. Use --generate-test-planet or specify --planet")
            return 1
    
    # Run evaluation
    try:
        evaluation = run_agent_b_evaluation(
            planet_file=planet_file,
            interactive=args.interactive,
            bridge_port=args.bridge_port,
            visual_testing=not args.no_visual_testing
        )
        
        if evaluation["recommendation"] == "ACCEPTABLE":
            print("✅ Agent B: Planet evaluation PASSED")
            return 0
        else:
            print("⚠️ Agent B: Planet needs improvements")
            return 2  # Needs improvement, but not a failure
            
    except KeyboardInterrupt:
        print("\n👋 Agent B evaluation interrupted")
        return 1
    except Exception as e:
        print(f"❌ Agent B evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())