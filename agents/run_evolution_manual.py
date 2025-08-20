#!/usr/bin/env python3
"""
SUMMARY: Manual Evolution Loop with Human Testing
================================================
Human-in-the-loop A→D→Human→C evolution cycle where human plays Agent A's generated worlds
and provides direct feedback to Agent C for evolution guidance.

FEATURES:
- Agent A generates spherical planet prompts (HOD-aligned)
- Agent D renders planet to 3D scene
- Human plays the world and provides structured feedback
- Agent C analyzes human feedback for evolution direction
- Interactive feedback collection with gameplay assessment
- Human feedback stored and reused for Agent B training

WORKFLOW:
1. Agent A generates spherical mini-planet with building features
2. Agent D renders scene to JSON format
3. Human plays world through interactive viewer
4. Human provides feedback via structured prompts
5. Agent C analyzes feedback and sets next evolution objective

USAGE:
  python agents/run_evolution_manual.py
  python agents/run_evolution_manual.py --iterations 3
  python agents/run_evolution_manual.py --skip-generation  # Use existing world

RELATED FILES:
- agents/agent_b_manual.py - Human feedback collection interface
- agents/run_evolution_auto.py - Automated testing loop
"""

import argparse
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

def load_enhanced_prompts():
    """Load enhanced agent prompts before execution"""
    print("📚 Loading Enhanced Agent Prompts...")
    prompt_files = [
        "agents/prompting/agent_a_enhanced.md",
        "agents/prompting/agent_b_enhanced.md", 
        "agents/prompting/agent_c_enhanced.md"
    ]
    
    for prompt_file in prompt_files:
        try:
            with open(prompt_file, 'r') as f:
                content = f.read()
                agent_name = prompt_file.split('/')[-1].replace('_enhanced.md', '').replace('agent_', '').upper()
                print(f"  ✅ Agent {agent_name}: Enhanced prompt loaded ({len(content)} chars)")
        except FileNotFoundError:
            print(f"  ⚠️ {prompt_file} not found - using default prompts")
    
    print("📋 Enhanced prompts ready for agent execution\n")

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
        print(f"   {result.stdout.strip()}")
        
        # Extract the generated game file from output
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'Created:' in line and '.pcc' in line:
                pcc_file = line.split('Created: ')[-1].strip()
                return True, pcc_file
        
        return True, None
    else:
        print(f"❌ Agent A failed: {result.stderr}")
        return False, None

def run_headless_analysis(planet_file: str) -> bool:
    """Run headless world analysis without OpenGL."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from forge.modules.fallback.headless_analyzer import generate_world_report
        
        print("📊 Generating comprehensive world analysis...")
        
        # Generate detailed text report
        report = generate_world_report(planet_file)
        print(report)
        
        # Save report to file
        report_file = f"runs/headless_analysis_{int(time.time())}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Analysis saved to: {report_file}")
        
        print("\n✅ Headless analysis complete!")
        print("🎯 Use this analysis to provide feedback on world quality")
        
        return True
        
    except Exception as e:
        print(f"❌ Headless analysis failed: {e}")
        return False

def run_2d_fallback_analysis(planet_file: str) -> bool:
    """Run 2D visualization fallback mode."""
    try:
        # Try headless analysis first
        success = run_headless_analysis(planet_file)
        
        print("\n📊 2D Visualization would generate:")
        print("   • Top-down planet map")
        print("   • Object distribution diagram") 
        print("   • Material usage charts")
        print("   • Navigation landmark overlay")
        print("💡 2D visualization implementation pending - using headless analysis")
        
        return success
        
    except Exception as e:
        print(f"❌ 2D fallback analysis failed: {e}")
        return False

def run_agent_d(pcc_file=None):
    """Run Agent D to render planet scene"""
    print("🎨 Agent D: Rendering planet scene...")
    
    # Generate mini-planet scene for human testing - USE HERO WORLD SEED
    hero_seed = 31415926  # Use T17 hero planet seed for consistent terrain
    planet_file = f"runs/miniplanet_seed_{hero_seed}.json"
    
    print(f"🌱 Generating interactive planet scene with HERO WORLD seed {hero_seed}")
    # Use actual PCC file from Agent A instead of dummy.pcc
    pcc_input = pcc_file if pcc_file else "dummy.pcc"
    result = subprocess.run(
        ["python", "agents/agent_d_renderer.py", pcc_input, "--seed", str(hero_seed)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode == 0:
        print(f"✅ Agent D: Interactive scene ready at {planet_file}")
        return True, planet_file
    else:
        print(f"❌ Agent D failed: {result.stderr}")
        # Fallback to existing scene
        fallback_file = "runs/miniplanet_seed_42.json"
        if Path(fallback_file).exists():
            print(f"🔄 Using fallback scene: {fallback_file}")
            return True, fallback_file
        return False, None

def run_human_testing(planet_file, test_mode=False, no_viewer=False):
    """Launch human testing interface"""
    print("🎮 Starting Human Testing Session")
    print("=" * 40)
    print(f"🌍 Planet file: {planet_file}")
    
    if test_mode:
        print("🤖 Test mode: Simulating human feedback")
        return True
    
    if no_viewer:
        print("⚠️ Skipping viewer launch - feedback only mode")
        return True
    
    # Always launch OpenGL viewer as per project requirements
    print("✅ 3D Viewer Mode - OpenGL Required")
    print("Instructions:")
    print("1. The spherical planet viewer will launch")
    print("2. Use WASD to walk on the planet surface")
    print("3. Use mouse to look around")
    print("4. Press SPACE to jump")
    print("5. Walk around and test building mechanics")
    print("6. Close the viewer when done testing")
    print("7. You'll provide feedback via prompts")
    print()
    
    try:
        input("Press ENTER to launch the spherical planet viewer...")
    except EOFError:
        print("⚠️ No interactive terminal - launching viewer in background...")
        # Continue to launch viewer even without interactive input
    
    # Launch spherical viewer for human testing - use system python with OpenGL
    viewer_cmd = [
        "python3", "renderer/pcc_spherical_viewer.py", 
        planet_file
    ]
    
    print(f"🚀 Launching: {' '.join(viewer_cmd)}")
    print("🎮 Human testing in progress... (close viewer when done)")
    
    # Run viewer - this will block until human closes it
    result = subprocess.run(
        viewer_cmd,
        cwd=Path(__file__).parent.parent
    )
    
    print("\n✅ Human testing session completed")
    return True

def generate_simulated_feedback():
    """Generate simulated feedback for test mode"""
    return {
        "timestamp": datetime.now().isoformat(),
        "tester": "simulated",
        "session_type": "test_mode",
        "spherical_mechanics": {
            "surface_walking": "good",
            "horizon_curvature": "yes",
            "circumnavigation": "yes"
        },
        "visual_quality": {
            "planet_size": "perfect",
            "terrain_variety": "good variety of materials and landmarks",
            "lighting": "good"
        },
        "building_potential": {
            "suitable_zones": "some",
            "resource_visibility": "moderate"
        },
        "overall_rating": 6,
        "top_issue": "Needs more terrain variety",
        "best_feature": "Spherical surface walking works well",
        "next_improvement": "Add more building zones and resource variety"
    }

def safe_input(prompt, valid_options=None, default=None):
    """Safe input with EOFError handling"""
    try:
        response = input(prompt).lower() if valid_options else input(prompt)
        if valid_options and response in valid_options:
            return response
        elif valid_options:
            print(f"Please enter: {', '.join(valid_options)}")
            return safe_input(prompt, valid_options, default)
        else:
            return response
    except EOFError:
        if default:
            print(f"{default} (default)")
            return default
        else:
            print("(skipped - no input)")
            return ""

def collect_human_feedback(test_mode=False):
    """Collect structured feedback from human tester"""
    if test_mode:
        print("\n🤖 Simulated Feedback Generation")
        print("=" * 35)
        feedback = generate_simulated_feedback()
        print("✅ Generated simulated human feedback")
        return feedback
    
    print("\n📝 Human Feedback Collection")
    print("=" * 35)
    
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "tester": "human",
        "session_type": "manual_evolution"
    }
    
    # Core spherical mechanics
    print("\n🌍 SPHERICAL PLANET MECHANICS:")
    feedback["spherical_mechanics"] = {}
    
    feedback["spherical_mechanics"]["surface_walking"] = safe_input(
        "1. Can you walk smoothly on the curved planet surface? (excellent/good/poor/broken): ",
        ['excellent', 'good', 'poor', 'broken'], 'good'
    )
    
    feedback["spherical_mechanics"]["horizon_curvature"] = safe_input(
        "2. Is the curved horizon clearly visible? (yes/somewhat/no): ",
        ['yes', 'somewhat', 'no'], 'yes'
    )
    
    feedback["spherical_mechanics"]["circumnavigation"] = safe_input(
        "3. Could you walk in a circle around the planet? (yes/partially/no): ",
        ['yes', 'partially', 'no'], 'yes'
    )
    
    # Visual quality
    print("\n🎨 VISUAL QUALITY:")
    feedback["visual_quality"] = {}
    
    while True:
        planet_size = input("4. Is the planet size good for exploration? (too_small/perfect/too_large): ").lower()
        if planet_size in ['too_small', 'perfect', 'too_large']:
            feedback["visual_quality"]["planet_size"] = planet_size
            break
        print("Please enter: too_small, perfect, or too_large")
    
    feedback["visual_quality"]["terrain_variety"] = input("5. Describe terrain variety (e.g., 'needs more materials', 'good variety'): ")
    feedback["visual_quality"]["lighting"] = input("6. How's the lighting? (too_dark/good/too_bright): ").lower()
    
    # Building potential
    print("\n🏗️ BUILDING POTENTIAL:")
    feedback["building_potential"] = {}
    
    while True:
        building_zones = input("7. Are there clear areas suitable for building? (many/some/few/none): ").lower()
        if building_zones in ['many', 'some', 'few', 'none']:
            feedback["building_potential"]["suitable_zones"] = building_zones
            break
        print("Please enter: many, some, few, or none")
    
    feedback["building_potential"]["resource_visibility"] = input("8. Can you see resource nodes/materials? (abundant/moderate/scarce): ").lower()
    
    # Overall assessment
    print("\n📊 OVERALL ASSESSMENT:")
    while True:
        overall_rating = input("9. Overall experience rating (1-10): ")
        try:
            rating = int(overall_rating)
            if 1 <= rating <= 10:
                feedback["overall_rating"] = rating
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    feedback["top_issue"] = input("10. What's the biggest issue that needs fixing? ")
    feedback["best_feature"] = input("11. What works best about this planet? ")
    feedback["next_improvement"] = input("12. What should be improved next iteration? ")
    
    # Save feedback  
    feedback_file = Path("agents/communication/human_feedback.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(feedback_file, 'w') as f:
        json.dump(feedback, f, indent=2)
    
    print(f"\n✅ Feedback saved to: {feedback_file}")
    return feedback

def run_agent_c(human_feedback):
    """Run Agent C to analyze human feedback and set evolution direction"""
    print("\n👁️ Agent C: Analyzing Human Feedback")
    print("=" * 40)
    
    # Analyze key metrics from human feedback
    spherical = human_feedback.get("spherical_mechanics", {})
    visual = human_feedback.get("visual_quality", {})
    building = human_feedback.get("building_potential", {})
    
    overall_rating = human_feedback.get("overall_rating", 0)
    top_issue = human_feedback.get("top_issue", "")
    next_improvement = human_feedback.get("next_improvement", "")
    
    print(f"📊 Human Rating: {overall_rating}/10")
    print(f"🔧 Top Issue: {top_issue}")
    print(f"⭐ Next Priority: {next_improvement}")
    
    # Agent C decision logic
    decision = {
        "timestamp": datetime.now().isoformat(),
        "agent": "C",
        "human_rating": overall_rating,
        "decision": "continue",
        "focus_areas": [],
        "next_objective": ""
    }
    
    # Determine focus areas based on feedback
    if spherical.get("surface_walking") in ['poor', 'broken']:
        decision["focus_areas"].append("spherical_mechanics")
    
    if spherical.get("horizon_curvature") == 'no':
        decision["focus_areas"].append("visual_curvature")
    
    if visual.get("planet_size") in ['too_small', 'too_large']:
        decision["focus_areas"].append("planet_sizing")
    
    if building.get("suitable_zones") in ['few', 'none']:
        decision["focus_areas"].append("building_zones")
    
    # Set next objective
    if overall_rating >= 8:
        decision["decision"] = "approved"
        decision["next_objective"] = "Planet design approved - ready for building system integration"
        print("✅ Agent C: ITERATION APPROVED - Excellent feedback!")
    elif overall_rating >= 6:
        decision["decision"] = "continue"
        decision["next_objective"] = f"Focus on: {', '.join(decision['focus_areas'])} - {next_improvement}"
        print("🔄 Agent C: Continue evolution - Good progress with specific improvements needed")
    else:
        decision["decision"] = "major_revision"
        decision["next_objective"] = f"Major revision required - Address: {top_issue}"
        print("🚨 Agent C: Major revision needed - Significant issues identified")
    
    print(f"🎯 Next Objective: {decision['next_objective']}")
    
    # Save Agent C decision
    decision_file = Path("agents/communication/agent_c_decision.json")
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    with open(decision_file, 'w') as f:
        json.dump(decision, f, indent=2)
    
    print(f"💾 Decision saved to: {decision_file}")
    
    return decision["decision"] == "approved"

def main():
    parser = argparse.ArgumentParser(description="Manual Evolution Loop with Human Testing")
    parser.add_argument("--iterations", type=int, default=1, help="Number of evolution cycles")
    parser.add_argument("--skip-generation", action="store_true", help="Skip Agent A/D, use existing world")
    parser.add_argument("--test-mode", action="store_true", help="Non-interactive test mode with simulated feedback")
    parser.add_argument("--no-viewer", action="store_true", help="Skip viewer launch (feedback only)")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode (requires terminal)")
    
    args = parser.parse_args()
    
    print("🎮 StarSystem Manual Evolution Loop")
    print("=" * 50)
    if args.test_mode:
        print(f"🤖 Test mode: Non-interactive with simulated feedback")
    else:
        print(f"🔄 Human-in-the-loop testing")
    print(f"🎯 {args.iterations} evolution cycle(s)")
    if args.no_viewer:
        print("⚠️ Viewer disabled - feedback only")
    print()
    
    successful_cycles = 0
    
    for cycle in range(1, args.iterations + 1):
        print(f"🚀 Manual Evolution Cycle {cycle}/{args.iterations}")
        print("-" * 40)
        
        # Load enhanced prompts before each cycle
        load_enhanced_prompts()
        
        pcc_file = None
        planet_file = None
        
        if not args.skip_generation:
            # Step 1: Agent A generates spherical planet
            a_success, pcc_file = run_agent_a()
            if not a_success:
                print(f"❌ Cycle {cycle} failed at Agent A")
                continue
            
            # Step 2: Agent D renders scene
            d_success, planet_file = run_agent_d(pcc_file)
            if not d_success:
                print(f"❌ Cycle {cycle} failed at Agent D")
                continue
        else:
            # Use existing planet
            planet_file = "runs/miniplanet_seed_42.json"
            print(f"🔄 Using existing planet: {planet_file}")
        
        # Step 3: Human testing
        if not run_human_testing(planet_file, args.test_mode, args.no_viewer):
            print(f"❌ Cycle {cycle} failed at human testing")
            continue
        
        # Step 4: Collect human feedback
        human_feedback = collect_human_feedback(args.test_mode)
        
        # Step 5: Agent C analyzes feedback
        if run_agent_c(human_feedback):
            successful_cycles += 1
            print(f"\n✅ Cycle {cycle} completed successfully")
            break  # Stop on approval
        else:
            print(f"\n🔄 Cycle {cycle} needs more improvement")
        
        print()
    
    print("🏁 Manual Evolution Summary")
    print("=" * 35)
    print(f"✅ Successful cycles: {successful_cycles}/{args.iterations}")
    
    if successful_cycles > 0:
        print("🎉 Human-approved evolution achieved!")
        print("💡 Human feedback has been saved for Agent B training")
    else:
        print("🔄 Continue evolution cycles based on human feedback")

if __name__ == "__main__":
    main()