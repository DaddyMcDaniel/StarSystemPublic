#!/usr/bin/env python3
"""
SUMMARY: Proper Manual Evolution Loop with Agent B Managing Human Testing
=========================================================================
Human-in-the-loop A→D→B(human)→C evolution cycle where Agent B manages the viewer
launch and human testing session, then provides structured input to Agent C.

WORKFLOW:
1. Agent A generates spherical mini-planet with building features
2. Agent D renders scene to JSON format
3. Agent B launches viewer, manages human testing, collects feedback
4. Agent C analyzes Agent B's report and human feedback for evolution direction

USAGE:
  python agents/run_evolution_manual_proper.py
  python agents/run_evolution_manual_proper.py --iterations 3
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

def run_agent_d(pcc_file=None):
    """Run Agent D to render planet scene"""
    print("🎨 Agent D: Rendering planet scene...")
    
    # Generate mini-planet scene for human testing
    manual_seed = int(time.time()) % 10000  # Keep seed manageable
    planet_file = f"runs/miniplanet_seed_{manual_seed}.json"
    
    print(f"🌱 Generating interactive planet scene with seed {manual_seed}")
    result = subprocess.run(
        ["python", "scripts/run_gl.py", "--seed", str(manual_seed), "--generate-only"],
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

def run_agent_b_manual(planet_file):
    """Run Agent B to manage human testing session"""
    print("🎮 Agent B: Managing Human Testing Session")
    print("=" * 45)
    
    # Agent B launches viewer and manages human interaction
    print("🚀 Agent B: Launching spherical planet viewer...")
    print("📋 Agent B: Instructions for human tester:")
    print("   • Use WASD to walk on planet surface")
    print("   • Use mouse to look around (infinite rotation)")
    print("   • Press SPACE to jump")
    print("   • Walk around entire planet to test navigation")
    print("   • Examine all generated objects closely")
    print("   • Test building potential in different areas")
    print("   • Close viewer when testing complete")
    print()
    
    try:
        input("🎯 Agent B: Press ENTER to launch viewer and begin testing session...")
    except EOFError:
        print("⚠️ No interactive terminal - using fallback mode")
        return generate_fallback_b_report(planet_file)
    
    # Launch spherical viewer
    viewer_cmd = [
        "python", "renderer/pcc_spherical_viewer.py", 
        planet_file
    ]
    
    print(f"🎬 Agent B: Launching viewer: {' '.join(viewer_cmd)}")
    print("⏳ Agent B: Monitoring human testing session...")
    
    # Run viewer - Agent B waits for human to finish
    result = subprocess.run(
        viewer_cmd,
        cwd=Path(__file__).parent.parent
    )
    
    print("\n✅ Agent B: Human testing session completed")
    print("📊 Agent B: Collecting test results and generating report...")
    
    # Agent B collects human feedback
    b_report = collect_agent_b_feedback(planet_file)
    
    return b_report

def generate_fallback_b_report(planet_file):
    """Generate fallback Agent B report when no interactive terminal"""
    return {
        "timestamp": datetime.now().isoformat(),
        "agent": "B",
        "session_type": "fallback_analysis",
        "planet_file": planet_file,
        "visual_assessment": {
            "object_recognition": "Unable to test - no interactive session",
            "navigation_quality": "Unable to test - no interactive session",
            "mesh_integration": "Requires visual inspection",
            "material_consistency": "Requires visual inspection"
        },
        "human_feedback_summary": "No human testing session available",
        "recommendations": [
            "Run in interactive terminal for proper Agent B testing",
            "Agent B requires human interaction for visual validation"
        ],
        "confidence": "low"
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

def collect_agent_b_feedback(planet_file):
    """Agent B collects structured feedback from human testing"""
    print("\n📝 Agent B: Human Feedback Collection Protocol")
    print("=" * 50)
    
    # Agent B's enhanced feedback collection based on visual validation
    b_report = {
        "timestamp": datetime.now().isoformat(),
        "agent": "B",
        "session_type": "manual_visual_validation",
        "planet_file": planet_file
    }
    
    # Visual Recognition Assessment
    print("\n🔍 AGENT B: VISUAL RECOGNITION ASSESSMENT")
    b_report["visual_recognition"] = {}
    
    b_report["visual_recognition"]["object_clarity"] = safe_input(
        "B1. Could you clearly identify what each object represents? (excellent/good/poor/unclear): ",
        ['excellent', 'good', 'poor', 'unclear'], 'good'
    )
    
    b_report["visual_recognition"]["material_authenticity"] = safe_input(
        "B2. Do materials look realistic for their type (crystal/temple/terrain)? (very_realistic/realistic/artificial/fake): ",
        ['very_realistic', 'realistic', 'artificial', 'fake'], 'realistic'
    )
    
    b_report["visual_recognition"]["scale_appropriateness"] = safe_input(
        "B3. Do object sizes feel appropriate and proportional? (perfect/good/too_large/too_small): ",
        ['perfect', 'good', 'too_large', 'too_small'], 'good'
    )
    
    # Movement and Navigation Assessment
    print("\n🚶 AGENT B: MOVEMENT QUALITY ASSESSMENT")
    b_report["navigation_quality"] = {}
    
    b_report["navigation_quality"]["surface_walking"] = safe_input(
        "B4. How smooth is walking on the curved planet surface? (perfect/smooth/choppy/broken): ",
        ['perfect', 'smooth', 'choppy', 'broken'], 'smooth'
    )
    
    b_report["navigation_quality"]["mouse_control"] = safe_input(
        "B5. How responsive is mouse camera control? (excellent/good/laggy/broken): ",
        ['excellent', 'good', 'laggy', 'broken'], 'good'
    )
    
    b_report["navigation_quality"]["collision_detection"] = safe_input(
        "B6. How well do collisions work with objects? (perfect/good/clipping/broken): ",
        ['perfect', 'good', 'clipping', 'broken'], 'good'
    )
    
    # Multi-angle Photography Results
    print("\n📸 AGENT B: VISUAL VALIDATION FROM MULTIPLE ANGLES")
    b_report["photography_analysis"] = {}
    
    b_report["photography_analysis"]["objects_from_all_angles"] = safe_input(
        "B7. Do objects look correct from all viewing angles? (yes_all/mostly/some_issues/major_problems): ",
        ['yes_all', 'mostly', 'some_issues', 'major_problems'], 'mostly'
    )
    
    b_report["photography_analysis"]["mesh_connections"] = safe_input(
        "B8. Are object-terrain connections seamless? (seamless/good/visible_gaps/major_gaps): ",
        ['seamless', 'good', 'visible_gaps', 'major_gaps'], 'good'
    )
    
    # Agent B's Professional Assessment
    print("\n🎯 AGENT B: PROFESSIONAL VISUAL VALIDATION SUMMARY")
    
    while True:
        visual_quality_score = input("B9. Overall visual quality score (1-10): ")
        try:
            score = int(visual_quality_score)
            if 1 <= score <= 10:
                b_report["overall_visual_score"] = score
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    b_report["critical_issues"] = input("B10. Most critical visual/navigation issue to fix: ")
    b_report["best_aspects"] = input("B11. Best visual/navigation aspects to preserve: ")
    b_report["improvement_priority"] = input("B12. Top priority for next iteration: ")
    
    # Agent B generates recommendations
    b_report["agent_b_recommendations"] = []
    
    if b_report["visual_recognition"]["object_clarity"] in ['poor', 'unclear']:
        b_report["agent_b_recommendations"].append("Improve object geometry for better recognition")
    
    if b_report["navigation_quality"]["surface_walking"] in ['choppy', 'broken']:
        b_report["agent_b_recommendations"].append("Fix spherical navigation mechanics")
    
    if b_report["photography_analysis"]["mesh_connections"] in ['visible_gaps', 'major_gaps']:
        b_report["agent_b_recommendations"].append("Enhance mesh-terrain edge alignment")
    
    # Save Agent B report
    b_report_file = Path("agents/communication/agent_b_manual_report.json")
    b_report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(b_report_file, 'w') as f:
        json.dump(b_report, f, indent=2)
    
    print(f"\n✅ Agent B: Report saved to {b_report_file}")
    print(f"📊 Agent B: Visual Quality Score: {b_report['overall_visual_score']}/10")
    print(f"🔧 Agent B: Priority: {b_report['improvement_priority']}")
    
    return b_report

def run_agent_c(b_report):
    """Run Agent C to analyze Agent B's report and set evolution direction"""
    print("\n👁️ Agent C: Analyzing Agent B Report and Human Feedback")
    print("=" * 55)
    
    # Agent C analyzes Agent B's structured report
    visual_score = b_report.get("overall_visual_score", 0)
    critical_issues = b_report.get("critical_issues", "")
    improvement_priority = b_report.get("improvement_priority", "")
    b_recommendations = b_report.get("agent_b_recommendations", [])
    
    print(f"📊 Agent B Visual Score: {visual_score}/10")
    print(f"🔧 Critical Issues: {critical_issues}")
    print(f"⭐ Agent B Priority: {improvement_priority}")
    print(f"💡 Agent B Recommendations: {len(b_recommendations)} items")
    
    # Agent C decision logic based on Agent B's assessment
    decision = {
        "timestamp": datetime.now().isoformat(),
        "agent": "C",
        "b_visual_score": visual_score,
        "decision": "continue",
        "focus_areas": [],
        "next_objective": "",
        "agent_a_modifications": []
    }
    
    # Determine focus areas from Agent B's detailed assessment
    visual_rec = b_report.get("visual_recognition", {})
    nav_quality = b_report.get("navigation_quality", {})
    photo_analysis = b_report.get("photography_analysis", {})
    
    if visual_rec.get("object_clarity") in ['poor', 'unclear']:
        decision["focus_areas"].append("object_geometry_clarity")
        decision["agent_a_modifications"].append("Enhance asset geometry for better visual recognition")
    
    if nav_quality.get("surface_walking") in ['choppy', 'broken']:
        decision["focus_areas"].append("spherical_navigation")
        decision["agent_a_modifications"].append("Improve planet surface navigation mechanics")
    
    if photo_analysis.get("mesh_connections") in ['visible_gaps', 'major_gaps']:
        decision["focus_areas"].append("mesh_edge_alignment")
        decision["agent_a_modifications"].append("Implement precise mesh-terrain edge connection")
    
    # Set evolution decision
    if visual_score >= 8:
        decision["decision"] = "approved"
        decision["next_objective"] = "Visual quality approved by Agent B - ready for advanced features"
        print("✅ Agent C: EVOLUTION APPROVED - Excellent visual validation!")
    elif visual_score >= 6:
        decision["decision"] = "continue"
        decision["next_objective"] = f"Address Agent B priorities: {improvement_priority}"
        print("🔄 Agent C: Continue evolution - Good progress with specific visual improvements needed")
    else:
        decision["decision"] = "major_revision"
        decision["next_objective"] = f"Major visual revision required: {critical_issues}"
        print("🚨 Agent C: Major revision needed - Significant visual issues identified by Agent B")
    
    print(f"🎯 Next Objective: {decision['next_objective']}")
    
    # Agent C proactive modifications (as per enhanced prompt)
    if decision["agent_a_modifications"]:
        print("\n🔧 Agent C: Proactive Agent A Modifications Required:")
        for i, mod in enumerate(decision["agent_a_modifications"], 1):
            print(f"   {i}. {mod}")
    
    # Save Agent C decision
    decision_file = Path("agents/communication/agent_c_decision.json")
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    with open(decision_file, 'w') as f:
        json.dump(decision, f, indent=2)
    
    print(f"💾 Agent C: Decision saved to {decision_file}")
    
    return decision["decision"] == "approved"

def main():
    parser = argparse.ArgumentParser(description="Proper Manual Evolution Loop with Agent B Managing Human Testing")
    parser.add_argument("--iterations", type=int, default=1, help="Number of evolution cycles")
    parser.add_argument("--skip-generation", action="store_true", help="Skip Agent A/D, use existing world")
    
    args = parser.parse_args()
    
    print("🎮 StarSystem Proper Manual Evolution Loop")
    print("=" * 55)
    print("🔄 Agent B manages human testing → Agent C analyzes")
    print(f"🎯 {args.iterations} evolution cycle(s)")
    print()
    
    successful_cycles = 0
    
    for cycle in range(1, args.iterations + 1):
        print(f"🚀 Manual Evolution Cycle {cycle}/{args.iterations}")
        print("-" * 45)
        
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
        
        # Step 3: Agent B manages human testing and feedback collection
        b_report = run_agent_b_manual(planet_file)
        if not b_report:
            print(f"❌ Cycle {cycle} failed at Agent B")
            continue
        
        # Step 4: Agent C analyzes Agent B's report
        if run_agent_c(b_report):
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
        print("🎉 Agent B visual validation approved!")
        print("💡 Agent B report and Agent C analysis saved for system improvement")
    else:
        print("🔄 Continue evolution cycles based on Agent B visual assessment")

if __name__ == "__main__":
    main()