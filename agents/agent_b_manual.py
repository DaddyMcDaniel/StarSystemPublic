#!/usr/bin/env python3
"""
SUMMARY: Agent B Manual - Human Testing Interface
================================================
Simplified Agent B that launches Agent A's generated worlds for human testing
and collects structured feedback to send to Agent C.

FEATURES:
- Launch spherical planet viewer for human interaction
- Collect structured feedback on spherical mechanics
- Assess building potential and visual quality
- Generate feedback reports compatible with Agent C
- Store human feedback for training automated Agent B

ROLE:
- Replaces automated Agent B in manual evolution loop
- Provides human-in-the-loop validation of Agent A's generations
- Ensures human feedback is properly formatted for Agent C analysis
- Maintains same interface as automated Agent B for seamless integration

USAGE:
  python agents/agent_b_manual.py <planet_file>
  python agents/agent_b_manual.py runs/miniplanet_seed_42.json
  python agents/agent_b_manual.py --latest  # Use latest generated planet

INTEGRATION:
- Called by agents/run_evolution_manual.py
- Outputs feedback to agents/communication/human_feedback.json
- Compatible with Agent C analysis pipeline
"""

import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

class ManualAgentB:
    def __init__(self, planet_file):
        self.planet_file = planet_file
        self.feedback_file = Path("agents/communication/human_feedback.json")
        self.report_file = Path("runs/latest/agent_b_manual_report.json")
        
        # Ensure output directories exist
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
    
    def launch_planet_viewer(self):
        """Launch spherical planet viewer for human testing"""
        print("🚀 Agent B Manual: Launching Planet Viewer")
        print("=" * 45)
        print(f"🌍 Planet: {self.planet_file}")
        print()
        print("🎮 CONTROLS:")
        print("  WASD - Walk on planet surface")
        print("  Mouse - Look around (first-person)")
        print("  SPACE - Jump")
        print("  R - Reset position")
        print("  TAB - Toggle mouse capture")
        print("  ESC - Exit viewer")
        print()
        print("🧪 TESTING OBJECTIVES:")
        print("  1. Test spherical surface walking")
        print("  2. Verify curved horizon visibility")
        print("  3. Attempt circumnavigation")
        print("  4. Assess building zone potential")
        print("  5. Evaluate visual quality")
        print()
        
        input("Press ENTER to launch viewer (close it when done testing)...")
        
        # Launch spherical viewer
        viewer_cmd = [
            "python", "renderer/pcc_spherical_viewer.py", 
            self.planet_file
        ]
        
        print(f"🌍 Starting viewer: {' '.join(viewer_cmd)}")
        
        try:
            # Run viewer - blocks until human closes it
            result = subprocess.run(
                viewer_cmd,
                cwd=Path(__file__).parent.parent
            )
            
            print("\n✅ Viewer session completed")
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Viewer interrupted by user")
            return True
        except Exception as e:
            print(f"\n❌ Viewer failed: {e}")
            return False
    
    def collect_structured_feedback(self):
        """Collect detailed human feedback on planet testing"""
        print("\n📝 Agent B Manual: Collecting Human Feedback")
        print("=" * 50)
        
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "agent": "B_manual",
            "planet_file": str(self.planet_file),
            "tester_type": "human",
            "testing_method": "manual_spherical_viewer"
        }
        
        # Core spherical planet mechanics
        print("\n🌍 SPHERICAL PLANET MECHANICS ASSESSMENT:")
        feedback["spherical_mechanics"] = self._assess_spherical_mechanics()
        
        # Visual quality evaluation
        print("\n🎨 VISUAL QUALITY EVALUATION:")
        feedback["visual_quality"] = self._assess_visual_quality()
        
        # Building and exploration potential
        print("\n🏗️ BUILDING & EXPLORATION POTENTIAL:")
        feedback["building_potential"] = self._assess_building_potential()
        
        # Overall assessment
        print("\n📊 OVERALL ASSESSMENT:")
        feedback["overall_assessment"] = self._collect_overall_assessment()
        
        return feedback
    
    def _assess_spherical_mechanics(self):
        """Assess core spherical planet mechanics"""
        mechanics = {}
        
        # Surface walking
        while True:
            walking = input("1. Surface walking quality (excellent/good/fair/poor/broken): ").lower()
            if walking in ['excellent', 'good', 'fair', 'poor', 'broken']:
                mechanics["surface_walking"] = walking
                break
            print("   Please enter: excellent, good, fair, poor, or broken")
        
        # Player spawn position
        while True:
            spawn = input("2. Player spawn position (safe/inside_object/floating/underground): ").lower()
            if spawn in ['safe', 'inside_object', 'floating', 'underground']:
                mechanics["spawn_position"] = spawn
                break
            print("   Please enter: safe, inside_object, floating, or underground")
        
        # Horizon curvature visibility
        while True:
            horizon = input("3. Curved horizon visibility (clearly_visible/somewhat_visible/not_visible): ").lower()
            if horizon in ['clearly_visible', 'somewhat_visible', 'not_visible']:
                mechanics["horizon_curvature"] = horizon
                break
            print("   Please enter: clearly_visible, somewhat_visible, or not_visible")
        
        # Circumnavigation capability
        while True:
            circumnav = input("4. Can you walk around the entire planet? (yes_easily/yes_difficult/partially/no): ").lower()
            if circumnav in ['yes_easily', 'yes_difficult', 'partially', 'no']:
                mechanics["circumnavigation"] = circumnav
                break
            print("   Please enter: yes_easily, yes_difficult, partially, or no")
        
        # Gravity orientation
        while True:
            gravity = input("5. Does 'up' always point away from planet center? (always/mostly/sometimes/never): ").lower()
            if gravity in ['always', 'mostly', 'sometimes', 'never']:
                mechanics["gravity_orientation"] = gravity
                break
            print("   Please enter: always, mostly, sometimes, or never")
        
        return mechanics
    
    def _assess_visual_quality(self):
        """Assess visual quality and aesthetics"""
        visual = {}
        
        # Planet size perception
        while True:
            size = input("6. Planet size feels (too_small/perfect/too_large): ").lower()
            if size in ['too_small', 'perfect', 'too_large']:
                visual["planet_size"] = size
                break
            print("   Please enter: too_small, perfect, or too_large")
        
        # Terrain variety
        while True:
            terrain = input("7. Terrain variety (excellent/good/basic/monotonous): ").lower()
            if terrain in ['excellent', 'good', 'basic', 'monotonous']:
                visual["terrain_variety"] = terrain
                break
            print("   Please enter: excellent, good, basic, or monotonous")
        
        # Material diversity
        visual["material_diversity"] = input("8. Describe material/color variety (e.g., 'good mix', 'needs more colors'): ")
        
        # Lighting quality
        while True:
            lighting = input("9. Lighting quality (excellent/good/adequate/poor): ").lower()
            if lighting in ['excellent', 'good', 'adequate', 'poor']:
                visual["lighting"] = lighting
                break
            print("   Please enter: excellent, good, adequate, or poor")
        
        # Atmospheric effects
        while True:
            atmosphere = input("10. Atmospheric effects (impressive/good/basic/none): ").lower()
            if atmosphere in ['impressive', 'good', 'basic', 'none']:
                visual["atmosphere"] = atmosphere
                break
            print("   Please enter: impressive, good, basic, or none")
        
        return visual
    
    def _assess_building_potential(self):
        """Assess building and exploration potential"""
        building = {}
        
        # Building zone availability
        while True:
            zones = input("11. Suitable building areas (abundant/adequate/limited/none): ").lower()
            if zones in ['abundant', 'adequate', 'limited', 'none']:
                building["building_zones"] = zones
                break
            print("   Please enter: abundant, adequate, limited, or none")
        
        # Resource visibility
        while True:
            resources = input("12. Resource nodes/materials visible (many/some/few/none): ").lower()
            if resources in ['many', 'some', 'few', 'none']:
                building["resource_visibility"] = resources
                break
            print("   Please enter: many, some, few, or none")
        
        # Landmark variety
        while True:
            landmarks = input("13. Landmark/structure variety (excellent/good/basic/none): ").lower()
            if landmarks in ['excellent', 'good', 'basic', 'none']:
                building["landmarks"] = landmarks
                break
            print("   Please enter: excellent, good, basic, or none")
        
        # Exploration interest
        building["exploration_appeal"] = input("14. What makes exploration interesting? (describe): ")
        
        return building
    
    def _collect_overall_assessment(self):
        """Collect overall assessment and priorities"""
        assessment = {}
        
        # Overall rating
        while True:
            try:
                rating = int(input("15. Overall experience rating (1-10): "))
                if 1 <= rating <= 10:
                    assessment["rating"] = rating
                    break
                else:
                    print("   Please enter a number between 1 and 10")
            except ValueError:
                print("   Please enter a valid number")
        
        # Top issue
        assessment["biggest_issue"] = input("16. Biggest issue that needs fixing: ")
        
        # Best feature
        assessment["best_feature"] = input("17. What works best about this planet: ")
        
        # Next priority
        assessment["next_priority"] = input("18. Highest priority for next iteration: ")
        
        # Readiness for building
        while True:
            ready = input("19. Ready for building system integration? (yes/needs_improvement/major_issues): ").lower()
            if ready in ['yes', 'needs_improvement', 'major_issues']:
                assessment["building_readiness"] = ready
                break
            print("   Please enter: yes, needs_improvement, or major_issues")
        
        return assessment
    
    def generate_agent_b_report(self, feedback):
        """Generate Agent B compatible report from human feedback"""
        assessment = feedback["overall_assessment"]
        mechanics = feedback["spherical_mechanics"]
        visual = feedback["visual_quality"]
        building = feedback["building_potential"]
        
        # Convert human feedback to Agent B report format
        report = {
            "agent": "B_manual",
            "timestamp": feedback["timestamp"],
            "planet_file": feedback["planet_file"],
            "tester": "human",
            "bridge_connected": False,
            "visual_testing_enabled": True,
            "evaluations": {
                "mechanics": {
                    "spherical_mechanics": "PASS" if mechanics["surface_walking"] in ['excellent', 'good'] else "FAIL",
                    "spawn_position": mechanics["spawn_position"],
                    "circumnavigation": "PASS" if mechanics["circumnavigation"] in ['yes_easily', 'yes_difficult'] else "FAIL",
                    "surface_gravity": "PASS" if mechanics["gravity_orientation"] in ['always', 'mostly'] else "FAIL",
                    "score": min(100, max(0, (assessment["rating"] * 10)))
                },
                "visual": {
                    "planet_curvature": "GOOD" if mechanics["horizon_curvature"] == "clearly_visible" else "BASIC",
                    "terrain_variety": visual["terrain_variety"].upper(),
                    "material_diversity": visual["material_diversity"],
                    "lighting_quality": visual["lighting"].upper(),
                    "atmospheric_effects": visual["atmosphere"].upper(),
                    "immersion_score": assessment["rating"] * 10
                },
                "building_potential": {
                    "suitable_zones": building["building_zones"],
                    "resource_availability": building["resource_visibility"],
                    "landmark_variety": building["landmarks"],
                    "exploration_appeal": building["exploration_appeal"],
                    "readiness": assessment["building_readiness"]
                }
            },
            "improvement_suggestions": self._generate_suggestions(feedback),
            "overall_score": assessment["rating"] * 10,
            "recommendation": self._get_recommendation(assessment["rating"]),
            "human_feedback": feedback
        }
        
        return report
    
    def _generate_suggestions(self, feedback):
        """Generate improvement suggestions based on feedback"""
        suggestions = []
        
        mechanics = feedback["spherical_mechanics"]
        visual = feedback["visual_quality"]
        building = feedback["building_potential"]
        assessment = feedback["overall_assessment"]
        
        # Mechanics suggestions
        if mechanics["surface_walking"] in ['poor', 'broken']:
            suggestions.append({
                "category": "mechanics",
                "priority": "critical",
                "suggestion": "Fix surface walking mechanics",
                "implementation": "Improve spherical gravity and surface normal calculations",
                "human_feedback": True
            })
        
        if mechanics["spawn_position"] != "safe":
            suggestions.append({
                "category": "spawn",
                "priority": "high", 
                "suggestion": "Fix player spawn position",
                "implementation": "Ensure spawn at safe equatorial position",
                "human_feedback": True
            })
        
        # Visual suggestions
        if visual["terrain_variety"] in ['basic', 'monotonous']:
            suggestions.append({
                "category": "visual",
                "priority": "medium",
                "suggestion": "Improve terrain variety",
                "implementation": "Add more material types and procedural placement",
                "human_feedback": True
            })
        
        if visual["planet_size"] == "too_small":
            suggestions.append({
                "category": "scale",
                "priority": "medium",
                "suggestion": "Increase planet size",
                "implementation": "Scale radius to 25-35 units for better exploration",
                "human_feedback": True
            })
        elif visual["planet_size"] == "too_large":
            suggestions.append({
                "category": "scale",
                "priority": "medium",
                "suggestion": "Reduce planet size",
                "implementation": "Scale radius to 20-25 units for manageable exploration",
                "human_feedback": True
            })
        
        # Building suggestions
        if building["building_zones"] in ['limited', 'none']:
            suggestions.append({
                "category": "building",
                "priority": "high",
                "suggestion": "Add more building-suitable areas",
                "implementation": "Create flat zones and platform-ready surfaces",
                "human_feedback": True
            })
        
        # Priority suggestion from human
        if assessment["next_priority"]:
            suggestions.append({
                "category": "human_priority",
                "priority": "high",
                "suggestion": assessment["next_priority"],
                "implementation": "Address human-identified priority",
                "human_feedback": True
            })
        
        return suggestions
    
    def _get_recommendation(self, rating):
        """Get overall recommendation based on rating"""
        if rating >= 8:
            return "APPROVED"
        elif rating >= 6:
            return "NEEDS_MINOR_IMPROVEMENT"
        elif rating >= 4:
            return "NEEDS_IMPROVEMENT"
        else:
            return "NEEDS_MAJOR_REVISION"
    
    def save_feedback(self, feedback, report):
        """Save feedback and report to files"""
        # Save raw human feedback
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        # Save Agent B compatible report
        with open(self.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Feedback saved to: {self.feedback_file}")
        print(f"💾 Report saved to: {self.report_file}")
    
    def run_manual_testing(self):
        """Run complete manual testing workflow"""
        print("🤖 Agent B Manual: Human Testing Interface")
        print("=" * 50)
        
        # Step 1: Launch viewer for human testing
        if not self.launch_planet_viewer():
            print("❌ Failed to launch planet viewer")
            return False
        
        # Step 2: Collect structured feedback
        feedback = self.collect_structured_feedback()
        
        # Step 3: Generate Agent B compatible report
        report = self.generate_agent_b_report(feedback)
        
        # Step 4: Save feedback and report
        self.save_feedback(feedback, report)
        
        # Step 5: Display summary
        self._display_summary(report)
        
        return True
    
    def _display_summary(self, report):
        """Display testing summary"""
        print("\n📊 Agent B Manual Testing Summary")
        print("=" * 40)
        print(f"Overall Score: {report['overall_score']}/100")
        print(f"Recommendation: {report['recommendation']}")
        print(f"Suggestions: {len(report['improvement_suggestions'])} improvements identified")
        
        if report['overall_score'] >= 80:
            print("✅ Excellent feedback - Planet ready for building integration!")
        elif report['overall_score'] >= 60:
            print("🔄 Good progress - Minor improvements needed")
        else:
            print("🚨 Significant improvements required")

def main():
    parser = argparse.ArgumentParser(description="Agent B Manual - Human Testing Interface")
    parser.add_argument("planet_file", nargs='?', help="Planet file to test")
    parser.add_argument("--latest", action="store_true", help="Use latest generated planet")
    
    args = parser.parse_args()
    
    # Determine planet file
    if args.latest:
        planet_file = "runs/miniplanet_seed_42.json"  # Default latest
    elif args.planet_file:
        planet_file = args.planet_file
    else:
        planet_file = "runs/miniplanet_seed_42.json"  # Default fallback
    
    if not Path(planet_file).exists():
        print(f"❌ Planet file not found: {planet_file}")
        return 1
    
    # Run manual testing
    agent_b = ManualAgentB(planet_file)
    success = agent_b.run_manual_testing()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())