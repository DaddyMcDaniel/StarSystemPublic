#!/usr/bin/env python3
"""
Feedback collection script - run after testing the planet
"""

import json
from datetime import datetime
from pathlib import Path

def main():
    print("📝 Spherical Planet Feedback Collection")
    print("=" * 45)
    print("Please provide your assessment of the spherical planet testing:")
    print()
    
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "tester": "human",
        "planet_file": "runs/miniplanet_seed_42.json",
        "testing_method": "manual_spherical_viewer"
    }
    
    # Core mechanics
    print("🌍 SPHERICAL PLANET MECHANICS:")
    feedback["surface_walking"] = input("1. Surface walking quality (excellent/good/fair/poor/broken): ")
    feedback["spawn_position"] = input("2. Player spawn position (safe/inside_object/floating/underground): ")
    feedback["horizon_curvature"] = input("3. Curved horizon visibility (clearly_visible/somewhat_visible/not_visible): ")
    feedback["circumnavigation"] = input("4. Can walk around entire planet? (yes_easily/yes_difficult/partially/no): ")
    feedback["gravity_orientation"] = input("5. Does 'up' point away from planet center? (always/mostly/sometimes/never): ")
    
    print("\n🎨 VISUAL QUALITY:")
    feedback["planet_size"] = input("6. Planet size feels (too_small/perfect/too_large): ")
    feedback["terrain_variety"] = input("7. Terrain variety (excellent/good/basic/monotonous): ")
    feedback["material_diversity"] = input("8. Material/color variety (describe briefly): ")
    feedback["lighting"] = input("9. Lighting quality (excellent/good/adequate/poor): ")
    feedback["atmosphere"] = input("10. Atmospheric effects (impressive/good/basic/none): ")
    
    print("\n🏗️ BUILDING POTENTIAL:")
    feedback["building_zones"] = input("11. Suitable building areas (abundant/adequate/limited/none): ")
    feedback["resource_visibility"] = input("12. Resource nodes visible (many/some/few/none): ")
    feedback["landmarks"] = input("13. Landmark variety (excellent/good/basic/none): ")
    feedback["exploration_appeal"] = input("14. What makes exploration interesting? ")
    
    print("\n📊 OVERALL ASSESSMENT:")
    while True:
        try:
            rating = int(input("15. Overall experience rating (1-10): "))
            if 1 <= rating <= 10:
                feedback["rating"] = rating
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    feedback["biggest_issue"] = input("16. Biggest issue that needs fixing: ")
    feedback["best_feature"] = input("17. What works best about this planet: ")
    feedback["next_priority"] = input("18. Highest priority for next iteration: ")
    feedback["building_readiness"] = input("19. Ready for building system? (yes/needs_improvement/major_issues): ")
    
    # Save feedback
    feedback_file = Path("agents/communication/human_feedback_manual.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(feedback_file, 'w') as f:
        json.dump(feedback, f, indent=2)
    
    print(f"\n💾 Feedback saved to: {feedback_file}")
    
    # Generate summary
    print("\n📊 FEEDBACK SUMMARY")
    print("=" * 25)
    print(f"Overall Rating: {feedback['rating']}/10")
    print(f"Surface Walking: {feedback['surface_walking']}")
    print(f"Horizon Curvature: {feedback['horizon_curvature']}")
    print(f"Circumnavigation: {feedback['circumnavigation']}")
    print(f"Planet Size: {feedback['planet_size']}")
    print(f"Building Readiness: {feedback['building_readiness']}")
    print()
    print(f"🔧 Top Issue: {feedback['biggest_issue']}")
    print(f"⭐ Best Feature: {feedback['best_feature']}")
    print(f"🎯 Next Priority: {feedback['next_priority']}")
    
    # Recommendation
    if feedback["rating"] >= 8:
        print("\n✅ EXCELLENT - Planet ready for building system integration!")
    elif feedback["rating"] >= 6:
        print("\n🔄 GOOD PROGRESS - Minor improvements needed")
    elif feedback["rating"] >= 4:
        print("\n⚠️ NEEDS WORK - Several issues to address")
    else:
        print("\n🚨 MAJOR REVISION NEEDED - Significant problems identified")
    
    print("\n🎉 Thank you for your feedback!")
    print("This will help improve the spherical planet system.")
    
    return 0

if __name__ == "__main__":
    exit(main())