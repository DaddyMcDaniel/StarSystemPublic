#!/usr/bin/env python3
"""
SUMMARY: Agent C - Supervisory & Governance Agent v1
===================================================
Strategic overseer and governance agent in the PCC-LanguageV2 evolution system.
Coordinates between Agent A (generator) and Agent B (validator) workflows.

ROLE & RESPONSIBILITIES:
- Setting clear iteration objectives that align with project goals
- Enforcing architectural constraints and design principles  
- Approving or rejecting work based on established acceptance criteria
- Maintaining system coherence and preventing scope creep
- Providing strategic direction for the evolution process
- Ensuring compliance with High-Order Benchmark requirements

KEY OVERSIGHT AREAS:
- Blueprint Chips: Approve node graph designs and prompt-to-code transformations
- Performance Pass: Set budget thresholds and approve performance criteria
- Schematic Cards: Validate export/import design and approve player sharing standards
- Building System: Approve grid architectures, hammer-style tools, and placement rules
- Planet Generation: Set standards for layered terrain, navigation, and alien aesthetics

INTEGRATION:
- Loads HOD from agents/memory/agent_c_prompt.txt
- Receives outputs from Agent A and validation reports from Agent B
- Provides final approval/rejection decisions with redirect plans
- Prioritizes creative freedom and unlimited building time (Halo Forge philosophy)

USAGE:
  python agents/agent_c_supervisor.py
  make agents  # Run full A→B→C handshake

RELATED FILES:
- agents/memory/agent_c_prompt.txt - High-Order Directive with oversight contracts
- Focus on robust planet generation foundation before advanced features
"""
import json
from pathlib import Path
from datetime import datetime

PCC_ROOT = Path(__file__).parent.parent
PCC_SPEC = PCC_ROOT / "docs/pcc_language_spec.md"
PCC_LOGS = PCC_ROOT / "agents/logs"

class PCCSupervisor:
    def analyze_test_results(self, test_data, cross_analysis):
        """Step 5: Analyze if game matched prompt perfectly"""
        perfect_execution = (
            test_data["gameplay_test"]["success_rate"] >= 0.95 and
            cross_analysis["accuracy_score"] >= 0.9 and
            len(cross_analysis["errors_found"]) == 0
        )
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "step": 5,
            "perfect_execution": perfect_execution,
            "ready_for_compression": perfect_execution,
            "agent": "C"
        }
        
        self._log(analysis)
        return analysis
    
    def compress_pcc_further(self, pcc_file, analysis_result):
        """Step 5: Compress successful PCC further"""
        if not analysis_result["ready_for_compression"]:
            return None
        
        compression_data = {
            "timestamp": datetime.now().isoformat(),
            "step": 5,
            "compression_ratio": 0.8,
            "improvements": ["remove_defaults", "merge_nodes"],
            "agent": "C"
        }
        
        self._log(compression_data)
        return compression_data
    
    def update_language_spec(self, compression_data):
        """Step 6: Update PCC language spec"""
        if not compression_data:
            return
        
        spec_update = {
            "timestamp": datetime.now().isoformat(),
            "step": 6,
            "spec_updates": ["Added compression patterns"],
            "agent": "C"
        }
        
        self._log(spec_update)
        return spec_update
    
    def coordinate_next_iteration(self):
        """Step 7: Start new evolution loop"""
        coordination = {
            "timestamp": datetime.now().isoformat(),
            "step": 7,
            "action": "start_new_loop",
            "agent": "C"
        }
        
        self._log(coordination)
        return True
    
    def analyze_terrain_feedback(self, agent_b_report, human_feedback):
        """Analyze terrain issues from Agent B and human feedback"""
        terrain_issues = []
        improvements = []
        
        # Parse human feedback for terrain issues
        if human_feedback and "critical_issues" in human_feedback:
            for issue in human_feedback["critical_issues"]:
                if "terrain" in issue.lower() or "smooth sphere" in issue.lower():
                    terrain_issues.append(issue)
                if "planet" in issue.lower() and "size" in issue.lower():
                    if "double" in issue.lower() or "bigger" in issue.lower():
                        improvements.append("increase_planet_size")
                if "grid" in issue.lower() and "halv" in issue.lower():
                    improvements.append("decrease_grid_size")
                if "flat areas" in issue.lower() or "hilly" in issue.lower():
                    improvements.append("increase_height_variation")
                if "building" in issue.lower():
                    improvements.append("more_building_zones")
        
        # Check Agent B score vs human score discrepancy
        agent_b_score = agent_b_report.get("visual_score", 0) if agent_b_report else 0
        human_score = human_feedback.get("visual_score", 0) if human_feedback else 0
        
        major_discrepancy = abs(agent_b_score - human_score) > 5
        
        return {
            "terrain_issues": terrain_issues,
            "terrain_improvements": improvements,
            "agent_b_vs_human_discrepancy": major_discrepancy,
            "priority": "HIGH" if terrain_issues else "MEDIUM"
        }
    
    def generate_agent_d_feedback(self, terrain_analysis):
        """Generate specific feedback for Agent D to improve terrain generation"""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "agent_c_analysis": terrain_analysis,
            "terrain_improvements": terrain_analysis["terrain_improvements"],
            "priority": terrain_analysis["priority"],
            "specific_actions": []
        }
        
        # Convert improvements to specific Agent D actions
        for improvement in terrain_analysis["terrain_improvements"]:
            if improvement == "increase_height_variation":
                feedback["specific_actions"].append({
                    "action": "increase_height_variation",
                    "parameter": "height_variation",
                    "change": "+0.2"
                })
            elif improvement == "more_building_zones":
                feedback["specific_actions"].append({
                    "action": "more_building_zones", 
                    "parameter": "building_zones",
                    "change": "+0.1"
                })
            elif improvement == "increase_planet_size":
                feedback["specific_actions"].append({
                    "action": "increase_planet_size",
                    "parameter": "default_radius", 
                    "change": "50.0"
                })
        
        # Save feedback for Agent D
        feedback_file = PCC_ROOT / "agents/communication/agent_d_feedback.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        return feedback
    
    def _log(self, entry):
        log_file = PCC_LOGS / "agent_c.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    supervisor = PCCSupervisor()
    print("👁️ Agent C: Supervisor active")
    
    # Check for feedback files and analyze
    communication_dir = PCC_ROOT / "agents/communication"
    
    human_feedback_file = communication_dir / "human_feedback.json"
    agent_b_file = communication_dir / "agent_b_report.json"  # Assuming this exists
    
    agent_b_report = None
    human_feedback = None
    
    if human_feedback_file.exists():
        with open(human_feedback_file) as f:
            human_feedback = json.load(f)
        print("📝 Found human feedback")
    
    if agent_b_file.exists():
        with open(agent_b_file) as f:
            agent_b_report = json.load(f)
        print("🤖 Found Agent B report")
    
    if human_feedback or agent_b_report:
        print("🧠 Analyzing terrain feedback...")
        terrain_analysis = supervisor.analyze_terrain_feedback(agent_b_report, human_feedback)
        
        if terrain_analysis["terrain_issues"]:
            print(f"⚠️ Found {len(terrain_analysis['terrain_issues'])} terrain issues")
            feedback = supervisor.generate_agent_d_feedback(terrain_analysis)
            print(f"📤 Generated Agent D feedback with {len(feedback['specific_actions'])} actions")
        else:
            print("✅ No major terrain issues found")