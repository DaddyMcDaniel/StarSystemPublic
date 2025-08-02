#!/usr/bin/env python3
"""
Master Evolution Loop Coordinator
Runs the complete 7-step PCC language evolution system
"""
import subprocess
import sys
from pathlib import Path

AGENTS_DIR = Path(__file__).parent

def run_single_evolution_cycle():
    """Run one complete evolution cycle"""
    print("🚀 Starting PCC Evolution Cycle")
    
    # Step 1-2: Agent A generates prompt and PCC
    print("🎮 Agent A: Generating game...")
    subprocess.run([sys.executable, AGENTS_DIR / "agent_a_generator.py"])
    
    # Step 3-4: Agent B tests and analyzes
    print("🔍 Agent B: Testing game...")
    subprocess.run([sys.executable, AGENTS_DIR / "agent_b_tester.py"])
    
    # Step 5-7: Agent C supervises and coordinates
    print("👁️ Agent C: Supervising...")
    subprocess.run([sys.executable, AGENTS_DIR / "agent_c_supervisor.py"])
    
    print("✅ Evolution cycle complete!")

if __name__ == "__main__":
    print("🎯 PCC Language Evolution System")
    print("🔄 7-Step Evolution Loop Active")
    run_single_evolution_cycle()