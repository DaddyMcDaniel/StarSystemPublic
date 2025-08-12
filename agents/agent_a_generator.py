#!/usr/bin/env python3
"""
SUMMARY: Agent A - Primary Code Generator v1
============================================
Fast, prolific code generator specializing in PCC language features and runtime improvements.
Primary agent in the A→D→B→C evolution cycle for StarSystem development.

ROLE & RESPONSIBILITIES:
- Rapid prototyping of PCC language features and Godot integration enhancements
- Generating deterministic code changes with proper seed management
- Creating testable implementations following established patterns
- Proposing surgical edits rather than broad architectural rewrites
- Building grid-based placement logic inspired by Terraria/Eco hammer tools

KEY CAPABILITIES:
- Blueprint Chips: Generate deterministic node graphs from natural language prompts
- Replay Systems: Create reproducible test scenarios with exact seed reproduction
- Building Systems: Design grid-based placement and validation logic
- Planet Generation: Generate layered, navigatable, alien-like mini-planets
- Performance-aware: Ensures generated code meets efficiency requirements

INTEGRATION:
- Loads HOD from agents/memory/agent_a_prompt.txt
- Outputs validated by Agent B (tester/validator)
- Supervised by Agent C (governance/coordination)
- Focus on unlimited creative building time (Halo Forge mode philosophy)

USAGE:
  python agents/agent_a_generator.py
  make agents  # Run full A→B→C handshake

RELATED FILES:
- agents/memory/agent_a_prompt.txt - High-Order Directive with contracts
- Week 2+ focus on idea → playable mini-planet workflow
"""
import random
import json
from pathlib import Path
from datetime import datetime

# PCC Language Project Paths
PCC_ROOT = Path(__file__).parent.parent
PCC_SPEC = PCC_ROOT / "docs/pcc_language_spec.md"
PCC_GAMES = PCC_ROOT / "tests/examples"
PCC_LOGS = PCC_ROOT / "agents/logs"

PCC_GAMES.mkdir(parents=True, exist_ok=True)
PCC_LOGS.mkdir(parents=True, exist_ok=True)

class PCCGameGenerator:
    def __init__(self):
        self.compression_level = 1.0  # Evolution metric
        self.successful_patterns = []  # Learned from feedback
        
    def generate_random_prompt(self):
        """Step 1: Generate random small game prompts in English"""
        game_types = ["maze runner", "platformer", "puzzle", "shooter", "adventure"]
        elements = ["coins", "enemies", "platforms", "doors", "keys", "portals"]
        objectives = ["collect all", "reach the end", "defeat boss", "solve puzzle", "escape"]
        
        game_type = random.choice(game_types)
        element = random.choice(elements)
        objective = random.choice(objectives)
        
        prompt = f"Create a {game_type} game where player must {objective} {element}"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": 1,
            "prompt": prompt,
            "agent": "A"
        }
        self._log(log_entry)
        
        return prompt
    
    def prompt_to_pcc_ast(self, prompt):
        """Step 2: Convert English prompt to pure PCC AST structure"""
        # This is where the magic happens - convert English to compressed AST
        
        # Basic PCC AST structure (will evolve based on feedback)
        pcc_ast = {
            "type": "GAME",
            "nodes": [
                {
                    "type": "WORLD",
                    "size": [10, 10, 1],
                    "entities": self._extract_entities(prompt)
                },
                {
                    "type": "PLAYER", 
                    "pos": [0, 0, 0],
                    "controls": ["WASD"]
                },
                {
                    "type": "OBJECTIVE",
                    "goal": self._extract_goal(prompt)
                }
            ],
            "compression_id": self._generate_compression_id()
        }
        
        # Save as .pcc file
        pcc_filename = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcc"
        pcc_path = PCC_GAMES / pcc_filename
        
        with open(pcc_path, 'w') as f:
            json.dump(pcc_ast, f, indent=2)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": 2,
            "prompt": prompt,
            "pcc_file": pcc_filename,
            "ast_nodes": len(pcc_ast["nodes"]),
            "compression_level": self.compression_level,
            "agent": "A"
        }
        self._log(log_entry)
        
        return pcc_path, pcc_ast
    
    def _extract_entities(self, prompt):
        """Extract game entities from English prompt"""
        entities = []
        
        if "maze" in prompt:
            entities.append({"type": "WALL", "pattern": "maze_gen"})
        if "coin" in prompt:
            entities.append({"type": "COLLECTIBLE", "id": "coin", "count": 10})
        if "enemy" in prompt:
            entities.append({"type": "ENEMY", "ai": "chase", "count": 3})
        if "platform" in prompt:
            entities.append({"type": "PLATFORM", "pattern": "jump_sequence"})
            
        return entities
    
    def _extract_goal(self, prompt):
        """Extract objective from English prompt"""
        if "collect all" in prompt:
            return {"type": "COLLECT_ALL", "target": "coin"}
        elif "reach the end" in prompt:
            return {"type": "REACH_POSITION", "pos": [9, 9, 0]}
        elif "defeat" in prompt:
            return {"type": "DEFEAT_ALL", "target": "enemy"}
        else:
            return {"type": "SURVIVE", "time": 60}
    
    def _generate_compression_id(self):
        """Generate unique compression identifier for evolution tracking"""
        return f"C{random.randint(1000, 9999)}"
    
    def _log(self, entry):
        """Log all agent activities"""
        log_file = PCC_LOGS / "agent_a.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    generator = PCCGameGenerator()
    prompt = generator.generate_random_prompt()
    print(f"🎮 Generated: {prompt}")
    pcc_file, ast = generator.prompt_to_pcc_ast(prompt)
    print(f"🔧 Created: {pcc_file}")