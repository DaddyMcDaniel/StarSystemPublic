# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the PCC-LanguageV2 project - a revolutionary AI-native programming language designed for ultra-compressed code generation by AI agents. PCC (Procedural-Compressed-Code) operates directly on AST nodes, eliminating parsing overhead and enabling alien-like compression ratios optimized for LLM code generation and evolution.

The project consists of three main systems:
1. **PCC Language & VM**: Ultra-compressed AST-based language with standalone C++ interpreter
2. **Forge Engine**: Modular game engine with collective intelligence integration  
3. **Agent Evolution System**: AI agents that generate, test, and evolve PCC games

## Common Commands

### Building the PCC VM
```bash
# Build C++ PCC Virtual Machine
mkdir -p build
cd build
cmake ..
make

# Run PCC VM
./bin/pcc_vm <filename.pcc>
./bin/pcc_vm --test        # Run built-in tests
```

### Python Environment Setup
```bash
# Setup main Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup AI tools environment (optional)
python3 -m venv ai_tools_env
source ai_tools_env/bin/activate
pip install -r ai_tools_requirements.txt

# Setup RL environment for Agent B (optional)
python3 -m venv .venv_rl
source .venv_rl/bin/activate
pip install -r requirements-rl.txt
```

### Running the Evolution System
```bash
# Run main evolution loop (Agent A→D→B→C cycle)
python agents/run_evolution_loop.py

# Run Agent B MVP with bridge
python run.py --agent-b-mvp --bridge-port 8765

# Test individual agents
python agents/agent_a_generator.py
python agents/agent_b_game_player.py
python agents/agent_c_supervisor.py
python agents/agent_d_renderer.py
```

### Rendering and Visualization
```bash
# OpenGL viewers (preferred)
python renderer/pcc_game_viewer.py <game_file>       # Interactive viewer with bridge
python renderer/pcc_fixed_viewer.py <game_file>      # Fixed camera viewer
python renderer/pcc_simple_viewer.py <game_file>     # Minimal smoke test viewer

# Forge engine demo
python forge_engine_demo.py

# Direct 3D viewer test
python test_3d_viewer.py
```

### Testing and Quality
```bash
# Run AI tools tests
python test_ai_tools.py

# Test specific components
python test_vm_execution.py
python test_integration.py
python test_forge_quick.py

# Test evolved games
python test_best_100gen_game.py
python test_final_evolved_game.py
```

## Architecture Overview

### PCC Language Stack
- **AST Nodes**: Direct AST programming eliminates parsing (`src/pcc_ast_standalone.h`)
- **C++ VM**: Standalone interpreter with no dependencies (`src/pcc_vm_simple.cpp`, `src/pcc_interpreter_standalone.h`)
- **Python Runtime**: Fallback PCC runtime in Forge (`forge/modules/pcc_runtime/pcc_runtime_module.py`)
- **Game Format**: `.pcc` files store compressed game ASTs as JSON

### Forge Engine Architecture
- **Modular Design**: Hot-swappable modules (`forge/modules/`)
  - Rendering: OpenGL-based 3D rendering
  - Physics: Game physics simulation
  - PCC Runtime: Executes PCC games
  - Audio/Animation: Planned modules
- **Collective Intelligence**: Claude agent collaboration (`agents/collective_subconscious.py`)
- **Core Engine**: Main loop and module management (`forge/core/engine.py`)

### Agent Evolution System
Four-agent cycle creates increasingly sophisticated games:

1. **Agent A (Generator)**: Creates PCC games using patterns and memory (`agents/agent_a_generator.py`)
2. **Agent D (Renderer)**: Converts PCC to 3D visual scenes (`agents/agent_d_renderer.py`)  
3. **Agent B (Tester)**: Plays games and provides quality assessment (`agents/agent_b_game_player.py`)
4. **Agent C (Supervisor)**: Provides feedback and evolves patterns (`agents/agent_c_supervisor.py`)

**Game Research System (GRS)**: Research-based assessment using game theory, AAA benchmarks, and human feedback patterns (`agents/game_research_system.py`, `grs_integration.py`)

### Key Data Flow
```
PCC Code → C++ VM / Python Runtime → 3D Scene → OpenGL Renderer → Agent B Analysis → Pattern Evolution
```

## Important File Locations

### Core Implementation
- `src/`: C++ PCC VM implementation
- `forge/`: Modular game engine
- `agents/`: AI agent evolution system
- `renderer/`: OpenGL visualization layer

### Data and Communication
- `agents/communication/`: Agent-to-agent data exchange
- `agents/memory/`: Agent memory and preferences 
- `agents/logs/`: Evolution cycle logs
- `tests/examples/`: Generated PCC games
- `rendered_games/`: Visual game outputs
- `reports/`: Evolution progress reports

### Configuration
- `.env`: Environment variables (API keys, paths)
- `config.json`: System configuration
- `agent_b_plan/`: Agent B specifications and gating
- Package management: `package.json` (Playwright), `requirements*.txt` (Python deps)

## Development Workflow

### Making Changes to PCC VM
1. Edit C++ files in `src/`
2. Rebuild: `cd build && make`
3. Test with: `./bin/pcc_vm --test`

### Agent Development
1. Agents auto-load memory from `agents/memory/`
2. Test individual agents before integration
3. Use GRS integration for research-based assessment
4. Check agent communication via `agents/communication/`

### Adding Forge Modules
1. Create module in `forge/modules/<module_name>/`
2. Implement `ForgeModule` interface
3. Register in engine configuration
4. Hot-swappable at runtime

### Rendering Pipeline
- Prefer OpenGL viewers over pygame (being phased out)
- Use C++ VM when available, Python runtime as fallback
- Bridge protocol for Agent B interaction (`renderer/pcc_game_viewer.py`)

## Key Technologies
- **Languages**: C++17 (VM), Python 3.8+ (agents/engine)
- **Graphics**: OpenGL, GLUT for 3D rendering
- **AI**: Integration with Claude collective intelligence
- **Build**: CMake for C++ components
- **Testing**: Built-in C++ tests, Python test scripts
- **Optional**: Reinforcement learning (stable-baselines3), Playwright for web automation

## Notes
- The system prefers the C++ VM (`bin/pcc_vm`) when built, falls back to Python runtime
- OpenGL is required for visual game testing and Agent B interaction
- Agent memory persists between evolution cycles in JSON format  
- Games are stored as compressed AST JSON in `.pcc` files
- The evolution system can run autonomous loops generating increasingly complex games