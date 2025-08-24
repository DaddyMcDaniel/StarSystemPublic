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
# AUTO LOOP: Automated A→D→B→C cycle with fullscreen screenshots
python run_auto_loop.py                    # Single cycle with fullscreen Agent B
python run_auto_loop.py --iterations 5     # Multiple automated cycles
python run_auto_loop.py --bridge-port 8765 # Custom bridge port

# MANUAL LOOP: Human-in-the-loop A→D→Human→C cycle  
python run_manual_loop.py                  # Human plays and provides feedback
python run_manual_loop.py --iterations 3   # Multiple manual cycles
python run_manual_loop.py --skip-generation # Use existing world
python run_manual_loop.py --test-mode      # Non-interactive with simulated feedback
python run_manual_loop.py --no-viewer      # Skip viewer, feedback only

# INDIVIDUAL AGENTS: Test components separately
python agents/agent_a_generator.py         # Generate spherical planet
python agents/agent_b_manual.py <planet>   # Human testing interface
python agents/agent_b_tester.py            # Automated testing

# LEGACY: Original evolution loop (deprecated)
python agents/run_evolution_loop.py
```

### Rendering and Visualization
```bash
# OpenGL viewers (preferred) - T19+ enhanced with LOD and lighting
python renderer/pcc_game_viewer.py <game_file>       # Interactive viewer with bridge
python renderer/pcc_fixed_viewer.py <game_file>      # Fixed camera viewer
python renderer/pcc_simple_viewer.py <game_file>     # Minimal smoke test viewer

# Screenshot capture (T16+)
python scripts/capture_viewer_screenshots.py         # Automated screenshot capture

# Forge engine demo
python forge_engine_demo.py

# Direct 3D viewer test
python test_3d_viewer.py
```

### Mini-Planet Workflow (T18-T22)
```bash
# Primary mini-planet generation command
./commands/LOP_miniplanet                    # Generate with random seed
./commands/LOP_miniplanet --seed 42          # Generate with specific seed
./commands/LOP_miniplanet --seed 42 --view   # Generate and launch viewer
./commands/LOP_miniplanet --quick --view     # Quick mode for testing

# Unified planet generation system
python unified_miniplanet.py --seed 12345    # Advanced planet generation
python create_mini_planet.py                 # Alternative planet creation
python generate_miniplanet.py                # Direct generation script

# Visual demonstration and testing
python terrain_visual_demo.py                # Terrain improvements demo
python demo_terrain_improvements.py          # T20+ terrain features
```

### Testing and Quality
```bash
# Run AI tools tests
python test_ai_tools.py

# Test specific components
python test_vm_execution.py
python test_integration.py
python test_forge_quick.py
python test_chunk_streaming.py              # Test T19+ streaming system

# Test evolved games
python test_best_100gen_game.py
python test_final_evolved_game.py

# Terrain quality validation (T22)
python -m agents.agent_d.validation.terrain_quality_validator
```

## Architecture Overview

### PCC Language Stack
- **AST Nodes**: Direct AST programming eliminates parsing (`src/pcc_ast_standalone.h`)
- **C++ VM**: Standalone interpreter with no dependencies (`src/pcc_vm_simple.cpp`, `src/pcc_interpreter_standalone.h`)
- **Python Runtime**: Fallback PCC runtime in Forge (`forge/modules/pcc_runtime/pcc_runtime_module.py`)
- **Game Format**: `.pcc` files store compressed game ASTs as JSON

### Forge Engine Architecture
- **Modular Design**: Hot-swappable modules (`forge/modules/`)
  - Rendering: OpenGL-based 3D rendering with T19+ LOD system
  - Physics: Game physics simulation
  - PCC Runtime: Executes PCC games
  - Audio/Animation: Planned modules
- **Collective Intelligence**: Claude agent collaboration (`agents/collective_subconscious.py`)
- **Core Engine**: Main loop and module management (`forge/core/engine.py`)

### Agent Evolution System
Four-agent cycle creates increasingly sophisticated games:

1. **Agent A (Generator)**: Creates PCC games using patterns and memory (`agents/agent_a_generator.py`)
2. **Agent D (Renderer)**: Converts PCC to 3D visual scenes (`agents/agent_d_renderer.py`)
   - **T18-T22 Enhanced**: Advanced terrain with lighting, validation, marching cubes
   - **Lighting System**: Shadow mapping, SSAO, tone mapping (`agents/agent_d/lighting/`)
   - **Validation**: Terrain quality assessment (`agents/agent_d/validation/`)
   - **Debug UI**: Marching cubes debug, LOD statistics (`agents/agent_d/debug_ui/`, `agents/agent_d/hud/`)
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
- `generated_planets/`: Mini-planet variations with metadata (`T18+`)
  - `metadata/`: Planet configuration and generation metadata
  - `meshes/`: Generated terrain mesh data
  - `textures/`: Planet surface textures
  - `previews/`: Generated preview images
- `planet_configs/`: Hero planet configurations
- `test_screenshots_t16/`: Automated screenshot captures

### Configuration
- `.env`: Environment variables (API keys, paths)
- `config.json`: System configuration
- `agent_b_plan/`: Agent B specifications and gating
- Package management: `package.json` (Playwright), `requirements*.txt` (Python deps)
- **Direct_Struct_Summary/**: Comprehensive system documentation and architecture summaries
  - `Main_Struct.md`: Complete technical overview
  - `core_terrain_system.json`: Terrain pipeline documentation
  - `rendering_pipeline.json`: OpenGL rendering system
  - `production_workflow.json`: Development workflows

## Git Workflow

### Committing and Pushing Changes
```bash
# Check status and see what files have changed
git status
git diff                    # Show unstaged changes
git diff --staged          # Show staged changes

# Pull latest changes first (important!)
git pull

# Add files to staging area
git add -A                 # Add all files (modified and untracked)
# OR selectively add:
git add <specific_files>   # Add specific files only

# Create comprehensive commit with proper message
git commit -m "$(cat <<'EOF'
feat(T18-T22): description of major changes

Detailed explanation of improvements:
• Key feature 1: description
• Key feature 2: description
• System enhancement: description

The system now supports [major capabilities].

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# Push to repository
git push
```

### Commit Message Patterns
Follow the established pattern seen in recent commits:
- `feat(T##):` for new features with iteration number
- `fix:` for bug fixes
- `refactor:` for code reorganization
- Include bullet points for major changes
- Always include Claude Code attribution

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

### Rendering Pipeline (T18-T22 Enhanced)
- **Advanced Terrain**: Marching cubes, SDF cave systems, seam prevention
- **LOD Management**: Runtime level-of-detail with adaptive resolution (`agents/agent_d/mesh/runtime_lod.py`)
- **Lighting System**: PBR materials, shadow mapping, SSAO, tone mapping
- **Chunk Streaming**: Optimized mesh loading with overlap prevention (`agents/agent_d/mesh/chunk_streamer.py`)
- **Quality Validation**: Automated terrain assessment and metrics
- Bridge protocol for Agent B interaction (`renderer/pcc_game_viewer.py`)
- Prefer OpenGL viewers over pygame (being phased out)
- Use C++ VM when available, Python runtime as fallback

## Key Technologies
- **Languages**: C++17 (VM), Python 3.8+ (agents/engine)
- **Graphics**: OpenGL, GLUT for 3D rendering with advanced shaders
- **Terrain**: Marching cubes, SDF evaluation, quadtree chunking
- **AI**: Integration with Claude collective intelligence
- **Build**: CMake for C++ components
- **Testing**: Built-in C++ tests, Python test scripts, terrain quality validation
- **Lighting**: Shadow mapping, SSAO, tone mapping, PBR materials
- **Optional**: Reinforcement learning (stable-baselines3), Playwright for web automation

## Project Documentation

### Direct_Struct_Summary System
Comprehensive technical documentation located in `Direct_Struct_Summary/`:
- **Main_Struct.md**: Complete system overview for external developers
- **core_terrain_system.json**: Detailed terrain pipeline documentation
- **rendering_pipeline.json**: OpenGL rendering architecture
- **agent_systems.json**: Agent collaboration and memory systems
- **production_workflow.json**: Development and testing workflows
- **quick_start_guide.json**: Getting started instructions

Refer to these files for in-depth technical details and system architecture.

## Notes
- The system prefers the C++ VM (`bin/pcc_vm`) when built, falls back to Python runtime
- OpenGL is required for visual game testing and Agent B interaction
- Agent memory persists between evolution cycles in JSON format  
- Games are stored as compressed AST JSON in `.pcc` files
- The evolution system can run autonomous loops generating increasingly complex games
- **T18-T22 Systems**: Advanced terrain generation with lighting, validation, and quality metrics
- **Mini-Planet Workflow**: Use `./commands/LOP_miniplanet` for primary generation interface
- **Seeded Generation**: Same seed produces identical, deterministic planets