# StarSystem + The Forge (Week 4 - OpenGL Independence)
Local-first, MCP-driven AI game evolution system with PCC language and OpenGL spherical planet viewer.

## Current Goals (Week 4)

🎯 **Perfect worldgen + seamless mesh pipeline + robust operation**
- ✅ Complete mesh pipeline with seamless edge connections
- ✅ Perfect spherical world navigation with mouse capture  
- ✅ OpenGL-independent operation with robust fallbacks
- ✅ Visual consistency across materials and assets
- 🔄 Enhanced world generation with grounded structures
- 🔄 Performance optimization for efficient rendering

## Quick Start

```bash
# Manual human testing loop (launches OpenGL viewer)
python run_manual_loop.py --iterations 1

# Automated testing loop
python run_auto_loop.py --iterations 5

# Test individual components
python agents/agent_a_generator.py         # Generate spherical planet
python3 renderer/pcc_spherical_viewer.py runs/miniplanet_seed_*.json  # View world
```

## Navigation Guide

### 🎮 Core Evolution System

**Manual Testing Loop (Human-in-the-loop)**
```bash
python run_manual_loop.py                    # Single cycle with human feedback
python run_manual_loop.py --iterations 5     # Multiple cycles
python run_manual_loop.py --test-mode        # Non-interactive simulation
```
*Workflow: Agent A generates → Agent D renders → OpenGL viewer opens → Human tests → Agent C analyzes feedback*

**Automated Testing Loop**
```bash
python run_auto_loop.py                      # Single automated cycle
python run_auto_loop.py --iterations 3       # Multiple cycles
python run_auto_loop.py --bridge-port 8765   # Custom bridge port
```

### 🤖 Individual Agents

**Agent A (Generator)** - Creates spherical planets with PCC language
```bash
python agents/agent_a_generator.py
# Outputs: tests/examples/game_TIMESTAMP.pcc
# Memory: agents/memory/agent_a_prompt.txt
```

**Agent B (Tester)** - Automated gameplay testing
```bash
python agents/agent_b_tester.py             # Automated testing
python agents/agent_b_manual.py <planet>    # Human testing interface
```

**Agent C (Supervisor)** - Feedback analysis and evolution guidance
```bash
# Auto-runs in loops, analyzes in agents/communication/agent_c_decision.json
```

**Agent D (Renderer)** - Converts PCC to 3D scenes
```bash
python agents/agent_d_renderer.py
# Outputs: runs/miniplanet_seed_*.json
```

### 🎨 Rendering & Visualization

**OpenGL Viewers** (Primary - requires OpenGL)
```bash
python3 renderer/pcc_spherical_viewer.py <scene_file>    # Spherical planet viewer
python3 renderer/pcc_game_viewer.py <scene_file>         # General game viewer
python3 renderer/pcc_simple_viewer.py <scene_file>       # Minimal smoke test
```

**Controls in OpenGL Viewer:**
- `WASD`: Walk on spherical surface
- `Mouse`: Look around (first-person)
- `Space`: Jump
- `Tab`: Toggle mouse capture
- `R`: Reset to north pole
- `ESC`: Exit

**Fallback Analysis** (When OpenGL unavailable)
```bash
python -c "from forge.modules.fallback.headless_analyzer import generate_world_report; print(generate_world_report('runs/miniplanet_seed_123.json'))"
```

### 🧪 Development & Testing

**Build PCC Virtual Machine**
```bash
mkdir -p build && cd build
cmake .. && make
./bin/pcc_vm <filename.pcc>                 # Run PCC games
./bin/pcc_vm --test                         # Built-in tests
```

**Test Individual Components**
```bash
python test_ai_tools.py                     # AI tools tests
python test_vm_execution.py                 # VM execution tests
python test_integration.py                  # Integration tests
python scripts/smoke_test.py                # Smoke tests
```

### 🌐 Playwright Automation

**Setup & Run Web Automation**
```bash
cd automation/playwright
npm install                                 # Install dependencies
node create_repo.js                        # Run automation script
# Outputs: automation/playwright/out/repo_raw.json
```

### 🐛 Debug & Bug Reports

**Find Existing Solutions**
```bash
find runs/*/bug_fixes -name "*.md"          # Search previous bug fixes
grep -r "error_message" runs/*/bug_fixes/   # Search by error
```

**Create New Bug Report** (when encountering new errors)
```bash
# Document in: runs/TIMESTAMP--bug-description/bug_fixes/ERROR_MESSAGE_fix.md
# Include: root cause, reproduction steps, solution, prevention
```

**View Recent Debug Files**
```bash
ls -la runs/latest/                         # Latest run artifacts
cat runs/latest/cleanup_plan.json          # Cleanup status
ls runs/latest/archive/                     # Archived debug files
```

### 💾 Agent Memory & Communication

**Agent Memory Files**
```bash
agents/memory/agent_a_prompt.txt            # Agent A instructions
agents/memory/agent_b_prompt.txt            # Agent B instructions  
agents/memory/agent_c_prompt.txt            # Agent C instructions
agents/memory/agent_d_prompt.txt            # Agent D instructions
agents/memory/collective_subconscious.json  # Shared memory
```

**Agent Communication**
```bash
agents/communication/agent_c_decision.json  # Latest C decisions
agents/communication/human_feedback.json    # Human feedback storage
agents/logs/agent_a.log                     # Agent A operation logs
```

### 📊 Data & Artifacts

**Generated Worlds**
```bash
tests/examples/game_*.pcc                   # Generated PCC games
runs/miniplanet_seed_*.json                 # Rendered 3D scenes
runs/assets/                                # Asset manifests
```

**Evolution Tracking**
```bash
runs/TIMESTAMP--*/run_manifest.json         # Run metadata
runs/latest -> most_recent_run/              # Latest run symlink
```

### 🔧 Configuration & Environment

**Environment Setup**
```bash
python3 -m venv .venv                       # Create project venv
source .venv/bin/activate                   # Activate venv
pip install -r requirements.txt             # Install dependencies (when available)
```

**Key Configuration Files**
```bash
CLAUDE.md                                   # Project instructions
.env                                        # Environment variables (create from env_template)
config/gates.native.yaml                   # Gate configurations
schemas/                                    # Input/output validation schemas
```

### 🏗️ Architecture Understanding

**PCC Language Stack**
- `src/` - C++ PCC Virtual Machine (standalone interpreter)
- `forge/modules/pcc_runtime/` - Python PCC runtime (fallback)
- `.pcc` files - Compressed game ASTs as JSON

**Forge Engine Modules**
- `forge/core/engine.py` - Main engine loop
- `forge/modules/rendering/` - OpenGL rendering
- `forge/modules/physics/` - Game physics
- `forge/modules/fallback/` - OpenGL-independent operations
- `forge/core/capabilities.py` - System capability detection

**Evolution System Flow**
```
Agent A (Generate) → Agent D (Render) → OpenGL Viewer → Human/Agent B (Test) → Agent C (Evolve)
                                      ↘ 
                                       Fallback Analysis (when OpenGL unavailable)
```

## Installation & Dependencies

**Core Requirements**
- Python 3.10+ 
- OpenGL (PyOpenGL + PyOpenGL_accelerate) for 3D viewing
- CMake for C++ VM compilation

**Optional Enhancements**
- PIL/Pillow for 2D fallback visualization
- Matplotlib for plotting
- Trimesh for mesh generation
- Node.js for Playwright automation

**Install Commands**
```bash
pip install PyOpenGL PyOpenGL_accelerate    # Essential for 3D viewing
pip install Pillow matplotlib trimesh       # Optional enhancements
```

## Week-4 Status

✅ **OpenGL Independence** - System operates with graceful fallbacks  
✅ **Spherical Navigation** - Perfect tiny planet mechanics with WASD movement  
✅ **Bug Fix System** - Documented error solutions in runs/*/bug_fixes/  
🔄 **Mesh Pipeline** - Asset generation and seamless connections (in progress)  
🔄 **Visual Consistency** - Material blending and edge alignment (in progress)

## Quick Debug Commands

```bash
# Check OpenGL status
python3 -c "from forge.core.capabilities import detector; detector.print_status()"

# Test viewer directly  
python3 renderer/pcc_spherical_viewer.py runs/miniplanet_seed_$(ls runs/miniplanet_seed_*.json | head -1 | cut -d_ -f3 | cut -d. -f1).json

# View latest evolution decision
cat agents/communication/agent_c_decision.json | jq .

# Check latest run artifacts
ls -la runs/latest/
```