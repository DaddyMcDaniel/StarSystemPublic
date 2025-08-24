# PCC-LanguageV2 System Architecture & Generation Pipeline
## Complete Technical Overview for External Programmers

---

## 🎯 **PROJECT OVERVIEW**

**PCC-LanguageV2** is an AI-native programming language and game generation system designed for ultra-compressed code generation by AI agents. The system consists of three main components:

1. **PCC Language & VM**: Ultra-compressed AST-based language with standalone C++ interpreter
2. **Forge Engine**: Modular game engine with collective intelligence integration  
3. **Agent Evolution System**: AI agents that generate, test, and evolve PCC games

### **Primary Goal**: Generate procedural mini-planets that are navigatable in 3D space with proper spherical terrain, seeded generation, and real-time rendering.

---

## 🤖 **FOUR-AGENT EVOLUTION SYSTEM**

The core of the project is a **four-agent collaborative loop** where AI agents work together to create increasingly sophisticated games:

### **Agent A (Generator)** - `agents/agent_a_generator.py`
- **Role**: Creates initial PCC game code and structures
- **Input**: Patterns from memory, evolution feedback from Agent C
- **Output**: `.pcc` files containing compressed game ASTs
- **Key Features**: 
  - Uses patterns from `agents/memory/agent_a_memory.json`
  - Generates game logic, planet specifications, and initial parameters
  - Responds to evolutionary pressure from Agent C feedback

### **Agent D (Renderer)** - `agents/agent_d_renderer.py` 
- **Role**: Converts PCC games into 3D visual scenes and terrain
- **Input**: PCC game files from Agent A, seed values
- **Output**: JSON scene files with 3D terrain, objects, and navigation data
- **Key Features**:
  - **T17 Hero Planet System**: Advanced terrain generation with ridged mountains, warped dunes, equatorial archipelagos
  - **Deterministic Seeded Generation**: Same seed always produces identical planets
  - **Terrain Zones**: Generates 40+ terrain zones with varied biomes (lush, volcanic, etc.)
  - **Building Generation**: Creates cottages and structures on planet surface
  - **SDF Cave Systems**: Gyroidal tunnel networks and distributed sphere cavities
  - **Performance Optimization**: LOD (Level of Detail) tuning for 60+ fps

**Technical Implementation:**
```bash
# Agent D Usage
python agents/agent_d_renderer.py --seed 1234 --output planet.json
```

**Output Format Example:**
```json
{
  "metadata": {
    "scene_type": "miniplanet",
    "seed": 1234,
    "agent_d_version": "1.0",
    "terrain_zones": 40
  },
  "terrain": {
    "type": "realistic_miniplanet",
    "radius": 59.07,
    "center": [0, 0, 0],
    "material": "volcanic_rock",
    "biome": "lush",
    "height_map": [...]
  }
}
```

### **Agent B (Tester)** - `agents/agent_b_tester.py` & `agents/agent_b_manual.py`
- **Role**: Plays generated games and provides quality assessment
- **Input**: Rendered scenes from Agent D
- **Output**: Quality metrics, feedback reports, navigation assessments
- **Key Features**:
  - **Automated Testing**: Runs games and measures performance, completeness
  - **Manual Testing**: Human-in-the-loop interface for detailed feedback
  - **Navigation Analysis**: Tests if planets are properly walkable/navigatable
  - **Visual Quality Assessment**: Evaluates terrain realism and rendering quality

### **Agent C (Supervisor)** - `agents/agent_c_supervisor.py`
- **Role**: Provides evolutionary feedback and improves generation patterns
- **Input**: Test results from Agent B, historical performance data
- **Output**: Feedback files for Agent A and D improvement
- **Key Features**:
  - **Pattern Evolution**: Identifies successful game generation patterns
  - **Quality Optimization**: Guides agents toward better terrain and gameplay
  - **Memory Management**: Updates agent memory with learned improvements

---

## 🌍 **TERRAIN GENERATION PIPELINE**

### **Current Working System (Agent D Pipeline)**
The project has **TWO different terrain systems** - one that works and one that's broken:

#### **✅ Working System: Agent D's T17 Hero Planet**
```
Seed → Agent D → Terrain Zones → Cubesphere Mesh → Height Displacement → Buildings → JSON Scene
```

**Key Components:**
- **`agents/agent_d/terrain/`**: Height field generation, noise systems
- **`agents/agent_d/mesh/cubesphere.py`**: Spherical planet mesh generation
- **`agents/agent_d/mesh/quadtree_chunking.py`**: LOD chunking system
- **`agents/agent_d/determinism/`**: Deterministic seed-based generation

**Generation Process:**
1. **Seed Input**: Integer value (e.g., 1234) determines all terrain characteristics
2. **Terrain Zone Generation**: Creates 40+ zones with different biomes and height maps
3. **Cubesphere Creation**: Generates spherical mesh using cube-to-sphere mapping
4. **Height Displacement**: Applies noise-based displacement for realistic terrain
5. **Object Placement**: Adds buildings, vegetation, and interactive elements
6. **Scene Export**: Saves as JSON with complete 3D scene data

#### **❌ Broken System: create_mini_planet.py**
- **Issue**: Falls back to flat procedural noise instead of Agent D's system
- **Problem**: Missing dependencies (`shading_basis`, terrain modules)
- **Result**: Generates flat planes instead of spherical planets
- **Error Messages**: 
  ```
  WARNING - CubeSphere module not available: No module named 'shading_basis'
  WARNING - Terrain modules not available, using procedural noise
  ```

### **Seed Generation System**

#### **Current Implementation:**
- **Random Seeds**: Auto-generated using `random.randint(1, 999999999)`
- **Deterministic**: Same seed always produces identical planets
- **Manual Override**: Can specify seed with `--seed 1234` parameter

#### **Seed-Based Terrain Parameters:**
```python
def generate_terrain_from_seed(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    return {
        "base_layer": {
            "amplitude": rng.uniform(3.0, 8.0),
            "frequency": rng.uniform(0.2, 0.5),
            "octaves": rng.randint(3, 6)
        },
        "mountain_layer": {
            "amplitude": rng.uniform(10.0, 25.0),
            "frequency": rng.uniform(0.05, 0.15)
        },
        "enable_caves": rng.random() < 0.3  # 30% chance
    }
```

---

## 🎮 **RENDERING & NAVIGATION SYSTEM**

### **OpenGL Viewers**
The project includes multiple OpenGL-based viewers for 3D navigation:

#### **Primary Viewer: `renderer/pcc_game_viewer.py`**
- **Purpose**: Interactive 3D navigation with full controls
- **Features**:
  - **Mouse Capture**: Click window to capture mouse for camera control
  - **WASD Movement**: Standard FPS-style navigation
  - **Smooth Physics**: Proper movement speed and jump mechanics
  - **Debug Tools**: Wireframe, normal visualization, performance HUD
  - **Chunk Streaming**: Runtime LOD system for large planets

**Controls:**
```
CLICK WINDOW - Capture mouse for camera control
TAB - Release/capture mouse
WASD - Move around
SPACE - Move up / Jump
C - Move down
+/- - Change movement speed
R - Reset camera position
F - Toggle wireframe mode
H - Toggle performance HUD
ESC - Quit
```

#### **Alternative Viewers:**
- **`pcc_fixed_viewer.py`**: Fixed camera for screenshots
- **`pcc_simple_viewer.py`**: Minimal smoke test viewer
- **`pcc_spherical_viewer.py`**: Specialized for spherical planets

### **Navigation Implementation**
- **Spherical Walking**: Players can walk completely around mini-planets
- **Gravity Simulation**: Proper gravity pointing toward planet center
- **Collision Detection**: Prevents walking through terrain
- **Camera System**: Smooth camera following with pitch/yaw controls

---

## 📊 **USER INTERFACE & HUD**

### **Debug UI Components**
- **`agents/agent_d/hud/lod_statistics_hud.py`**: Shows LOD performance metrics
- **`agents/agent_d/debug_ui/debug_toggles.py`**: Runtime debug option toggles
- **Performance HUD**: FPS counter, chunk loading stats, memory usage

### **Current UI Features**
- **Info Overlay**: Shows controls and objectives
- **Coordinate Axes**: RGB lines for spatial orientation
- **Target Objectives**: Blue cubes as navigation goals
- **Real-time Stats**: Chunk loading, terrain zones, object counts

---

## 🚀 **COMMAND-LINE INTERFACE**

### **Primary Entry Point: `commands/LOP_miniplanet`**
```bash
# Basic generation (random seed)
./commands/LOP_miniplanet

# Specific seed generation
./commands/LOP_miniplanet --seed 1234

# Generate and launch viewer
./commands/LOP_miniplanet --seed 1234 --view

# Quick mode (reduced quality for testing)
./commands/LOP_miniplanet --quick --view
```

### **Current Implementation Flow:**
```bash
LOP_miniplanet --seed 42 --view
    ↓
Agent D: python agents/agent_d_renderer.py --seed 42 --output planet.json
    ↓
Viewer: python renderer/pcc_game_viewer.py planet.json
```

### **Evolution Loop Scripts**
- **`run_auto_loop.py`**: Automated A→D→B→C evolution cycle
- **`run_manual_loop.py`**: Human-in-the-loop testing with Agent B manual interface

---

## 🏗️ **BUILD SYSTEM & DEPENDENCIES**

### **C++ PCC Virtual Machine**
```bash
# Build C++ VM
mkdir -p build
cd build
cmake ..
make

# Run PCC games
./bin/pcc_vm game.pcc
./bin/pcc_vm --test
```

### **Python Environment Setup**
```bash
# Main environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: RL environment for Agent B
python3 -m venv .venv_rl
source .venv_rl/bin/activate
pip install -r requirements-rl.txt
```

### **Required Dependencies**
- **OpenGL**: For 3D rendering (`python-opengl`)
- **NumPy**: Mathematical operations
- **JSON**: Data serialization
- **CMake**: C++ build system

---

## 📁 **DATA FORMATS & STORAGE**

### **Planet Data Files**
- **Location**: `runs/miniplanet_seed_XXXX.json`
- **Format**: JSON with terrain, objects, and metadata
- **Example**: `runs/miniplanet_seed_9999.json` (working reference)

### **PCC Game Files**
- **Location**: `tests/examples/game_YYYYMMDD_HHMMSS.pcc`
- **Format**: JSON containing compressed AST structures
- **Purpose**: Input for Agent D rendering

### **Agent Memory**
- **`agents/memory/agent_d_memory.json`**: Agent D's learned patterns
- **`agents/memory/agent_a_memory.json`**: Agent A's generation patterns
- **`agents/communication/`**: Inter-agent data exchange

---

## 🎯 **LONG-TERM GOALS & ROADMAP**

### **Immediate Goals**
1. **Fix Terrain Rendering**: Resolve flat plane issue, show proper spherical planets
2. **Seed Consistency**: Ensure same seeds produce identical, deterministic results
3. **Navigation Polish**: Smooth camera controls and physics
4. **Performance**: Optimize chunk loading for large planets

### **Medium-Term Goals**
1. **Enhanced Terrain**: More biome variety, realistic physics
2. **Interactive Elements**: Clickable objects, resource gathering
3. **Multiplayer Support**: Multiple players on same planet
4. **AI Evolution**: Self-improving terrain generation through Agent C feedback

### **Long-Term Vision**
1. **Complete Game Engine**: Full PCC language support with complex game logic
2. **Procedural Universes**: Multiple planets, space travel, solar systems
3. **AI-Generated Content**: NPCs, quests, and storylines via agent collaboration
4. **Performance Scaling**: Support for massive worlds with streaming technology

---

## 🐛 **CURRENT ISSUES & DEBUGGING**

### **Primary Issues (As of Current State)**
1. **Flat Terrain Bug**: Planets render as flat planes instead of spheres
   - **Root Cause**: `create_mini_planet.py` uses fallback system instead of Agent D
   - **Solution**: Use Agent D directly (`agents/agent_d_renderer.py`)

2. **Floating Objects**: Random cubes appearing in sphere zones
   - **Likely Cause**: Building/ore placement system in Agent D
   - **Investigation Needed**: Check building placement algorithms

3. **Viewer Integration**: Mismatch between generated data and viewer expectations
   - **Issue**: Different data formats between systems
   - **Solution**: Standardize on Agent D's working format

### **Debug Workflow**
1. **Test Agent D Directly**: 
   ```bash
   python agents/agent_d_renderer.py --seed 42 --output test.json
   ```
2. **Compare Working Examples**: Use `runs/miniplanet_seed_9999.json` as reference
3. **Check Dependencies**: Ensure OpenGL and terrain modules are available
4. **Validate Data Format**: Verify JSON structure matches viewer expectations

---

## 📋 **DEVELOPMENT WORKFLOW**

### **For New Developers**
1. **Setup Environment**: Follow build instructions in CLAUDE.md
2. **Test Working System**: Run `python renderer/pcc_game_viewer.py runs/miniplanet_seed_9999.json`
3. **Understand Agent D**: Study `agents/agent_d_renderer.py` (the working system)
4. **Test Generation**: Use `./commands/LOP_miniplanet --seed 123 --view`
5. **Debug Issues**: Compare generated output with working examples

### **Key Files for Understanding**
- **`CLAUDE.md`**: Complete project documentation and usage
- **`agents/agent_d_renderer.py`**: Working terrain generation system
- **`renderer/pcc_game_viewer.py`**: Primary 3D viewer
- **`runs/miniplanet_seed_9999.json`**: Reference working mini-planet
- **`commands/LOP_miniplanet`**: Main user entry point

### **Testing Commands**
```bash
# Test working reference
python renderer/pcc_game_viewer.py runs/miniplanet_seed_9999.json

# Test new generation
./commands/LOP_miniplanet --seed 42 --view

# Test Agent D directly
python agents/agent_d_renderer.py --seed 42 --output test.json
python renderer/pcc_game_viewer.py test.json
```

---

## 🔧 **TECHNICAL ARCHITECTURE**

```
User Command (LOP_miniplanet)
    ↓
Agent A (Generate PCC) → Agent D (Render Terrain) → Agent B (Test) → Agent C (Evolve)
    ↓                        ↓                         ↓               ↓
PCC Game File           3D Scene JSON              Quality Report    Feedback
    ↓                        ↓
                      OpenGL Viewer
                           ↓
                    Interactive Navigation
```

This system represents a unique approach to procedural content generation where AI agents collaborate to create, test, and evolve increasingly sophisticated game worlds with full 3D navigation and realistic terrain generation.