# Project Structure Cleanup - Evidence Log
**Started**: 2025-08-24  
**Budget**: 90 minutes total for Stage-1 sweep  
**Status**: Phase 1 COMPLETED, Phase 2+ IN PROGRESS

## COMPLETED PHASES - DO NOT LOSE THIS INFORMATION

### Phase 1: Inventory (COMPLETED - 15 min)

#### File Count Evidence
- **Total files**: 14,621
- **Python files**: 4,398 (.py)
- **C/C++ files**: 92 (.cpp, .c, .h, .hpp)
- **Compiled Python**: 4,312 (.pyc)
- **Binary files**: 1,847 (.bin)
- **JSON files**: 712 (.json)

**Evidence location**: `/tmp/files.txt` (file list)

#### Extension Histogram (Top 20)
```
4398 py      <- Dominant language: Python
4312 pyc     <- Many compiled Python files 
1847 bin     <- Significant binary presence
712 json     <- Heavy JSON usage (planet data)
392 js       <- Some JavaScript 
250 pyi      <- Type stubs present
107 png      <- Image assets
60 f90       <- Fortran files (unexpected)
56 txt       <- Documentation
53 pcc       <- PCC game files
52 c         <- C source
51 md        <- Markdown docs
```

#### Python Entrypoints (VERIFIED - 104 total)
**Evidence location**: `/tmp/py_entry.txt`

**PRIMARY ENTRYPOINTS** (Critical paths):
- `unified_miniplanet.py:670` - **Main planet generator**
- `renderer/pcc_game_viewer.py:1791` - **Primary viewer**
- `agents/agent_d_renderer.py:748` - **Agent D terrain system**
- `create_mini_planet.py:676` - **Alternative planet creator**
- `run_auto_loop.py:30` - **Evolution automation**
- `run_manual_loop.py:30` - **Manual evolution**

**AGENT SYSTEM ENTRYPOINTS**:
- `agents/agent_a_generator.py:393` - Game generator
- `agents/agent_b_tester.py:622` - Game tester  
- `agents/agent_c_supervisor.py:184` - Evolution supervisor

**TEST/VALIDATION ENTRYPOINTS** (22 identified):
- `agents/agent_d/test_*` - Agent D test suite
- `agents/agent_d/validate_*` - Validation scripts
- `test_*.py` - Integration tests

#### C/C++ Entrypoints (VERIFIED - 2 total)
**Evidence location**: `/tmp/c_entry.txt`
- `src/pcc_vm_simple.cpp:372` - **PCC Virtual Machine**
- `agents/agent_d/mesh/cubesphere.cpp:293` - **Cubesphere generator**

#### CI/Linter/Test Discovery (VERIFIED)
**GitHub Actions found**:
- `.github/workflows/claude-code-review.yml` - Code review automation
- `.github/workflows/claude.yml` - Claude integration

**Test patterns identified**:
- 22 test files in `agents/agent_d/test_*`
- 8 validation scripts in `agents/agent_d/validate_*`
- Test modules follow `test_*.py` convention

**Linter configs**: **NONE FOUND** (RISK: No code quality enforcement)
**Requirements files**: **NONE FOUND IN ROOT** (HIGH RISK: Dependency management unclear)
**Package manifests**: Only `./automation/playwright/package.json` (Playwright testing only)

**PHASE 1 COMPLETE** - Evidence preserved in `inventory.json`, `entrypoints.json`, `findings.json`

## CURRENT PHASE STATUS

### Phase 2: SBOM Ground Truth (COMPLETED)
- [x] Parse requirements*.txt, pyproject.toml, package.json
- [x] Cross-check imports in core modules  
- [x] Generate sbom.json + drift analysis

**CRITICAL FINDINGS**:
- **NO Python dependency manifests** found in root directory
- **HIGH RISK**: Core dependencies (numpy, PyOpenGL, PIL) have no version control
- **Import drift detected**: 3 critical libraries used without pinned versions
- Only manifest: `automation/playwright/package.json` (Playwright ^1.54.2)

**Evidence preserved**: `sbom.json` with complete dependency analysis

### Phase 3: Anchor Read (COMPLETED)
- [x] CLAUDE.md architecture extraction
- [x] unified_miniplanet.py analysis  
- [x] renderer/viewer schema analysis
- [x] agents/agent_d pipeline analysis
- [x] Mermaid diagram generation

**ARCHITECTURE FINDINGS** (`file:path:line` evidence):
- **CLAUDE.md:87-96**: Primary mini-planet workflow is `./commands/LOP_miniplanet` → T18-T22 system
- **unified_miniplanet.py:42-69**: Complete pipeline `Seed → Terrain → Mesh → Objects → Viewer`
- **renderer/pcc_game_viewer.py:21-38**: OpenGL viewer with numpy, PyOpenGL dependencies (UNMANAGED)
- **agents/agent_d_renderer.py:1-39**: T17 Hero Planet system with Agent D mesh pipeline

**DUAL TERRAIN SYSTEMS CONFIRMED**:
1. **unified_miniplanet.py** (Newer): Direct seed→JSON pipeline, outputs `unified_planet_seed_N.json`
2. **agents/agent_d_renderer.py** (Agent D): T17 Hero Planet, outputs scene.json in different schema

**SCHEMA DRIFT RISK**: Two different JSON schemas for planet data - renderer compatibility unclear

**Evidence preserved**: `docs/architecture_improved.mmd` with complete dataflow diagram (subgraphs, schema validator, deprecated paths marked)

### Phase 4: Risk Passes (COMPLETED)
- [x] Security sweep (secrets, unsafe calls)
- [x] Reliability sweep (error handling)  
- [x] Schema drift analysis

**CRITICAL RISKS IDENTIFIED**:
- **🔴 HIGH**: Global state usage in `agents/agent_d/marching_cubes/cave_viewer_extension.py` (8+ global vars)
- **🔴 HIGH**: Missing dependency management (numpy, PyOpenGL, PIL unmanaged)
- **🔴 HIGH**: Schema drift between unified_miniplanet.py and Agent D systems  
- **🟢 LOW**: No hardcoded secrets found ✅
- **🟢 LOW**: Subprocess usage safe (no shell=True) ✅

**Evidence preserved**: `docs/risk_register.md` with complete risk analysis

### Phase 5: Acceptance Criteria (COMPLETED)
- [x] FPS navigation checks
- [x] Debug HUD validation
- [x] Mini-planet generation standards
- [x] Schema contract definition
- [x] Smoke test specifications

**STAGE-1 ASSESSMENT**:
- **✅ FPS Navigation**: Implemented with gravity simulation and spherical controls
- **🟡 Debug HUD**: Partial implementation, LOD statistics available, need to verify all metrics
- **✅ Mini-Planet Generation**: Seed-deterministic, cubesphere mesh, per-vertex normals present
- **❌ Schema Contract**: FAILED - Schema drift between unified_miniplanet.py and Agent D systems

**STAGE-1 STATUS**: ❌ BLOCKED by schema validation requirement

**Evidence preserved**: `docs/stage1_acceptance.md` with complete criteria and smoke tests

## KEY FINDINGS SO FAR

### ARCHITECTURE INSIGHTS
1. **Python-dominant** codebase (4,398 .py files)
2. **Dual terrain systems** detected:
   - `unified_miniplanet.py` (newer?)
   - `agents/agent_d_renderer.py` (Agent D system)
3. **Complex test suite** (30+ test/validation files)
4. **Binary-heavy** (1,847 .bin files - investigate)

### IMMEDIATE RISKS IDENTIFIED
1. **No linter configuration** detected
2. **Schema drift potential** between planet generators
3. **Complex entrypoint matrix** (104 Python, 2 C++)

### ARTIFACTS ORGANIZED
**Documentation moved to `docs/` folder**:
- `docs/inventory.json` - File counts and extensions  
- `docs/entrypoints.json` - Python/C++ entry points
- `docs/findings.json` - CI/linter/test discovery
- `docs/sbom.json` - Dependency analysis with drift
- `docs/architecture_improved.mmd` - Enhanced Mermaid diagram

## STAGE-1 CLEANUP COMPLETE ✅

**All 5 Phases completed in 90-minute budget**  
**Status**: Foundation assessment complete, with actionable findings

### NEXT ACTIONS - IMMEDIATE QUICK WINS
1. **Create requirements.txt** (HIGH RISK - missing dependency management)
2. **Implement schema validator** (STAGE-1 BLOCKER - dual planet schemas) 
3. **Refactor cave_viewer_extension.py globals** (HIGH RISK - 8+ global variables)

**Estimated effort**: 2-4 hours to unblock Stage-1 Foundation

### ARTIFACTS SUMMARY  
**All evidence preserved in `docs/` folder**:
- `inventory.json`, `entrypoints.json`, `findings.json`, `sbom.json` - Systematic analysis
- `architecture_improved.mmd` - Complete system diagram with subgraphs
- `risk_register.md` - 6 risks identified (3 HIGH, 1 MEDIUM, 2 LOW)
- `stage1_acceptance.md` - Pass/fail criteria with 3 automated smoke tests

**This log MUST be preserved** - complete evidence chain from Phases 1-5 documented to prevent T01-T16 information loss.