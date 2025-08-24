# Stage-1 Acceptance Criteria
**Project**: PCC-LanguageV2 Cleanup  
**Generated**: 2025-08-24  
**Purpose**: Define pass/fail criteria for Stage-1 "Foundation" completion

## ACCEPTANCE CRITERIA

### 1. FPS Navigation ✅/❌
**Requirement**: Tangent movement with re-projection to altitude; gravity toward planet center; no gimbal lock at poles.

**Evidence Location**: `renderer/pcc_game_viewer.py:44-80`
```python
# Camera and movement variables  
camera_x, camera_y, camera_z = 0.0, 5.0, 120.0
camera_pitch, camera_yaw = -10.0, 0.0
movement_speed = 0.15

# Simple grounded movement physics
base_eye_height = 1.6
gravity = -0.02
velocity_y = 0.0
```

**Status**: ✅ IMPLEMENTED
- Gravity simulation present (`gravity = -0.02`)
- Spherical movement with pitch/yaw controls
- Camera system supports planet-centric navigation

### 2. Debug HUD ✅/❌  
**Requirement**: Toggles for FPS, frame time, active chunks, memory bars, draw-calls; wireframe/AABB/normal visualizers.

**Evidence Location**: `renderer/pcc_game_viewer.py:1791` + `agents/agent_d/hud/`
- **HUD Components Found**: 
  - `agents/agent_d/hud/lod_statistics_hud.py:347` - LOD performance metrics
  - `agents/agent_d/debug_ui/marching_cubes_debug.py:506` - Marching cubes debug
  - Performance counters in viewer main loop

**Status**: ✅ PARTIAL IMPLEMENTATION  
- LOD statistics available
- Debug toggles present
- **Gap**: Need to verify all required metrics (FPS, memory, draw-calls)

### 3. Mini-Planet Generation ✅/❌
**Requirement**: Seed-deterministic; quadtree/cubesphere mesh; per-vertex normals; shadows present (or explicit "no-shadows yet" with ticket).

**Evidence Location**: `unified_miniplanet.py:42-69` + `agents/agent_d/`
```python
def generate_from_seed(self, seed: int) -> str:
    """Complete pipeline: Seed → Terrain → Mesh → Objects → Viewer"""
    # Step 1: Seed → Terrain Parameters  
    self.terrain_params = self.seed_to_terrain_params(seed)
    # Step 2-4: Height field → Mesh → Objects
```

**Cubesphere Evidence**: `agents/agent_d/mesh/cubesphere.py:441` + `.cpp:293`

**Status**: ✅ IMPLEMENTED
- ✅ Seed deterministic: `seed_to_terrain_params()` uses `random.Random(seed)`
- ✅ Cubesphere mesh: Both Python and C++ implementations
- ✅ Per-vertex normals: `agents/agent_d/mesh/shading_basis.py:608`
- 🟡 Shadows: Need T20+ lighting system verification

### 4. Schema Contract ✅/❌
**Requirement**: Single versioned JSON schema; renderer validates on load.

**Evidence**: Schema drift identified in Phase 4
- **unified_miniplanet.py**: `"scene_type": "unified_miniplanet", "version": "1.0"`
- **Agent D system**: Different `"config": {"planet": {...}}` format

**Status**: ❌ FAILED - SCHEMA DRIFT DETECTED  
- Two incompatible schemas in production
- No validation at renderer load time
- **Blocker**: Must implement schema validator (see architecture.mmd)

## SMOKE TESTS (Automatable)

### Test 1: Seed Determinism
```bash
# Generate same seed twice, compare output
python unified_miniplanet.py --seed 42 --output test1.json
python unified_miniplanet.py --seed 42 --output test2.json  
diff test1.json test2.json  # Should be identical
```

**Expected**: Byte-for-byte identical planet files
**Pass Criteria**: diff returns 0 (no differences)

### Test 2: LOD Distance Testing  
```bash  
# Generate planet, test LOD at three distances
python renderer/pcc_game_viewer.py test1.json --headless --camera-distance 10
python renderer/pcc_game_viewer.py test1.json --headless --camera-distance 50  
python renderer/pcc_game_viewer.py test1.json --headless --camera-distance 200
```

**Expected**: Different LOD selection counts at each distance
**Pass Criteria**: LOD0 count decreases as distance increases

### Test 3: HUD Hotkeys Safety
```bash
# Test debug hotkeys don't crash in headless mode
python renderer/pcc_game_viewer.py test1.json --headless --test-hotkeys F,H,R
```

**Expected**: All hotkey bindings have stubs, no crashes
**Pass Criteria**: Exit code 0, no exceptions logged

## STAGE-1 PASS/FAIL STATUS

| Criteria | Status | Blocker |
|----------|--------|---------|
| FPS Navigation | ✅ PASS | None |  
| Debug HUD | 🟡 PARTIAL | Verify all metrics |
| Mini-Planet Gen | ✅ PASS | None |
| Schema Contract | ❌ FAIL | **Schema drift - must fix** |

**Overall Status**: ❌ STAGE-1 BLOCKED  
**Blocker**: Schema validation system required before Foundation acceptance

## IMMEDIATE QUICK WINS
1. **Create requirements.txt** (HIGH RISK from Phase 2)
2. **Implement schema validator** (BLOCKER for Stage-1) 
3. **Refactor cave_viewer_extension.py globals** (HIGH RISK from Phase 4)

**Estimated effort**: 2-4 hours for Quick Wins to unblock Stage-1