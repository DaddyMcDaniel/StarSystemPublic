# Manual Loop Debugging Rules

## NEVER Use Test Mode for Agent Pipeline Debugging

**Rule**: When debugging the agent evolution pipeline, especially terrain generation issues, NEVER use `--test-mode` flag.

### Why Test Mode Fails:
- Test mode attempts to run without OpenGL viewer
- All OpenGL viewers require actual display/graphics capability 
- Running without viewer causes "OpenGL not available" errors
- Test mode bypasses the actual human assessment loop that is needed for debugging

### Correct Debugging Approach:
1. Run `python run_manual_loop.py` (without flags)
2. Ensure OpenGL environment is available
3. Use actual human interaction for proper assessment
4. Allow the viewer to launch properly for visual terrain assessment

### OpenGL Requirements:
- All viewers (`pcc_game_viewer.py`, `pcc_spherical_viewer.py`, `pcc_fixed_viewer.py`, `pcc_simple_viewer.py`) require OpenGL
- Test mode cannot substitute for actual visual assessment
- Terrain generation issues can only be properly diagnosed with visual inspection

### Agent Pipeline Debug Process:
1. Fix any code errors first (like Agent D renderer zone_weights mismatch)
2. Run full manual loop with OpenGL viewer
3. Human assesses terrain visually 
4. Collect real feedback for agent evolution
5. No automated testing can replace visual terrain assessment