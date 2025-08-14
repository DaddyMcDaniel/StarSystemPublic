Summary: Week-4 Low-Order Prompt (executor). Implement permanent OpenGL-independent operation with robust fallbacks for manual and auto loops. Complete mesh pipeline integration and visual consistency systems.

```xml
<low-order-prompt version="4.0" role="executor">
  <context>
    <phase week="4" seed="${GLOBAL_SEED}" focus="opengl_independence"/>
    <repo root="${ABS_PATH_TO_REPO}">
      <branch>main</branch>
      <status>clean|dirty</status>
      <name>starsystem</name>
    </repo>
    <contracts_source>agents/prompting/high_order_benchmark_week4.md</contracts_source>
  </context>

  <objective>
    Deliver permanent OpenGL-independent operation: robust fallbacks for manual/auto loops, complete mesh pipeline, seamless visual consistency, and comprehensive testing without 3D rendering dependencies.
  </objective>

  <critical_fixes>
    <opengl_independence>
      <problem>Manual and auto loops fail when OpenGL unavailable</problem>
      <solution>Implement capability detection, fallback modes, alternative testing</solution>
      <priority>critical</priority>
    </opengl_independence>
    <mesh_pipeline>
      <problem>Asset manifest created but mesh generation/loading incomplete</problem>
      <solution>Complete end-to-end pipeline with trimesh integration</solution>
      <priority>high</priority>
    </mesh_pipeline>
    <visual_consistency>
      <problem>Mesh edges don't connect seamlessly, material inconsistencies</problem>
      <solution>Edge alignment algorithms, material transition systems</solution>
      <priority>high</priority>
    </visual_consistency>
  </critical_fixes>

  <plan>
    <steps>
      <step id="S1">Capability Detection System: detect OpenGL, PIL, matplotlib availability; auto-select operation mode</step>
      <step id="S2">Fallback Rendering: headless world analysis, 2D map generation, text-based reporting</step>
      <step id="S3">Manual/Auto Loop Fixes: integrate fallback modes, clear error messages, alternative testing paths</step>
      <step id="S4">Complete Mesh Pipeline: asset generation using trimesh, GLB loading, seamless edge connections</step>
      <step id="S5">Visual Consistency System: material transitions, edge alignment validation, thematic coherence</step>
      <step id="S6">Enhanced Navigation: perfect spherical walking, infinite mouse rotation, collision refinements</step>
      <step id="S7">Testing Infrastructure: comprehensive validation without OpenGL dependency</step>
      <step id="S8">Performance Optimization: efficient mesh batching, memory management, fast generation</step>
    </steps>
    
    <implementation_priority>
      <p1>OpenGL independence (S1, S2, S3) - fixes critical user-blocking issues</p1>
      <p2>Mesh pipeline completion (S4, S5) - enables visual quality goals</p2>
      <p3>Navigation perfection (S6) - enhances user experience</p3>
      <p4>Testing and optimization (S7, S8) - ensures robustness</p4>
    </implementation_priority>
  </plan>

  <files_to_create>
    <!-- Capability Detection and Fallback Systems -->
    <file path="forge/core/capabilities.py">Detect OpenGL, PIL, matplotlib; select operation mode</file>
    <file path="forge/modules/fallback/headless_analyzer.py">World structure analysis without rendering</file>
    <file path="forge/modules/fallback/map_2d_generator.py">Generate 2D visualizations using PIL/matplotlib</file>
    <file path="forge/modules/fallback/text_reporter.py">Detailed text-based world reports</file>
    
    <!-- Complete Mesh Pipeline -->
    <file path="scripts/providers/provider_trimesh_local.py">Local mesh generation using trimesh</file>
    <file path="forge/modules/meshes/loader.py">GLB loading and caching system</file>
    <file path="forge/modules/meshes/edge_connector.py">Seamless edge connection algorithms</file>
    <file path="forge/modules/meshes/material_blender.py">Smooth material transitions</file>
    
    <!-- Enhanced Testing -->
    <file path="tests/test_opengl_independence.py">Validate all modes work without OpenGL</file>
    <file path="tests/test_mesh_consistency.py">Validate seamless connections and materials</file>
    <file path="tests/test_navigation_perfect.py">Validate spherical navigation quality</file>
    
    <!-- Updated Loop Systems -->
    <file path="agents/run_evolution_auto_enhanced.py">Auto loop with robust fallbacks</file>
    <file path="agents/run_evolution_manual_enhanced.py">Manual loop with fallback testing</file>
  </files_to_create>

  <files_to_edit>
    <!-- Fix Existing Loop Systems -->
    <file path="agents/run_evolution_auto.py">Add capability detection, fallback modes</file>
    <file path="agents/run_evolution_manual.py">Integrate headless testing options</file>
    <file path="launch_manual_test.py">Add fallback visualization modes</file>
    
    <!-- Enhance Rendering Pipeline -->
    <file path="renderer/pcc_spherical_viewer.py">Integrate mesh loading, perfect navigation</file>
    <file path="scripts/run_gl.py">Add capability checks, fallback options</file>
    
    <!-- Complete Mesh Generation -->
    <file path="scripts/generate_meshes.py">Complete implementation with trimesh</file>
    <file path="agents/agent_a_generator.py">Enhance asset manifest with consistency metadata</file>
  </files_to_edit>

  <operation_modes>
    <mode name="full" requirements="OpenGL + PyOpenGL">
      <description>Complete 3D rendering with mesh assets</description>
      <capabilities>spherical_viewer, mesh_rendering, real_time_navigation</capabilities>
    </mode>
    <mode name="headless" requirements="None">
      <description>World analysis and validation without rendering</description>
      <capabilities>structure_analysis, placement_validation, text_reporting</capabilities>
    </mode>
    <mode name="2d_fallback" requirements="PIL or matplotlib">
      <description>2D visualization and map generation</description>
      <capabilities>top_down_maps, cross_sections, asset_diagrams</capabilities>
    </mode>
    <mode name="minimal" requirements="Python stdlib only">
      <description>Basic world generation and JSON analysis</description>
      <capabilities>world_generation, data_validation, basic_reporting</capabilities>
    </mode>
  </operation_modes>

  <visual_consistency_requirements>
    <edge_connections>
      <requirement>Adjacent mesh vertices align within 1mm tolerance</requirement>
      <requirement>UV coordinates match at shared edges</requirement>
      <requirement>Normal vectors blend smoothly across boundaries</requirement>
    </edge_connections>
    <material_transitions>
      <requirement>Color gradients flow naturally between assets</requirement>
      <requirement>Texture scales match at connection points</requirement>
      <requirement>Lighting response consistent across materials</requirement>
    </material_transitions>
    <thematic_coherence>
      <requirement>Asset styles match environmental context</requirement>
      <requirement>Scale relationships feel natural</requirement>
      <requirement>Wear patterns and aging consistent</requirement>
    </thematic_coherence>
  </visual_consistency_requirements>

  <navigation_perfection_specs>
    <spherical_walking>
      <spec>Player remains locked to sphere surface at all times</spec>
      <spec>Movement vectors properly projected to curved surface</spec>
      <spec>No jitter or sticking when walking over terrain features</spec>
      <spec>Smooth transitions over mesh boundaries</spec>
    </spherical_walking>
    <mouse_control>
      <spec>Infinite rotation without boundary limits</spec>
      <spec>Consistent sensitivity across all viewing angles</spec>
      <spec>Smooth mouse warp to maintain continuous rotation</spec>
      <spec>No drift or acceleration artifacts</spec>
    </mouse_control>
    <collision_system>
      <spec>Safe spawn with adequate clearance radius</spec>
      <spec>Precise mesh-based collision detection</spec>
      <spec>Smooth sliding along surfaces</spec>
      <spec>No clipping through thin objects</spec>
    </collision_system>
  </navigation_perfection_specs>

  <testing_strategy>
    <automated_tests>
      <test>Capability detection correctly identifies available libraries</test>
      <test>All operation modes function without errors</test>
      <test>Mesh generation produces valid GLB files</test>
      <test>Edge connections meet tolerance requirements</test>
      <test>Spherical navigation passes precision tests</test>
    </automated_tests>
    <integration_tests>
      <test>Manual loop works in all operation modes</test>
      <test>Auto loop provides meaningful feedback without OpenGL</test>
      <test>World generation produces consistent results</test>
      <test>Performance targets met across all modes</test>
    </integration_tests>
    <user_experience_tests>
      <test>Clear error messages when capabilities missing</test>
      <test>Graceful degradation to appropriate fallback mode</test>
      <test>Meaningful output in headless operation</test>
      <test>Smooth navigation experience in 3D mode</test>
    </user_experience_tests>
  </testing_strategy>

  <commands shell="bash">
    <![CDATA[
set -euo pipefail

# Create core capability detection
mkdir -p forge/core forge/modules/fallback forge/modules/meshes
mkdir -p scripts/providers tests
mkdir -p runs/fallback_assets

# Capability detection stub
cat > forge/core/capabilities.py <<'EOF'
"""System capability detection and operation mode selection."""
import importlib
import os
from typing import Dict, List, Tuple

class CapabilityDetector:
    def __init__(self):
        self.capabilities = self._detect_capabilities()
        self.operation_mode = self._select_operation_mode()
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect available libraries and system capabilities."""
        caps = {}
        
        # OpenGL detection
        try:
            import OpenGL.GL
            caps['opengl'] = True
        except ImportError:
            caps['opengl'] = False
        
        # PIL detection
        try:
            import PIL.Image
            caps['pil'] = True
        except ImportError:
            caps['pil'] = False
        
        # Matplotlib detection
        try:
            import matplotlib.pyplot
            caps['matplotlib'] = True
        except ImportError:
            caps['matplotlib'] = False
        
        # Display check (Linux)
        caps['display'] = bool(os.environ.get('DISPLAY'))
        
        return caps
    
    def _select_operation_mode(self) -> str:
        """Select best available operation mode."""
        if self.capabilities['opengl'] and self.capabilities['display']:
            return 'full'
        elif self.capabilities['pil'] or self.capabilities['matplotlib']:
            return '2d_fallback'
        else:
            return 'headless'
    
    def get_mode_description(self) -> str:
        """Get human-readable description of current mode."""
        descriptions = {
            'full': 'Full 3D rendering with OpenGL',
            '2d_fallback': '2D visualization fallback',
            'headless': 'Headless analysis mode'
        }
        return descriptions.get(self.operation_mode, 'Unknown mode')

# Global instance
detector = CapabilityDetector()
EOF

# Headless analyzer stub
cat > forge/modules/fallback/headless_analyzer.py <<'EOF'
"""Headless world analysis without rendering requirements."""
import json
from typing import Dict, Any, List

class HeadlessWorldAnalyzer:
    def __init__(self, world_data: Dict[str, Any]):
        self.world_data = world_data
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze world structure without rendering."""
        terrain = self.world_data.get('terrain', {})
        objects = self.world_data.get('objects', [])
        
        return {
            'terrain_type': terrain.get('type'),
            'planet_radius': terrain.get('radius'),
            'object_count': len(objects),
            'material_types': list(set(obj.get('material', 'unknown') for obj in objects)),
            'spawn_safety': self._check_spawn_safety(),
            'navigability_score': self._calculate_navigability()
        }
    
    def _check_spawn_safety(self) -> Dict[str, Any]:
        """Check if spawn positions are safe."""
        # Simplified spawn safety check
        return {
            'safe_spawn_zones': 8,  # Mock data
            'collision_risks': 0,
            'clearance_adequate': True
        }
    
    def _calculate_navigability(self) -> float:
        """Calculate how navigable the world is."""
        # Mock navigability calculation
        return 0.85

def analyze_world_headless(world_file: str) -> Dict[str, Any]:
    """Analyze world file without rendering."""
    try:
        with open(world_file, 'r') as f:
            world_data = json.load(f)
        
        analyzer = HeadlessWorldAnalyzer(world_data)
        return analyzer.analyze_structure()
    except Exception as e:
        return {'error': str(e), 'success': False}
EOF

echo "Week-4 OpenGL independence foundations created."
echo "Next: Complete mesh pipeline and enhanced navigation systems."
    ]]>
  </commands>

  <verify>
    <checks>
      <check>Capability detection system created and functional</check>
      <check>Headless analysis mode available</check>
      <check>Foundation files created for mesh pipeline completion</check>
    </checks>
  </verify>

  <results>
    <summary>Week-4 OpenGL independence foundation established; capability detection active; ready for complete mesh pipeline and navigation perfection.</summary>
  </results>
</low-order-prompt>
```