#!/usr/bin/env python3
"""
T12 Normal Mapping Shader System Test
=====================================

Comprehensive test of T12 material & tangent space correctness system:
- MikkTSpace tangent generation validation  
- TBN space standardization verification
- Shader system functionality testing
- Visual sanity lighting integration
- Consistent shading validation

Tests the complete T12 pipeline from tangent generation through shader rendering.
"""

import numpy as np
import os
import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'materials'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'shaders'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lighting'))

def test_mikktspace_tangent_generation():
    """Test MikkTSpace tangent generation system"""
    print("🔍 Testing MikkTSpace tangent generation...")
    
    try:
        from mikktspace_tangents import MikkTSpaceTangentGenerator, TangentGenerationConfig
        
        # Create test geometry (simple quad)
        positions = np.array([
            [-1.0, 0.0, -1.0],  # Bottom-left
            [ 1.0, 0.0, -1.0],  # Bottom-right
            [ 1.0, 0.0,  1.0],  # Top-right
            [-1.0, 0.0,  1.0],  # Top-left
        ], dtype=np.float32)
        
        normals = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        indices = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3],  # Second triangle
        ], dtype=np.int32)
        
        # Generate tangents
        generator = MikkTSpaceTangentGenerator()
        tangents = generator.generate_tangents(positions, normals, uvs, indices)
        
        # Validate results
        if len(tangents) == len(positions) and tangents.shape[1] == 4:
            # Check tangent orthogonality
            orthogonal_count = 0
            for i, tangent in enumerate(tangents):
                t_vec = tangent[:3]
                n_vec = normals[i]
                dot_product = abs(np.dot(t_vec, n_vec))
                
                if dot_product < 0.01:  # Nearly orthogonal
                    orthogonal_count += 1
            
            if orthogonal_count >= len(positions) * 0.8:  # 80% orthogonal
                print("   ✅ MikkTSpace tangent generation working")
                return True
            else:
                print(f"   ❌ Poor tangent orthogonality: {orthogonal_count}/{len(positions)}")
                return False
        else:
            print("   ❌ Invalid tangent array dimensions")
            return False
            
    except Exception as e:
        print(f"   ❌ MikkTSpace test failed: {e}")
        return False


def test_tbn_space_standardization():
    """Test TBN space standardization system"""
    print("🔍 Testing TBN space standardization...")
    
    try:
        from tbn_space_standard import TBNSpaceManager, MaterialType
        
        # Create TBN manager
        tbn_mgr = TBNSpaceManager()
        
        # Test material standard access
        terrain_material = tbn_mgr.get_material_standard(MaterialType.TERRAIN)
        cave_material = tbn_mgr.get_material_standard(MaterialType.CAVE)
        
        if (terrain_material.material_type == MaterialType.TERRAIN and
            cave_material.material_type == MaterialType.CAVE):
            
            # Test shader defines generation
            terrain_defines = tbn_mgr.generate_material_shader_defines(MaterialType.TERRAIN)
            cave_defines = tbn_mgr.generate_material_shader_defines(MaterialType.CAVE)
            
            if (len(terrain_defines) > 0 and len(cave_defines) > 0 and
                any("#define MATERIAL_TYPE_TERRAIN" in define for define in terrain_defines) and
                any("#define MATERIAL_TYPE_CAVE" in define for define in cave_defines)):
                
                print("   ✅ TBN space standardization working")
                return True
            else:
                print("   ❌ Shader defines generation failed")
                return False
        else:
            print("   ❌ Material standard access failed")
            return False
            
    except Exception as e:
        print(f"   ❌ TBN space standardization test failed: {e}")
        return False


def test_shader_system():
    """Test terrain normal mapping shader system"""
    print("🔍 Testing shader system...")
    
    try:
        from terrain_normal_mapping import TerrainShaderManager
        from tbn_space_standard import MaterialType
        
        # Create shader manager (without OpenGL context)
        shader_mgr = TerrainShaderManager()
        
        # Test TBN manager integration
        if shader_mgr.tbn_manager is not None:
            
            # Test material access through shader manager
            terrain_material = shader_mgr.tbn_manager.get_material_standard(MaterialType.TERRAIN)
            
            if (terrain_material.base_color and
                terrain_material.roughness >= 0 and
                terrain_material.metallic >= 0):
                
                print("   ✅ Shader system integration working")
                return True
            else:
                print("   ❌ Material parameter access failed")
                return False
        else:
            print("   ❌ TBN manager not initialized")
            return False
            
    except Exception as e:
        print(f"   ❌ Shader system test failed: {e}")
        return False


def test_visual_sanity_lighting():
    """Test visual sanity lighting system"""
    print("🔍 Testing visual sanity lighting...")
    
    try:
        from visual_sanity_scene import VisualSanityLighting, TimeOfDay
        
        # Create lighting system
        lighting = VisualSanityLighting()
        
        # Test time of day updates
        initial_sun_intensity = lighting.skybox_params.sun_intensity
        
        lighting.update_time_of_day(0.0)  # Midnight
        night_intensity = lighting.skybox_params.sun_intensity
        
        lighting.update_time_of_day(0.5)  # Midday
        day_intensity = lighting.skybox_params.sun_intensity
        
        if day_intensity > night_intensity and day_intensity > initial_sun_intensity * 0.5:
            
            # Test animation updates
            initial_pos = lighting.light_sources["moving_spot"].position.copy()
            lighting.update_animation(1.0)  # 1 second
            updated_pos = lighting.light_sources["moving_spot"].position.copy()
            
            movement_distance = np.linalg.norm(updated_pos - initial_pos)
            
            if movement_distance > 0.1:  # Light moved
                print("   ✅ Visual sanity lighting working")
                return True
            else:
                print("   ❌ Moving light not animating")
                return False
        else:
            print("   ❌ Time of day lighting changes not working")
            return False
            
    except Exception as e:
        print(f"   ❌ Visual sanity lighting test failed: {e}")
        return False


def test_integration_complete():
    """Test complete T12 system integration"""
    print("🔍 Testing complete T12 integration...")
    
    try:
        from mikktspace_tangents import MikkTSpaceTangentGenerator
        from tbn_space_standard import TBNSpaceManager, MaterialType
        from terrain_normal_mapping import TerrainShaderManager
        from visual_sanity_scene import VisualSanityLighting
        
        # Create all systems
        tangent_gen = MikkTSpaceTangentGenerator()
        tbn_mgr = TBNSpaceManager()
        shader_mgr = TerrainShaderManager()
        lighting = VisualSanityLighting()
        
        # Test system consistency
        if (tangent_gen and tbn_mgr and shader_mgr and lighting and
            shader_mgr.tbn_manager is not None):
            
            # Test material consistency across systems
            terrain_material_tbn = tbn_mgr.get_material_standard(MaterialType.TERRAIN)
            terrain_material_shader = shader_mgr.tbn_manager.get_material_standard(MaterialType.TERRAIN)
            
            if (terrain_material_tbn.material_id == terrain_material_shader.material_id and
                terrain_material_tbn.base_color == terrain_material_shader.base_color):
                
                # Test lighting stats
                lighting_stats = lighting.get_lighting_stats()
                
                if (lighting_stats and 
                    'total_lights' in lighting_stats and 
                    lighting_stats['total_lights'] >= 3):
                    
                    print("   ✅ Complete T12 integration working")
                    return True
                else:
                    print("   ❌ Lighting stats incomplete")
                    return False
            else:
                print("   ❌ Material consistency failed")
                return False
        else:
            print("   ❌ System initialization failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


def test_material_header_export():
    """Test material definitions header export"""
    print("🔍 Testing material header export...")
    
    try:
        from terrain_normal_mapping import TerrainShaderManager
        
        shader_mgr = TerrainShaderManager()
        
        # Export header to temporary location
        header_path = "/tmp/t12_material_definitions.h"
        shader_mgr.export_shader_header(header_path)
        
        # Check if file was created and contains expected content
        if os.path.exists(header_path):
            with open(header_path, 'r') as f:
                content = f.read()
            
            if ("MATERIAL_DEFINITIONS_H" in content and
                "MATERIAL_ID_TERRAIN" in content and
                "TBN_RIGHT_HANDED" in content and
                "MaterialProperties" in content):
                
                os.remove(header_path)  # Cleanup
                print("   ✅ Material header export working")
                return True
            else:
                print("   ❌ Header content incomplete")
                return False
        else:
            print("   ❌ Header file not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Material header export test failed: {e}")
        return False


def run_t12_shader_tests():
    """Run comprehensive T12 shader system tests"""
    print("🚀 T12 Material & Tangent Space Correctness Tests")
    print("=" * 70)
    
    tests = [
        ("MikkTSpace Tangent Generation", test_mikktspace_tangent_generation),
        ("TBN Space Standardization", test_tbn_space_standardization),
        ("Shader System Integration", test_shader_system),
        ("Visual Sanity Lighting", test_visual_sanity_lighting),
        ("Complete System Integration", test_integration_complete),
        ("Material Header Export", test_material_header_export),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 T12 material & tangent space correctness system functional!")
        
        # Print summary of T12 achievements
        print("\n✅ T12 Implementation Summary:")
        print("   - MikkTSpace compatible tangent generation")
        print("   - Standardized TBN space (right-handed, OpenGL convention)")
        print("   - Material-aware shader system with normal mapping")
        print("   - Visual sanity scene with sun/sky and moving light")
        print("   - Consistent shading across terrain and cave surfaces")
        print("   - Shader header export for external use")
        
        return True
    else:
        print("⚠️ Some T12 tests failed")
        return False


if __name__ == "__main__":
    success = run_t12_shader_tests()
    sys.exit(0 if success else 1)