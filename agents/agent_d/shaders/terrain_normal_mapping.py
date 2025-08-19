#!/usr/bin/env python3
"""
Terrain Normal Mapping Shader System - T12
==========================================

Implements normal mapping shaders for terrain and cave surfaces using
standardized TBN space and material parameters from T12 system.

Features:
- Per-pixel normal mapping for terrain and cave materials
- TBN matrix construction from vertex tangents
- Material-aware shader parameter handling  
- Consistent lighting across surface types
- Integration with MikkTSpace tangent generation

Usage:
    from terrain_normal_mapping import TerrainShaderManager
    
    shader_mgr = TerrainShaderManager()
    shader_mgr.use_material_shader("terrain")
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Import TBN space system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'materials'))
from tbn_space_standard import TBNSpaceManager, MaterialType, MaterialStandard

try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileShader, compileProgram
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


class TerrainShaderManager:
    """Manages OpenGL shaders for terrain normal mapping"""
    
    def __init__(self):
        """Initialize terrain shader manager"""
        self.tbn_manager = TBNSpaceManager()
        self.shader_programs = {}
        self.current_shader = None
        self.material_uniforms = {}
        
        # Initialize shaders if OpenGL is available
        if OPENGL_AVAILABLE:
            self._create_shader_programs()
    
    def _create_shader_programs(self):
        """Create OpenGL shader programs for different materials"""
        
        # Common vertex shader for all materials
        vertex_shader_source = self._get_vertex_shader_source()
        
        # Create shader programs for each material type
        for material_type in MaterialType:
            fragment_source = self._get_fragment_shader_source(material_type)
            
            try:
                vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
                fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
                
                program = compileProgram(vertex_shader, fragment_shader)
                self.shader_programs[material_type.value] = program
                
                print(f"✅ Created shader program for {material_type.value}")
                
            except Exception as e:
                print(f"❌ Failed to create shader for {material_type.value}: {e}")
    
    def _get_vertex_shader_source(self) -> str:
        """Get vertex shader source with TBN matrix calculation"""
        return """
#version 330 core

// Vertex attributes
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec4 aTangent;  // xyz = tangent, w = handedness

// Uniforms
uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;

// Material uniforms
uniform vec2 uUvScale;
uniform vec2 uUvOffset;
uniform float uUvRotation;

// Outputs to fragment shader
out vec3 FragPos;
out vec2 TexCoord;
out vec3 Normal;
out mat3 TBN;  // Tangent-Bitangent-Normal matrix
out vec3 TangentLightPos;
out vec3 TangentViewPos;
out vec3 TangentFragPos;

// Light uniforms  
uniform vec3 uLightPos;
uniform vec3 uViewPos;

void main() {
    // Transform position
    vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
    FragPos = worldPos.xyz;
    gl_Position = uProjectionMatrix * uViewMatrix * worldPos;
    
    // Transform and apply UV operations
    vec2 uv = aTexCoord;
    uv = uv * uUvScale + uUvOffset;
    
    // Apply UV rotation if needed
    if (uUvRotation != 0.0) {
        float cosRot = cos(uUvRotation);
        float sinRot = sin(uUvRotation);
        mat2 rotMatrix = mat2(cosRot, -sinRot, sinRot, cosRot);
        uv = rotMatrix * (uv - 0.5) + 0.5;
    }
    TexCoord = uv;
    
    // Transform normal
    Normal = normalize(uNormalMatrix * aNormal);
    
    // Build TBN matrix for tangent space normal mapping
    vec3 T = normalize(uNormalMatrix * aTangent.xyz);
    vec3 B = normalize(cross(Normal, T) * aTangent.w);  // Use handedness
    TBN = mat3(T, B, Normal);
    
    // Transform light and view positions to tangent space
    TangentLightPos = transpose(TBN) * uLightPos;
    TangentViewPos = transpose(TBN) * uViewPos;
    TangentFragPos = transpose(TBN) * FragPos;
}
"""
    
    def _get_fragment_shader_source(self, material_type: MaterialType) -> str:
        """Get fragment shader source for specific material type"""
        
        # Get material-specific defines
        defines = self.tbn_manager.generate_material_shader_defines(material_type)
        
        defines_str = "\n".join(defines) + "\n"
        
        return f"""
#version 330 core

{defines_str}

// Inputs from vertex shader
in vec3 FragPos;
in vec2 TexCoord;
in vec3 Normal;
in mat3 TBN;
in vec3 TangentLightPos;
in vec3 TangentViewPos;
in vec3 TangentFragPos;

// Material uniforms
uniform vec3 uBaseColor;
uniform float uRoughness;
uniform float uMetallic;
uniform float uNormalStrength;
uniform float uAmbientOcclusion;
uniform float uEmissionStrength;
uniform float uSubsurfaceScattering;

// Texture uniforms (if available)
uniform sampler2D uDiffuseTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uRoughnessTexture;
uniform bool uHasDiffuseTexture;
uniform bool uHasNormalTexture;
uniform bool uHasRoughnessTexture;

// Lighting uniforms
uniform vec3 uLightColor;
uniform float uLightIntensity;
uniform vec3 uAmbientLight;

// Output
out vec4 FragColor;

// Utility functions
vec3 getNormalFromMap() {{
    if (!uHasNormalTexture) {{
        return normalize(Normal);
    }}
    
    // Sample normal map
    vec3 normal = texture(uNormalTexture, TexCoord).xyz;
    
    // Convert from [0,1] to [-1,1]  
    normal = normal * 2.0 - 1.0;
    
    // Apply normal strength
    normal.xy *= uNormalStrength;
    
#ifdef FLIP_NORMAL_GREEN
    normal.y = -normal.y;  // DirectX compatibility
#endif
    
    // Transform to world space using TBN matrix
    normal = normalize(TBN * normal);
    return normal;
}}

// Simple PBR-inspired lighting model
vec3 calculateLighting(vec3 albedo, vec3 normal, float roughness, float metallic) {{
    // Light direction in world space
    vec3 lightDir = normalize(TangentLightPos - TangentFragPos);
    vec3 viewDir = normalize(TangentViewPos - TangentFragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    
    // Diffuse
    float NdotL = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = albedo * NdotL;
    
    // Specular (simplified)
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 1.0 / (roughness + 0.01));
    vec3 specular = vec3(spec) * (1.0 - roughness) * uLightIntensity;
    
    // Metallic workflow approximation
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    specular *= F0;
    
    // Ambient
    vec3 ambient = uAmbientLight * albedo * uAmbientOcclusion;
    
    return ambient + (diffuse + specular) * uLightColor * uLightIntensity;
}}

void main() {{
    // Get base color
    vec3 albedo = uBaseColor;
    if (uHasDiffuseTexture) {{
        albedo *= texture(uDiffuseTexture, TexCoord).rgb;
    }}
    
    // Get normal from normal map
    vec3 normal = getNormalFromMap();
    
    // Get roughness
    float roughness = uRoughness;
    if (uHasRoughnessTexture) {{
        roughness *= texture(uRoughnessTexture, TexCoord).r;
    }}
    
    // Calculate lighting
    vec3 color = calculateLighting(albedo, normal, roughness, uMetallic);
    
    // Add emission if present
    if (uEmissionStrength > 0.0) {{
        color += albedo * uEmissionStrength;
    }}
    
    // Material-specific adjustments
#ifdef MATERIAL_TYPE_CAVE
    // Caves are generally darker and more ambient
    color *= 0.8;
    color += uAmbientLight * 0.3;
#endif

#ifdef MATERIAL_TYPE_TERRAIN  
    // Terrain gets full lighting
    // No adjustments needed
#endif

#ifdef MATERIAL_TYPE_OVERHANG
    // Overhangs have enhanced normal details
    // Already handled by normal strength
#endif

#ifdef MATERIAL_TYPE_FUSED
    // Fused materials blend characteristics
    color = mix(color, color * 0.9, 0.5);
#endif
    
    // Apply subsurface scattering approximation
    if (uSubsurfaceScattering > 0.0) {{
        vec3 subsurface = albedo * uSubsurfaceScattering * uLightColor;
        color = mix(color, color + subsurface, 0.3);
    }}
    
    FragColor = vec4(color, 1.0);
}}
"""
    
    def use_material_shader(self, material_type: str) -> bool:
        """Use shader program for specific material type"""
        if not OPENGL_AVAILABLE:
            return False
            
        if material_type not in self.shader_programs:
            print(f"❌ No shader program for material type: {material_type}")
            return False
        
        program = self.shader_programs[material_type]
        glUseProgram(program)
        self.current_shader = program
        
        # Set material-specific uniforms
        material = self.tbn_manager.get_material_standard(MaterialType(material_type))
        self._set_material_uniforms(material)
        
        return True
    
    def _set_material_uniforms(self, material: MaterialStandard):
        """Set material uniforms for current shader"""
        if not self.current_shader:
            return
        
        try:
            # Material properties
            self._set_uniform_vec3("uBaseColor", material.base_color)
            self._set_uniform_float("uRoughness", material.roughness)
            self._set_uniform_float("uMetallic", material.metallic)
            self._set_uniform_float("uNormalStrength", material.normal_strength)
            self._set_uniform_float("uAmbientOcclusion", material.ambient_occlusion)
            self._set_uniform_float("uEmissionStrength", material.emission_strength)
            self._set_uniform_float("uSubsurfaceScattering", material.subsurface_scattering)
            
            # UV parameters
            self._set_uniform_vec2("uUvScale", material.uv_scale)
            self._set_uniform_vec2("uUvOffset", material.uv_offset)
            self._set_uniform_float("uUvRotation", material.uv_rotation)
            
            # Texture availability flags (default to false)
            self._set_uniform_bool("uHasDiffuseTexture", False)
            self._set_uniform_bool("uHasNormalTexture", False)
            self._set_uniform_bool("uHasRoughnessTexture", False)
            
        except Exception as e:
            print(f"❌ Failed to set material uniforms: {e}")
    
    def _set_uniform_vec3(self, name: str, value: Tuple[float, float, float]):
        """Set vec3 uniform"""
        location = glGetUniformLocation(self.current_shader, name)
        if location >= 0:
            glUniform3f(location, *value)
    
    def _set_uniform_vec2(self, name: str, value: Tuple[float, float]):
        """Set vec2 uniform"""
        location = glGetUniformLocation(self.current_shader, name)
        if location >= 0:
            glUniform2f(location, *value)
    
    def _set_uniform_float(self, name: str, value: float):
        """Set float uniform"""
        location = glGetUniformLocation(self.current_shader, name)
        if location >= 0:
            glUniform1f(location, value)
    
    def _set_uniform_bool(self, name: str, value: bool):
        """Set boolean uniform"""
        location = glGetUniformLocation(self.current_shader, name)
        if location >= 0:
            glUniform1i(location, 1 if value else 0)
    
    def set_matrices(self, model_matrix: np.ndarray, view_matrix: np.ndarray, 
                    projection_matrix: np.ndarray):
        """Set transformation matrices"""
        if not self.current_shader:
            return
        
        try:
            # Model matrix
            location = glGetUniformLocation(self.current_shader, "uModelMatrix")
            if location >= 0:
                glUniformMatrix4fv(location, 1, GL_FALSE, model_matrix.astype(np.float32))
            
            # View matrix
            location = glGetUniformLocation(self.current_shader, "uViewMatrix")
            if location >= 0:
                glUniformMatrix4fv(location, 1, GL_FALSE, view_matrix.astype(np.float32))
            
            # Projection matrix
            location = glGetUniformLocation(self.current_shader, "uProjectionMatrix")
            if location >= 0:
                glUniformMatrix4fv(location, 1, GL_FALSE, projection_matrix.astype(np.float32))
            
            # Normal matrix (inverse transpose of model matrix upper 3x3)
            normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T
            location = glGetUniformLocation(self.current_shader, "uNormalMatrix")
            if location >= 0:
                glUniformMatrix3fv(location, 1, GL_FALSE, normal_matrix.astype(np.float32))
                
        except Exception as e:
            print(f"❌ Failed to set matrices: {e}")
    
    def set_lighting(self, light_pos: Tuple[float, float, float], 
                    light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                    light_intensity: float = 1.0,
                    ambient_light: Tuple[float, float, float] = (0.1, 0.1, 0.1),
                    view_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """Set lighting parameters"""
        if not self.current_shader:
            return
        
        try:
            self._set_uniform_vec3("uLightPos", light_pos)
            self._set_uniform_vec3("uLightColor", light_color)
            self._set_uniform_float("uLightIntensity", light_intensity)
            self._set_uniform_vec3("uAmbientLight", ambient_light)
            self._set_uniform_vec3("uViewPos", view_pos)
            
        except Exception as e:
            print(f"❌ Failed to set lighting uniforms: {e}")
    
    def export_shader_header(self, output_path: str):
        """Export shader definitions to header file"""
        self.tbn_manager.export_material_definitions(output_path)
        print(f"✅ Exported shader definitions to {output_path}")
    
    def cleanup(self):
        """Clean up OpenGL resources"""
        if not OPENGL_AVAILABLE:
            return
            
        for program in self.shader_programs.values():
            glDeleteProgram(program)
        
        self.shader_programs.clear()
        self.current_shader = None


if __name__ == "__main__":
    # Test terrain shader system (without OpenGL context)
    print("🚀 T12 Terrain Normal Mapping Shader System")
    print("=" * 60)
    
    # Create shader manager
    shader_mgr = TerrainShaderManager()
    
    # Test material access
    print("📋 Available Materials:")
    for material_type in MaterialType:
        material = shader_mgr.tbn_manager.get_material_standard(material_type)
        print(f"   {material_type.value}: {material.base_color}, rough={material.roughness:.2f}")
    
    # Test shader defines generation
    terrain_defines = shader_mgr.tbn_manager.generate_material_shader_defines(MaterialType.TERRAIN)
    print(f"\n🔧 Terrain Shader Defines:")
    for define in terrain_defines:
        print(f"   {define}")
    
    # Export material definitions
    header_path = "/tmp/material_definitions.h"
    shader_mgr.export_shader_header(header_path)
    
    print("\n✅ Terrain normal mapping shader system initialized")
    
    if OPENGL_AVAILABLE:
        print("✅ OpenGL shaders ready for use")
    else:
        print("⚠️ OpenGL not available - shaders defined but not compiled")