#!/usr/bin/env python3
"""
TBN Space Standardization - T12
================================

Defines standard conventions for Tangent-Bitangent-Normal (TBN) space
across all materials and mesh types to ensure consistent normal mapping
and lighting behavior.

Standards:
- Right-handed coordinate system: T × B = N (positive handedness)
- OpenGL normal map convention: +X=right, +Y=up, +Z=forward
- Consistent UV mapping orientation across terrain and caves
- Material type identification and shader parameter mapping

Usage:
    from tbn_space_standard import TBNSpaceManager, MaterialStandard
    
    tbn_mgr = TBNSpaceManager()
    material = tbn_mgr.get_material_standard("terrain")
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Import tangent generation system
sys.path.append(os.path.dirname(__file__))
from mikktspace_tangents import MikkTSpaceTangentGenerator, TangentGenerationConfig, MikkTSpaceHandedness


class NormalMapConvention(Enum):
    """Normal map coordinate system conventions"""
    OPENGL = "opengl"        # +X=right, +Y=up, +Z=forward (standard)
    DIRECTX = "directx"      # +X=right, +Y=down, +Z=forward (flipped Y)


class MaterialType(Enum):
    """Material types with different shading requirements"""
    TERRAIN = "terrain"
    CAVE = "cave"
    OVERHANG = "overhang" 
    FUSED = "fused"
    WATER = "water"
    VEGETATION = "vegetation"


@dataclass
class MaterialStandard:
    """Standard material definition with TBN space parameters"""
    material_type: MaterialType
    material_id: int
    
    # Visual properties
    base_color: Tuple[float, float, float]
    roughness: float
    metallic: float
    normal_strength: float = 1.0
    
    # TBN space configuration
    normal_map_convention: NormalMapConvention = NormalMapConvention.OPENGL
    handedness: MikkTSpaceHandedness = MikkTSpaceHandedness.RIGHT_HANDED
    flip_green_channel: bool = False  # For DirectX compatibility
    
    # UV mapping parameters  
    uv_scale: Tuple[float, float] = (1.0, 1.0)
    uv_offset: Tuple[float, float] = (0.0, 0.0)
    uv_rotation: float = 0.0  # Radians
    
    # Lighting parameters
    ambient_occlusion: float = 1.0
    emission_strength: float = 0.0
    subsurface_scattering: float = 0.0
    
    def get_shader_parameters(self) -> Dict[str, Any]:
        """Get shader parameters for this material"""
        return {
            'material_id': self.material_id,
            'base_color': self.base_color,
            'roughness': self.roughness,
            'metallic': self.metallic,
            'normal_strength': self.normal_strength,
            'uv_scale': self.uv_scale,
            'uv_offset': self.uv_offset,
            'uv_rotation': self.uv_rotation,
            'flip_green_channel': self.flip_green_channel,
            'ambient_occlusion': self.ambient_occlusion,
            'emission_strength': self.emission_strength,
            'subsurface_scattering': self.subsurface_scattering
        }


@dataclass
class TBNMatrix:
    """TBN matrix with validation and transformation utilities"""
    tangent: np.ndarray    # [3] tangent vector
    bitangent: np.ndarray  # [3] bitangent vector  
    normal: np.ndarray     # [3] normal vector
    handedness: float      # +1.0 or -1.0
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 3x3 TBN transformation matrix"""
        return np.column_stack([self.tangent, self.bitangent, self.normal])
    
    @property
    def inverse_matrix(self) -> np.ndarray:
        """Get inverse TBN matrix for transforming from world to tangent space"""
        return self.matrix.T  # Orthogonal matrix, so transpose = inverse
    
    def transform_to_tangent_space(self, world_vector: np.ndarray) -> np.ndarray:
        """Transform vector from world space to tangent space"""
        return self.inverse_matrix @ world_vector
    
    def transform_from_tangent_space(self, tangent_vector: np.ndarray) -> np.ndarray:
        """Transform vector from tangent space to world space"""
        return self.matrix @ tangent_vector
    
    def validate_orthogonality(self, tolerance: float = 1e-3) -> bool:
        """Validate TBN vectors are orthogonal"""
        # Check dot products
        tb_dot = abs(np.dot(self.tangent, self.bitangent))
        tn_dot = abs(np.dot(self.tangent, self.normal))
        bn_dot = abs(np.dot(self.bitangent, self.normal))
        
        return all(dot < tolerance for dot in [tb_dot, tn_dot, bn_dot])
    
    def validate_handedness(self) -> bool:
        """Validate handedness matches cross product"""
        computed_bitangent = np.cross(self.tangent, self.normal)
        expected_bitangent = self.bitangent * self.handedness
        
        # Check if computed bitangent matches expected direction
        dot_product = np.dot(computed_bitangent, expected_bitangent)
        return dot_product > 0.99  # Should be very close to 1.0


class TBNSpaceManager:
    """Manages TBN space standardization across all materials"""
    
    def __init__(self):
        """Initialize TBN space manager with material standards"""
        self.material_standards = self._create_default_material_standards()
        self.tangent_generator = MikkTSpaceTangentGenerator()
        
        # Global TBN space configuration
        self.global_convention = NormalMapConvention.OPENGL
        self.global_handedness = MikkTSpaceHandedness.RIGHT_HANDED
    
    def _create_default_material_standards(self) -> Dict[MaterialType, MaterialStandard]:
        """Create default material standards"""
        standards = {}
        
        # Terrain material
        standards[MaterialType.TERRAIN] = MaterialStandard(
            material_type=MaterialType.TERRAIN,
            material_id=0,
            base_color=(0.6, 0.4, 0.2),  # Brown terrain
            roughness=0.8,
            metallic=0.0,
            normal_strength=1.0,
            uv_scale=(4.0, 4.0),  # Tile texture 4x
            ambient_occlusion=0.8
        )
        
        # Cave material
        standards[MaterialType.CAVE] = MaterialStandard(
            material_type=MaterialType.CAVE,
            material_id=1,
            base_color=(0.3, 0.3, 0.4),  # Dark gray cave
            roughness=0.9,
            metallic=0.1,
            normal_strength=0.8,
            uv_scale=(2.0, 2.0),  # Less tiling for caves
            ambient_occlusion=0.6  # Darker caves
        )
        
        # Overhang material
        standards[MaterialType.OVERHANG] = MaterialStandard(
            material_type=MaterialType.OVERHANG,
            material_id=2,
            base_color=(0.4, 0.3, 0.2),  # Dark brown overhang
            roughness=0.7,
            metallic=0.0,
            normal_strength=1.2,  # More pronounced normals
            uv_scale=(3.0, 3.0)
        )
        
        # Fused material (terrain + caves)
        standards[MaterialType.FUSED] = MaterialStandard(
            material_type=MaterialType.FUSED,
            material_id=3,
            base_color=(0.5, 0.35, 0.25),  # Blend of terrain and cave
            roughness=0.85,
            metallic=0.05,
            normal_strength=0.9,
            uv_scale=(3.0, 3.0)
        )
        
        return standards
    
    def get_material_standard(self, material_type: MaterialType) -> MaterialStandard:
        """Get material standard for given type"""
        return self.material_standards.get(material_type, self.material_standards[MaterialType.TERRAIN])
    
    def create_tbn_matrices_for_mesh(self, positions: np.ndarray, normals: np.ndarray,
                                   uvs: np.ndarray, indices: np.ndarray,
                                   material_type: MaterialType = MaterialType.TERRAIN) -> List[TBNMatrix]:
        """Create TBN matrices for entire mesh with material-specific settings"""
        
        # Get material standard
        material = self.get_material_standard(material_type)
        
        # Configure tangent generator for this material
        config = TangentGenerationConfig(
            handedness=material.handedness,
            normalize_uvs=True
        )
        
        generator = MikkTSpaceTangentGenerator(config)
        
        # Generate tangents
        tangents = generator.generate_tangents(positions, normals, uvs, indices)
        
        # Create TBN matrices
        tbn_matrices = []
        for i in range(len(positions)):
            tangent_vec = tangents[i, :3]
            handedness = tangents[i, 3]
            normal_vec = normals[i]
            
            # Compute bitangent
            bitangent_vec = np.cross(normal_vec, tangent_vec) * handedness
            
            tbn_matrix = TBNMatrix(
                tangent=tangent_vec,
                bitangent=bitangent_vec,
                normal=normal_vec,
                handedness=handedness
            )
            
            tbn_matrices.append(tbn_matrix)
        
        return tbn_matrices
    
    def validate_mesh_tbn_consistency(self, tbn_matrices: List[TBNMatrix]) -> Dict[str, Any]:
        """Validate TBN consistency across a mesh"""
        results = {
            'total_vertices': len(tbn_matrices),
            'orthogonal_count': 0,
            'valid_handedness_count': 0,
            'right_handed_count': 0,
            'left_handed_count': 0,
            'max_orthogonality_error': 0.0,
            'avg_orthogonality_error': 0.0
        }
        
        total_error = 0.0
        
        for tbn in tbn_matrices:
            # Check orthogonality
            if tbn.validate_orthogonality():
                results['orthogonal_count'] += 1
            
            # Check handedness
            if tbn.validate_handedness():
                results['valid_handedness_count'] += 1
            
            # Count handedness distribution
            if tbn.handedness > 0:
                results['right_handed_count'] += 1
            else:
                results['left_handed_count'] += 1
            
            # Calculate orthogonality error
            t, b, n = tbn.tangent, tbn.bitangent, tbn.normal
            error = max(abs(np.dot(t, b)), abs(np.dot(t, n)), abs(np.dot(b, n)))
            total_error += error
            results['max_orthogonality_error'] = max(results['max_orthogonality_error'], error)
        
        results['avg_orthogonality_error'] = total_error / len(tbn_matrices) if tbn_matrices else 0
        
        return results
    
    def convert_normal_map_convention(self, normal_map: np.ndarray, 
                                    from_convention: NormalMapConvention,
                                    to_convention: NormalMapConvention) -> np.ndarray:
        """Convert normal map between different conventions"""
        if from_convention == to_convention:
            return normal_map
        
        converted = normal_map.copy()
        
        # DirectX to OpenGL: flip Y channel
        if from_convention == NormalMapConvention.DIRECTX and to_convention == NormalMapConvention.OPENGL:
            if len(converted.shape) == 3 and converted.shape[2] >= 2:
                converted[:, :, 1] = 1.0 - converted[:, :, 1]  # Flip green channel
        
        # OpenGL to DirectX: flip Y channel
        elif from_convention == NormalMapConvention.OPENGL and to_convention == NormalMapConvention.DIRECTX:
            if len(converted.shape) == 3 and converted.shape[2] >= 2:
                converted[:, :, 1] = 1.0 - converted[:, :, 1]  # Flip green channel
        
        return converted
    
    def generate_material_shader_defines(self, material_type: MaterialType) -> List[str]:
        """Generate shader preprocessor defines for material type"""
        material = self.get_material_standard(material_type)
        defines = []
        
        defines.append(f"#define MATERIAL_ID {material.material_id}")
        defines.append(f"#define MATERIAL_TYPE_{material.material_type.value.upper()}")
        
        if material.normal_map_convention == NormalMapConvention.DIRECTX:
            defines.append("#define NORMAL_MAP_DIRECTX")
        else:
            defines.append("#define NORMAL_MAP_OPENGL")
        
        if material.handedness == MikkTSpaceHandedness.LEFT_HANDED:
            defines.append("#define LEFT_HANDED_TBN")
        else:
            defines.append("#define RIGHT_HANDED_TBN")
        
        if material.flip_green_channel:
            defines.append("#define FLIP_NORMAL_GREEN")
        
        return defines
    
    def export_material_definitions(self, output_path: str):
        """Export material definitions to shader header file"""
        try:
            with open(output_path, 'w') as f:
                f.write("// Auto-generated material definitions - T12\n")
                f.write("// TBN Space Standardization\n\n")
                
                f.write("#ifndef MATERIAL_DEFINITIONS_H\n")
                f.write("#define MATERIAL_DEFINITIONS_H\n\n")
                
                # Material ID constants
                f.write("// Material ID constants\n")
                for material_type, standard in self.material_standards.items():
                    define_name = f"MATERIAL_ID_{material_type.value.upper()}"
                    f.write(f"#define {define_name} {standard.material_id}\n")
                f.write("\n")
                
                # TBN space constants
                f.write("// TBN space configuration\n")
                f.write("#define TBN_RIGHT_HANDED 1.0\n")
                f.write("#define TBN_LEFT_HANDED -1.0\n")
                f.write("#define NORMAL_MAP_OPENGL 0\n")
                f.write("#define NORMAL_MAP_DIRECTX 1\n\n")
                
                # Material properties structure
                f.write("// Material properties (match with uniform buffer)\n")
                f.write("struct MaterialProperties {\n")
                f.write("    vec3 baseColor;\n")
                f.write("    float roughness;\n")
                f.write("    float metallic;\n")
                f.write("    float normalStrength;\n")
                f.write("    vec2 uvScale;\n")
                f.write("    vec2 uvOffset;\n")
                f.write("    float uvRotation;\n")
                f.write("    float ambientOcclusion;\n")
                f.write("    float emissionStrength;\n")
                f.write("    float subsurfaceScattering;\n")
                f.write("};\n\n")
                
                # Utility functions
                f.write("// TBN space utility functions\n")
                f.write("vec3 transformToTangentSpace(vec3 worldVector, mat3 tbn) {\n")
                f.write("    return transpose(tbn) * worldVector;\n")
                f.write("}\n\n")
                
                f.write("vec3 transformFromTangentSpace(vec3 tangentVector, mat3 tbn) {\n")
                f.write("    return tbn * tangentVector;\n")
                f.write("}\n\n")
                
                f.write("#endif // MATERIAL_DEFINITIONS_H\n")
            
            print(f"✅ Exported material definitions to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to export material definitions: {e}")


class NormalMapProcessor:
    """Processes normal maps for consistent TBN space usage"""
    
    def __init__(self, tbn_manager: TBNSpaceManager):
        """Initialize with TBN space manager"""
        self.tbn_manager = tbn_manager
    
    def process_normal_map_for_material(self, normal_map: np.ndarray, 
                                      material_type: MaterialType) -> np.ndarray:
        """Process normal map for specific material type"""
        material = self.tbn_manager.get_material_standard(material_type)
        
        # Convert to target convention
        processed = self.tbn_manager.convert_normal_map_convention(
            normal_map, 
            NormalMapConvention.OPENGL,  # Assume input is OpenGL
            material.normal_map_convention
        )
        
        # Apply material-specific strength scaling
        if len(processed.shape) == 3 and processed.shape[2] >= 3:
            # Scale X and Y channels by normal strength, keep Z
            processed[:, :, 0] = (processed[:, :, 0] - 0.5) * material.normal_strength + 0.5
            processed[:, :, 1] = (processed[:, :, 1] - 0.5) * material.normal_strength + 0.5
        
        return processed
    
    def create_default_normal_map(self, width: int = 256, height: int = 256) -> np.ndarray:
        """Create default flat normal map (pointing up in tangent space)"""
        # Default normal is (0, 0, 1) in tangent space
        # Encoded as (0.5, 0.5, 1.0) in [0,1] range
        normal_map = np.zeros((height, width, 3), dtype=np.float32)
        normal_map[:, :, 0] = 0.5  # X = 0 (no tangent deviation)
        normal_map[:, :, 1] = 0.5  # Y = 0 (no bitangent deviation) 
        normal_map[:, :, 2] = 1.0  # Z = 1 (pointing up in tangent space)
        
        return normal_map


if __name__ == "__main__":
    # Test TBN space standardization
    print("🚀 T12 TBN Space Standardization System")
    print("=" * 60)
    
    # Create TBN manager
    tbn_manager = TBNSpaceManager()
    
    # Test material standards
    print("📋 Material Standards:")
    for material_type in MaterialType:
        standard = tbn_manager.get_material_standard(material_type)
        print(f"   {material_type.value}: ID={standard.material_id}, "
              f"Color={standard.base_color}, Rough={standard.roughness}")
    
    # Test TBN generation for simple geometry
    positions = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0]
    ], dtype=np.float32)
    
    normals = np.array([
        [0, 0, 1], [0, 0, 1], [0, 0, 1]
    ], dtype=np.float32)
    
    uvs = np.array([
        [0, 0], [1, 0], [0.5, 1]
    ], dtype=np.float32)
    
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    
    # Generate TBN matrices
    tbn_matrices = tbn_manager.create_tbn_matrices_for_mesh(
        positions, normals, uvs, indices, MaterialType.TERRAIN
    )
    
    print(f"\n✅ Generated TBN matrices: {len(tbn_matrices)}")
    
    # Validate TBN consistency
    validation = tbn_manager.validate_mesh_tbn_consistency(tbn_matrices)
    print(f"\n📊 TBN Validation:")
    print(f"   Orthogonal: {validation['orthogonal_count']}/{validation['total_vertices']}")
    print(f"   Valid handedness: {validation['valid_handedness_count']}/{validation['total_vertices']}")
    print(f"   Right-handed: {validation['right_handed_count']}")
    print(f"   Max orthogonality error: {validation['max_orthogonality_error']:.6f}")
    
    # Test shader defines
    terrain_defines = tbn_manager.generate_material_shader_defines(MaterialType.TERRAIN)
    print(f"\n🔧 Terrain Shader Defines:")
    for define in terrain_defines:
        print(f"   {define}")
    
    print("\n✅ TBN space standardization system functional")