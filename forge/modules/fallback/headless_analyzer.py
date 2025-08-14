"""Headless world analysis without OpenGL requirements - Week 4 enhanced."""
import json
import math
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

class HeadlessWorldAnalyzer:
    """Analyzes world structure and provides detailed reports without rendering."""
    
    def __init__(self, world_data: Dict[str, Any]):
        self.world_data = world_data
        self.analysis_results = {}
    
    def analyze_complete_structure(self) -> Dict[str, Any]:
        """Complete analysis of world structure, objects, and gameplay potential."""
        terrain = self.world_data.get('terrain', {})
        objects = self.world_data.get('objects', [])
        metadata = self.world_data.get('metadata', {})
        
        analysis = {
            "metadata_analysis": self._analyze_metadata(metadata),
            "terrain_analysis": self._analyze_terrain(terrain),
            "object_analysis": self._analyze_objects(objects),
            "spatial_analysis": self._analyze_spatial_distribution(objects, terrain),
            "building_analysis": self._analyze_buildings(objects),
            "navigation_analysis": self._analyze_navigation_potential(objects, terrain),
            "gameplay_analysis": self._analyze_gameplay_potential(objects, terrain),
            "material_analysis": self._analyze_materials(objects),
            "complexity_score": self._calculate_complexity_score(objects),
            "recommendations": self._generate_recommendations(objects, terrain)
        }
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_metadata(self, metadata: Dict) -> Dict[str, Any]:
        """Analyze world metadata and generation info."""
        return {
            "scene_type": metadata.get('scene_type', 'unknown'),
            "generation_seed": metadata.get('seed', 'unknown'),
            "generation_time": metadata.get('generated_at', 'unknown'),
            "layer": metadata.get('layer', 'surface'),
            "has_terrain_zones": metadata.get('terrain_zones', 0) > 0,
            "agent_d_version": metadata.get('agent_d_version', 'unknown')
        }
    
    def _analyze_terrain(self, terrain: Dict) -> Dict[str, Any]:
        """Analyze terrain structure and properties."""
        terrain_type = terrain.get('type', 'unknown')
        radius = terrain.get('radius', 0)
        center = terrain.get('center', [0, 0, 0])
        
        analysis = {
            "type": terrain_type,
            "radius": radius,
            "diameter": radius * 2,
            "circumference": 2 * math.pi * radius if radius > 0 else 0,
            "surface_area": 4 * math.pi * radius**2 if radius > 0 else 0,
            "center": center,
            "material": terrain.get('material', 'unknown'),
            "is_navigable": terrain_type in ["sphere", "plane"],
            "height_map_available": "height_map" in terrain
        }
        
        if "height_map" in terrain:
            height_map = terrain["height_map"]
            analysis["height_zones"] = len(height_map) if isinstance(height_map, list) else 0
            analysis["terrain_variation"] = "high" if analysis["height_zones"] > 20 else "medium" if analysis["height_zones"] > 10 else "low"
        else:
            analysis["terrain_variation"] = "none"
        
        return analysis
    
    def _analyze_objects(self, objects: List[Dict]) -> Dict[str, Any]:
        """Analyze all objects in the world."""
        if not objects:
            return {"total_count": 0, "types": {}, "materials": {}}
        
        # Count by type
        type_counts = {}
        material_counts = {}
        size_stats = {"min": float('inf'), "max": 0, "avg": 0}
        
        total_volume = 0
        
        for obj in objects:
            obj_type = obj.get('type', 'unknown')
            material = obj.get('material', 'unknown')
            
            # Count types
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            # Count materials
            material_counts[material] = material_counts.get(material, 0) + 1
            
            # Calculate volume for size stats
            if obj_type == "CUBE":
                size = obj.get('size', [1, 1, 1])
                volume = size[0] * size[1] * size[2]
            elif obj_type == "SPHERE":
                radius = obj.get('radius', 1)
                volume = (4/3) * math.pi * radius**3
            elif obj_type == "MESH":
                # Estimate from footprint
                footprint = obj.get('size', [1, 1, 1])
                volume = footprint[0] * footprint[1] * footprint[2]
            else:
                volume = 1.0  # Default
            
            total_volume += volume
            size_stats["min"] = min(size_stats["min"], volume)
            size_stats["max"] = max(size_stats["max"], volume)
        
        size_stats["avg"] = total_volume / len(objects)
        size_stats["total_volume"] = total_volume
        
        return {
            "total_count": len(objects),
            "types": type_counts,
            "materials": material_counts,
            "size_statistics": size_stats,
            "density_per_unit_area": len(objects) / (4 * math.pi * 50**2),  # Assuming radius 50
            "dominant_type": max(type_counts.keys(), key=type_counts.get) if type_counts else "none",
            "material_diversity": len(material_counts),
            "has_complex_meshes": "MESH" in type_counts
        }
    
    def _analyze_buildings(self, objects: List[Dict]) -> Dict[str, Any]:
        """Analyze building and structure objects specifically."""
        buildings = [obj for obj in objects if obj.get('type') == 'MESH' or 
                    obj.get('material', '').startswith(('building_', 'structure_'))]
        
        if not buildings:
            return {"building_count": 0, "building_types": [], "component_analysis": {}}
        
        building_analysis = {
            "building_count": len(buildings),
            "building_types": [],
            "component_analysis": {},
            "material_usage": {},
            "size_distribution": []
        }
        
        total_components = 0
        
        for building in buildings:
            # Analyze building components
            components = building.get('components', [])
            if components:
                total_components += len(components)
                
                # Analyze component materials
                for comp in components:
                    material = comp.get('material', 'unknown')
                    building_analysis["material_usage"][material] = building_analysis["material_usage"].get(material, 0) + 1
                
                # Track building complexity
                building_analysis["component_analysis"][building.get('asset_id', 'unknown')] = {
                    "component_count": len(components),
                    "materials_used": building.get('materials_used', []),
                    "footprint": building.get('size', [1, 1, 1])
                }
            
            # Track building types
            material = building.get('material', '')
            if material.startswith('building_'):
                building_type = material.replace('building_', '')
                building_analysis["building_types"].append(building_type)
        
        building_analysis["average_components_per_building"] = total_components / len(buildings) if buildings else 0
        building_analysis["total_components"] = total_components
        building_analysis["complexity_level"] = ("high" if total_components > 100 else 
                                               "medium" if total_components > 50 else "low")
        
        return building_analysis
    
    def _analyze_spatial_distribution(self, objects: List[Dict], terrain: Dict) -> Dict[str, Any]:
        """Analyze how objects are distributed spatially."""
        if not objects:
            return {"distribution": "empty"}
        
        positions = [obj.get('pos', [0, 0, 0]) for obj in objects]
        radius = terrain.get('radius', 50)
        
        # Calculate distances from center
        distances = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in positions]
        
        # Analyze distribution patterns
        surface_objects = sum(1 for d in distances if abs(d - radius) < 5)  # Within 5 units of surface
        floating_objects = sum(1 for d in distances if d > radius + 5)
        underground_objects = sum(1 for d in distances if d < radius - 2)
        
        # Calculate spatial clustering
        clustering_score = self._calculate_clustering(positions)
        
        return {
            "surface_objects": surface_objects,
            "floating_objects": floating_objects,
            "underground_objects": underground_objects,
            "surface_adherence": surface_objects / len(objects),
            "clustering_score": clustering_score,
            "average_distance_from_center": sum(distances) / len(distances),
            "spatial_spread": max(distances) - min(distances),
            "distribution_quality": "good" if surface_objects / len(objects) > 0.8 else "poor"
        }
    
    def _analyze_navigation_potential(self, objects: List[Dict], terrain: Dict) -> Dict[str, Any]:
        """Analyze how navigable the world is."""
        radius = terrain.get('radius', 50)
        
        # Check for navigation landmarks
        beacons = sum(1 for obj in objects if 'beacon' in obj.get('material', ''))
        temples = sum(1 for obj in objects if 'temple' in obj.get('material', ''))
        landmarks = sum(1 for obj in objects if 'landmark' in obj.get('material', ''))
        
        # Check spawn safety
        spawn_zones = self._identify_safe_spawn_zones(objects, radius)
        
        return {
            "navigation_landmarks": beacons + temples + landmarks,
            "beacon_count": beacons,
            "temple_count": temples,
            "landmark_count": landmarks,
            "safe_spawn_zones": len(spawn_zones),
            "circumnavigability": "good" if terrain.get('type') == 'sphere' else "limited",
            "navigation_quality": ("excellent" if beacons > 0 and temples > 0 else
                                 "good" if landmarks > 5 else "basic")
        }
    
    def _analyze_gameplay_potential(self, objects: List[Dict], terrain: Dict) -> Dict[str, Any]:
        """Analyze potential for interesting gameplay."""
        resources = sum(1 for obj in objects if 'resource' in obj.get('material', ''))
        structures = sum(1 for obj in objects if 'structure' in obj.get('material', ''))
        caves = sum(1 for obj in objects if 'cave' in obj.get('material', ''))
        buildings = sum(1 for obj in objects if obj.get('type') == 'MESH')
        
        # Calculate landmarks count
        landmarks = sum(1 for obj in objects if 'landmark' in obj.get('material', ''))
        
        # Gameplay elements score
        exploration_score = (resources * 0.1 + caves * 0.3 + landmarks * 0.2) / len(objects) if objects else 0
        building_score = buildings / len(objects) if objects else 0
        variety_score = len(set(obj.get('material', '') for obj in objects)) / 20  # Normalize to materials
        
        overall_score = (exploration_score + building_score + variety_score) / 3
        
        return {
            "resource_nodes": resources,
            "building_opportunities": structures + buildings,
            "exploration_features": caves + landmarks,
            "exploration_score": exploration_score,
            "building_score": building_score,
            "variety_score": variety_score,
            "overall_gameplay_score": overall_score,
            "gameplay_rating": ("excellent" if overall_score > 0.7 else
                              "good" if overall_score > 0.4 else
                              "basic" if overall_score > 0.2 else "poor")
        }
    
    def _analyze_materials(self, objects: List[Dict]) -> Dict[str, Any]:
        """Analyze material usage and consistency."""
        materials = [obj.get('material', 'unknown') for obj in objects]
        material_counts = {}
        
        for material in materials:
            material_counts[material] = material_counts.get(material, 0) + 1
        
        # Categorize materials
        terrain_materials = sum(1 for m in materials if m.startswith('terrain_'))
        resource_materials = sum(1 for m in materials if m.startswith('resource_'))
        structure_materials = sum(1 for m in materials if m.startswith('structure_'))
        building_materials = sum(1 for m in materials if m.startswith('building_'))
        
        return {
            "total_materials": len(set(materials)),
            "material_distribution": material_counts,
            "terrain_materials": terrain_materials,
            "resource_materials": resource_materials,
            "structure_materials": structure_materials,
            "building_materials": building_materials,
            "dominant_material": max(material_counts.keys(), key=material_counts.get) if material_counts else "none",
            "material_balance": "good" if len(set(materials)) > 5 else "limited"
        }
    
    def _calculate_complexity_score(self, objects: List[Dict]) -> float:
        """Calculate overall world complexity score."""
        if not objects:
            return 0.0
        
        # Factors contributing to complexity
        object_count_score = min(len(objects) / 100, 1.0)  # Normalize to 100 objects
        type_diversity_score = len(set(obj.get('type') for obj in objects)) / 5  # Max 5 types
        material_diversity_score = len(set(obj.get('material') for obj in objects)) / 20  # Max 20 materials
        mesh_complexity_score = sum(1 for obj in objects if obj.get('type') == 'MESH') / len(objects)
        
        complexity = (object_count_score + type_diversity_score + 
                     material_diversity_score + mesh_complexity_score) / 4
        
        return min(complexity, 1.0)
    
    def _generate_recommendations(self, objects: List[Dict], terrain: Dict) -> List[str]:
        """Generate recommendations for world improvement."""
        recommendations = []
        
        if len(objects) < 20:
            recommendations.append("Add more objects to increase world density")
        
        mesh_objects = sum(1 for obj in objects if obj.get('type') == 'MESH')
        if mesh_objects == 0:
            recommendations.append("Add complex MESH buildings for more interesting architecture")
        
        materials = set(obj.get('material', '') for obj in objects)
        if len(materials) < 8:
            recommendations.append("Increase material variety for more visual diversity")
        
        resources = sum(1 for obj in objects if 'resource' in obj.get('material', ''))
        if resources < 5:
            recommendations.append("Add more resource nodes for gameplay incentives")
        
        if not any('beacon' in obj.get('material', '') for obj in objects):
            recommendations.append("Add navigation beacons for better wayfinding")
        
        return recommendations
    
    def _identify_safe_spawn_zones(self, objects: List[Dict], radius: float) -> List[Dict]:
        """Identify areas safe for player spawning."""
        # Simplified spawn zone identification
        zones = []
        for angle in range(0, 360, 30):  # Check every 30 degrees
            theta = math.radians(angle)
            test_pos = [radius * math.cos(theta), 0, radius * math.sin(theta)]
            
            # Check if area is clear of objects
            is_clear = True
            for obj in objects:
                obj_pos = obj.get('pos', [0, 0, 0])
                distance = math.sqrt(sum((a - b)**2 for a, b in zip(test_pos, obj_pos)))
                if distance < 3.0:  # Too close to object
                    is_clear = False
                    break
            
            if is_clear:
                zones.append({"position": test_pos, "angle": angle})
        
        return zones
    
    def _calculate_clustering(self, positions: List[List[float]]) -> float:
        """Calculate spatial clustering score."""
        if len(positions) < 2:
            return 0.0
        
        # Calculate average distance between objects
        total_distance = 0
        pair_count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1, pos2 = positions[i], positions[j]
                distance = math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))
                total_distance += distance
                pair_count += 1
        
        average_distance = total_distance / pair_count if pair_count > 0 else 0
        
        # Normalize clustering score (lower distance = higher clustering)
        clustering_score = max(0, 1.0 - (average_distance / 50))  # Normalize to planet size
        
        return clustering_score
    
    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        if not self.analysis_results:
            self.analyze_complete_structure()
        
        report_lines = []
        report_lines.append("🌍 HEADLESS WORLD ANALYSIS REPORT")
        report_lines.append("=" * 50)
        
        # Metadata section
        meta = self.analysis_results["metadata_analysis"]
        report_lines.append(f"📊 Scene Type: {meta['scene_type']}")
        report_lines.append(f"🌱 Generation Seed: {meta['generation_seed']}")
        report_lines.append(f"⏰ Generated: {meta['generation_time']}")
        report_lines.append("")
        
        # Terrain section
        terrain = self.analysis_results["terrain_analysis"]
        report_lines.append("🏔️ TERRAIN ANALYSIS")
        report_lines.append(f"   Type: {terrain['type']}")
        report_lines.append(f"   Radius: {terrain['radius']:.1f} units")
        report_lines.append(f"   Surface Area: {terrain['surface_area']:.0f} square units")
        report_lines.append(f"   Navigability: {'✅' if terrain['is_navigable'] else '❌'}")
        report_lines.append(f"   Terrain Variation: {terrain['terrain_variation']}")
        report_lines.append("")
        
        # Objects section
        objects = self.analysis_results["object_analysis"]
        report_lines.append("🎯 OBJECT ANALYSIS")
        report_lines.append(f"   Total Objects: {objects['total_count']}")
        report_lines.append(f"   Dominant Type: {objects['dominant_type']}")
        report_lines.append(f"   Material Diversity: {objects['material_diversity']} types")
        report_lines.append(f"   Complex Meshes: {'✅' if objects['has_complex_meshes'] else '❌'}")
        report_lines.append("")
        
        # Buildings section
        buildings = self.analysis_results["building_analysis"]
        if buildings["building_count"] > 0:
            report_lines.append("🏠 BUILDING ANALYSIS")
            report_lines.append(f"   Building Count: {buildings['building_count']}")
            report_lines.append(f"   Total Components: {buildings['total_components']}")
            report_lines.append(f"   Complexity Level: {buildings['complexity_level']}")
            report_lines.append("")
        
        # Navigation section
        nav = self.analysis_results["navigation_analysis"]
        report_lines.append("🧭 NAVIGATION ANALYSIS")
        report_lines.append(f"   Navigation Quality: {nav['navigation_quality']}")
        report_lines.append(f"   Landmarks: {nav['navigation_landmarks']}")
        report_lines.append(f"   Safe Spawn Zones: {nav['safe_spawn_zones']}")
        report_lines.append("")
        
        # Gameplay section
        gameplay = self.analysis_results["gameplay_analysis"]
        report_lines.append("🎮 GAMEPLAY ANALYSIS")
        report_lines.append(f"   Overall Rating: {gameplay['gameplay_rating']}")
        report_lines.append(f"   Resource Nodes: {gameplay['resource_nodes']}")
        report_lines.append(f"   Exploration Features: {gameplay['exploration_features']}")
        report_lines.append("")
        
        # Recommendations section
        recommendations = self.analysis_results["recommendations"]
        if recommendations:
            report_lines.append("💡 RECOMMENDATIONS")
            for rec in recommendations:
                report_lines.append(f"   • {rec}")
            report_lines.append("")
        
        # Summary section
        complexity = self.analysis_results["complexity_score"]
        report_lines.append("📋 SUMMARY")
        report_lines.append(f"   Complexity Score: {complexity:.2f}/1.0")
        report_lines.append(f"   World Quality: {'Excellent' if complexity > 0.8 else 'Good' if complexity > 0.6 else 'Moderate' if complexity > 0.4 else 'Basic'}")
        
        return "\n".join(report_lines)

def analyze_world_headless(world_file: str) -> Dict[str, Any]:
    """Analyze world file without rendering dependencies."""
    try:
        with open(world_file, 'r') as f:
            world_data = json.load(f)
        
        analyzer = HeadlessWorldAnalyzer(world_data)
        return analyzer.analyze_complete_structure()
    except Exception as e:
        return {'error': str(e), 'success': False}

def generate_world_report(world_file: str, output_file: Optional[str] = None) -> str:
    """Generate and optionally save detailed world report."""
    try:
        with open(world_file, 'r') as f:
            world_data = json.load(f)
        
        analyzer = HeadlessWorldAnalyzer(world_data)
        analyzer.analyze_complete_structure()
        report = analyzer.generate_text_report()
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    except Exception as e:
        return f"❌ Error generating report: {e}"

def generate_simulated_feedback(world_file: str) -> Dict[str, str]:
    """Generate simulated human feedback based on headless analysis."""
    try:
        analysis = analyze_world_headless(world_file)
        
        if 'error' in analysis:
            # Return poor feedback if analysis failed
            return {
                'spherical_mechanics': 'broken',
                'curved_horizon': 'no', 
                'walk_around_planet': 'no',
                'planet_size': 'broken',
                'visual_quality': 'poor',
                'building_interaction': 'broken',
                'gameplay_clarity': 'poor',
                'overall_experience': 'poor',
                'rating': '2'
            }
        
        # Extract key analysis results
        terrain = analysis.get('terrain_analysis', {})
        objects = analysis.get('object_analysis', {})
        buildings = analysis.get('building_analysis', {})
        navigation = analysis.get('navigation_analysis', {})
        gameplay = analysis.get('gameplay_analysis', {})
        complexity = analysis.get('complexity_score', 0)
        
        feedback = {}
        
        # Spherical mechanics based on terrain
        if terrain.get('type') == 'sphere' and terrain.get('is_navigable'):
            feedback['spherical_mechanics'] = 'excellent'
        else:
            feedback['spherical_mechanics'] = 'poor'
            
        # Curved horizon visibility
        if terrain.get('type') == 'sphere':
            feedback['curved_horizon'] = 'yes'
        else:
            feedback['curved_horizon'] = 'no'
            
        # Walk around planet
        if terrain.get('type') == 'sphere' and navigation.get('navigation_quality') in ['excellent', 'good']:
            feedback['walk_around_planet'] = 'yes'
        else:
            feedback['walk_around_planet'] = 'partially'
            
        # Planet size assessment
        radius = terrain.get('radius', 0)
        if radius < 30:
            feedback['planet_size'] = 'too_small'
        elif radius > 80:
            feedback['planet_size'] = 'too_large' 
        else:
            feedback['planet_size'] = 'perfect'
            
        # Visual quality based on complexity and diversity
        if complexity > 0.6 and objects.get('material_diversity', 0) > 8:
            feedback['visual_quality'] = 'excellent'
        elif complexity > 0.4:
            feedback['visual_quality'] = 'good'
        else:
            feedback['visual_quality'] = 'basic'
            
        # Building interaction
        building_count = buildings.get('building_count', 0)
        if building_count > 3:
            feedback['building_interaction'] = 'excellent'
        elif building_count > 0:
            feedback['building_interaction'] = 'good'
        else:
            feedback['building_interaction'] = 'limited'
            
        # Gameplay clarity
        gameplay_rating = gameplay.get('gameplay_rating', 'poor')
        if gameplay_rating == 'excellent':
            feedback['gameplay_clarity'] = 'excellent'
        elif gameplay_rating in ['good', 'basic']:
            feedback['gameplay_clarity'] = 'good'
        else:
            feedback['gameplay_clarity'] = 'unclear'
            
        # Overall experience
        if complexity > 0.5 and terrain.get('type') == 'sphere':
            feedback['overall_experience'] = 'excellent'
        elif complexity > 0.3:
            feedback['overall_experience'] = 'good'
        else:
            feedback['overall_experience'] = 'basic'
            
        # Rating (2-10 scale)
        base_rating = 2 + int(complexity * 8)
        if terrain.get('type') == 'sphere':
            base_rating += 1  # Bonus for spherical worlds
        feedback['rating'] = str(min(10, max(2, base_rating)))
        
        return feedback
        
    except Exception as e:
        print(f"❌ Error generating simulated feedback: {e}")
        return {
            'spherical_mechanics': 'broken',
            'curved_horizon': 'no',
            'walk_around_planet': 'no', 
            'planet_size': 'broken',
            'visual_quality': 'poor',
            'building_interaction': 'broken',
            'gameplay_clarity': 'poor',
            'overall_experience': 'poor',
            'rating': '2'
        }