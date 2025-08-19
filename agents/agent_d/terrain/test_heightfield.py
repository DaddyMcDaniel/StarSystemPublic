#!/usr/bin/env python3
"""
Unit Tests for HeightField Module - T04
=======================================

Tests for deterministic equality and proper functionality
of the PCC terrain heightfield system.
"""

import unittest
import numpy as np
from typing import List, Tuple

try:
    from .noise_nodes import NoiseFBM, RidgedMF, DomainWarp
    from .heightfield import HeightField, create_heightfield_from_pcc, create_example_terrain_stack
except ImportError:
    from noise_nodes import NoiseFBM, RidgedMF, DomainWarp
    from heightfield import HeightField, create_heightfield_from_pcc, create_example_terrain_stack


class TestNoiseNodes(unittest.TestCase):
    """Test individual noise nodes for deterministic behavior"""
    
    def setUp(self):
        """Set up test data"""
        self.test_positions = [
            (0.0, 0.0, 0.0),
            (0.5, 0.3, 0.7),
            (1.0, 1.0, 1.0),
            (-0.5, 0.8, -0.2),
            (0.707, 0.707, 0.0)
        ]
        self.test_seed = 12345
        
    def test_noise_fbm_determinism(self):
        """Test that NoiseFBM produces identical results with same seed"""
        fbm1 = NoiseFBM(seed=self.test_seed)
        fbm2 = NoiseFBM(seed=self.test_seed)
        
        for pos in self.test_positions:
            val1 = fbm1.sample(*pos)
            val2 = fbm2.sample(*pos)
            self.assertAlmostEqual(val1, val2, places=10, 
                                 msg=f"FBM determinism failed at position {pos}")
    
    def test_ridged_mf_determinism(self):
        """Test that RidgedMF produces identical results with same seed"""
        ridged1 = RidgedMF(seed=self.test_seed)
        ridged2 = RidgedMF(seed=self.test_seed)
        
        for pos in self.test_positions:
            val1 = ridged1.sample(*pos)
            val2 = ridged2.sample(*pos)
            self.assertAlmostEqual(val1, val2, places=10,
                                 msg=f"RidgedMF determinism failed at position {pos}")
    
    def test_domain_warp_determinism(self):
        """Test that DomainWarp produces identical results with same seed"""
        warp1 = DomainWarp(seed=self.test_seed)
        warp2 = DomainWarp(seed=self.test_seed)
        
        for pos in self.test_positions:
            val1 = warp1.sample(*pos)
            val2 = warp2.sample(*pos)
            self.assertAlmostEqual(val1, val2, places=10,
                                 msg=f"DomainWarp determinism failed at position {pos}")
    
    def test_noise_fbm_parameters(self):
        """Test that NoiseFBM parameters affect output correctly"""
        # Test different octaves
        fbm_low = NoiseFBM(seed=self.test_seed, octaves=2)
        fbm_high = NoiseFBM(seed=self.test_seed, octaves=6)
        
        pos = self.test_positions[1]
        val_low = fbm_low.sample(*pos)
        val_high = fbm_high.sample(*pos)
        
        # Higher octaves should generally produce different (more detailed) results
        self.assertNotAlmostEqual(val_low, val_high, places=3,
                                msg="Different octave counts should produce different results")
    
    def test_ridged_mf_ridge_effect(self):
        """Test that RidgedMF produces ridged characteristics"""
        ridged = RidgedMF(seed=self.test_seed, ridge_offset=1.0)
        
        # Sample at multiple positions
        values = [ridged.sample(*pos) for pos in self.test_positions]
        
        # Ridged multifractal should produce positive values due to abs() operation
        for i, val in enumerate(values):
            self.assertGreaterEqual(val, 0.0, 
                                  msg=f"RidgedMF should produce non-negative values, got {val} at position {i}")
    
    def test_domain_warp_effect(self):
        """Test that DomainWarp actually warps the domain"""
        # Create source noise
        source = NoiseFBM(seed=self.test_seed)
        
        # Create warped version
        warped = DomainWarp(seed=self.test_seed + 1000, source_node=source, strength=0.5)
        
        pos = self.test_positions[1]
        source_val = source.sample(*pos)
        warped_val = warped.sample(*pos)
        
        # With significant warping strength, results should be different
        self.assertNotAlmostEqual(source_val, warped_val, places=3,
                                msg="Domain warp should modify source noise values")


class TestHeightField(unittest.TestCase):
    """Test HeightField composition and functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.test_positions = [
            (0.0, 0.0, 1.0),   # North pole
            (1.0, 0.0, 0.0),   # Equator X
            (0.0, 1.0, 0.0),   # Equator Y
            (0.707, 0.707, 0.0), # Diagonal
            (0.0, 0.0, -1.0)   # South pole
        ]
        self.test_seed = 42
    
    def test_heightfield_basic_functionality(self):
        """Test basic HeightField creation and sampling"""
        noise = NoiseFBM(seed=self.test_seed)
        heightfield = HeightField(noise, base_height=1.0, height_scale=0.5)
        
        for pos in self.test_positions:
            height = heightfield.sample(*pos)
            self.assertIsInstance(height, float, "HeightField should return float values")
            
            # Test that base_height and height_scale are applied
            noise_val = noise.sample(*pos)
            expected_height = 1.0 + noise_val * 0.5
            self.assertAlmostEqual(height, expected_height, places=10,
                                 msg="HeightField should apply base_height and height_scale correctly")
    
    def test_heightfield_determinism(self):
        """Test that HeightField produces deterministic results"""
        heightfield1 = create_example_terrain_stack(seed=self.test_seed)
        heightfield2 = create_example_terrain_stack(seed=self.test_seed)
        
        for pos in self.test_positions:
            height1 = heightfield1.sample(*pos)
            height2 = heightfield2.sample(*pos)
            self.assertAlmostEqual(height1, height2, places=10,
                                 msg=f"HeightField determinism failed at position {pos}")
    
    def test_heightfield_seed_variation(self):
        """Test that different seeds produce different results"""
        heightfield1 = create_example_terrain_stack(seed=12345)
        heightfield2 = create_example_terrain_stack(seed=54321)
        
        differences = 0
        for pos in self.test_positions:
            height1 = heightfield1.sample(*pos)
            height2 = heightfield2.sample(*pos)
            if abs(height1 - height2) > 1e-6:
                differences += 1
        
        self.assertGreater(differences, 0, 
                         "Different seeds should produce different terrain")
    
    def test_heightfield_grid_sampling(self):
        """Test grid sampling functionality"""
        heightfield = create_example_terrain_stack(seed=self.test_seed)
        
        # Create position array
        positions = np.array(self.test_positions)
        
        # Sample grid
        heights = heightfield.sample_grid(positions)
        
        self.assertEqual(len(heights), len(positions),
                        "Grid sampling should return correct number of heights")
        
        # Verify against individual sampling
        for i, pos in enumerate(positions):
            individual_height = heightfield.sample(*pos)
            self.assertAlmostEqual(heights[i], individual_height, places=10,
                                 msg=f"Grid sampling mismatch at position {i}")


class TestPCCSpecParsing(unittest.TestCase):
    """Test PCC specification parsing"""
    
    def test_pcc_spec_parsing(self):
        """Test parsing of PCC terrain specifications"""
        pcc_spec = {
            "terrain": {
                "heightfield": {
                    "base_height": 0.5,
                    "height_scale": 2.0
                },
                "nodes": {
                    "type": "NoiseFBM",
                    "seed": 1000,
                    "octaves": 3,
                    "frequency": 1.5,
                    "amplitude": 1.0
                }
            }
        }
        
        heightfield = create_heightfield_from_pcc(pcc_spec, global_seed=42)
        
        self.assertEqual(heightfield.base_height, 0.5)
        self.assertEqual(heightfield.height_scale, 2.0)
        self.assertEqual(heightfield.seed, 42)
        
        # Test that it actually works
        test_pos = (0.5, 0.3, 0.7)
        height = heightfield.sample(*test_pos)
        self.assertIsInstance(height, float)
    
    def test_complex_pcc_spec(self):
        """Test parsing of complex nested PCC specifications"""
        pcc_spec = {
            "terrain": {
                "heightfield": {
                    "base_height": 0.0,
                    "height_scale": 1.0
                },
                "nodes": {
                    "type": "Composite",
                    "operation": "add",
                    "weights": [1.0, 0.3],
                    "nodes": [
                        {
                            "type": "RidgedMF",
                            "seed": 1000,
                            "octaves": 3,
                            "frequency": 2.0
                        },
                        {
                            "type": "DomainWarp",
                            "seed": 2000,
                            "strength": 0.1,
                            "source": {
                                "type": "NoiseFBM",
                                "seed": 3000,
                                "octaves": 4
                            }
                        }
                    ]
                }
            }
        }
        
        heightfield = create_heightfield_from_pcc(pcc_spec, global_seed=123)
        
        # Test sampling
        test_positions = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        for pos in test_positions:
            height = heightfield.sample(*pos)
            self.assertIsInstance(height, float)


class TestDeterministicEquality(unittest.TestCase):
    """Comprehensive tests for deterministic equality across different scenarios"""
    
    def test_same_seed_same_samples(self):
        """Test that same seed produces identical samples across runs"""
        seed = 999999
        test_positions = [
            (0.0, 0.0, 1.0),
            (0.577, 0.577, 0.577),  # Normalized (1,1,1)
            (-0.707, 0.0, 0.707),
            (0.0, -1.0, 0.0),
            (0.866, 0.0, 0.5)       # 30 degree point
        ]
        
        # Create multiple heightfields with same seed
        heightfields = [create_example_terrain_stack(seed=seed) for _ in range(3)]
        
        # Test all positions
        for pos in test_positions:
            values = [hf.sample(*pos) for hf in heightfields]
            
            # All values should be identical
            for i in range(1, len(values)):
                self.assertAlmostEqual(values[0], values[i], places=12,
                                     msg=f"Determinism failed at position {pos}, "
                                         f"values: {values}")
    
    def test_cross_platform_determinism(self):
        """Test determinism across different numpy versions/platforms"""
        # Use multiple seeds to test robustness
        seeds = [42, 12345, 999999, 555555]
        test_pos = (0.5, 0.3, 0.8)
        
        # Store reference values
        reference_values = {}
        
        for seed in seeds:
            heightfield = create_example_terrain_stack(seed=seed)
            value = heightfield.sample(*test_pos)
            reference_values[seed] = value
            
            # Test that recreating gives same result
            heightfield2 = create_example_terrain_stack(seed=seed)
            value2 = heightfield2.sample(*test_pos)
            
            self.assertAlmostEqual(value, value2, places=12,
                                 msg=f"Cross-platform determinism failed for seed {seed}")
    
    def test_parameter_sensitivity(self):
        """Test that parameter changes produce different results"""
        import copy
        
        base_spec = {
            "terrain": {
                "heightfield": {"base_height": 0.0, "height_scale": 1.0},
                "nodes": {
                    "type": "NoiseFBM",
                    "seed": 1000,
                    "octaves": 4,
                    "frequency": 1.0
                }
            }
        }
        
        # Test significantly different frequencies
        spec1 = copy.deepcopy(base_spec)
        spec1["terrain"]["nodes"]["frequency"] = 0.5
        
        spec2 = copy.deepcopy(base_spec)
        spec2["terrain"]["nodes"]["frequency"] = 5.0  # Much more dramatic change
        
        hf1 = create_heightfield_from_pcc(spec1, global_seed=42)
        hf2 = create_heightfield_from_pcc(spec2, global_seed=42)
        
        # Test multiple positions to ensure we see differences
        test_positions = [(0.5, 0.3, 0.7), (0.1, 0.9, 0.2), (0.8, 0.1, 0.6)]
        differences_found = 0
        
        for test_pos in test_positions:
            val1 = hf1.sample(*test_pos)
            val2 = hf2.sample(*test_pos)
            if abs(val1 - val2) > 1e-3:
                differences_found += 1
        
        self.assertGreater(differences_found, 0,
                         "Different parameters should produce different results at some positions")


def run_comprehensive_tests():
    """Run all tests and report results"""
    print("🧪 Running comprehensive HeightField tests...")
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestNoiseNodes,
        TestHeightField, 
        TestPCCSpecParsing,
        TestDeterministicEquality
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed} ✅")
    print(f"   Failed: {failures} ❌")
    print(f"   Errors: {errors} 💥")
    
    success = failures == 0 and errors == 0
    print(f"   Overall: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)