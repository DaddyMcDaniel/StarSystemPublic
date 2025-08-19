#!/usr/bin/env python3
"""
Test Script Cleanup Utility
============================

Consolidates and manages test scripts for T06-T10 systems.
Use this to clean up temporary test files once features are stable.

Usage:
    python cleanup_test_scripts.py --list      # List all test scripts
    python cleanup_test_scripts.py --cleanup   # Remove temporary test files  
    python cleanup_test_scripts.py --run-core  # Run core validation tests
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

def find_test_scripts():
    """Find all test scripts in the agent_d directory"""
    agent_d_path = Path(__file__).parent
    test_scripts = {
        'core': [],
        'comprehensive': [],
        'temporary': []
    }
    
    # Core validation scripts (keep these)
    core_patterns = [
        'test_t*_core.py',
        'test_integration.py', 
        'test_vm_execution.py'
    ]
    
    # Comprehensive test suites (can be cleaned up)
    comprehensive_patterns = [
        'test_*_system.py',
        'test_marching_cubes.py',
        'test_runtime_lod.py', 
        'test_crack_prevention.py'
    ]
    
    # Temporary/generated test files
    temp_patterns = [
        'test_chunk_*.py',
        'test_temp_*.py',
        'test_debug_*.py'
    ]
    
    for pattern in core_patterns:
        test_scripts['core'].extend(agent_d_path.rglob(pattern))
    
    for pattern in comprehensive_patterns:
        test_scripts['comprehensive'].extend(agent_d_path.rglob(pattern))
        
    for pattern in temp_patterns:
        test_scripts['temporary'].extend(agent_d_path.rglob(pattern))
    
    return test_scripts

def list_test_scripts():
    """List all test scripts organized by category"""
    scripts = find_test_scripts()
    
    print("🧪 Test Script Inventory")
    print("=" * 40)
    
    for category, files in scripts.items():
        if files:
            print(f"\n📋 {category.title()} Tests:")
            for file in files:
                rel_path = file.relative_to(Path(__file__).parent)
                size_kb = file.stat().st_size / 1024
                print(f"   {rel_path} ({size_kb:.1f} KB)")
    
    total_files = sum(len(files) for files in scripts.values())
    print(f"\nTotal test scripts: {total_files}")

def cleanup_test_scripts():
    """Remove temporary and comprehensive test scripts"""
    scripts = find_test_scripts()
    
    print("🧹 Cleaning up test scripts...")
    
    files_to_remove = scripts['temporary'] + scripts['comprehensive']
    
    if not files_to_remove:
        print("   No files to clean up")
        return
    
    print(f"   Found {len(files_to_remove)} files to remove:")
    
    for file in files_to_remove:
        rel_path = file.relative_to(Path(__file__).parent)
        try:
            file.unlink()
            print(f"   ✅ Removed {rel_path}")
        except Exception as e:
            print(f"   ❌ Failed to remove {rel_path}: {e}")
    
    # Also remove any test output directories
    test_dirs = ['test_chunks*', 'test_cave*', 'test_t*_output']
    agent_d_path = Path(__file__).parent
    
    for pattern in test_dirs:
        for test_dir in agent_d_path.glob(pattern):
            if test_dir.is_dir():
                try:
                    shutil.rmtree(test_dir)
                    print(f"   ✅ Removed directory {test_dir.name}")
                except Exception as e:
                    print(f"   ❌ Failed to remove {test_dir.name}: {e}")

def run_core_tests():
    """Run core validation tests to ensure systems are working"""
    print("🚀 Running Core Validation Tests")
    print("=" * 40)
    
    scripts = find_test_scripts()
    core_tests = scripts['core']
    
    if not core_tests:
        print("   No core tests found")
        return False
    
    passed = 0
    for test_script in core_tests:
        rel_path = test_script.relative_to(Path(__file__).parent)
        print(f"\n🔍 Running {rel_path}...")
        
        try:
            # Change to script directory and run
            original_cwd = os.getcwd()
            os.chdir(test_script.parent)
            
            result = os.system(f"python {test_script.name}")
            
            os.chdir(original_cwd)
            
            if result == 0:
                print(f"   ✅ {rel_path} passed")
                passed += 1
            else:
                print(f"   ❌ {rel_path} failed")
                
        except Exception as e:
            print(f"   ❌ {rel_path} error: {e}")
    
    print(f"\n📊 Core Tests: {passed}/{len(core_tests)} passed")
    return passed == len(core_tests)

def main():
    """Main cleanup utility function"""
    parser = argparse.ArgumentParser(description="Test script cleanup utility")
    parser.add_argument('--list', action='store_true', help='List all test scripts')
    parser.add_argument('--cleanup', action='store_true', help='Remove temporary test files')
    parser.add_argument('--run-core', action='store_true', help='Run core validation tests')
    
    args = parser.parse_args()
    
    if args.list:
        list_test_scripts()
    elif args.cleanup:
        cleanup_test_scripts()
    elif args.run_core:
        success = run_core_tests()
        sys.exit(0 if success else 1)
    else:
        print("Use --help for available options")
        list_test_scripts()

if __name__ == "__main__":
    main()