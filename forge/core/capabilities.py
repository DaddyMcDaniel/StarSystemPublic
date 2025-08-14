"""System capability detection and operation mode selection for Week 4."""
import importlib
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class CapabilityDetector:
    """Detects available system capabilities and selects optimal operation mode."""
    
    def __init__(self):
        self.capabilities = self._detect_capabilities()
        self.operation_mode = self._select_operation_mode()
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect available libraries and system capabilities."""
        caps = {}
        
        # OpenGL detection (full check)
        try:
            import OpenGL.GL
            import OpenGL.GLU
            import OpenGL.GLUT
            caps['opengl'] = True
        except ImportError:
            caps['opengl'] = False
        
        # PIL detection
        try:
            import PIL.Image
            import PIL.ImageDraw
            caps['pil'] = True
        except ImportError:
            caps['pil'] = False
        
        # Matplotlib detection
        try:
            import matplotlib.pyplot
            import matplotlib.patches
            caps['matplotlib'] = True
        except ImportError:
            caps['matplotlib'] = False
        
        # Trimesh for mesh generation
        try:
            import trimesh
            caps['trimesh'] = True
        except ImportError:
            caps['trimesh'] = False
        
        # Display check (Linux/Unix)
        caps['display'] = bool(os.environ.get('DISPLAY'))
        
        # Interactive terminal check
        caps['interactive'] = hasattr(sys.stdin, 'isatty') and sys.stdin.isatty()
        
        return caps
    
    def _select_operation_mode(self) -> str:
        """Select best available operation mode based on capabilities."""
        if self.capabilities['opengl'] and self.capabilities['display']:
            return 'full'
        elif self.capabilities['pil'] or self.capabilities['matplotlib']:
            return '2d_fallback'
        elif self.capabilities['interactive']:
            return 'headless_interactive'
        else:
            return 'headless_batch'
    
    def get_mode_description(self) -> str:
        """Get human-readable description of current mode."""
        descriptions = {
            'full': 'Full 3D rendering with OpenGL',
            '2d_fallback': '2D visualization fallback mode',
            'headless_interactive': 'Headless analysis with interactive prompts',
            'headless_batch': 'Headless batch processing mode'
        }
        return descriptions.get(self.operation_mode, 'Unknown mode')
    
    def get_missing_capabilities(self) -> List[str]:
        """Get list of missing capabilities for diagnostics."""
        missing = []
        if not self.capabilities['opengl']:
            missing.append('OpenGL (PyOpenGL package)')
        if not self.capabilities['display']:
            missing.append('Display server (DISPLAY environment variable)')
        if not self.capabilities['pil']:
            missing.append('PIL/Pillow for 2D fallback')
        if not self.capabilities['matplotlib']:
            missing.append('Matplotlib for 2D plotting')
        if not self.capabilities['trimesh']:
            missing.append('Trimesh for mesh generation')
        return missing
    
    def can_run_3d_viewer(self) -> bool:
        """Check if 3D viewer can run."""
        return self.capabilities['opengl'] and self.capabilities['display']
    
    def can_generate_meshes(self) -> bool:
        """Check if mesh generation is possible."""
        return self.capabilities['trimesh']
    
    def can_create_2d_visualization(self) -> bool:
        """Check if 2D visualization is possible."""
        return self.capabilities['pil'] or self.capabilities['matplotlib']
    
    def get_recommended_install_commands(self) -> List[str]:
        """Get installation commands for missing capabilities."""
        commands = []
        missing = self.get_missing_capabilities()
        
        if 'OpenGL (PyOpenGL package)' in missing:
            commands.append('pip install PyOpenGL PyOpenGL_accelerate')
        if 'PIL/Pillow for 2D fallback' in missing:
            commands.append('pip install Pillow')
        if 'Matplotlib for 2D plotting' in missing:
            commands.append('pip install matplotlib')
        if 'Trimesh for mesh generation' in missing:
            commands.append('pip install trimesh')
        if 'Display server (DISPLAY environment variable)' in missing:
            commands.append('# Set up X11 forwarding or local display server')
        
        return commands

# Global instance
detector = CapabilityDetector()

def check_capabilities_and_recommend() -> None:
    """Check capabilities and print recommendations if needed."""
    print(f"🔍 System Capabilities Check")
    print(f"📋 Operation Mode: {detector.get_mode_description()}")
    print(f"✅ Available: OpenGL={detector.capabilities['opengl']}, "
          f"Display={detector.capabilities['display']}, "
          f"PIL={detector.capabilities['pil']}, "
          f"Trimesh={detector.capabilities['trimesh']}")
    
    missing = detector.get_missing_capabilities()
    if missing:
        print(f"⚠️ Missing capabilities: {', '.join(missing)}")
        print("📦 Installation commands:")
        for cmd in detector.get_recommended_install_commands():
            print(f"   {cmd}")
        print()
    else:
        print("✅ All capabilities available!")
        print()
