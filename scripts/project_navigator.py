#!/usr/bin/env python3
"""
SUMMARY: Project Navigator & Context Management Tool v1
======================================================
Comprehensive tool for navigating and understanding the StarSystem codebase structure.
Provides quick access to file summaries, dependencies, and project organization.

KEY FEATURES:
- File summary extraction from <100-line headers across all source files
- Dependency mapping between schemas, tools, agents, and configurations
- Quick search by file type, purpose, or related functionality
- Project structure visualization with component relationships
- Context-aware file recommendations based on current work area

COMMANDS:
- summary: Extract and display file summaries with filtering options
- deps: Show dependency relationships between project components
- search: Find files by content, purpose, or relationship patterns
- structure: Visualize project organization and component hierarchy
- context: Get relevant files for specific development tasks

USAGE:
  python scripts/project_navigator.py summary --type schema
  python scripts/project_navigator.py deps --file mcp_server/server.py
  python scripts/project_navigator.py search --purpose "building system"
  python scripts/project_navigator.py context --task "agent development"

OUTPUT:
- Formatted summaries with file paths and key information
- Dependency graphs showing component relationships
- Contextual file recommendations for development workflows

RELATED FILES:
- All project files with summary headers (identified automatically)
- Complements human feedback CLI and schema validation workflows
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

class ProjectNavigator:
    def __init__(self, root_path: str = "."):
        self.root = Path(root_path).resolve()
        self.file_summaries = {}
        self.dependencies = defaultdict(set)
        self.reverse_deps = defaultdict(set)
        self._load_project_data()
    
    def _load_project_data(self):
        """Scan project and extract file summaries and dependencies."""
        print(f"🔍 Scanning project at {self.root}...")
        
        # Patterns for different file types
        code_patterns = ['*.py', '*.js', '*.ts', '*.cpp', '*.h']
        config_patterns = ['*.yaml', '*.yml', '*.json']
        doc_patterns = ['*.md', '*.txt']
        
        all_files = []
        for pattern in code_patterns + config_patterns + doc_patterns:
            all_files.extend(self.root.rglob(pattern))
        
        # Filter out common ignore patterns
        ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'build', 'dist'}
        
        for file_path in all_files:
            if any(ignore_dir in file_path.parts for ignore_dir in ignore_dirs):
                continue
                
            if file_path.is_file():
                self._extract_file_info(file_path)
        
        print(f"📊 Found {len(self.file_summaries)} files with summaries")
    
    def _extract_file_info(self, file_path: Path):
        """Extract summary and dependencies from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract summary from first 100 lines
            lines = content.split('\n')[:100]
            summary = self._extract_summary(lines, file_path.suffix)
            
            if summary:
                rel_path = file_path.relative_to(self.root)
                self.file_summaries[str(rel_path)] = {
                    'path': rel_path,
                    'summary': summary,
                    'type': self._classify_file(file_path),
                    'size_lines': len(content.split('\n')),
                    'modified': file_path.stat().st_mtime
                }
                
                # Extract dependencies
                deps = self._extract_dependencies(content, file_path)
                for dep in deps:
                    self.dependencies[str(rel_path)].add(dep)
                    self.reverse_deps[dep].add(str(rel_path))
        
        except Exception as e:
            # Skip files that can't be read
            pass
    
    def _extract_summary(self, lines: List[str], file_ext: str) -> Optional[Dict]:
        """Extract SUMMARY block from file header."""
        summary_start = None
        summary_lines = []
        
        # Look for SUMMARY marker in first 20 lines
        for i, line in enumerate(lines[:20]):
            if 'SUMMARY:' in line:
                summary_start = i
                break
        
        if summary_start is None:
            return None
        
        # Extract summary block
        in_summary = False
        title = ""
        description = ""
        key_features = []
        usage = []
        related_files = []
        
        for line in lines[summary_start:summary_start+50]:  # Max 50 lines of summary
            line = line.strip()
            
            # Remove comment markers
            line = re.sub(r'^[#\/\*\"\'\'\'\']+\s*', '', line)
            line = re.sub(r'\*\/$', '', line)
            
            if not line:
                continue
            
            if 'SUMMARY:' in line:
                title = line.split('SUMMARY:', 1)[1].strip()
                in_summary = True
                continue
            
            if line.startswith('=') and in_summary:
                continue  # Skip separator lines
            
            if any(keyword in line.upper() for keyword in ['KEY FEATURES:', 'KEY COMPONENTS:', 'TOOLS IMPLEMENTED:']):
                current_section = 'features'
                continue
            elif any(keyword in line.upper() for keyword in ['USAGE:', 'COMMANDS:']):
                current_section = 'usage'
                continue
            elif 'RELATED FILES:' in line.upper():
                current_section = 'related'
                continue
            elif line.startswith('"""') or line.startswith("'''"):
                break  # End of docstring
            
            # Categorize content
            if 'current_section' not in locals():
                if description == "":
                    description = line
                else:
                    description += " " + line
            elif current_section == 'features' and line.startswith('-'):
                key_features.append(line[1:].strip())
            elif current_section == 'usage' and line.strip():
                usage.append(line.strip())
            elif current_section == 'related' and line.startswith('-'):
                related_files.append(line[1:].strip())
        
        return {
            'title': title,
            'description': description[:200] + '...' if len(description) > 200 else description,
            'key_features': key_features[:5],  # Limit to top 5
            'usage': usage[:3],  # Limit to top 3
            'related_files': related_files[:5]  # Limit to top 5
        }
    
    def _classify_file(self, file_path: Path) -> str:
        """Classify file by type and purpose."""
        path_parts = file_path.parts
        suffix = file_path.suffix
        
        if 'schema' in path_parts:
            return 'schema'
        elif 'agent' in path_parts or 'agents' in path_parts:
            return 'agent'
        elif 'mcp' in path_parts:
            return 'mcp_tool'
        elif suffix in ['.yaml', '.yml']:
            return 'config'
        elif 'test' in file_path.name or 'test' in path_parts:
            return 'test'
        elif suffix == '.py':
            return 'script'
        elif suffix == '.md':
            return 'documentation'
        elif suffix in ['.cpp', '.h']:
            return 'native_code'
        else:
            return 'other'
    
    def _extract_dependencies(self, content: str, file_path: Path) -> Set[str]:
        """Extract file dependencies from imports and references."""
        deps = set()
        
        # Python imports
        import_patterns = [
            r'from\s+([\w.]+)',
            r'import\s+([\w.]+)',
            r'from\s+\.([\w.]+)',
        ]
        
        # File references
        file_patterns = [
            r'["\']([\w/.-]+\.(?:py|js|ts|yaml|yml|json|md))["\']',
            r'([\w/.-]+\.schema\.json)',
            r'([\w/.-]+/[\w.-]+\.py)',
        ]
        
        for pattern in import_patterns + file_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dep = match.group(1)
                if '.' in dep and not dep.endswith('.json'):
                    # Convert module path to file path
                    dep_path = dep.replace('.', '/') + '.py'
                    if (self.root / dep_path).exists():
                        deps.add(dep_path)
                elif dep.endswith(('.py', '.js', '.ts', '.yaml', '.yml', '.json', '.md')):
                    if (self.root / dep).exists():
                        deps.add(dep)
        
        return deps
    
    def get_summary(self, file_type: Optional[str] = None, search_term: Optional[str] = None) -> List[Dict]:
        """Get file summaries with optional filtering."""
        results = []
        
        for file_path, info in self.file_summaries.items():
            if file_type and info['type'] != file_type:
                continue
                
            if search_term:
                search_lower = search_term.lower()
                if not any(search_lower in str(v).lower() for v in info.values() if isinstance(v, (str, list))):
                    continue
            
            results.append(info)
        
        return sorted(results, key=lambda x: (x['type'], x['path']))
    
    def get_dependencies(self, file_path: str) -> Dict[str, Set[str]]:
        """Get dependencies for a specific file."""
        return {
            'depends_on': self.dependencies.get(file_path, set()),
            'depended_by': self.reverse_deps.get(file_path, set())
        }
    
    def search_files(self, purpose: Optional[str] = None, component: Optional[str] = None) -> List[Dict]:
        """Search files by purpose or component."""
        results = []
        
        for file_path, info in self.file_summaries.items():
            match = False
            
            if purpose:
                purpose_lower = purpose.lower()
                if any(purpose_lower in str(v).lower() for v in info['summary'].values() if v):
                    match = True
            
            if component:
                component_lower = component.lower()
                if component_lower in file_path.lower():
                    match = True
            
            if match:
                results.append(info)
        
        return sorted(results, key=lambda x: x['path'])
    
    def get_context_files(self, task: str) -> List[Dict]:
        """Get relevant files for a specific development task."""
        task_lower = task.lower()
        context_map = {
            'agent': ['agent', 'prompt', 'memory'],
            'schema': ['schema', 'validation', 'json'],
            'mcp': ['mcp', 'server', 'tool'],
            'building': ['building', 'grid', 'placement', 'terraria'],
            'feedback': ['feedback', 'human', 'cli', 'rubric'],
            'test': ['test', 'smoke', 'validation', 'pipeline']
        }
        
        relevant_keywords = []
        for task_type, keywords in context_map.items():
            if task_type in task_lower:
                relevant_keywords.extend(keywords)
        
        if not relevant_keywords:
            relevant_keywords = [task_lower]
        
        results = []
        for file_path, info in self.file_summaries.items():
            relevance_score = 0
            content_str = json.dumps(info).lower()
            
            for keyword in relevant_keywords:
                relevance_score += content_str.count(keyword)
            
            if relevance_score > 0:
                info['relevance_score'] = relevance_score
                results.append(info)
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:10]

def main():
    parser = argparse.ArgumentParser(description='StarSystem Project Navigator')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show file summaries')
    summary_parser.add_argument('--type', help='Filter by file type')
    summary_parser.add_argument('--search', help='Search term in summaries')
    
    # Dependencies command
    deps_parser = subparsers.add_parser('deps', help='Show file dependencies')
    deps_parser.add_argument('file', help='File path to analyze')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search files by purpose')
    search_parser.add_argument('--purpose', help='Search by purpose/functionality')
    search_parser.add_argument('--component', help='Search by component name')
    
    # Context command
    context_parser = subparsers.add_parser('context', help='Get context files for task')
    context_parser.add_argument('task', help='Development task (agent, schema, mcp, etc.)')
    
    args = parser.parse_args()
    
    nav = ProjectNavigator()
    
    if args.command == 'summary':
        results = nav.get_summary(args.type, args.search)
        print(f"\n📋 File Summaries ({len(results)} found)")
        print("=" * 50)
        
        for info in results:
            print(f"\n📄 {info['path']} ({info['type']})")
            if info['summary']['title']:
                print(f"   {info['summary']['title']}")
            if info['summary']['description']:
                print(f"   {info['summary']['description']}")
            
            if info['summary']['key_features']:
                print("   Features:", ", ".join(info['summary']['key_features'][:2]))
    
    elif args.command == 'deps':
        deps = nav.get_dependencies(args.file)
        print(f"\n🔗 Dependencies for {args.file}")
        print("=" * 50)
        
        if deps['depends_on']:
            print("\nDepends on:")
            for dep in sorted(deps['depends_on']):
                print(f"  → {dep}")
        
        if deps['depended_by']:
            print("\nDepended on by:")
            for dep in sorted(deps['depended_by']):
                print(f"  ← {dep}")
    
    elif args.command == 'search':
        results = nav.search_files(args.purpose, args.component)
        print(f"\n🔍 Search Results ({len(results)} found)")
        print("=" * 50)
        
        for info in results:
            print(f"\n📄 {info['path']} ({info['type']})")
            if info['summary']['title']:
                print(f"   {info['summary']['title']}")
    
    elif args.command == 'context':
        results = nav.get_context_files(args.task)
        print(f"\n🎯 Context Files for '{args.task}' ({len(results)} found)")
        print("=" * 50)
        
        for info in results:
            print(f"\n📄 {info['path']} ({info['type']}) [Score: {info['relevance_score']}]")
            if info['summary']['title']:
                print(f"   {info['summary']['title']}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()