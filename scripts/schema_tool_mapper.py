#!/usr/bin/env python3
"""
SUMMARY: Schema-Tool Relationship Mapper v1
============================================
Analyzes and visualizes relationships between schemas, MCP tools, and validation workflows.
Essential for understanding the Week 2+ schema freeze and tool integration architecture.

KEY FEATURES:
- Schema dependency analysis: Which tools use which schemas for input/output validation
- Tool chain mapping: Shows complete workflows from generators → validators → builders
- URN resolution: Maps URN schema IDs to actual file locations and usage
- Validation coverage: Identifies tools missing proper schema validation
- Integration testing: Suggests test scenarios based on tool chains

ANALYSIS TYPES:
- schema-usage: Show which tools reference specific schemas
- tool-deps: Map tool input/output schema dependencies
- validation-gaps: Find tools without proper schema validation
- workflow-chains: Identify complete tool execution workflows
- urn-map: Resolve URN schema IDs to file paths and usage

USAGE:
  python scripts/schema_tool_mapper.py schema-usage --schema blueprint_chip
  python scripts/schema_tool_mapper.py tool-deps --tool godot.apply_patch
  python scripts/schema_tool_mapper.py workflow-chains --start generators
  python scripts/schema_tool_mapper.py validation-gaps

OUTPUT:
- Visual relationship maps showing schema ↔ tool connections
- Workflow diagrams for complete tool chains
- Coverage reports for validation completeness
- Recommendations for missing integrations

RELATED FILES:
- schemas/*.v1.schema.json - All schema definitions with URN IDs
- mcp_server/server.py - Tool implementations and registrations
- Essential for Week 2+ schema freeze compliance validation
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import argparse

class SchemaToolMapper:
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root).resolve()
        self.schemas = {}
        self.tools = {}
        self.schema_usage = defaultdict(set)
        self.tool_schemas = defaultdict(dict)
        self.urn_mapping = {}
        
        self._load_schemas()
        self._load_tools()
        self._analyze_relationships()
    
    def _load_schemas(self):
        """Load all schema files and extract metadata."""
        schema_dir = self.root / "schemas"
        if not schema_dir.exists():
            print(f"⚠️  Schema directory not found: {schema_dir}")
            return
        
        for schema_file in schema_dir.glob("*.schema.json"):
            try:
                with open(schema_file, 'r') as f:
                    content = f.read()
                    # Extract JSON part (skip comment header)
                    json_start = content.find('{')
                    if json_start == -1:
                        continue
                    
                    schema_data = json.loads(content[json_start:])
                    
                    schema_name = schema_file.stem
                    self.schemas[schema_name] = {
                        'file': schema_file,
                        'data': schema_data,
                        'urn': schema_data.get('$id', ''),
                        'version': schema_data.get('version', 'unknown'),
                        'title': schema_data.get('title', schema_name)
                    }
                    
                    # Map URN to schema name
                    if '$id' in schema_data:
                        self.urn_mapping[schema_data['$id']] = schema_name
                        
            except Exception as e:
                print(f"⚠️  Failed to load schema {schema_file}: {e}")
        
        print(f"📋 Loaded {len(self.schemas)} schemas")
    
    def _load_tools(self):
        """Extract tool definitions from MCP server and other sources."""
        # Load from MCP server
        mcp_server_file = self.root / "mcp_server" / "server.py"
        if mcp_server_file.exists():
            self._extract_tools_from_file(mcp_server_file, "mcp_server")
        
        # Load from high-order benchmark (tool specifications)
        hob_file = self.root / "agents" / "prompting" / "high_order_benchmark_week2.md"
        if hob_file.exists():
            self._extract_tools_from_hob(hob_file)
        
        print(f"🔧 Found {len(self.tools)} tools")
    
    def _extract_tools_from_file(self, file_path: Path, source: str):
        """Extract tool definitions from Python source."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for tool registration patterns
            tool_patterns = [
                r'@server\.list_tools\(\)\s*async def (\w+)',
                r'def\s+(\w+_\w+)\s*\(',
                r'["\']([\w.]+)["\']\s*:.*tool',
            ]
            
            for pattern in tool_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    tool_name = match.group(1)
                    if '.' not in tool_name and '_' in tool_name:
                        # Convert underscore to dot notation
                        tool_name = tool_name.replace('_', '.', 1)
                    
                    self.tools[tool_name] = {
                        'name': tool_name,
                        'source': source,
                        'file': file_path,
                        'input_schema': None,
                        'output_schema': None
                    }
                    
        except Exception as e:
            print(f"⚠️  Failed to extract tools from {file_path}: {e}")
    
    def _extract_tools_from_hob(self, file_path: Path):
        """Extract tool specifications from high-order benchmark."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for tool definitions in XML format
            tool_pattern = r'<tool name="([^"]+)"[^>]*in_schema="([^"]+)"[^>]*out_schema="([^"]+)"[^>]*/?>'  
            matches = re.finditer(tool_pattern, content)
            
            for match in matches:
                tool_name = match.group(1)
                input_schema = match.group(2)
                output_schema = match.group(3)
                
                if tool_name not in self.tools:
                    self.tools[tool_name] = {
                        'name': tool_name,
                        'source': 'specification',
                        'file': file_path
                    }
                
                self.tools[tool_name]['input_schema'] = input_schema
                self.tools[tool_name]['output_schema'] = output_schema
                
        except Exception as e:
            print(f"⚠️  Failed to extract tools from HOB {file_path}: {e}")
    
    def _analyze_relationships(self):
        """Analyze relationships between schemas and tools."""
        for tool_name, tool_info in self.tools.items():
            # Map schema references
            input_schema = tool_info.get('input_schema')
            output_schema = tool_info.get('output_schema')
            
            if input_schema:
                # Resolve URN or direct reference
                schema_name = self._resolve_schema_reference(input_schema)
                if schema_name:
                    self.schema_usage[schema_name].add(tool_name)
                    self.tool_schemas[tool_name]['input'] = schema_name
            
            if output_schema:
                schema_name = self._resolve_schema_reference(output_schema)
                if schema_name:
                    self.schema_usage[schema_name].add(tool_name)
                    self.tool_schemas[tool_name]['output'] = schema_name
    
    def _resolve_schema_reference(self, schema_ref: str) -> Optional[str]:
        """Resolve schema reference to actual schema name."""
        # Direct URN lookup
        if schema_ref in self.urn_mapping:
            return self.urn_mapping[schema_ref]
        
        # Extract schema name from URN
        if schema_ref.startswith('urn:starsystem:schema:'):
            # Format: urn:starsystem:schema:name:v1
            parts = schema_ref.split(':')
            if len(parts) >= 4:
                schema_base = parts[3]
                # Look for matching schema
                for schema_name in self.schemas:
                    if schema_base in schema_name:
                        return schema_name
        
        # Direct schema ID lookup  
        if schema_ref.startswith('$id:'):
            ref_name = schema_ref[4:]
            for schema_name in self.schemas:
                if ref_name in schema_name:
                    return schema_name
        
        # Partial name match
        for schema_name in self.schemas:
            if schema_ref in schema_name or schema_name in schema_ref:
                return schema_name
        
        return None
    
    def get_schema_usage(self, schema_name: str) -> Dict:
        """Get tools that use a specific schema."""
        if schema_name not in self.schemas:
            return {'error': f'Schema {schema_name} not found'}
        
        schema_info = self.schemas[schema_name]
        using_tools = list(self.schema_usage.get(schema_name, set()))
        
        return {
            'schema': schema_info,
            'tools_using': using_tools,
            'usage_count': len(using_tools)
        }
    
    def get_tool_dependencies(self, tool_name: str) -> Dict:
        """Get schema dependencies for a specific tool."""
        if tool_name not in self.tools:
            return {'error': f'Tool {tool_name} not found'}
        
        tool_info = self.tools[tool_name]
        schemas_used = self.tool_schemas.get(tool_name, {})
        
        return {
            'tool': tool_info,
            'input_schema': schemas_used.get('input'),
            'output_schema': schemas_used.get('output'),
            'schema_details': {
                'input': self.schemas.get(schemas_used.get('input', ''), {}),
                'output': self.schemas.get(schemas_used.get('output', ''), {})
            }
        }
    
    def find_validation_gaps(self) -> List[Dict]:
        """Find tools without proper schema validation."""
        gaps = []
        
        for tool_name, tool_info in self.tools.items():
            schemas_used = self.tool_schemas.get(tool_name, {})
            
            issues = []
            if not schemas_used.get('input'):
                issues.append('Missing input schema')
            if not schemas_used.get('output'):
                issues.append('Missing output schema')
            
            if issues:
                gaps.append({
                    'tool': tool_name,
                    'issues': issues,
                    'source': tool_info.get('source', 'unknown')
                })
        
        return gaps
    
    def find_workflow_chains(self, start_prefix: str) -> List[List[str]]:
        """Find complete tool workflow chains starting with prefix."""
        chains = []
        
        # Find starting tools
        starting_tools = [name for name in self.tools.keys() if name.startswith(start_prefix)]
        
        for start_tool in starting_tools:
            chain = self._build_chain(start_tool, set())
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def _build_chain(self, tool_name: str, visited: Set[str]) -> List[str]:
        """Recursively build tool chain based on schema flow."""
        if tool_name in visited:
            return [tool_name]  # Avoid cycles
        
        visited.add(tool_name)
        chain = [tool_name]
        
        # Find tools that use this tool's output schema as input
        tool_schemas = self.tool_schemas.get(tool_name, {})
        output_schema = tool_schemas.get('output')
        
        if output_schema:
            for next_tool, next_schemas in self.tool_schemas.items():
                if next_tool not in visited and next_schemas.get('input') == output_schema:
                    next_chain = self._build_chain(next_tool, visited.copy())
                    if len(next_chain) > 1:
                        chain.extend(next_chain[1:])  # Skip duplicate tool name
                        break
        
        return chain
    
    def get_urn_mapping(self) -> Dict[str, str]:
        """Get complete URN to schema name mapping."""
        return dict(self.urn_mapping)

def main():
    parser = argparse.ArgumentParser(description='Schema-Tool Relationship Mapper')
    subparsers = parser.add_subparsers(dest='command', help='Analysis commands')
    
    # Schema usage command
    usage_parser = subparsers.add_parser('schema-usage', help='Show tool usage for schema')
    usage_parser.add_argument('schema', help='Schema name to analyze')
    
    # Tool dependencies command
    deps_parser = subparsers.add_parser('tool-deps', help='Show schema dependencies for tool')
    deps_parser.add_argument('tool', help='Tool name to analyze')
    
    # Validation gaps command
    gaps_parser = subparsers.add_parser('validation-gaps', help='Find tools missing schema validation')
    
    # Workflow chains command
    chains_parser = subparsers.add_parser('workflow-chains', help='Find complete tool workflows')
    chains_parser.add_argument('start', help='Starting tool prefix (e.g., generators, validators)')
    
    # URN mapping command
    urn_parser = subparsers.add_parser('urn-map', help='Show URN to schema mapping')
    
    args = parser.parse_args()
    
    mapper = SchemaToolMapper()
    
    if args.command == 'schema-usage':
        result = mapper.get_schema_usage(args.schema)
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            schema = result['schema']
            print(f"\n📋 Schema: {schema['title']} ({args.schema})")
            print(f"   URN: {schema['urn']}")
            print(f"   Version: {schema['version']}")
            print(f"\n🔧 Tools using this schema ({result['usage_count']}):")
            for tool in result['tools_using']:
                print(f"   • {tool}")
    
    elif args.command == 'tool-deps':
        result = mapper.get_tool_dependencies(args.tool)
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            tool = result['tool']
            print(f"\n🔧 Tool: {tool['name']}")
            print(f"   Source: {tool['source']}")
            
            if result['input_schema']:
                print(f"\n📥 Input Schema: {result['input_schema']}")
                input_details = result['schema_details']['input']
                if input_details:
                    print(f"   URN: {input_details.get('urn', 'N/A')}")
            
            if result['output_schema']:
                print(f"\n📤 Output Schema: {result['output_schema']}")
                output_details = result['schema_details']['output']
                if output_details:
                    print(f"   URN: {output_details.get('urn', 'N/A')}")
    
    elif args.command == 'validation-gaps':
        gaps = mapper.find_validation_gaps()
        print(f"\n⚠️  Validation Gaps Found ({len(gaps)}):")
        print("=" * 50)
        
        for gap in gaps:
            print(f"\n🔧 {gap['tool']} ({gap['source']})")
            for issue in gap['issues']:
                print(f"   ❌ {issue}")
    
    elif args.command == 'workflow-chains':
        chains = mapper.find_workflow_chains(args.start)
        print(f"\n🔗 Workflow Chains starting with '{args.start}' ({len(chains)} found):")
        print("=" * 60)
        
        for i, chain in enumerate(chains, 1):
            print(f"\nChain {i}: {' → '.join(chain)}")
    
    elif args.command == 'urn-map':
        urn_map = mapper.get_urn_mapping()
        print(f"\n🆔 URN to Schema Mapping ({len(urn_map)} entries):")
        print("=" * 60)
        
        for urn, schema_name in sorted(urn_map.items()):
            print(f"\n{urn}")
            print(f"  → {schema_name}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()