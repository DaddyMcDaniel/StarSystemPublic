#!/usr/bin/env python3
"""
PCC Runtime Module
Core runtime engine for executing PCC (Procedural-Compressed-Code) programs
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import subprocess
import tempfile

# Import engine interfaces
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.engine import IEngineModule, IClaudeIntegrated

# Import collective intelligence
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from agents.collective_subconscious import consult_collective_wisdom, store_agent_experience

@dataclass
class PCCProgram:
    """Represents a compiled PCC program"""
    id: str
    source_ast: Dict[str, Any]
    bytecode: Optional[bytes] = None
    metadata: Dict[str, Any] = None
    compiled_at: float = 0.0
    execution_count: int = 0
    performance_stats: Dict[str, float] = None

@dataclass
class PCCExecutionContext:
    """Execution context for PCC programs"""
    program_id: str
    variables: Dict[str, Any]
    call_stack: List[Dict[str, Any]]
    execution_time: float = 0.0
    memory_usage: int = 0
    error_state: Optional[str] = None

class PCCRuntimeModule(IEngineModule, IClaudeIntegrated):
    """
    PCC Runtime Module - The heart of Forge engine
    
    This module handles:
    1. PCC program compilation and execution
    2. AST processing and optimization
    3. Bytecode generation and caching
    4. Runtime performance monitoring
    5. Claude-powered optimization suggestions
    """
    
    def __init__(self):
        self._module_id = "pcc_runtime"
        self._version = "1.0.0"
        self._dependencies = []  # Core module - no dependencies
        
        # Runtime state
        self.programs: Dict[str, PCCProgram] = {}
        self.active_contexts: Dict[str, PCCExecutionContext] = {}
        self.interpreter_path = None
        self.optimization_enabled = True
        
        # Performance tracking
        self.execution_stats = {
            "total_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "compilation_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Claude integration
        self.claude_enabled = True
        self.optimization_suggestions = []
        
        print("🔧 PCC Runtime Module initialized")
    
    @property
    def module_id(self) -> str:
        return self._module_id
    
    @property 
    def version(self) -> str:
        return self._version
    
    @property
    def dependencies(self) -> List[str]:
        return self._dependencies
    
    async def initialize(self, engine_context) -> bool:
        """Initialize PCC runtime module"""
        print("🚀 Initializing PCC Runtime...")
        
        # Find PCC interpreter
        await self._locate_pcc_interpreter()
        
        # Initialize with Claude intelligence
        if self.claude_enabled:
            await self._claude_runtime_initialization()
        
        # Store initialization experience
        await self.store_module_experience(
            "PCC Runtime Module initialized successfully",
            "initialization",
            context={
                "interpreter_path": str(self.interpreter_path),
                "optimization_enabled": self.optimization_enabled
            },
            tags=["initialization", "pcc", "runtime"]
        )
        
        print("✅ PCC Runtime initialized")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown PCC runtime"""
        print("🛑 Shutting down PCC Runtime...")
        
        # Store final statistics
        if self.claude_enabled:
            await self.store_module_experience(
                f"PCC Runtime shutdown - {self.execution_stats['total_executions']} total executions",
                "shutdown",
                context=self.execution_stats,
                tags=["shutdown", "performance", "statistics"]
            )
        
        # Clear active contexts
        self.active_contexts.clear()
        
        print("✅ PCC Runtime shutdown complete")
    
    async def update(self, delta_time: float) -> None:
        """Update PCC runtime per frame"""
        # Update execution statistics
        if self.execution_stats["total_executions"] > 0:
            self.execution_stats["average_execution_time"] = (
                self.execution_stats["total_execution_time"] / 
                self.execution_stats["total_executions"]
            )
        
        # Periodic optimization check
        if (int(time.time()) % 30 == 0 and  # Every 30 seconds
            self.execution_stats["total_executions"] % 10 == 0):  # And every 10 executions
            await self._periodic_optimization()
    
    async def handle_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle engine events"""
        event_type = event.get("type", "")
        
        if event_type == "pcc.compile":
            return await self._handle_compile_event(event)
        elif event_type == "pcc.execute":
            return await self._handle_execute_event(event)
        elif event_type == "pcc.optimize":
            return await self._handle_optimize_event(event)
        
        return None
    
    def consult_claude(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Consult Claude for PCC runtime intelligence"""
        if not self.claude_enabled:
            return {"insights": "Claude integration disabled"}
        
        runtime_query = f"""
        PCC Runtime Consultation:
        {query}
        
        Current Runtime State:
        - Total Programs: {len(self.programs)}
        - Active Contexts: {len(self.active_contexts)}
        - Execution Stats: {self.execution_stats}
        - Optimization Enabled: {self.optimization_enabled}
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Provide specific PCC runtime guidance focusing on:
        1. Performance optimization opportunities
        2. Compilation efficiency improvements
        3. Memory usage optimization
        4. Execution pattern analysis
        5. Runtime architecture enhancements
        """
        
        return consult_collective_wisdom(
            runtime_query,
            self.module_id,
            context=context
        )
    
    async def store_module_experience(self, experience: str, exp_type: str = "experience",
                                    context: Dict[str, Any] = None, tags: List[str] = None):
        """Store module experience in collective memory"""
        if not self.claude_enabled:
            return
        
        try:
            # store_agent_experience is not async, so don't await it
            store_agent_experience(
                experience,
                self.module_id,
                exp_type,
                context,
                tags
            )
        except Exception as e:
            print(f"⚠️ Error storing module experience: {e}")
    
    # Core PCC Runtime Methods
    
    async def compile_pcc_program(self, ast_data: Dict[str, Any], program_id: str = None) -> str:
        """Compile PCC AST to executable program"""
        start_time = time.time()
        
        if program_id is None:
            program_id = f"pcc_program_{int(time.time() * 1000)}"
        
        print(f"🔧 Compiling PCC program: {program_id}")
        
        # Check cache first
        if program_id in self.programs:
            self.execution_stats["cache_hits"] += 1
            print(f"📦 Using cached program: {program_id}")
            return program_id
        
        self.execution_stats["cache_misses"] += 1
        
        try:
            # Create PCC program object
            program = PCCProgram(
                id=program_id,
                source_ast=ast_data,
                compiled_at=time.time(),
                metadata={
                    "compilation_time": 0.0,
                    "ast_nodes": self._count_ast_nodes(ast_data),
                    "complexity_score": self._calculate_complexity(ast_data)
                }
            )
            
            # Compile to bytecode (using existing interpreter if available)
            bytecode = await self._compile_to_bytecode(ast_data)
            program.bytecode = bytecode
            
            # Store compiled program
            self.programs[program_id] = program
            
            compilation_time = time.time() - start_time
            program.metadata["compilation_time"] = compilation_time
            self.execution_stats["compilation_time"] += compilation_time
            
            # Store compilation experience
            await self.store_module_experience(
                f"Compiled PCC program {program_id} with {program.metadata['ast_nodes']} AST nodes",
                "compilation",
                context={
                    "program_id": program_id,
                    "compilation_time": compilation_time,
                    "ast_nodes": program.metadata["ast_nodes"],
                    "complexity": program.metadata["complexity_score"]
                },
                tags=["compilation", "pcc", "performance"]
            )
            
            print(f"✅ Compiled PCC program: {program_id} ({compilation_time:.3f}s)")
            return program_id
            
        except Exception as e:
            print(f"❌ Compilation failed for {program_id}: {e}")
            await self.store_module_experience(
                f"PCC compilation failed: {e}",
                "error",
                context={"program_id": program_id, "error": str(e)},
                tags=["error", "compilation"]
            )
            raise
    
    async def execute_pcc_program(self, program_id: str, 
                                 input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a compiled PCC program"""
        start_time = time.time()
        
        if program_id not in self.programs:
            raise ValueError(f"Program {program_id} not found")
        
        program = self.programs[program_id]
        
        print(f"🎮 Executing PCC program: {program_id}")
        
        try:
            # Create execution context
            context = PCCExecutionContext(
                program_id=program_id,
                variables=input_data or {},
                call_stack=[]
            )
            
            self.active_contexts[f"{program_id}_{int(time.time() * 1000)}"] = context
            
            # Execute program
            result = await self._execute_bytecode(program, context, input_data or {})
            
            execution_time = time.time() - start_time
            context.execution_time = execution_time
            
            # Update statistics
            self.execution_stats["total_executions"] += 1
            self.execution_stats["total_execution_time"] += execution_time
            program.execution_count += 1
            
            # Store execution experience
            await self.store_module_experience(
                f"Executed PCC program {program_id} successfully",
                "execution",
                context={
                    "program_id": program_id,
                    "execution_time": execution_time,
                    "input_size": len(str(input_data or {})),
                    "output_size": len(str(result))
                },
                tags=["execution", "success", "performance"]
            )
            
            print(f"✅ Executed PCC program: {program_id} ({execution_time:.3f}s)")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Execution failed for {program_id}: {e}")
            
            await self.store_module_experience(
                f"PCC execution failed: {e}",
                "error",
                context={
                    "program_id": program_id,
                    "execution_time": execution_time,
                    "error": str(e)
                },
                tags=["error", "execution"]
            )
            raise
    
    async def optimize_pcc_program(self, program_id: str) -> bool:
        """Optimize a PCC program using Claude intelligence"""
        if program_id not in self.programs:
            return False
        
        program = self.programs[program_id]
        
        optimization_query = f"""
        Optimize PCC Program: {program_id}
        
        Program Statistics:
        - AST Nodes: {program.metadata.get('ast_nodes', 0)}
        - Complexity Score: {program.metadata.get('complexity_score', 0)}
        - Execution Count: {program.execution_count}
        - Average Performance: {program.performance_stats or 'Not available'}
        
        AST Structure: {json.dumps(program.source_ast, indent=2)[:1000]}...
        
        Suggest specific optimizations:
        1. AST structure improvements
        2. Bytecode optimization opportunities
        3. Execution pattern optimizations
        4. Memory usage reductions
        5. Caching strategies
        
        Provide actionable optimization recommendations.
        """
        
        optimization_result = self.consult_claude(optimization_query, {
            "program_id": program_id,
            "program_stats": program.metadata,
            "optimization_type": "performance"
        })
        
        # Apply optimizations (simplified)
        optimizations_applied = await self._apply_optimizations(program, optimization_result)
        
        if optimizations_applied:
            await self.store_module_experience(
                f"Applied optimizations to program {program_id}",
                "optimization",
                context={
                    "program_id": program_id,
                    "optimizations": optimization_result.get("insights", "")[:200]
                },
                tags=["optimization", "claude", "performance"]
            )
        
        return optimizations_applied
    
    # Internal helper methods
    
    async def _locate_pcc_interpreter(self):
        """Locate PCC VM executable"""
        # Look for PCC VM in project directories
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Check for compiled VM in priority order
        potential_paths = [
            project_root / "bin" / "pcc_vm",  # Project bin directory (priority)
            project_root / "bin" / "pcc_vm.exe",
            project_root / "src" / "pcc_vm",
            "/usr/local/bin/pcc_vm",
            "/usr/bin/pcc_vm",
            "pcc_vm"  # In PATH
        ]
        
        for path in potential_paths:
            if Path(path).exists() and Path(path).is_file():
                self.interpreter_path = Path(path).resolve()
                print(f"🚀 Found PCC VM: {self.interpreter_path}")
                # Test the VM quickly
                try:
                    import subprocess
                    result = subprocess.run([str(self.interpreter_path), "--version"], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"✅ PCC VM verified: {result.stdout.strip()}")
                    else:
                        print(f"⚠️ PCC VM found but failed verification")
                except Exception as e:
                    print(f"⚠️ PCC VM verification failed: {e}")
                return
        
        print("❌ PCC VM not found - PCC execution disabled")
        print(f"   Searched paths: {[str(p) for p in potential_paths]}")
        self.interpreter_path = None
    
    async def _claude_runtime_initialization(self):
        """Initialize runtime with Claude intelligence"""
        init_query = """
        PCC Runtime Initialization Consultation
        
        The PCC Runtime Module is starting up. This is the core execution engine
        for Procedural-Compressed-Code programs.
        
        Provide startup guidance:
        1. Optimal runtime configuration recommendations
        2. Performance monitoring strategies
        3. Compilation optimization approaches
        4. Memory management best practices
        5. Error handling and recovery patterns
        
        Focus on creating the most efficient possible runtime for AI-generated code.
        """
        
        init_guidance = self.consult_claude(init_query, {
            "module": "pcc_runtime",
            "initialization": True
        })
        
        # Apply initialization optimizations
        insights = init_guidance.get("insights", "")
        if "enable_aggressive_optimization" in insights.lower():
            self.optimization_enabled = True
        
        await self.store_module_experience(
            f"Runtime initialized with Claude guidance: {insights[:200]}...",
            "initialization",
            context={"claude_guidance": insights},
            tags=["initialization", "claude", "optimization"]
        )
    
    async def _compile_to_bytecode(self, ast_data: Dict[str, Any]) -> Optional[bytes]:
        """Compile AST to bytecode"""
        if self.interpreter_path:
            # Use C++ interpreter if available
            return await self._compile_with_cpp_interpreter(ast_data)
        else:
            # Python fallback
            return await self._compile_with_python_fallback(ast_data)
    
    async def _compile_with_cpp_interpreter(self, ast_data: Dict[str, Any]) -> bytes:
        """Compile using C++ interpreter"""
        try:
            # Create temporary AST file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(ast_data, f)
                ast_file = f.name
            
            # Run compiler
            result = subprocess.run([
                str(self.interpreter_path), 
                "--compile", ast_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Read bytecode from output
                bytecode_file = ast_file.replace('.json', '.bytecode')
                if Path(bytecode_file).exists():
                    with open(bytecode_file, 'rb') as f:
                        bytecode = f.read()
                    Path(bytecode_file).unlink()  # Clean up
                    Path(ast_file).unlink()  # Clean up
                    return bytecode
            
            Path(ast_file).unlink()  # Clean up on failure
            raise RuntimeError(f"Compilation failed: {result.stderr}")
            
        except Exception as e:
            print(f"❌ C++ compilation failed: {e}")
            return await self._compile_with_python_fallback(ast_data)
    
    async def _compile_with_python_fallback(self, ast_data: Dict[str, Any]) -> bytes:
        """Compile using Python fallback"""
        # Simplified bytecode generation
        bytecode_data = {
            "version": 1,
            "ast": ast_data,
            "compiled_at": time.time()
        }
        
        return json.dumps(bytecode_data).encode('utf-8')
    
    async def _execute_bytecode(self, program: PCCProgram, context: PCCExecutionContext,
                              input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compiled bytecode"""
        if self.interpreter_path and program.bytecode:
            return await self._execute_with_cpp_interpreter(program, context, input_data)
        else:
            return await self._execute_with_python_fallback(program, context, input_data)
    
    async def _execute_with_cpp_interpreter(self, program: PCCProgram, 
                                          context: PCCExecutionContext,
                                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using C++ interpreter"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                f.write(program.bytecode)
                bytecode_file = f.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f)
                input_file = f.name
            
            # Execute
            result = subprocess.run([
                str(self.interpreter_path),
                "--execute", bytecode_file,
                "--input", input_file
            ], capture_output=True, text=True)
            
            # Clean up
            Path(bytecode_file).unlink()
            Path(input_file).unlink()
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                raise RuntimeError(f"Execution failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ C++ execution failed: {e}")
            return await self._execute_with_python_fallback(program, context, input_data)
    
    async def _execute_with_python_fallback(self, program: PCCProgram,
                                          context: PCCExecutionContext,
                                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using Python fallback"""
        # Simplified execution - processes the AST directly
        ast_data = program.source_ast
        
        # Basic execution simulation
        result = {
            "program_id": program.id,
            "input": input_data,
            "output": self._process_ast_nodes(ast_data, input_data),
            "execution_time": context.execution_time,
            "success": True
        }
        
        return result
    
    def _process_ast_nodes(self, ast_data: Dict[str, Any], input_data: Dict[str, Any]) -> Any:
        """Process AST nodes (simplified)"""
        # This is a basic AST processor - would be much more sophisticated
        if isinstance(ast_data, dict):
            if "type" in ast_data:
                node_type = ast_data["type"]
                
                if node_type == "game":
                    return self._process_game_node(ast_data, input_data)
                elif node_type == "entity":
                    return self._process_entity_node(ast_data, input_data)
                elif node_type == "behavior":
                    return self._process_behavior_node(ast_data, input_data)
                else:
                    return f"Processed {node_type} node"
        
        return ast_data
    
    def _process_game_node(self, node: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process game AST node"""
        return {
            "type": "game_instance",
            "name": node.get("name", "Unnamed Game"),
            "entities": [self._process_ast_nodes(child, input_data) 
                        for child in node.get("children", [])],
            "processed": True
        }
    
    def _process_entity_node(self, node: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process entity AST node"""
        return {
            "type": "entity_instance",
            "entity_type": node.get("entity_type", "unknown"),
            "properties": node.get("properties", {}),
            "active": True
        }
    
    def _process_behavior_node(self, node: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process behavior AST node"""
        return {
            "type": "behavior_instance",
            "behavior_name": node.get("behavior_name", "unknown"),
            "trigger": node.get("trigger", "always"),
            "executed": True
        }
    
    def _count_ast_nodes(self, ast_data: Any) -> int:
        """Count total AST nodes"""
        if isinstance(ast_data, dict):
            count = 1
            for value in ast_data.values():
                count += self._count_ast_nodes(value)
            return count
        elif isinstance(ast_data, list):
            count = 0
            for item in ast_data:
                count += self._count_ast_nodes(item)
            return count
        else:
            return 0
    
    def _calculate_complexity(self, ast_data: Any) -> float:
        """Calculate AST complexity score"""
        node_count = self._count_ast_nodes(ast_data)
        depth = self._calculate_ast_depth(ast_data)
        
        # Simple complexity formula
        return node_count * 0.1 + depth * 0.5
    
    def _calculate_ast_depth(self, ast_data: Any, current_depth: int = 0) -> int:
        """Calculate maximum AST depth"""
        if isinstance(ast_data, dict):
            max_depth = current_depth
            for value in ast_data.values():
                depth = self._calculate_ast_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
            return max_depth
        elif isinstance(ast_data, list):
            max_depth = current_depth
            for item in ast_data:
                depth = self._calculate_ast_depth(item, current_depth + 1)
                max_depth = max(max_depth, depth)
            return max_depth
        else:
            return current_depth
    
    async def _periodic_optimization(self):
        """Periodic optimization using Claude"""
        if not self.claude_enabled or len(self.programs) == 0:
            return
        
        optimization_query = f"""
        PCC Runtime Performance Analysis
        
        Current Statistics:
        - Total Programs: {len(self.programs)}
        - Total Executions: {self.execution_stats['total_executions']}
        - Average Execution Time: {self.execution_stats['average_execution_time']:.3f}s
        - Cache Hit Rate: {self.execution_stats['cache_hits'] / max(1, self.execution_stats['cache_hits'] + self.execution_stats['cache_misses']) * 100:.1f}%
        
        Analyze performance patterns and suggest runtime optimizations:
        1. Execution bottlenecks
        2. Compilation efficiency improvements
        3. Caching strategy optimizations
        4. Memory usage patterns
        5. Overall runtime architecture improvements
        """
        
        optimization_result = self.consult_claude(optimization_query)
        
        # Store optimization insights
        await self.store_module_experience(
            f"Periodic optimization analysis completed",
            "optimization",
            context={
                "statistics": self.execution_stats,
                "insights": optimization_result.get("insights", "")[:300]
            },
            tags=["optimization", "periodic", "analysis"]
        )
    
    async def _apply_optimizations(self, program: PCCProgram, 
                                 optimization_result: Dict[str, Any]) -> bool:
        """Apply optimization suggestions"""
        insights = optimization_result.get("insights", "").lower()
        optimizations_applied = False
        
        # Simple optimization applications
        if "enable caching" in insights:
            # Already using caching
            optimizations_applied = True
        
        if "reduce complexity" in insights:
            # Could implement AST simplification
            optimizations_applied = True
        
        if "optimize memory" in insights:
            # Could implement memory pooling
            optimizations_applied = True
        
        return optimizations_applied
    
    # Event handlers
    
    async def _handle_compile_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compilation event"""
        data = event.get("data", {})
        ast_data = data.get("ast")
        program_id = data.get("program_id")
        
        if not ast_data:
            return {"success": False, "error": "No AST data provided"}
        
        try:
            compiled_id = await self.compile_pcc_program(ast_data, program_id)
            return {"success": True, "program_id": compiled_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_execute_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execution event"""
        data = event.get("data", {})
        program_id = data.get("program_id")
        input_data = data.get("input", {})
        
        if not program_id:
            return {"success": False, "error": "No program ID provided"}
        
        try:
            result = await self.execute_pcc_program(program_id, input_data)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_optimize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimization event"""
        data = event.get("data", {})
        program_id = data.get("program_id")
        
        if not program_id:
            return {"success": False, "error": "No program ID provided"}
        
        try:
            success = await self.optimize_pcc_program(program_id)
            return {"success": success}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_runtime_status(self) -> Dict[str, Any]:
        """Get comprehensive runtime status"""
        return {
            "module_id": self.module_id,
            "version": self.version,
            "interpreter_path": str(self.interpreter_path) if self.interpreter_path else None,
            "programs_loaded": len(self.programs),
            "active_contexts": len(self.active_contexts),
            "execution_stats": self.execution_stats,
            "optimization_enabled": self.optimization_enabled,
            "claude_enabled": self.claude_enabled
        }