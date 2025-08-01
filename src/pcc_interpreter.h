/**************************************************************************/
/*  pcc_interpreter.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef PCC_INTERPRETER_H
#define PCC_INTERPRETER_H

#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "pcc_parser.h"

// Forward declarations
class PCCASTNode;

// Interpreter class for executing PCC AST
class PCCInterpreter : public RefCounted {
    GDCLASS(PCCInterpreter, RefCounted);

private:
    // Execution context
    struct ExecutionContext {
        HashMap<StringName, Variant> variables;
        HashMap<StringName, Ref<PCCASTNode>> functions;
        ExecutionContext *parent = nullptr;
        
        ExecutionContext() {}
        ExecutionContext(ExecutionContext *p_parent) : parent(p_parent) {}
    };
    
    ExecutionContext *current_context;
    List<String> errors;
    bool has_error = false;
    
    // Execution methods
    Variant execute_node(Ref<PCCASTNode> node);
    Variant execute_program(Ref<PCCASTNode> node);
    Variant execute_function_def(Ref<PCCASTNode> node);
    Variant execute_function_call(Ref<PCCASTNode> node);
    Variant execute_variable_decl(Ref<PCCASTNode> node);
    Variant execute_assignment(Ref<PCCASTNode> node);
    Variant execute_binary_op(Ref<PCCASTNode> node);
    Variant execute_unary_op(Ref<PCCASTNode> node);
    Variant execute_if_statement(Ref<PCCASTNode> node);
    Variant execute_while_loop(Ref<PCCASTNode> node);
    Variant execute_for_loop(Ref<PCCASTNode> node);
    Variant execute_return(Ref<PCCASTNode> node);
    Variant execute_literal(Ref<PCCASTNode> node);
    Variant execute_identifier(Ref<PCCASTNode> node);
    Variant execute_block(Ref<PCCASTNode> node);
    Variant execute_table(Ref<PCCASTNode> node);
    Variant execute_table_access(Ref<PCCASTNode> node);
    
    // Helper methods
    void push_context();
    void pop_context();
    void set_variable(const StringName &name, const Variant &value);
    Variant get_variable(const StringName &name);
    void set_function(const StringName &name, Ref<PCCASTNode> function);
    Ref<PCCASTNode> get_function(const StringName &name);
    void error(const String &message);
    bool is_truthy(const Variant &value);

public:
    // Execute a parsed AST
    Variant execute(Ref<PCCASTNode> ast);
    
    // Get execution errors
    const List<String> &get_errors() const { return errors; }
    
    // Clear errors
    void clear_errors() { errors.clear(); has_error = false; }
    
    // Check if execution had errors
    bool has_execution_error() const { return has_error; }
    
    PCCInterpreter();
    ~PCCInterpreter();
};

#endif // VRON_INTERPRETER_H 