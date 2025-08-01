/**************************************************************************/
/*  pcc_interpreter.cpp                                                  */
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

#include "pcc_interpreter.h"
#include "core/string/string_builder.h"

PCCInterpreter::PCCInterpreter() {
    current_context = new ExecutionContext();
}

PCCInterpreter::~PCCInterpreter() {
    // Clean up contexts
    while (current_context) {
        ExecutionContext *parent = current_context->parent;
        delete current_context;
        current_context = parent;
    }
}

Variant PCCInterpreter::execute(Ref<PCCASTNode> ast) {
    if (ast.is_null()) {
        error("Cannot execute null AST");
        return Variant();
    }
    
    clear_errors();
    return execute_node(ast);
}

Variant PCCInterpreter::execute_node(Ref<PCCASTNode> node) {
    if (node.is_null()) {
        return Variant();
    }
    
    switch (node->get_type()) {
        case AST_PROGRAM:
            return execute_program(node);
        case AST_FUNCTION_DEF:
            return execute_function_def(node);
        case AST_FUNCTION_CALL:
            return execute_function_call(node);
        case AST_VARIABLE_DECL:
            return execute_variable_decl(node);
        case AST_ASSIGNMENT:
            return execute_assignment(node);
        case AST_BINARY_OP:
            return execute_binary_op(node);
        case AST_UNARY_OP:
            return execute_unary_op(node);
        case AST_IF_STATEMENT:
            return execute_if_statement(node);
        case AST_WHILE_LOOP:
            return execute_while_loop(node);
        case AST_FOR_LOOP:
            return execute_for_loop(node);
        case AST_RETURN:
            return execute_return(node);
        case AST_LITERAL:
            return execute_literal(node);
        case AST_IDENTIFIER:
            return execute_identifier(node);
        case AST_BLOCK:
            return execute_block(node);
        case AST_TABLE:
            return execute_table(node);
        case AST_TABLE_ACCESS:
            return execute_table_access(node);
        default:
            error("Unknown AST node type: " + itos(node->get_type()));
            return Variant();
    }
}

Variant PCCInterpreter::execute_program(Ref<PCCASTNode> node) {
    // Execute all statements in the program
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_function_def(Ref<PCCASTNode> node) {
    // Store function definition in current context
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_function_call(Ref<PCCASTNode> node) {
    // Execute function call
    // For now, handle built-in functions like print
    return Variant();
}

Variant PCCInterpreter::execute_variable_decl(Ref<PCCASTNode> node) {
    // Execute variable declaration
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_assignment(Ref<PCCASTNode> node) {
    // Execute assignment
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_binary_op(Ref<PCCASTNode> node) {
    // Execute binary operation
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_unary_op(Ref<PCCASTNode> node) {
    // Execute unary operation
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_if_statement(Ref<PCCASTNode> node) {
    // Execute if statement
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_while_loop(Ref<PCCASTNode> node) {
    // Execute while loop
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_for_loop(Ref<PCCASTNode> node) {
    // Execute for loop
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_return(Ref<PCCASTNode> node) {
    // Execute return statement
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_literal(Ref<PCCASTNode> node) {
    // Execute literal value
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_identifier(Ref<PCCASTNode> node) {
    // Execute identifier lookup
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_block(Ref<PCCASTNode> node) {
    // Execute block of statements
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_table(Ref<PCCASTNode> node) {
    // Execute table creation
    // For now, just return success
    return Variant();
}

Variant PCCInterpreter::execute_table_access(Ref<PCCASTNode> node) {
    // Execute table access
    // For now, just return success
    return Variant();
}

void PCCInterpreter::push_context() {
    current_context = new ExecutionContext(current_context);
}

void PCCInterpreter::pop_context() {
    if (current_context && current_context->parent) {
        ExecutionContext *parent = current_context->parent;
        delete current_context;
        current_context = parent;
    }
}

void PCCInterpreter::set_variable(const StringName &name, const Variant &value) {
    if (current_context) {
        current_context->variables[name] = value;
    }
}

Variant PCCInterpreter::get_variable(const StringName &name) {
    ExecutionContext *context = current_context;
    while (context) {
        if (context->variables.has(name)) {
            return context->variables[name];
        }
        context = context->parent;
    }
    return Variant();
}

void PCCInterpreter::set_function(const StringName &name, Ref<PCCASTNode> function) {
    if (current_context) {
        current_context->functions[name] = function;
    }
}

Ref<PCCASTNode> PCCInterpreter::get_function(const StringName &name) {
    ExecutionContext *context = current_context;
    while (context) {
        if (context->functions.has(name)) {
            return context->functions[name];
        }
        context = context->parent;
    }
    return Ref<PCCASTNode>();
}

void PCCInterpreter::error(const String &message) {
    errors.push_back(message);
    has_error = true;
}

bool PCCInterpreter::is_truthy(const Variant &value) {
    if (value.is_null()) {
        return false;
    }
    
    if (value.get_type() == Variant::BOOL) {
        return value.operator bool();
    }
    
    if (value.get_type() == Variant::INT) {
        return value.operator int() != 0;
    }
    
    if (value.get_type() == Variant::FLOAT) {
        return value.operator double() != 0.0;
    }
    
    if (value.get_type() == Variant::STRING) {
        return !value.operator String().is_empty();
    }
    
    // For other types, consider them truthy if not null
    return true;
} 