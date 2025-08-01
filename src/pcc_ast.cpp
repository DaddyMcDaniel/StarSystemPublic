/**************************************************************************/
/*  pcc_ast.cpp                                                          */
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

#include "pcc_ast.h"

// PCCASTNode implementation
void PCCASTNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_type_int"), &PCCASTNode::get_type_int);
    ClassDB::bind_method(D_METHOD("get_type_name"), &PCCASTNode::get_type_name);
    ClassDB::bind_method(D_METHOD("get_line"), &PCCASTNode::get_line);
    ClassDB::bind_method(D_METHOD("get_column"), &PCCASTNode::get_column);
    ClassDB::bind_method(D_METHOD("is_valid"), &PCCASTNode::is_valid);
}

String PCCASTNode::get_type_name() const {
    switch (node_type) {
        case AST_PROGRAM: return "AST_PROGRAM";
        case AST_FUNCTION_DEF: return "AST_FUNCTION_DEF";
        case AST_FUNCTION_CALL: return "AST_FUNCTION_CALL";
        case AST_VARIABLE_DECL: return "AST_VARIABLE_DECL";
        case AST_ASSIGNMENT: return "AST_ASSIGNMENT";
        case AST_BINARY_OP: return "AST_BINARY_OP";
        case AST_UNARY_OP: return "AST_UNARY_OP";
        case AST_IF_STATEMENT: return "AST_IF_STATEMENT";
        case AST_WHILE_LOOP: return "AST_WHILE_LOOP";
        case AST_FOR_LOOP: return "AST_FOR_LOOP";
        case AST_RETURN: return "AST_RETURN";
        case AST_LITERAL: return "AST_LITERAL";
        case AST_IDENTIFIER: return "AST_IDENTIFIER";
        case AST_BLOCK: return "AST_BLOCK";
        case AST_TABLE: return "AST_TABLE";
        case AST_TABLE_ACCESS: return "AST_TABLE_ACCESS";
        default: return "AST_UNKNOWN";
    }
}

// ProgramNode implementation
void ProgramNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("add_statement", "statement"), &ProgramNode::add_statement);
    ClassDB::bind_method(D_METHOD("get_statements"), &ProgramNode::get_statements);
    ClassDB::bind_method(D_METHOD("to_string"), &ProgramNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &ProgramNode::get_children);
}

void ProgramNode::add_statement(Ref<PCCASTNode> statement) {
    if (statement.is_valid()) {
        statements.push_back(statement);
    }
}

String ProgramNode::to_string() const {
    StringBuilder sb;
    sb.append("Program(");
    sb.append(itos(statements.size()));
    sb.append(" statements)");
    return sb.as_string();
}

Array ProgramNode::get_children() const {
    Array children;
    for (const Ref<PCCASTNode> &stmt : statements) {
        children.push_back(stmt);
    }
    return children;
}

Array ProgramNode::get_statements() const {
    Array stmt_array;
    for (const Ref<PCCASTNode> &stmt : statements) {
        stmt_array.push_back(stmt);
    }
    return stmt_array;
}

// FunctionDefNode implementation
void FunctionDefNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_name", "name"), &FunctionDefNode::set_name);
    ClassDB::bind_method(D_METHOD("get_name"), &FunctionDefNode::get_name);
    ClassDB::bind_method(D_METHOD("add_parameter", "param"), &FunctionDefNode::add_parameter);
    ClassDB::bind_method(D_METHOD("get_parameters"), &FunctionDefNode::get_parameters);
    ClassDB::bind_method(D_METHOD("set_body", "body"), &FunctionDefNode::set_body);
    ClassDB::bind_method(D_METHOD("get_body"), &FunctionDefNode::get_body);
    ClassDB::bind_method(D_METHOD("to_string"), &FunctionDefNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &FunctionDefNode::get_children);
}

String FunctionDefNode::to_string() const {
    StringBuilder sb;
    sb.append("FunctionDef(");
    sb.append(name);
    sb.append(", ");
    sb.append(itos(parameters.size()));
    sb.append(" params)");
    return sb.as_string();
}

Array FunctionDefNode::get_children() const {
    Array children;
    if (body.is_valid()) {
        children.push_back(body);
    }
    return children;
}

// FunctionCallNode implementation
void FunctionCallNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_name", "name"), &FunctionCallNode::set_name);
    ClassDB::bind_method(D_METHOD("get_name"), &FunctionCallNode::get_name);
    ClassDB::bind_method(D_METHOD("add_argument", "arg"), &FunctionCallNode::add_argument);
    ClassDB::bind_method(D_METHOD("get_arguments"), &FunctionCallNode::get_arguments);
    ClassDB::bind_method(D_METHOD("to_string"), &FunctionCallNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &FunctionCallNode::get_children);
}

String FunctionCallNode::to_string() const {
    StringBuilder sb;
    sb.append("FunctionCall(");
    sb.append(name);
    sb.append(", ");
    sb.append(itos(arguments.size()));
    sb.append(" args)");
    return sb.as_string();
}

Array FunctionCallNode::get_arguments() const {
    Array arg_array;
    for (const Ref<PCCASTNode> &arg : arguments) {
        arg_array.push_back(arg);
    }
    return arg_array;
}

Array FunctionCallNode::get_children() const {
    Array children;
    for (const Ref<PCCASTNode> &arg : arguments) {
        children.push_back(arg);
    }
    return children;
}

// VariableDeclNode implementation
void VariableDeclNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_name", "name"), &VariableDeclNode::set_name);
    ClassDB::bind_method(D_METHOD("get_name"), &VariableDeclNode::get_name);
    ClassDB::bind_method(D_METHOD("set_initializer", "init"), &VariableDeclNode::set_initializer);
    ClassDB::bind_method(D_METHOD("get_initializer"), &VariableDeclNode::get_initializer);
    ClassDB::bind_method(D_METHOD("set_local", "local"), &VariableDeclNode::set_local);
    ClassDB::bind_method(D_METHOD("is_local_var"), &VariableDeclNode::is_local_var);
    ClassDB::bind_method(D_METHOD("to_string"), &VariableDeclNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &VariableDeclNode::get_children);
}

String VariableDeclNode::to_string() const {
    StringBuilder sb;
    sb.append(is_local ? "LocalVar(" : "Var(");
    sb.append(name);
    if (initializer.is_valid()) {
        sb.append(" = ...)");
    } else {
        sb.append(")");
    }
    return sb.as_string();
}

Array VariableDeclNode::get_children() const {
    Array children;
    if (initializer.is_valid()) {
        children.push_back(initializer);
    }
    return children;
}

// AssignmentNode implementation
void AssignmentNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_name", "name"), &AssignmentNode::set_name);
    ClassDB::bind_method(D_METHOD("get_name"), &AssignmentNode::get_name);
    ClassDB::bind_method(D_METHOD("set_value", "value"), &AssignmentNode::set_value);
    ClassDB::bind_method(D_METHOD("get_value"), &AssignmentNode::get_value);
    ClassDB::bind_method(D_METHOD("to_string"), &AssignmentNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &AssignmentNode::get_children);
}

String AssignmentNode::to_string() const {
    StringBuilder sb;
    sb.append("Assignment(");
    sb.append(name);
    sb.append(" = ...)");
    return sb.as_string();
}

Array AssignmentNode::get_children() const {
    Array children;
    if (value.is_valid()) {
        children.push_back(value);
    }
    return children;
}

// BinaryOpNode implementation
void BinaryOpNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_operator", "op"), &BinaryOpNode::set_operator);
    ClassDB::bind_method(D_METHOD("get_operator"), &BinaryOpNode::get_operator);
    ClassDB::bind_method(D_METHOD("set_left", "left"), &BinaryOpNode::set_left);
    ClassDB::bind_method(D_METHOD("get_left"), &BinaryOpNode::get_left);
    ClassDB::bind_method(D_METHOD("set_right", "right"), &BinaryOpNode::set_right);
    ClassDB::bind_method(D_METHOD("get_right"), &BinaryOpNode::get_right);
    ClassDB::bind_method(D_METHOD("to_string"), &BinaryOpNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &BinaryOpNode::get_children);
}

String BinaryOpNode::to_string() const {
    StringBuilder sb;
    sb.append("BinaryOp(");
    sb.append(op);
    sb.append(")");
    return sb.as_string();
}

Array BinaryOpNode::get_children() const {
    Array children;
    if (left.is_valid()) {
        children.push_back(left);
    }
    if (right.is_valid()) {
        children.push_back(right);
    }
    return children;
}

// UnaryOpNode implementation
void UnaryOpNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_operator", "op"), &UnaryOpNode::set_operator);
    ClassDB::bind_method(D_METHOD("get_operator"), &UnaryOpNode::get_operator);
    ClassDB::bind_method(D_METHOD("set_operand", "operand"), &UnaryOpNode::set_operand);
    ClassDB::bind_method(D_METHOD("get_operand"), &UnaryOpNode::get_operand);
    ClassDB::bind_method(D_METHOD("to_string"), &UnaryOpNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &UnaryOpNode::get_children);
}

String UnaryOpNode::to_string() const {
    StringBuilder sb;
    sb.append("UnaryOp(");
    sb.append(op);
    sb.append(")");
    return sb.as_string();
}

Array UnaryOpNode::get_children() const {
    Array children;
    if (operand.is_valid()) {
        children.push_back(operand);
    }
    return children;
}

// IfStatementNode implementation
void IfStatementNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_condition", "condition"), &IfStatementNode::set_condition);
    ClassDB::bind_method(D_METHOD("get_condition"), &IfStatementNode::get_condition);
    ClassDB::bind_method(D_METHOD("set_then_body", "body"), &IfStatementNode::set_then_body);
    ClassDB::bind_method(D_METHOD("get_then_body"), &IfStatementNode::get_then_body);
    ClassDB::bind_method(D_METHOD("set_else_body", "body"), &IfStatementNode::set_else_body);
    ClassDB::bind_method(D_METHOD("get_else_body"), &IfStatementNode::get_else_body);
    ClassDB::bind_method(D_METHOD("add_elseif", "condition", "body"), &IfStatementNode::add_elseif);
    ClassDB::bind_method(D_METHOD("get_elseif_conditions"), &IfStatementNode::get_elseif_conditions);
    ClassDB::bind_method(D_METHOD("get_elseif_bodies"), &IfStatementNode::get_elseif_bodies);
    ClassDB::bind_method(D_METHOD("to_string"), &IfStatementNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &IfStatementNode::get_children);
}

void IfStatementNode::add_elseif(Ref<PCCASTNode> p_condition, Ref<PCCASTNode> p_body) {
    elseif_conditions.push_back(p_condition);
    elseif_bodies.push_back(p_body);
}

Array IfStatementNode::get_elseif_conditions() const {
    Array cond_array;
    for (const Ref<PCCASTNode> &cond : elseif_conditions) {
        cond_array.push_back(cond);
    }
    return cond_array;
}

Array IfStatementNode::get_elseif_bodies() const {
    Array body_array;
    for (const Ref<PCCASTNode> &body : elseif_bodies) {
        body_array.push_back(body);
    }
    return body_array;
}

String IfStatementNode::to_string() const {
    StringBuilder sb;
    sb.append("IfStatement(");
    sb.append(itos(elseif_conditions.size()));
    sb.append(" elseif)");
    return sb.as_string();
}

Array IfStatementNode::get_children() const {
    Array children;
    if (condition.is_valid()) {
        children.push_back(condition);
    }
    if (then_body.is_valid()) {
        children.push_back(then_body);
    }
    for (const Ref<PCCASTNode> &cond : elseif_conditions) {
        children.push_back(cond);
    }
    for (const Ref<PCCASTNode> &body : elseif_bodies) {
        children.push_back(body);
    }
    if (else_body.is_valid()) {
        children.push_back(else_body);
    }
    return children;
}

// WhileLoopNode implementation
void WhileLoopNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_condition", "condition"), &WhileLoopNode::set_condition);
    ClassDB::bind_method(D_METHOD("get_condition"), &WhileLoopNode::get_condition);
    ClassDB::bind_method(D_METHOD("set_body", "body"), &WhileLoopNode::set_body);
    ClassDB::bind_method(D_METHOD("get_body"), &WhileLoopNode::get_body);
    ClassDB::bind_method(D_METHOD("to_string"), &WhileLoopNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &WhileLoopNode::get_children);
}

String WhileLoopNode::to_string() const {
    return "WhileLoop";
}

Array WhileLoopNode::get_children() const {
    Array children;
    if (condition.is_valid()) {
        children.push_back(condition);
    }
    if (body.is_valid()) {
        children.push_back(body);
    }
    return children;
}

// ForLoopNode implementation
void ForLoopNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_iterator", "iterator"), &ForLoopNode::set_iterator);
    ClassDB::bind_method(D_METHOD("get_iterator"), &ForLoopNode::get_iterator);
    ClassDB::bind_method(D_METHOD("set_start", "start"), &ForLoopNode::set_start);
    ClassDB::bind_method(D_METHOD("get_start"), &ForLoopNode::get_start);
    ClassDB::bind_method(D_METHOD("set_end", "end"), &ForLoopNode::set_end);
    ClassDB::bind_method(D_METHOD("get_end"), &ForLoopNode::get_end);
    ClassDB::bind_method(D_METHOD("set_step", "step"), &ForLoopNode::set_step);
    ClassDB::bind_method(D_METHOD("get_step"), &ForLoopNode::get_step);
    ClassDB::bind_method(D_METHOD("set_body", "body"), &ForLoopNode::set_body);
    ClassDB::bind_method(D_METHOD("get_body"), &ForLoopNode::get_body);
    ClassDB::bind_method(D_METHOD("to_string"), &ForLoopNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &ForLoopNode::get_children);
}

String ForLoopNode::to_string() const {
    StringBuilder sb;
    sb.append("ForLoop(");
    sb.append(iterator);
    sb.append(")");
    return sb.as_string();
}

Array ForLoopNode::get_children() const {
    Array children;
    if (start.is_valid()) {
        children.push_back(start);
    }
    if (end.is_valid()) {
        children.push_back(end);
    }
    if (step.is_valid()) {
        children.push_back(step);
    }
    if (body.is_valid()) {
        children.push_back(body);
    }
    return children;
}

// ReturnNode implementation
void ReturnNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_value", "value"), &ReturnNode::set_value);
    ClassDB::bind_method(D_METHOD("get_value"), &ReturnNode::get_value);
    ClassDB::bind_method(D_METHOD("to_string"), &ReturnNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &ReturnNode::get_children);
}

String ReturnNode::to_string() const {
    return value.is_valid() ? "Return(...)" : "Return";
}

Array ReturnNode::get_children() const {
    Array children;
    if (value.is_valid()) {
        children.push_back(value);
    }
    return children;
}

// LiteralNode implementation
void LiteralNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_value", "value"), &LiteralNode::set_value);
    ClassDB::bind_method(D_METHOD("get_value"), &LiteralNode::get_value);
    ClassDB::bind_method(D_METHOD("to_string"), &LiteralNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &LiteralNode::get_children);
}

String LiteralNode::to_string() const {
    StringBuilder sb;
    sb.append("Literal(");
    sb.append(value);
    sb.append(")");
    return sb.as_string();
}

Array LiteralNode::get_children() const {
    return Array(); // Literals have no children
}

// IdentifierNode implementation
void IdentifierNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_name", "name"), &IdentifierNode::set_name);
    ClassDB::bind_method(D_METHOD("get_name"), &IdentifierNode::get_name);
    ClassDB::bind_method(D_METHOD("to_string"), &IdentifierNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &IdentifierNode::get_children);
}

String IdentifierNode::to_string() const {
    StringBuilder sb;
    sb.append("Identifier(");
    sb.append(name);
    sb.append(")");
    return sb.as_string();
}

Array IdentifierNode::get_children() const {
    return Array(); // Identifiers have no children
}

// BlockNode implementation
void BlockNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("add_statement", "statement"), &BlockNode::add_statement);
    ClassDB::bind_method(D_METHOD("get_statements"), &BlockNode::get_statements);
    ClassDB::bind_method(D_METHOD("to_string"), &BlockNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &BlockNode::get_children);
}

void BlockNode::add_statement(Ref<PCCASTNode> statement) {
    if (statement.is_valid()) {
        statements.push_back(statement);
    }
}

String BlockNode::to_string() const {
    StringBuilder sb;
    sb.append("Block(");
    sb.append(itos(statements.size()));
    sb.append(" statements)");
    return sb.as_string();
}

Array BlockNode::get_statements() const {
    Array stmt_array;
    for (const Ref<PCCASTNode> &stmt : statements) {
        stmt_array.push_back(stmt);
    }
    return stmt_array;
}

Array BlockNode::get_children() const {
    Array children;
    for (const Ref<PCCASTNode> &stmt : statements) {
        children.push_back(stmt);
    }
    return children;
}

// TableNode implementation
void TableNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("add_entry", "key", "value"), &TableNode::add_entry);
    ClassDB::bind_method(D_METHOD("get_keys"), &TableNode::get_keys);
    ClassDB::bind_method(D_METHOD("get_values"), &TableNode::get_values);
    ClassDB::bind_method(D_METHOD("to_string"), &TableNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &TableNode::get_children);
}

void TableNode::add_entry(Ref<PCCASTNode> key, Ref<PCCASTNode> value) {
    keys.push_back(key);
    values.push_back(value);
}

String TableNode::to_string() const {
    StringBuilder sb;
    sb.append("Table(");
    sb.append(itos(keys.size()));
    sb.append(" entries)");
    return sb.as_string();
}

Array TableNode::get_keys() const {
    Array key_array;
    for (const Ref<PCCASTNode> &key : keys) {
        key_array.push_back(key);
    }
    return key_array;
}

Array TableNode::get_values() const {
    Array value_array;
    for (const Ref<PCCASTNode> &value : values) {
        value_array.push_back(value);
    }
    return value_array;
}

Array TableNode::get_children() const {
    Array children;
    for (const Ref<PCCASTNode> &key : keys) {
        children.push_back(key);
    }
    for (const Ref<PCCASTNode> &value : values) {
        children.push_back(value);
    }
    return children;
}

// TableAccessNode implementation
void TableAccessNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_table", "table"), &TableAccessNode::set_table);
    ClassDB::bind_method(D_METHOD("get_table"), &TableAccessNode::get_table);
    ClassDB::bind_method(D_METHOD("set_key", "key"), &TableAccessNode::set_key);
    ClassDB::bind_method(D_METHOD("get_key"), &TableAccessNode::get_key);
    ClassDB::bind_method(D_METHOD("set_dot_access", "dot"), &TableAccessNode::set_dot_access);
    ClassDB::bind_method(D_METHOD("is_dot_operator"), &TableAccessNode::is_dot_operator);
    ClassDB::bind_method(D_METHOD("to_string"), &TableAccessNode::to_string);
    ClassDB::bind_method(D_METHOD("get_children"), &TableAccessNode::get_children);
}

String TableAccessNode::to_string() const {
    StringBuilder sb;
    sb.append("TableAccess(");
    sb.append(is_dot_access ? "dot" : "bracket");
    sb.append(")");
    return sb.as_string();
}

Array TableAccessNode::get_children() const {
    Array children;
    if (table.is_valid()) {
        children.push_back(table);
    }
    if (key.is_valid()) {
        children.push_back(key);
    }
    return children;
} 