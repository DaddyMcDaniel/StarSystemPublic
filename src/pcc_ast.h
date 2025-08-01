/**************************************************************************/
/*  pcc_ast.h                                                            */
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

#ifndef PCC_AST_H
#define PCC_AST_H

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/string/string_builder.h"

// Forward declarations
class PCCASTNode;

// AST Node types (moved from pcc_parser.h)
enum PCCASTNodeType {
    AST_PROGRAM,
    AST_FUNCTION_DEF,
    AST_FUNCTION_CALL,
    AST_VARIABLE_DECL,
    AST_ASSIGNMENT,
    AST_BINARY_OP,
    AST_UNARY_OP,
    AST_IF_STATEMENT,
    AST_WHILE_LOOP,
    AST_FOR_LOOP,
    AST_RETURN,
    AST_LITERAL,
    AST_IDENTIFIER,
    AST_BLOCK,
    AST_TABLE,
    AST_TABLE_ACCESS
};

// AST Node base class
class PCCASTNode : public RefCounted {
    GDCLASS(PCCASTNode, RefCounted);

protected:
    PCCASTNodeType node_type;
    int line;
    int column;

    static void _bind_methods();

public:
    PCCASTNode() : node_type(AST_PROGRAM), line(0), column(0) {}
    PCCASTNode(PCCASTNodeType p_type, int p_line = 0, int p_column = 0) 
        : node_type(p_type), line(p_line), column(p_column) {}
    
    virtual ~PCCASTNode() {}
    
    PCCASTNodeType get_type() const { return node_type; }
    int get_type_int() const { return static_cast<int>(node_type); }
    String get_type_name() const;
    int get_line() const { return line; }
    int get_column() const { return column; }
    
    virtual String to_string() const { return get_type_name(); }
    virtual Array get_children() const { return Array(); }
    virtual bool is_valid() const { return true; }
};

// Specific AST Node subclasses
class ProgramNode : public PCCASTNode {
    GDCLASS(ProgramNode, PCCASTNode);

private:
    Vector<Ref<PCCASTNode>> statements;

protected:
    static void _bind_methods();

public:
    ProgramNode() : PCCASTNode(AST_PROGRAM) {}
    
    void add_statement(Ref<PCCASTNode> statement);
    Array get_statements() const;
    
    String to_string() const;
    Array get_children() const;
};

class FunctionDefNode : public PCCASTNode {
    GDCLASS(FunctionDefNode, PCCASTNode);

private:
    String name;
    Vector<String> parameters;
    Ref<PCCASTNode> body;

protected:
    static void _bind_methods();

public:
    FunctionDefNode() : PCCASTNode(AST_FUNCTION_DEF) {}
    
    void set_name(const String &p_name) { name = p_name; }
    String get_name() const { return name; }
    
    void add_parameter(const String &param) { parameters.push_back(param); }
    const Vector<String> &get_parameters() const { return parameters; }
    
    void set_body(Ref<PCCASTNode> p_body) { body = p_body; }
    Ref<PCCASTNode> get_body() const { return body; }
    
    String to_string() const;
    Array get_children() const;
};

class FunctionCallNode : public PCCASTNode {
    GDCLASS(FunctionCallNode, PCCASTNode);

private:
    String name;
    Vector<Ref<PCCASTNode>> arguments;

protected:
    static void _bind_methods();

public:
    FunctionCallNode() : PCCASTNode(AST_FUNCTION_CALL) {}
    
    void set_name(const String &p_name) { name = p_name; }
    String get_name() const { return name; }
    
    void add_argument(Ref<PCCASTNode> arg) { arguments.push_back(arg); }
    Array get_arguments() const;
    
    String to_string() const;
    Array get_children() const;
};

class VariableDeclNode : public PCCASTNode {
    GDCLASS(VariableDeclNode, PCCASTNode);

private:
    String name;
    Ref<PCCASTNode> initializer;
    bool is_local;

protected:
    static void _bind_methods();

public:
    VariableDeclNode() : PCCASTNode(AST_VARIABLE_DECL), is_local(false) {}
    
    void set_name(const String &p_name) { name = p_name; }
    String get_name() const { return name; }
    
    void set_initializer(Ref<PCCASTNode> init) { initializer = init; }
    Ref<PCCASTNode> get_initializer() const { return initializer; }
    
    void set_local(bool p_local) { is_local = p_local; }
    bool is_local_var() const { return is_local; }
    
    String to_string() const;
    Array get_children() const;
};

class AssignmentNode : public PCCASTNode {
    GDCLASS(AssignmentNode, PCCASTNode);

private:
    String name;
    Ref<PCCASTNode> value;

protected:
    static void _bind_methods();

public:
    AssignmentNode() : PCCASTNode(AST_ASSIGNMENT) {}
    
    void set_name(const String &p_name) { name = p_name; }
    String get_name() const { return name; }
    
    void set_value(Ref<PCCASTNode> p_value) { value = p_value; }
    Ref<PCCASTNode> get_value() const { return value; }
    
    String to_string() const;
    Array get_children() const;
};

class BinaryOpNode : public PCCASTNode {
    GDCLASS(BinaryOpNode, PCCASTNode);

private:
    String op;
    Ref<PCCASTNode> left;
    Ref<PCCASTNode> right;

protected:
    static void _bind_methods();

public:
    BinaryOpNode() : PCCASTNode(AST_BINARY_OP) {}
    
    void set_operator(const String &p_op) { op = p_op; }
    String get_operator() const { return op; }
    
    void set_left(Ref<PCCASTNode> p_left) { left = p_left; }
    Ref<PCCASTNode> get_left() const { return left; }
    
    void set_right(Ref<PCCASTNode> p_right) { right = p_right; }
    Ref<PCCASTNode> get_right() const { return right; }
    
    String to_string() const;
    Array get_children() const;
};

class UnaryOpNode : public PCCASTNode {
    GDCLASS(UnaryOpNode, PCCASTNode);

private:
    String op;
    Ref<PCCASTNode> operand;

protected:
    static void _bind_methods();

public:
    UnaryOpNode() : PCCASTNode(AST_UNARY_OP) {}
    
    void set_operator(const String &p_op) { op = p_op; }
    String get_operator() const { return op; }
    
    void set_operand(Ref<PCCASTNode> p_operand) { operand = p_operand; }
    Ref<PCCASTNode> get_operand() const { return operand; }
    
    String to_string() const;
    Array get_children() const;
};

class IfStatementNode : public PCCASTNode {
    GDCLASS(IfStatementNode, PCCASTNode);

private:
    Ref<PCCASTNode> condition;
    Ref<PCCASTNode> then_body;
    Ref<PCCASTNode> else_body;
    Vector<Ref<PCCASTNode>> elseif_conditions;
    Vector<Ref<PCCASTNode>> elseif_bodies;

protected:
    static void _bind_methods();

public:
    IfStatementNode() : PCCASTNode(AST_IF_STATEMENT) {}
    
    void set_condition(Ref<PCCASTNode> p_condition) { condition = p_condition; }
    Ref<PCCASTNode> get_condition() const { return condition; }
    
    void set_then_body(Ref<PCCASTNode> p_body) { then_body = p_body; }
    Ref<PCCASTNode> get_then_body() const { return then_body; }
    
    void set_else_body(Ref<PCCASTNode> p_body) { else_body = p_body; }
    Ref<PCCASTNode> get_else_body() const { return else_body; }
    
    void add_elseif(Ref<PCCASTNode> p_condition, Ref<PCCASTNode> p_body);
    Array get_elseif_conditions() const;
    Array get_elseif_bodies() const;
    
    String to_string() const;
    Array get_children() const;
};

class WhileLoopNode : public PCCASTNode {
    GDCLASS(WhileLoopNode, PCCASTNode);

private:
    Ref<PCCASTNode> condition;
    Ref<PCCASTNode> body;

protected:
    static void _bind_methods();

public:
    WhileLoopNode() : PCCASTNode(AST_WHILE_LOOP) {}
    
    void set_condition(Ref<PCCASTNode> p_condition) { condition = p_condition; }
    Ref<PCCASTNode> get_condition() const { return condition; }
    
    void set_body(Ref<PCCASTNode> p_body) { body = p_body; }
    Ref<PCCASTNode> get_body() const { return body; }
    
    String to_string() const;
    Array get_children() const;
};

class ForLoopNode : public PCCASTNode {
    GDCLASS(ForLoopNode, PCCASTNode);

private:
    String iterator;
    Ref<PCCASTNode> start;
    Ref<PCCASTNode> end;
    Ref<PCCASTNode> step; // Optional
    Ref<PCCASTNode> body;

protected:
    static void _bind_methods();

public:
    ForLoopNode() : PCCASTNode(AST_FOR_LOOP) {}
    
    void set_iterator(const String &p_iterator) { iterator = p_iterator; }
    String get_iterator() const { return iterator; }
    
    void set_start(Ref<PCCASTNode> p_start) { start = p_start; }
    Ref<PCCASTNode> get_start() const { return start; }
    
    void set_end(Ref<PCCASTNode> p_end) { end = p_end; }
    Ref<PCCASTNode> get_end() const { return end; }
    
    void set_step(Ref<PCCASTNode> p_step) { step = p_step; }
    Ref<PCCASTNode> get_step() const { return step; }
    
    void set_body(Ref<PCCASTNode> p_body) { body = p_body; }
    Ref<PCCASTNode> get_body() const { return body; }
    
    String to_string() const;
    Array get_children() const;
};

class ReturnNode : public PCCASTNode {
    GDCLASS(ReturnNode, PCCASTNode);

private:
    Ref<PCCASTNode> value;

protected:
    static void _bind_methods();

public:
    ReturnNode() : PCCASTNode(AST_RETURN) {}
    
    void set_value(Ref<PCCASTNode> p_value) { value = p_value; }
    Ref<PCCASTNode> get_value() const { return value; }
    
    String to_string() const;
    Array get_children() const;
};

class LiteralNode : public PCCASTNode {
    GDCLASS(LiteralNode, PCCASTNode);

private:
    Variant value;

protected:
    static void _bind_methods();

public:
    LiteralNode() : PCCASTNode(AST_LITERAL) {}
    
    void set_value(const Variant &p_value) { value = p_value; }
    Variant get_value() const { return value; }
    
    String to_string() const;
    Array get_children() const;
};

class IdentifierNode : public PCCASTNode {
    GDCLASS(IdentifierNode, PCCASTNode);

private:
    String name;

protected:
    static void _bind_methods();

public:
    IdentifierNode() : PCCASTNode(AST_IDENTIFIER) {}
    
    void set_name(const String &p_name) { name = p_name; }
    String get_name() const { return name; }
    
    String to_string() const;
    Array get_children() const;
};

class BlockNode : public PCCASTNode {
    GDCLASS(BlockNode, PCCASTNode);

private:
    Vector<Ref<PCCASTNode>> statements;

protected:
    static void _bind_methods();

public:
    BlockNode() : PCCASTNode(AST_BLOCK) {}
    
    void add_statement(Ref<PCCASTNode> statement);
    Array get_statements() const;
    
    String to_string() const;
    Array get_children() const;
};

class TableNode : public PCCASTNode {
    GDCLASS(TableNode, PCCASTNode);

private:
    Vector<Ref<PCCASTNode>> keys;
    Vector<Ref<PCCASTNode>> values;

protected:
    static void _bind_methods();

public:
    TableNode() : PCCASTNode(AST_TABLE) {}
    
    void add_entry(Ref<PCCASTNode> key, Ref<PCCASTNode> value);
    Array get_keys() const;
    Array get_values() const;
    
    String to_string() const;
    Array get_children() const;
};

class TableAccessNode : public PCCASTNode {
    GDCLASS(TableAccessNode, PCCASTNode);

private:
    Ref<PCCASTNode> table;
    Ref<PCCASTNode> key;
    bool is_dot_access;

protected:
    static void _bind_methods();

public:
    TableAccessNode() : PCCASTNode(AST_TABLE_ACCESS), is_dot_access(false) {}
    
    void set_table(Ref<PCCASTNode> p_table) { table = p_table; }
    Ref<PCCASTNode> get_table() const { return table; }
    
    void set_key(Ref<PCCASTNode> p_key) { key = p_key; }
    Ref<PCCASTNode> get_key() const { return key; }
    
    void set_dot_access(bool p_dot) { is_dot_access = p_dot; }
    bool is_dot_operator() const { return is_dot_access; }
    
    String to_string() const;
    Array get_children() const;
};

#endif // VRON_AST_H 