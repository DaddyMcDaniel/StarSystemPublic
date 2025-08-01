/**************************************************************************/
/*  pcc_parser.cpp                                                       */
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

#include "pcc_parser.h"
#include "pcc_ast.h"
#include "core/string/string_builder.h"

void PCCParser::_bind_methods() {
    ClassDB::bind_method(D_METHOD("parse", "source_code"), &PCCParser::parse);
    ClassDB::bind_method(D_METHOD("parse_file", "source"), &PCCParser::parse_file);
    ClassDB::bind_method(D_METHOD("get_errors_array"), &PCCParser::get_errors_array);
    ClassDB::bind_method(D_METHOD("clear_errors"), &PCCParser::clear_errors);
    ClassDB::bind_method(D_METHOD("debug_print_ast", "ast", "indent"), &PCCParser::debug_print_ast);
}

Array PCCParser::get_errors_array() const {
    Array error_array;
    for (const String &error : errors) {
        error_array.push_back(error);
    }
    return error_array;
}

Ref<PCCASTNode> PCCParser::parse_file(const String &p_source) {
    clear_errors();
    return parse(p_source);
}

String PCCParser::debug_print_ast(Ref<PCCASTNode> p_ast, int p_indent) const {
    if (!p_ast.is_valid()) {
        return "null";
    }
    
    StringBuilder sb;
    String indent_str = String("  ").repeat(p_indent);
    
    sb.append(indent_str);
    sb.append(p_ast->get_type_name());
    sb.append(": ");
    
    // Try to get the specific node type and print details
    if (p_ast->get_type() == AST_PROGRAM) {
        Ref<ProgramNode> program = p_ast;
        if (program.is_valid()) {
            sb.append(program->to_string());
        }
    } else if (p_ast->get_type() == AST_FUNCTION_DEF) {
        Ref<FunctionDefNode> func = p_ast;
        if (func.is_valid()) {
            sb.append(func->to_string());
        }
    } else if (p_ast->get_type() == AST_FUNCTION_CALL) {
        Ref<FunctionCallNode> call = p_ast;
        if (call.is_valid()) {
            sb.append(call->to_string());
        }
    } else if (p_ast->get_type() == AST_VARIABLE_DECL) {
        Ref<VariableDeclNode> var = p_ast;
        if (var.is_valid()) {
            sb.append(var->to_string());
        }
    } else if (p_ast->get_type() == AST_ASSIGNMENT) {
        Ref<AssignmentNode> assign = p_ast;
        if (assign.is_valid()) {
            sb.append(assign->to_string());
        }
    } else if (p_ast->get_type() == AST_BINARY_OP) {
        Ref<BinaryOpNode> binop = p_ast;
        if (binop.is_valid()) {
            sb.append(binop->to_string());
        }
    } else if (p_ast->get_type() == AST_UNARY_OP) {
        Ref<UnaryOpNode> unop = p_ast;
        if (unop.is_valid()) {
            sb.append(unop->to_string());
        }
    } else if (p_ast->get_type() == AST_IF_STATEMENT) {
        Ref<IfStatementNode> ifstmt = p_ast;
        if (ifstmt.is_valid()) {
            sb.append(ifstmt->to_string());
        }
    } else if (p_ast->get_type() == AST_WHILE_LOOP) {
        Ref<WhileLoopNode> whileloop = p_ast;
        if (whileloop.is_valid()) {
            sb.append(whileloop->to_string());
        }
    } else if (p_ast->get_type() == AST_FOR_LOOP) {
        Ref<ForLoopNode> forloop = p_ast;
        if (forloop.is_valid()) {
            sb.append(forloop->to_string());
        }
    } else if (p_ast->get_type() == AST_RETURN) {
        Ref<ReturnNode> ret = p_ast;
        if (ret.is_valid()) {
            sb.append(ret->to_string());
        }
    } else if (p_ast->get_type() == AST_LITERAL) {
        Ref<LiteralNode> lit = p_ast;
        if (lit.is_valid()) {
            sb.append(lit->to_string());
        }
    } else if (p_ast->get_type() == AST_IDENTIFIER) {
        Ref<IdentifierNode> ident = p_ast;
        if (ident.is_valid()) {
            sb.append(ident->to_string());
        }
    } else if (p_ast->get_type() == AST_BLOCK) {
        Ref<BlockNode> block = p_ast;
        if (block.is_valid()) {
            sb.append(block->to_string());
        }
    } else if (p_ast->get_type() == AST_TABLE) {
        Ref<TableNode> table = p_ast;
        if (table.is_valid()) {
            sb.append(table->to_string());
        }
    } else if (p_ast->get_type() == AST_TABLE_ACCESS) {
        Ref<TableAccessNode> access = p_ast;
        if (access.is_valid()) {
            sb.append(access->to_string());
        }
    }
    
    sb.append("\n");
    
    // Print children
    Array children = p_ast->get_children();
    for (int i = 0; i < children.size(); i++) {
        Ref<PCCASTNode> child = children[i];
        if (child.is_valid()) {
            sb.append(debug_print_ast(child, p_indent + 1));
        }
    }
    
    return sb.as_string();
}

// Basic parser implementation
Ref<PCCASTNode> PCCParser::parse(const String &p_source_code) {
    print_line("PCCParser::parse called with source length: " + itos(p_source_code.length()));
    source_code = p_source_code;
    current_token = 0;
    errors.clear();
    tokens.clear();
    print_line("Parser state initialized");
    
    // Tokenize the source code
    print_line("Starting tokenization...");
    tokenize();
    print_line("Tokenization completed, token count: " + itos(tokens.size()));
    
    if (!errors.is_empty()) {
        print_line("Errors found during tokenization, returning null");
        return Ref<PCCASTNode>();
    }
    
    // Parse the program
    print_line("Starting program parsing...");
    auto result = parse_program();
    print_line("Program parsing completed");
    return result;
}

Ref<PCCASTNode> PCCParser::parse_program() {
    Ref<ProgramNode> program = memnew(ProgramNode);
    
    while (current_token < tokens.size() && tokens[current_token].type != TOKEN_EOF) {
        Ref<PCCASTNode> statement = parse_statement();
        if (statement.is_valid()) {
            program->add_statement(statement);
        } else {
            // Skip to next statement on error
            while (current_token < tokens.size() && 
                   tokens[current_token].type != TOKEN_EOF &&
                   !tokens[current_token].value.ends_with("\n")) {
                advance();
            }
        }
    }
    
    return program;
}

void PCCParser::tokenize() {
    print_line("tokenize() started");
    int pos = 0;
    int line = 1;
    int column = 1;
    
    while (pos < source_code.length()) {
        char32_t ch = source_code[pos];
        
        // Skip whitespace
        if (ch == ' ' || ch == '\t' || ch == '\r') {
            pos++;
            column++;
            continue;
        }
        
        // Handle newlines
        if (ch == '\n') {
            pos++;
            line++;
            column = 1;
            continue;
        }
        
        // Skip comments
        if (ch == '-' && pos + 1 < source_code.length() && source_code[pos + 1] == '-') {
            print_line("Found comment, skipping...");
            // Skip the comment by advancing past the newline
            while (pos < source_code.length() && source_code[pos] != '\n') {
                pos++;
            }
            if (pos < source_code.length()) {
                pos++; // Skip the newline
                line++;
                column = 1;
            }
            print_line("Comment skipped, new pos: " + itos(pos));
            continue;
        }
        
        // Read identifier or keyword
        if (is_identifier_start(ch)) {
            print_line("Reading identifier at pos " + itos(pos));
            PCCToken token = read_identifier(pos, line, column);
            tokens.push_back(token);
            column += token.value.length();
            print_line("Added identifier token: " + token.value);
            continue;
        }
        
        // Read number
        if (is_digit(ch)) {
            print_line("Reading number at pos " + itos(pos));
            PCCToken token = read_number(pos, line, column);
            tokens.push_back(token);
            column += token.value.length();
            print_line("Added number token: " + token.value);
            continue;
        }
        
        // Read string
        if (ch == '"') {
            print_line("Reading string at pos " + itos(pos));
            PCCToken token = read_string(pos, line, column);
            tokens.push_back(token);
            column += token.value.length();
            print_line("Added string token: " + token.value);
            continue;
        }
        
        // Read operator or punctuation
        print_line("Reading operator at pos " + itos(pos));
        PCCToken token = read_operator(pos, line, column);
        tokens.push_back(token);
        column += token.value.length();
        print_line("Added operator token: " + token.value);
    }
    
    // Add EOF token
    tokens.push_back(PCCToken(TOKEN_EOF, "", line, column));
    print_line("Tokenization completed, total tokens: " + itos(tokens.size()));
}

void PCCParser::skip_comment() {
    print_line("skip_comment() called");
    int pos = 0;
    while (pos < source_code.length()) {
        char32_t ch = source_code[pos];
        if (ch == '\n') {
            break;
        }
        pos++;
    }
    source_code = source_code.substr(pos);
    print_line("Comment skipped, remaining length: " + itos(source_code.length()));
}

PCCToken PCCParser::read_identifier(int &pos, int line, int column) {
    StringBuilder sb;
    
    while (pos < source_code.length() && is_identifier_char(source_code[pos])) {
        sb.append(String::chr(source_code[pos]));
        pos++;
    }
    
    String identifier = sb.as_string();
    
    // Check if it's a keyword
    if (identifier == "function") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "if") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "then") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "else") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "elseif") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "end") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "while") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "do") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "for") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "return") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "local") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "and") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "or") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "not") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "true") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "false") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    } else if (identifier == "nil") {
        return PCCToken(TOKEN_KEYWORD, identifier, line, column);
    }
    
    return PCCToken(TOKEN_IDENTIFIER, identifier, line, column);
}

PCCToken PCCParser::read_number(int &pos, int line, int column) {
    StringBuilder sb;
    
    while (pos < source_code.length() && is_digit(source_code[pos])) {
        sb.append(String::chr(source_code[pos]));
        pos++;
    }
    
    // Handle decimal point
    if (pos < source_code.length() && source_code[pos] == '.') {
        sb.append(String::chr(source_code[pos]));
        pos++;
        while (pos < source_code.length() && is_digit(source_code[pos])) {
            sb.append(String::chr(source_code[pos]));
            pos++;
        }
    }
    
    return PCCToken(TOKEN_NUMBER, sb.as_string(), line, column);
}

PCCToken PCCParser::read_string(int &pos, int line, int column) {
    StringBuilder sb;
    pos++; // Skip opening quote
    
    while (pos < source_code.length() && source_code[pos] != '"') {
        if (source_code[pos] == '\\') {
            pos++; // Skip escape character
            if (pos < source_code.length()) {
                sb.append(String::chr(source_code[pos]));
            }
        } else {
            sb.append(String::chr(source_code[pos]));
        }
        pos++;
    }
    
    if (pos < source_code.length()) {
        pos++; // Skip closing quote
    }
    
    return PCCToken(TOKEN_STRING, sb.as_string(), line, column);
}

PCCToken PCCParser::read_operator(int &pos, int line, int column) {
    char32_t ch = source_code[pos];
    pos++;
    
    // Handle two-character operators
    if (pos < source_code.length()) {
        char32_t next_ch = source_code[pos];
        String two_char = String::chr(ch) + String::chr(next_ch);
        
        if (two_char == "==" || two_char == "~=" || two_char == "<=" || 
            two_char == ">=" || two_char == "..") {
            pos++;
            return PCCToken(TOKEN_OPERATOR, two_char, line, column);
        }
    }
    
    // Single character operators and punctuation
    String single_char = String::chr(ch);
    if (single_char == "+" || single_char == "-" || single_char == "*" || 
        single_char == "/" || single_char == "%" || single_char == "=" ||
        single_char == "<" || single_char == ">" || single_char == "(" ||
        single_char == ")" || single_char == "{" || single_char == "}" ||
        single_char == "[" || single_char == "]" || single_char == "," ||
        single_char == ";" || single_char == ".") {
        return PCCToken(TOKEN_OPERATOR, single_char, line, column);
    }
    
    error("Unknown character: " + single_char);
    return PCCToken(TOKEN_OPERATOR, single_char, line, column);
}

bool PCCParser::is_identifier_start(char32_t ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
}

bool PCCParser::is_identifier_char(char32_t ch) {
    return is_identifier_start(ch) || is_digit(ch);
}

bool PCCParser::is_digit(char32_t ch) {
    return ch >= '0' && ch <= '9';
}

Ref<PCCASTNode> PCCParser::parse_statement() {
    if (current_token >= tokens.size()) {
        print_line("No more tokens to parse");
        return Ref<PCCASTNode>();
    }
    
    PCCToken token = tokens[current_token];
    print_line("Parsing statement, current token: " + token.value);
    
    if (token.type == TOKEN_KEYWORD) {
        if (token.value == "function") {
            print_line("Parsing function definition");
            return parse_function_def();
        } else if (token.value == "if") {
            print_line("Parsing if statement");
            return parse_if_statement();
        } else if (token.value == "while") {
            print_line("Parsing while loop");
            return parse_while_loop();
        } else if (token.value == "for") {
            print_line("Parsing for loop");
            return parse_for_loop();
        } else if (token.value == "local") {
            print_line("Parsing variable declaration");
            return parse_variable_decl();
        } else if (token.value == "return") {
            print_line("Parsing return statement");
            return parse_return();
        }
    } else if (token.type == TOKEN_IDENTIFIER) {
        print_line("Parsing assignment or call");
        return parse_assignment_or_call();
    }
    
    print_line("Unexpected token: " + token.value);
    error("Unexpected token: " + token.value);
    advance();
    return Ref<PCCASTNode>();
}

Ref<PCCASTNode> PCCParser::parse_expression() {
    return parse_logical_or();
}

Ref<PCCASTNode> PCCParser::parse_logical_or() {
    Ref<PCCASTNode> left = parse_logical_and();
    
    while (current_token < tokens.size() && tokens[current_token].value == "or") {
        String op = tokens[current_token].value;
        advance(); // consume 'or'
        
        Ref<PCCASTNode> right = parse_logical_and();
        if (right.is_valid()) {
            Ref<BinaryOpNode> binop = memnew(BinaryOpNode);
            binop->set_operator(op);
            binop->set_left(left);
            binop->set_right(right);
            left = binop;
        }
    }
    
    return left;
}

Ref<PCCASTNode> PCCParser::parse_logical_and() {
    Ref<PCCASTNode> left = parse_equality();
    
    while (current_token < tokens.size() && tokens[current_token].value == "and") {
        String op = tokens[current_token].value;
        advance(); // consume 'and'
        
        Ref<PCCASTNode> right = parse_equality();
        if (right.is_valid()) {
            Ref<BinaryOpNode> binop = memnew(BinaryOpNode);
            binop->set_operator(op);
            binop->set_left(left);
            binop->set_right(right);
            left = binop;
        }
    }
    
    return left;
}

Ref<PCCASTNode> PCCParser::parse_equality() {
    Ref<PCCASTNode> left = parse_comparison();
    
    while (current_token < tokens.size() && 
           (tokens[current_token].value == "==" || tokens[current_token].value == "~=")) {
        String op = tokens[current_token].value;
        advance(); // consume operator
        
        Ref<PCCASTNode> right = parse_comparison();
        if (right.is_valid()) {
            Ref<BinaryOpNode> binop = memnew(BinaryOpNode);
            binop->set_operator(op);
            binop->set_left(left);
            binop->set_right(right);
            left = binop;
        }
    }
    
    return left;
}

Ref<PCCASTNode> PCCParser::parse_comparison() {
    Ref<PCCASTNode> left = parse_term();
    
    while (current_token < tokens.size() && 
           (tokens[current_token].value == "<" || tokens[current_token].value == ">" ||
            tokens[current_token].value == "<=" || tokens[current_token].value == ">=")) {
        String op = tokens[current_token].value;
        advance(); // consume operator
        
        Ref<PCCASTNode> right = parse_term();
        if (right.is_valid()) {
            Ref<BinaryOpNode> binop = memnew(BinaryOpNode);
            binop->set_operator(op);
            binop->set_left(left);
            binop->set_right(right);
            left = binop;
        }
    }
    
    return left;
}

Ref<PCCASTNode> PCCParser::parse_term() {
    Ref<PCCASTNode> left = parse_factor();
    
    while (current_token < tokens.size() && 
           (tokens[current_token].value == "+" || tokens[current_token].value == "-")) {
        String op = tokens[current_token].value;
        advance(); // consume operator
        
        Ref<PCCASTNode> right = parse_factor();
        if (right.is_valid()) {
            Ref<BinaryOpNode> binop = memnew(BinaryOpNode);
            binop->set_operator(op);
            binop->set_left(left);
            binop->set_right(right);
            left = binop;
        }
    }
    
    return left;
}

Ref<PCCASTNode> PCCParser::parse_factor() {
    Ref<PCCASTNode> left = parse_primary();
    
    while (current_token < tokens.size() && 
           (tokens[current_token].value == "*" || tokens[current_token].value == "/" ||
            tokens[current_token].value == "%")) {
        String op = tokens[current_token].value;
        advance(); // consume operator
        
        Ref<PCCASTNode> right = parse_primary();
        if (right.is_valid()) {
            Ref<BinaryOpNode> binop = memnew(BinaryOpNode);
            binop->set_operator(op);
            binop->set_left(left);
            binop->set_right(right);
            left = binop;
        }
    }
    
    return left;
}

Ref<PCCASTNode> PCCParser::parse_primary() {
    if (current_token >= tokens.size()) {
        error("Unexpected end of input");
        return Ref<PCCASTNode>();
    }
    
    PCCToken token = tokens[current_token];
    
    if (token.type == TOKEN_IDENTIFIER) {
        Ref<IdentifierNode> ident = memnew(IdentifierNode);
        ident->set_name(token.value);
        advance();
        return ident;
    } else if (token.type == TOKEN_NUMBER) {
        Ref<LiteralNode> lit = memnew(LiteralNode);
        // Try to parse as integer first, then float
        if (token.value.contains(".")) {
            lit->set_value(token.value.to_float());
        } else {
            lit->set_value(token.value.to_int());
        }
        advance();
        return lit;
    } else if (token.type == TOKEN_STRING) {
        Ref<LiteralNode> lit = memnew(LiteralNode);
        lit->set_value(token.value);
        advance();
        return lit;
    } else if (token.value == "(") {
        advance(); // consume '('
        Ref<PCCASTNode> expr = parse_expression();
        
        if (current_token < tokens.size() && tokens[current_token].value == ")") {
            advance(); // consume ')'
        } else {
            error("Expected ')'");
        }
        
        return expr;
    } else if (token.value == "{" && token.type == TOKEN_PUNCTUATION) {
        return parse_table();
    } else if (token.value == "not") {
        advance(); // consume 'not'
        Ref<PCCASTNode> operand = parse_primary();
        if (operand.is_valid()) {
            Ref<UnaryOpNode> unop = memnew(UnaryOpNode);
            unop->set_operator("not");
            unop->set_operand(operand);
            return unop;
        }
    }
    
    error("Unexpected token: " + token.value);
    advance();
    return Ref<PCCASTNode>();
}

Ref<PCCASTNode> PCCParser::parse_function_def() {
    advance(); // consume 'function'
    
    if (current_token >= tokens.size() || tokens[current_token].type != TOKEN_IDENTIFIER) {
        error("Expected function name");
        return Ref<PCCASTNode>();
    }
    
    Ref<FunctionDefNode> func = memnew(FunctionDefNode);
    func->set_name(tokens[current_token].value);
    advance(); // consume function name
    
    // Parse parameters
    if (current_token < tokens.size() && tokens[current_token].value == "(") {
        advance(); // consume '('
        
        while (current_token < tokens.size() && tokens[current_token].value != ")") {
            if (tokens[current_token].type == TOKEN_IDENTIFIER) {
                func->add_parameter(tokens[current_token].value);
                advance();
                
                if (current_token < tokens.size() && tokens[current_token].value == ",") {
                    advance(); // consume ','
                }
            } else {
                error("Expected parameter name");
                break;
            }
        }
        
        if (current_token < tokens.size() && tokens[current_token].value == ")") {
            advance(); // consume ')'
        }
    }
    
    // Parse function body
    if (current_token < tokens.size() && tokens[current_token].value == "end") {
        advance(); // consume 'end'
    } else {
        Ref<PCCASTNode> body = parse_block();
        if (body.is_valid()) {
            func->set_body(body);
        }
    }
    
    return func;
}

Ref<PCCASTNode> PCCParser::parse_if_statement() {
    advance(); // consume 'if'
    
    Ref<IfStatementNode> ifstmt = memnew(IfStatementNode);
    
    // Parse condition
    Ref<PCCASTNode> condition = parse_expression();
    if (condition.is_valid()) {
        ifstmt->set_condition(condition);
    }
    
    // Parse 'then'
    if (current_token < tokens.size() && tokens[current_token].value == "then") {
        advance(); // consume 'then'
    } else {
        error("Expected 'then'");
    }
    
    // Parse then body
    Ref<PCCASTNode> then_body = parse_block();
    if (then_body.is_valid()) {
        ifstmt->set_then_body(then_body);
    }
    
    // Parse elseif clauses
    while (current_token < tokens.size() && tokens[current_token].value == "elseif") {
        advance(); // consume 'elseif'
        
        Ref<PCCASTNode> elseif_condition = parse_expression();
        if (elseif_condition.is_valid()) {
            if (current_token < tokens.size() && tokens[current_token].value == "then") {
                advance(); // consume 'then'
            }
            
            Ref<PCCASTNode> elseif_body = parse_block();
            if (elseif_body.is_valid()) {
                ifstmt->add_elseif(elseif_condition, elseif_body);
            }
        }
    }
    
    // Parse else clause
    if (current_token < tokens.size() && tokens[current_token].value == "else") {
        advance(); // consume 'else'
        Ref<PCCASTNode> else_body = parse_block();
        if (else_body.is_valid()) {
            ifstmt->set_else_body(else_body);
        }
    }
    
    // Parse 'end'
    if (current_token < tokens.size() && tokens[current_token].value == "end") {
        advance(); // consume 'end'
    } else {
        error("Expected 'end'");
    }
    
    return ifstmt;
}

Ref<PCCASTNode> PCCParser::parse_while_loop() {
    advance(); // consume 'while'
    
    Ref<WhileLoopNode> whileloop = memnew(WhileLoopNode);
    
    // Parse condition
    Ref<PCCASTNode> condition = parse_expression();
    if (condition.is_valid()) {
        whileloop->set_condition(condition);
    }
    
    // Parse 'do'
    if (current_token < tokens.size() && tokens[current_token].value == "do") {
        advance(); // consume 'do'
    } else {
        error("Expected 'do'");
    }
    
    // Parse body
    Ref<PCCASTNode> body = parse_block();
    if (body.is_valid()) {
        whileloop->set_body(body);
    }
    
    // Parse 'end'
    if (current_token < tokens.size() && tokens[current_token].value == "end") {
        advance(); // consume 'end'
    } else {
        error("Expected 'end'");
    }
    
    return whileloop;
}

Ref<PCCASTNode> PCCParser::parse_for_loop() {
    advance(); // consume 'for'
    
    Ref<ForLoopNode> forloop = memnew(ForLoopNode);
    
    // Parse iterator
    if (current_token < tokens.size() && tokens[current_token].type == TOKEN_IDENTIFIER) {
        forloop->set_iterator(tokens[current_token].value);
        advance(); // consume iterator name
    } else {
        error("Expected iterator name");
        return Ref<PCCASTNode>();
    }
    
    // Parse '='
    if (current_token < tokens.size() && tokens[current_token].value == "=") {
        advance(); // consume '='
    } else {
        error("Expected '='");
        return Ref<PCCASTNode>();
    }
    
    // Parse start value
    Ref<PCCASTNode> start = parse_expression();
    if (start.is_valid()) {
        forloop->set_start(start);
    }
    
    // Parse ','
    if (current_token < tokens.size() && tokens[current_token].value == ",") {
        advance(); // consume ','
    } else {
        error("Expected ','");
        return Ref<PCCASTNode>();
    }
    
    // Parse end value
    Ref<PCCASTNode> end = parse_expression();
    if (end.is_valid()) {
        forloop->set_end(end);
    }
    
    // Parse optional step
    if (current_token < tokens.size() && tokens[current_token].value == ",") {
        advance(); // consume ','
        Ref<PCCASTNode> step = parse_expression();
        if (step.is_valid()) {
            forloop->set_step(step);
        }
    }
    
    // Parse 'do'
    if (current_token < tokens.size() && tokens[current_token].value == "do") {
        advance(); // consume 'do'
    } else {
        error("Expected 'do'");
        return Ref<PCCASTNode>();
    }
    
    // Parse body
    Ref<PCCASTNode> body = parse_block();
    if (body.is_valid()) {
        forloop->set_body(body);
    }
    
    // Parse 'end'
    if (current_token < tokens.size() && tokens[current_token].value == "end") {
        advance(); // consume 'end'
    } else {
        error("Expected 'end'");
    }
    
    return forloop;
}

Ref<PCCASTNode> PCCParser::parse_return() {
    advance(); // consume 'return'
    
    Ref<ReturnNode> ret = memnew(ReturnNode);
    
    if (current_token < tokens.size() && tokens[current_token].value != "end") {
        Ref<PCCASTNode> value = parse_expression();
        if (value.is_valid()) {
            ret->set_value(value);
        }
    }
    
    return ret;
}

Ref<PCCASTNode> PCCParser::parse_variable_decl() {
    advance(); // consume 'local'
    
    if (current_token >= tokens.size() || tokens[current_token].type != TOKEN_IDENTIFIER) {
        error("Expected variable name");
        return Ref<PCCASTNode>();
    }
    
    Ref<VariableDeclNode> var = memnew(VariableDeclNode);
    var->set_name(tokens[current_token].value);
    var->set_local(true);
    advance(); // consume variable name
    
    // Parse initializer
    if (current_token < tokens.size() && tokens[current_token].value == "=") {
        advance(); // consume '='
        Ref<PCCASTNode> init = parse_expression();
        if (init.is_valid()) {
            var->set_initializer(init);
        }
    }
    
    return var;
}

Ref<PCCASTNode> PCCParser::parse_assignment_or_call() {
    String name = tokens[current_token].value;
    advance(); // consume identifier
    
    if (current_token < tokens.size() && tokens[current_token].value == "=") {
        // Assignment
        advance(); // consume '='
        Ref<PCCASTNode> value = parse_expression();
        
        Ref<AssignmentNode> assign = memnew(AssignmentNode);
        assign->set_name(name);
        assign->set_value(value);
        return assign;
    } else if (current_token < tokens.size() && tokens[current_token].value == "(") {
        // Function call
        Ref<FunctionCallNode> call = memnew(FunctionCallNode);
        call->set_name(name);
        advance(); // consume '('
        
        // Parse arguments
        while (current_token < tokens.size() && tokens[current_token].value != ")") {
            Ref<PCCASTNode> arg = parse_expression();
            if (arg.is_valid()) {
                call->add_argument(arg);
            }
            
            if (current_token < tokens.size() && tokens[current_token].value == ",") {
                advance(); // consume ','
            }
        }
        
        if (current_token < tokens.size() && tokens[current_token].value == ")") {
            advance(); // consume ')'
        }
        
        return call;
    } else {
        // Just an identifier
        Ref<IdentifierNode> ident = memnew(IdentifierNode);
        ident->set_name(name);
        return ident;
    }
}

Ref<PCCASTNode> PCCParser::parse_table() {
    advance(); // consume '{'
    
    Ref<TableNode> table = memnew(TableNode);
    
    while (current_token < tokens.size() && tokens[current_token].value != "}") {
        Ref<PCCASTNode> key = parse_expression();
        Ref<PCCASTNode> value;
        
        if (current_token < tokens.size() && tokens[current_token].value == "=") {
            advance(); // consume '='
            value = parse_expression();
        } else {
            // Implicit key (array-style)
            value = key;
            key = Ref<PCCASTNode>(); // Will be set to array index later
        }
        
        if (value.is_valid()) {
            table->add_entry(key, value);
        }
        
        if (current_token < tokens.size() && tokens[current_token].value == ",") {
            advance(); // consume ','
        }
    }
    
    if (current_token < tokens.size() && tokens[current_token].value == "}") {
        advance(); // consume '}'
    } else {
        error("Expected '}'");
    }
    
    return table;
}

Ref<PCCASTNode> PCCParser::parse_block() {
    Ref<BlockNode> block = memnew(BlockNode);
    
    while (current_token < tokens.size() && 
           tokens[current_token].type != TOKEN_EOF &&
           tokens[current_token].value != "end" &&
           tokens[current_token].value != "else" &&
           tokens[current_token].value != "elseif") {
        
        Ref<PCCASTNode> statement = parse_statement();
        if (statement.is_valid()) {
            block->add_statement(statement);
        } else {
            // Skip to next statement on error
            while (current_token < tokens.size() && 
                   tokens[current_token].type != TOKEN_EOF &&
                   !tokens[current_token].value.ends_with("\n")) {
                advance();
            }
        }
    }
    
    return block;
}

bool PCCParser::match(PCCTokenType type) {
    if (current_token >= tokens.size()) {
        return false;
    }
    return tokens[current_token].type == type;
}

bool PCCParser::match_keyword(PCCKeyword keyword) {
    if (current_token >= tokens.size() || tokens[current_token].type != TOKEN_KEYWORD) {
        return false;
    }
    
    String value = tokens[current_token].value;
    switch (keyword) {
        case KW_FUNCTION: return value == "function";
        case KW_IF: return value == "if";
        case KW_THEN: return value == "then";
        case KW_ELSE: return value == "else";
        case KW_ELSEIF: return value == "elseif";
        case KW_END: return value == "end";
        case KW_WHILE: return value == "while";
        case KW_DO: return value == "do";
        case KW_FOR: return value == "for";
        case KW_RETURN: return value == "return";
        case KW_LOCAL: return value == "local";
        case KW_AND: return value == "and";
        case KW_OR: return value == "or";
        case KW_NOT: return value == "not";
        case KW_TRUE: return value == "true";
        case KW_FALSE: return value == "false";
        case KW_NIL: return value == "nil";
        default: return false;
    }
}

void PCCParser::advance() {
    if (current_token < tokens.size()) {
        current_token++;
    }
}

PCCToken PCCParser::peek() {
    if (current_token < tokens.size()) {
        return tokens[current_token];
    }
    return PCCToken(TOKEN_EOF, "", 1, 1);
}

PCCToken PCCParser::current() {
    if (current_token < tokens.size()) {
        return tokens[current_token];
    }
    return PCCToken(TOKEN_EOF, "", 1, 1);
}

void PCCParser::error(const String &message) {
    errors.push_back(message);
}

// Ultra-compact parsing methods
Ref<PCCASTNode> PCCParser::parse_ultra_compact_statement() {
    // Check for ultra-compact control structures first
    if (current_token < tokens.size()) {
        String token_value = tokens[current_token].value;
        
        if (token_value == "f") {
            return parse_ultra_compact_function();
        } else if (token_value == "r") {
            return parse_ultra_compact_return();
        } else if (token_value == "?") {
            return parse_ultra_compact_if();
        } else if (token_value == "~") {
            return parse_ultra_compact_while();
        } else if (token_value == "[") {
            return parse_ultra_compact_for();
        }
    }
    
    // Fall back to regular expression parsing
    return parse_expression();
}

Ref<PCCASTNode> PCCParser::parse_ultra_compact_control() {
    return parse_ultra_compact_statement();
}

Ref<PCCASTNode> PCCParser::parse_ultra_compact_if() {
    Ref<IfStatementNode> if_node = memnew(IfStatementNode);
    
    advance(); // consume '?'
    
    // Parse condition
    Ref<PCCASTNode> condition = parse_expression();
    if_node->set_condition(condition);
    
    // Expect ':'
    if (current_token < tokens.size() && tokens[current_token].value == ":") {
        advance(); // consume ':'
    } else {
        error("Expected ':' after condition in ultra-compact if");
        return Ref<PCCASTNode>();
    }
    
    // Parse then block
    Ref<PCCASTNode> then_block = parse_block();
    if_node->set_then_body(then_block);
    
    // Check for else if (';' followed by condition and ':')
    while (current_token < tokens.size() && tokens[current_token].value == ";") {
        advance(); // consume ';'
        
        // Check if next token is a condition (not a block)
        if (current_token < tokens.size() && tokens[current_token].value != "{") {
            Ref<PCCASTNode> elseif_condition = parse_expression();
            
            if (current_token < tokens.size() && tokens[current_token].value == ":") {
                advance(); // consume ':'
                Ref<PCCASTNode> elseif_body = parse_block();
                if_node->add_elseif(elseif_condition, elseif_body);
            } else {
                error("Expected ':' after elseif condition");
                break;
            }
        } else {
            // This is the else block
            Ref<PCCASTNode> else_block = parse_block();
            if_node->set_else_body(else_block);
            break;
        }
    }
    
    return if_node;
}

Ref<PCCASTNode> PCCParser::parse_ultra_compact_while() {
    Ref<WhileLoopNode> while_node = memnew(WhileLoopNode);
    
    advance(); // consume '~'
    
    // Parse condition
    Ref<PCCASTNode> condition = parse_expression();
    while_node->set_condition(condition);
    
    // Expect ':'
    if (current_token < tokens.size() && tokens[current_token].value == ":") {
        advance(); // consume ':'
    } else {
        error("Expected ':' after condition in ultra-compact while");
        return Ref<PCCASTNode>();
    }
    
    // Parse body
    Ref<PCCASTNode> body = parse_block();
    while_node->set_body(body);
    
    return while_node;
}

Ref<PCCASTNode> PCCParser::parse_ultra_compact_for() {
    Ref<ForLoopNode> for_node = memnew(ForLoopNode);
    
    advance(); // consume '['
    
    // Parse iterator
    if (current_token < tokens.size() && tokens[current_token].type == TOKEN_IDENTIFIER) {
        String iterator = tokens[current_token].value;
        for_node->set_iterator(iterator);
        advance(); // consume iterator
    } else {
        error("Expected identifier for iterator in ultra-compact for");
        return Ref<PCCASTNode>();
    }
    
    // Expect '='
    if (current_token < tokens.size() && tokens[current_token].value == "=") {
        advance(); // consume '='
    } else {
        error("Expected '=' after iterator in ultra-compact for");
        return Ref<PCCASTNode>();
    }
    
    // Parse start expression
    Ref<PCCASTNode> start = parse_expression();
    for_node->set_start(start);
    
    // Expect ','
    if (current_token < tokens.size() && tokens[current_token].value == ",") {
        advance(); // consume ','
    } else {
        error("Expected ',' after start expression in ultra-compact for");
        return Ref<PCCASTNode>();
    }
    
    // Parse end expression
    Ref<PCCASTNode> end = parse_expression();
    for_node->set_end(end);
    
    // Optional step expression
    if (current_token < tokens.size() && tokens[current_token].value == ",") {
        advance(); // consume ','
        Ref<PCCASTNode> step = parse_expression();
        for_node->set_step(step);
    }
    
    // Expect ']'
    if (current_token < tokens.size() && tokens[current_token].value == "]") {
        advance(); // consume ']'
    } else {
        error("Expected ']' after for loop parameters");
        return Ref<PCCASTNode>();
    }
    
    // Parse body
    Ref<PCCASTNode> body = parse_block();
    for_node->set_body(body);
    
    return for_node;
}

Ref<PCCASTNode> PCCParser::parse_ultra_compact_function() {
    Ref<FunctionDefNode> func_node = memnew(FunctionDefNode);
    
    advance(); // consume 'f'
    
    // Parse function name
    if (current_token < tokens.size() && tokens[current_token].type == TOKEN_IDENTIFIER) {
        String name = tokens[current_token].value;
        func_node->set_name(name);
        advance(); // consume name
    } else {
        error("Expected function name after 'f'");
        return Ref<PCCASTNode>();
    }
    
    // Expect '('
    if (current_token < tokens.size() && tokens[current_token].value == "(") {
        advance(); // consume '('
    } else {
        error("Expected '(' after function name");
        return Ref<PCCASTNode>();
    }
    
    // Parse parameters
    while (current_token < tokens.size() && tokens[current_token].type == TOKEN_IDENTIFIER) {
        String param = tokens[current_token].value;
        func_node->add_parameter(param);
        advance(); // consume parameter
        
        if (current_token < tokens.size() && tokens[current_token].value == ",") {
            advance(); // consume ','
        } else {
            break;
        }
    }
    
    // Expect ')'
    if (current_token < tokens.size() && tokens[current_token].value == ")") {
        advance(); // consume ')'
    } else {
        error("Expected ')' after function parameters");
        return Ref<PCCASTNode>();
    }
    
    // Parse function body
    Ref<PCCASTNode> body = parse_block();
    func_node->set_body(body);
    
    return func_node;
}

Ref<PCCASTNode> PCCParser::parse_ultra_compact_return() {
    Ref<ReturnNode> return_node = memnew(ReturnNode);
    
    advance(); // consume 'r'
    
    // Optional return value
    if (current_token < tokens.size() && 
        tokens[current_token].type != TOKEN_EOF &&
        !tokens[current_token].value.ends_with("\n")) {
        Ref<PCCASTNode> value = parse_expression();
        return_node->set_value(value);
    }
    
    return return_node;
} 