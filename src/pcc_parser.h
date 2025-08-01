/**************************************************************************/
/*  pcc_parser.h                                                         */
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

#ifndef PCC_PARSER_H
#define PCC_PARSER_H

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "pcc_ast.h"

// Forward declarations
class PCCASTNode;

// Token types for the parser
enum PCCTokenType {
    TOKEN_EOF,
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_STRING,
    TOKEN_KEYWORD,
    TOKEN_OPERATOR,
    TOKEN_PUNCTUATION
};

// Keywords
enum PCCKeyword {
    KW_FUNCTION,
    KW_IF,
    KW_THEN,
    KW_ELSE,
    KW_ELSEIF,
    KW_WHILE,
    KW_DO,
    KW_FOR,
    KW_RETURN,
    KW_LOCAL,
    KW_AND,
    KW_OR,
    KW_NOT,
    KW_TRUE,
    KW_FALSE,
    KW_NIL,
    KW_END,
    // Ultra-compact keywords
    KW_F,      // function
    KW_R,      // return
    KW_Q,      // if (question mark)
    KW_W,      // while (wave)
    KW_L,      // for loop
    KW_T,      // true
    KW_N       // nil
};

// Token structure
struct PCCToken {
    PCCTokenType type;
    String value;
    int line;
    int column;
    
    PCCToken() : type(TOKEN_EOF), line(0), column(0) {}
    PCCToken(PCCTokenType p_type, const String &p_value, int p_line, int p_column) 
        : type(p_type), value(p_value), line(p_line), column(p_column) {}
};

// Parser class
class PCCParser : public RefCounted {
    GDCLASS(PCCParser, RefCounted);

private:
    String source_code;
    Vector<PCCToken> tokens;
    int current_token;
    List<String> errors;
    
    // Tokenization
    void tokenize();
    void skip_whitespace();
    void skip_comment();
    PCCToken read_identifier(int &pos, int line, int column);
    PCCToken read_number(int &pos, int line, int column);
    PCCToken read_string(int &pos, int line, int column);
    PCCToken read_operator(int &pos, int line, int column);
    
    // Helper methods for tokenization
    bool is_identifier_start(char32_t ch);
    bool is_identifier_char(char32_t ch);
    bool is_digit(char32_t ch);
    
    // Parsing methods
    Ref<PCCASTNode> parse_program();
    Ref<PCCASTNode> parse_statement();
    Ref<PCCASTNode> parse_expression();
    Ref<PCCASTNode> parse_logical_or();
    Ref<PCCASTNode> parse_logical_and();
    Ref<PCCASTNode> parse_equality();
    Ref<PCCASTNode> parse_comparison();
    Ref<PCCASTNode> parse_term();
    Ref<PCCASTNode> parse_factor();
    Ref<PCCASTNode> parse_primary();
    Ref<PCCASTNode> parse_function_def();
    Ref<PCCASTNode> parse_function_call();
    Ref<PCCASTNode> parse_if_statement();
    Ref<PCCASTNode> parse_while_loop();
    Ref<PCCASTNode> parse_for_loop();
    Ref<PCCASTNode> parse_return();
    Ref<PCCASTNode> parse_variable_decl();
    Ref<PCCASTNode> parse_assignment_or_call();
    Ref<PCCASTNode> parse_block();
    Ref<PCCASTNode> parse_table();
    
    // Ultra-compact parsing methods
    Ref<PCCASTNode> parse_ultra_compact_statement();
    Ref<PCCASTNode> parse_ultra_compact_control();
    Ref<PCCASTNode> parse_ultra_compact_if();
    Ref<PCCASTNode> parse_ultra_compact_while();
    Ref<PCCASTNode> parse_ultra_compact_for();
    Ref<PCCASTNode> parse_ultra_compact_function();
    Ref<PCCASTNode> parse_ultra_compact_return();
    
    // Helper methods
    bool match(PCCTokenType type);
    bool match_keyword(PCCKeyword keyword);
    void advance();
    PCCToken peek();
    PCCToken current();
    void error(const String &message);

protected:
    static void _bind_methods();

public:
    // Parse the source code and return AST
    Ref<PCCASTNode> parse(const String &p_source_code);
    
    // Parse file with error reporting
    Ref<PCCASTNode> parse_file(const String &p_source);
    
    // Get parsing errors as Array for GDScript compatibility
    Array get_errors_array() const;
    
    // Get parsing errors
    const List<String> &get_errors() const { return errors; }
    
    // Clear errors
    void clear_errors() { errors.clear(); }
    
    // Debug print AST
    String debug_print_ast(Ref<PCCASTNode> p_ast, int p_indent = 0) const;
};

#endif // VRON_PARSER_H 