#ifndef PCC_INTERPRETER_STANDALONE_H
#define PCC_INTERPRETER_STANDALONE_H

#include "pcc_ast_standalone.h"
#include <iostream>

namespace PCC {

class PCCInterpreter {
private:
    struct ExecutionContext {
        HashMap<StringName, Variant> variables;
        HashMap<StringName, Ref<PCCASTNode>> functions;
        ExecutionContext* parent = nullptr;
        
        ExecutionContext() {}
        ExecutionContext(ExecutionContext* p_parent) : parent(p_parent) {}
    };
    
    ExecutionContext* current_context;
    List<String> errors;
    bool has_error = false;

public:
    PCCInterpreter() {
        current_context = new ExecutionContext();
    }
    
    ~PCCInterpreter() {
        while (current_context) {
            ExecutionContext* parent = current_context->parent;
            delete current_context;
            current_context = parent;
        }
    }

    // Main execution method
    Variant execute(Ref<PCCASTNode> ast) {
        if (ast.is_null()) {
            error("Cannot execute null AST");
            return Variant{};
        }
        
        clear_errors();
        return execute_node(ast);
    }

    // Execute a single AST node
    Variant execute_node(Ref<PCCASTNode> node) {
        if (node.is_null()) {
            return Variant{};
        }

        switch (node->type) {
            case AST_PROGRAM:
                return execute_program(node);
            case AST_LITERAL:
                return execute_literal(node);
            case AST_IDENTIFIER:
                return execute_identifier(node);
            case AST_FUNCTION_CALL:
                return execute_function_call(node);
            case AST_BINARY_OP:
                return execute_binary_op(node);
            case AST_BLOCK:
                return execute_block(node);
            case AST_VARIABLE_DECL:
                return execute_variable_decl(node);
            case AST_ASSIGNMENT:
                return execute_assignment(node);
            case AST_IF_STATEMENT:
                return execute_if_statement(node);
            case AST_WHILE_LOOP:
                return execute_while_loop(node);
            case AST_RETURN:
                return execute_return(node);
            default:
                return Variant{};
        }
    }

private:
    Variant execute_program(Ref<PCCASTNode> node) {
        Variant result{};
        for (int i = 0; i < node->get_child_count(); i++) {
            result = execute_node(node->get_child(i));
            if (has_error) break;
        }
        return result;
    }

    Variant execute_literal(Ref<PCCASTNode> node) {
        return node->value;
    }

    Variant execute_identifier(Ref<PCCASTNode> node) {
        return get_variable(node->token_value);
    }

    Variant execute_function_call(Ref<PCCASTNode> node) {
        String func_name = node->token_value;
        
        // Built-in functions
        if (func_name == "print") {
            if (node->get_child_count() > 0) {
                Variant arg = execute_node(node->get_child(0));
                std::cout << VariantUtils::to_string(arg) << std::endl;
            }
            return Variant{};
        }
        
        return Variant{};
    }

    Variant execute_binary_op(Ref<PCCASTNode> node) {
        if (node->get_child_count() != 2) {
            error("Binary operation needs exactly 2 operands");
            return Variant{};
        }

        Variant left = execute_node(node->get_child(0));
        Variant right = execute_node(node->get_child(1));
        String op = node->token_value;

        // Simple arithmetic for numbers
        if (std::holds_alternative<int64_t>(left) && std::holds_alternative<int64_t>(right)) {
            int64_t l = std::get<int64_t>(left);
            int64_t r = std::get<int64_t>(right);
            
            if (op == "+") return Variant{l + r};
            if (op == "-") return Variant{l - r};
            if (op == "*") return Variant{l * r};
            if (op == "/") return Variant{r != 0 ? l / r : 0};
            if (op == "==") return Variant{l == r};
            if (op == "<") return Variant{l < r};
            if (op == ">") return Variant{l > r};
        }

        return Variant{};
    }

    Variant execute_block(Ref<PCCASTNode> node) {
        Variant result{};
        for (int i = 0; i < node->get_child_count(); i++) {
            result = execute_node(node->get_child(i));
            if (has_error) break;
        }
        return result;
    }

    Variant execute_variable_decl(Ref<PCCASTNode> node) {
        return Variant{};
    }

    Variant execute_assignment(Ref<PCCASTNode> node) {
        return Variant{};
    }

    Variant execute_if_statement(Ref<PCCASTNode> node) {
        return Variant{};
    }

    Variant execute_while_loop(Ref<PCCASTNode> node) {
        return Variant{};
    }

    Variant execute_return(Ref<PCCASTNode> node) {
        return Variant{};
    }

    // Variable management
    void set_variable(const StringName& name, const Variant& value) {
        current_context->variables[name] = value;
    }

    Variant get_variable(const StringName& name) {
        ExecutionContext* ctx = current_context;
        while (ctx) {
            auto it = ctx->variables.find(name);
            if (it != ctx->variables.end()) {
                return it->second;
            }
            ctx = ctx->parent;
        }
        return Variant{};
    }

    // Error handling
    void error(const String& message) {
        errors.push_back(message);
        has_error = true;
        std::cerr << "PCC Error: " << message << std::endl;
    }

public:
    const List<String>& get_errors() const { return errors; }
    void clear_errors() { errors.clear(); has_error = false; }
    bool has_execution_error() const { return has_error; }
};

} // namespace PCC

#endif // PCC_INTERPRETER_STANDALONE_H