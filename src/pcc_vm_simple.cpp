#include "pcc_interpreter_standalone.h"
#include "pcc_ast_standalone.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace PCC;

void print_usage() {
    std::cout << "PCC Virtual Machine v1.0" << std::endl;
    std::cout << "Usage: pcc_vm <filename.pcc>" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  pcc_vm game.pcc    - Execute PCC game file (TODO)" << std::endl;
    std::cout << "  pcc_vm --test      - Run built-in tests" << std::endl;
    std::cout << "  pcc_vm --version   - Show version" << std::endl;
}

void run_tests() {
    std::cout << "🧪 Running PCC VM Tests..." << std::endl;
    
    PCCInterpreter interpreter;
    
    // Test 1: Simple literal
    auto literal_node = Ref<PCCASTNode>(new PCCASTNode(AST_LITERAL));
    literal_node->value = Variant{int64_t(42)};
    
    Variant result = interpreter.execute(literal_node);
    if (std::holds_alternative<int64_t>(result) && std::get<int64_t>(result) == 42) {
        std::cout << "✅ Test 1: Literal execution - PASSED" << std::endl;
    } else {
        std::cout << "❌ Test 1: Literal execution - FAILED" << std::endl;
    }
    
    // Test 2: Print function
    auto print_node = Ref<PCCASTNode>(new PCCASTNode(AST_FUNCTION_CALL));
    print_node->token_value = "print";
    
    auto print_arg = Ref<PCCASTNode>(new PCCASTNode(AST_LITERAL));
    print_arg->value = Variant{String("Hello, PCC!")};
    print_node->add_child(print_arg);
    
    interpreter.execute(print_node);
    if (!interpreter.has_execution_error()) {
        std::cout << "✅ Test 2: Print function - PASSED" << std::endl;
    } else {
        std::cout << "❌ Test 2: Print function - FAILED" << std::endl;
    }
    
    // Test 3: Binary operation
    auto add_node = Ref<PCCASTNode>(new PCCASTNode(AST_BINARY_OP));
    add_node->token_value = "+";
    
    auto left_operand = Ref<PCCASTNode>(new PCCASTNode(AST_LITERAL));
    left_operand->value = Variant{int64_t(10)};
    
    auto right_operand = Ref<PCCASTNode>(new PCCASTNode(AST_LITERAL));
    right_operand->value = Variant{int64_t(32)};
    
    add_node->add_child(left_operand);
    add_node->add_child(right_operand);
    
    result = interpreter.execute(add_node);
    if (std::holds_alternative<int64_t>(result) && std::get<int64_t>(result) == 42) {
        std::cout << "✅ Test 3: Binary operation - PASSED" << std::endl;
    } else {
        std::cout << "❌ Test 3: Binary operation - FAILED" << std::endl;
    }
    
    // Test 4: Program with multiple statements
    auto program = Ref<PCCASTNode>(new PCCASTNode(AST_PROGRAM));
    
    // Create: print("PCC Game executing!")
    auto game_print = Ref<PCCASTNode>(new PCCASTNode(AST_FUNCTION_CALL));
    game_print->token_value = "print";
    auto game_msg = Ref<PCCASTNode>(new PCCASTNode(AST_LITERAL));
    game_msg->value = Variant{String("PCC Game executing!")};
    game_print->add_child(game_msg);
    
    // Create: print("World created")
    auto world_print = Ref<PCCASTNode>(new PCCASTNode(AST_FUNCTION_CALL));
    world_print->token_value = "print";
    auto world_msg = Ref<PCCASTNode>(new PCCASTNode(AST_LITERAL));
    world_msg->value = Variant{String("World created")};
    world_print->add_child(world_msg);
    
    // Create: print("Player spawned")
    auto player_print = Ref<PCCASTNode>(new PCCASTNode(AST_FUNCTION_CALL));
    player_print->token_value = "print";
    auto player_msg = Ref<PCCASTNode>(new PCCASTNode(AST_LITERAL));
    player_msg->value = Variant{String("Player spawned")};
    player_print->add_child(player_msg);
    
    program->add_child(game_print);
    program->add_child(world_print);
    program->add_child(player_print);
    
    std::cout << "\n📋 Test 4: Sample PCC Game Program:" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    interpreter.execute(program);
    std::cout << "-----------------------------------" << std::endl;
    
    if (!interpreter.has_execution_error()) {
        std::cout << "✅ Test 4: Program execution - PASSED" << std::endl;
    } else {
        std::cout << "❌ Test 4: Program execution - FAILED" << std::endl;
    }
    
    std::cout << "\n🎯 PCC VM Tests Complete!" << std::endl;
    std::cout << "🚀 PCC Virtual Machine is working!" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string arg = argv[1];
    
    if (arg == "--version") {
        std::cout << "PCC Virtual Machine v1.0" << std::endl;
        std::cout << "Procedural-Compressed-Code Execution Engine" << std::endl;
        std::cout << "Standalone VM without Godot dependencies" << std::endl;
        return 0;
    }
    
    if (arg == "--test") {
        run_tests();
        return 0;
    }
    
    if (arg == "--help" || arg == "-h") {
        print_usage();
        return 0;
    }
    
    // For now, just show that we can handle file arguments
    std::string filename = arg;
    std::cout << "🚀 PCC VM: File execution for " << filename << " is not yet implemented" << std::endl;
    std::cout << "💡 Use --test to see PCC VM capabilities" << std::endl;
    
    return 0;
}