#ifndef PCC_AST_STANDALONE_H
#define PCC_AST_STANDALONE_H

#include "pcc_types.h"

namespace PCC {

// AST Node types
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
    AST_TABLE_ACCESS,
    AST_ARRAY,
    AST_ARRAY_ACCESS,
    AST_PRINT
};

// Forward declaration
class PCCASTNode;

// AST Node class
class PCCASTNode : public RefCounted {
public:
    PCCASTNodeType type;
    Variant value;
    String token_value;
    int line_number;
    int column_number;
    List<Ref<PCCASTNode>> children;
    HashMap<String, Variant> attributes;

    PCCASTNode(PCCASTNodeType node_type = AST_PROGRAM) 
        : type(node_type), line_number(0), column_number(0) {}
    
    virtual ~PCCASTNode() = default;

    // Add child node
    void add_child(Ref<PCCASTNode> child) {
        if (child.get()) {
            children.push_back(child);
        }
    }

    // Get child by index
    Ref<PCCASTNode> get_child(int index) const {
        if (index >= 0 && index < children.size()) {
            return children[index];
        }
        return Ref<PCCASTNode>();
    }

    // Get number of children
    int get_child_count() const {
        return children.size();
    }

    // Set/get attributes
    void set_attribute(const String& name, const Variant& value) {
        attributes[name] = value;
    }

    Variant get_attribute(const String& name) const {
        auto it = attributes.find(name);
        return (it != attributes.end()) ? it->second : Variant{};
    }

    // Utility methods
    String to_string() const {
        return String("PCCASTNode(") + std::to_string((int)type) + ")";
    }
};

} // namespace PCC

#endif // PCC_AST_STANDALONE_H