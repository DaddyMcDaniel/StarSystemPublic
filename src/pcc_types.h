#ifndef PCC_TYPES_H
#define PCC_TYPES_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>

namespace PCC {

// Standalone types replacing Godot dependencies
using String = std::string;
using StringName = std::string;

// Forward declaration for Variant to avoid circular dependency
struct VariantData;

// Variant type that can hold different data types
using Variant = std::variant<
    std::nullptr_t,
    bool,
    int64_t,
    double,
    String
>;

// Reference counting base class (simplified)
class RefCounted {
public:
    RefCounted() : ref_count(1) {}
    virtual ~RefCounted() = default;
    
    void ref() { ref_count++; }
    bool unref() { 
        ref_count--;
        if (ref_count <= 0) {
            delete this;
            return true;
        }
        return false;
    }
    
    int get_ref_count() const { return ref_count; }

private:
    int ref_count;
};

// Smart pointer for reference counted objects
template<typename T>
class Ref {
public:
    Ref() : ptr(nullptr) {}
    Ref(T* p) : ptr(p) { if (ptr) ptr->ref(); }
    Ref(const Ref& other) : ptr(other.ptr) { if (ptr) ptr->ref(); }
    
    ~Ref() { if (ptr) ptr->unref(); }
    
    Ref& operator=(const Ref& other) {
        if (ptr) ptr->unref();
        ptr = other.ptr;
        if (ptr) ptr->ref();
        return *this;
    }
    
    T* operator->() const { return ptr; }
    T& operator*() const { return *ptr; }
    T* get() const { return ptr; }
    
    bool is_null() const { return ptr == nullptr; }
    
private:
    T* ptr;
};

// List type (simplified)
template<typename T>
using List = std::vector<T>;

// HashMap type 
template<typename K, typename V>
using HashMap = std::unordered_map<K, V>;

// Utility functions for Variant
class VariantUtils {
public:
    static bool is_truthy(const Variant& v) {
        return std::visit([](auto&& arg) -> bool {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                return false;
            } else if constexpr (std::is_same_v<T, bool>) {
                return arg;
            } else if constexpr (std::is_arithmetic_v<T>) {
                return arg != 0;
            } else if constexpr (std::is_same_v<T, String>) {
                return !arg.empty();
            } else {
                return true;
            }
        }, v);
    }
    
    static String to_string(const Variant& v) {
        return std::visit([](auto&& arg) -> String {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                return "null";
            } else if constexpr (std::is_same_v<T, bool>) {
                return arg ? "true" : "false";
            } else if constexpr (std::is_arithmetic_v<T>) {
                return std::to_string(arg);
            } else if constexpr (std::is_same_v<T, String>) {
                return arg;
            } else {
                return "[complex]";
            }
        }, v);
    }
};

} // namespace PCC

#endif // PCC_TYPES_H