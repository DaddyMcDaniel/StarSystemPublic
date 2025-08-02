cmake_minimum_required(VERSION 3.16)
project(PCC_Language)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(src)

# Create bin directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# PCC VM executable (simplified without jsoncpp for now)
add_executable(pcc_vm
    src/pcc_vm_simple.cpp
)

# Set output directory
set_target_properties(pcc_vm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS pcc_vm DESTINATION bin)

# Tests
enable_testing()
add_test(NAME pcc_vm_test COMMAND pcc_vm --test)