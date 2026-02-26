#pragma once

#include <string>
#include <vector>
#include <list>
#include <stdexcept>
#include "StateOperators.c"

// In state, each variable is defined as a function of other variables.
// These functions are defined by the user in the project file.
// After user equations are parsed from strings, they are converted to UnresolvedStateEquation objects.
enum UnresolvedStateEquationComponentType {
    UNRESOLVED_CONSTANT,
    UNRESOLVED_OPERATOR,
    UNRESOLVED_LOCAL_VARIABLE,
    UNRESOLVED_GLOBAL_VARIABLE,
    UNRESOLVED_PARENT_VARIABLE,
    UNRESOLVED_CUDA_TAG,
};

struct UnresolvedContent {
    float constant;
    StateOperator op;
    std::string variable_name;
    int cuda_tag;
};

class UnresolvedStateEquationComponent {
public:
    UnresolvedStateEquationComponentType type;
    UnresolvedContent content;

    UnresolvedStateEquationComponent(const std::string& space_delimited_string);
    std::string to_string() const;
};

class UnresolvedStateEquation {
public:
    std::vector<UnresolvedStateEquationComponent> components;

    UnresolvedStateEquation(const std::string& equation_string);
    std::list<std::string> get_local_dependencies() const;
    std::string to_string() const;
};
