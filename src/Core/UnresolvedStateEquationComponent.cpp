#include "StateOperators.c"

// In state, each variable is defined as a function of other variables.
// These functions are defined by the user in the project file.
// After user equations are parsed from strings, they are converted to UnresolvedStateEquation objects.
enum UnresolvedStateEquationComponentType {
    CONSTANT,
    OPERATOR,
    LOCAL_VARIABLE,
    GLOBAL_VARIABLE,
    PARENT_VARIABLE,
    CUDA_TAG,
};
struct UnresolvedStateEquationComponent {
    UnresolvedStateEquationComponentType type;
    union {
        double constant;
        StateOperator op;
        std::string variable_name;
        char cuda_tag;
    };
};

// Examples:
// 1e-4 -> constant
// <local_var> -> local variable
// {global_var} -> global variable
// [parent_var] -> parent variable
// (f) -> cuda tag
// >= -> operator
UnresolvedStateEquationComponent parse_unresolved_state_equation_constant(std::string space_delimited_string) {
    UnresolvedStateEquationComponent component;

    if (space_delimited_string.length() == 0) {
        throw std::runtime_error("Empty string cannot be parsed as UnresolvedStateEquationComponent.");
    }

    if (space_delimited_string.find(' ') != std::string::npos) {
        throw std::runtime_error("String with spaces cannot be parsed as UnresolvedStateEquationComponent: " + space_delimited_string);
    }

    // Check for local variable
    if (space_delimited_string.front() == '<' && space_delimited_string.back() == '>') {
        component.type = LOCAL_VARIABLE;
        component.variable_name = space_delimited_string.substr(1, space_delimited_string.length() - 2);
        return component;
    }
    // Check for global variable
    if (space_delimited_string.front() == '{' && space_delimited_string.back() == '}') {
        component.type = GLOBAL_VARIABLE;
        component.variable_name = space_delimited_string.substr(1, space_delimited_string.length() - 2);
        return component;
    }
    // Check for parent variable
    if (space_delimited_string.front() == '[' && space_delimited_string.back() == ']') {
        component.type = PARENT_VARIABLE;
        component.variable_name = space_delimited_string.substr(1, space_delimited_string.length() - 2);
        return component;
    }
    // Check for CUDA tag
    if (space_delimited_string.front() == '(' && space_delimited_string.back() == ')' && space_delimited_string.length() == 3) {
        component.type = CUDA_TAG;
        component.cuda_tag = space_delimited_string[1];
        return component;
    }

    // Check for constant
    char* end;
    double constant = strtod(space_delimited_string.c_str(), &end);
    if (*end == '\0') {
        component.type = CONSTANT;
        component.constant = constant;
        return component;
    }

    // Otherwise, it must be an operator
    StateOperator op = parse_state_operator(space_delimited_string.c_str());
    component.type = OPERATOR;
    component.op = op;
    return component;
}
