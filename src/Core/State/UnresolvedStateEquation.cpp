#pragma once

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

    UnresolvedStateEquationComponent(std::string space_delimited_string) {
        if (space_delimited_string.length() == 0) {
            throw std::runtime_error("Empty string cannot be parsed as UnresolvedStateEquationComponent.");
        }

        if (space_delimited_string.find(' ') != std::string::npos) {
            throw std::runtime_error("String with spaces cannot be parsed as UnresolvedStateEquationComponent: " + space_delimited_string);
        }

        // Check for local variable
        if (space_delimited_string.front() == '<' && space_delimited_string.back() == '>') {
            type = UNRESOLVED_LOCAL_VARIABLE;
            content.variable_name = space_delimited_string.substr(1, space_delimited_string.length() - 2);
            return;
        }
        // Check for global variable
        if (space_delimited_string.front() == '{' && space_delimited_string.back() == '}') {
            type = UNRESOLVED_GLOBAL_VARIABLE;
            content.variable_name = space_delimited_string.substr(1, space_delimited_string.length() - 2);
            return;
        }
        // Check for parent variable
        if (space_delimited_string.front() == '[' && space_delimited_string.back() == ']') {
            type = UNRESOLVED_PARENT_VARIABLE;
            content.variable_name = space_delimited_string.substr(1, space_delimited_string.length() - 2);
            return;
        }
        // Check for CUDA tag
        if (space_delimited_string.front() == '(' && space_delimited_string.back() == ')' && space_delimited_string.length() == 3) {
            type = UNRESOLVED_CUDA_TAG;
            content.cuda_tag = space_delimited_string[1] - 'a';
            return;
        }

        // Check for constant
        char* end;
        float constant_ = strtof(space_delimited_string.c_str(), &end);
        if (*end == '\0') {
            type = UNRESOLVED_CONSTANT;
            content.constant = constant_;
            return;
        }

        // Otherwise, it must be an operator
        type = UNRESOLVED_OPERATOR;
        content.op = parse_state_operator(space_delimited_string.c_str());
    }

    std::string to_string() const {
        switch (type) {
            case UNRESOLVED_CONSTANT:
                return std::to_string(content.constant);
            case UNRESOLVED_OPERATOR:
                return state_operator_to_string(content.op);
            case UNRESOLVED_LOCAL_VARIABLE:
                return "<" + content.variable_name + ">";
            case UNRESOLVED_GLOBAL_VARIABLE:
                return "{" + content.variable_name + "}";
            case UNRESOLVED_PARENT_VARIABLE:
                return "[" + content.variable_name + "]";
            case UNRESOLVED_CUDA_TAG:
                return "(" + std::string(1, content.cuda_tag + 'a') + ")";
            default:
                return "";
        }
    }
};

class UnresolvedStateEquation {
public:
    std::vector<UnresolvedStateEquationComponent> components;

    UnresolvedStateEquation(std::string equation_string) {
        std::istringstream iss(equation_string);
        std::string token;
        while (iss >> token) {
            components.push_back(UnresolvedStateEquationComponent(token));
        }
    }

    std::list<std::string> get_local_dependencies() const {
        std::list<std::string> dependencies;
        for (const auto& component : components) {
            if (component.type == UNRESOLVED_LOCAL_VARIABLE) {
                dependencies.push_back(component.content.variable_name);
            }
        }
        return dependencies;
    }

    std::string to_string() const {
        std::ostringstream oss;
        for (size_t i = 0; i < components.size(); ++i) {
            oss << components[i].to_string();
            if (i < components.size() - 1) {
                oss << " ";
            }
        }
        return oss.str();
    }
};

