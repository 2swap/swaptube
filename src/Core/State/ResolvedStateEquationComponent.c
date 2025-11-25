#pragma once

#include "StateOperators.c"

enum ResolvedStateEquationComponentType {
    RESOLVED_CONSTANT,
    RESOLVED_OPERATOR,
    RESOLVED_CUDA_TAG,
};
struct ResolvedStateEquationComponent {
    ResolvedStateEquationComponentType type;
    union ResolvedContent {
        double constant;
        StateOperator op;
        char cuda_tag;
    } content;
    ResolvedStateEquationComponent(double c) : type(RESOLVED_CONSTANT) { content.constant = c; }
    ResolvedStateEquationComponent(StateOperator o) : type(RESOLVED_OPERATOR) { content.op = o; }
    ResolvedStateEquationComponent(char tag) : type(RESOLVED_CUDA_TAG) { content.cuda_tag = tag; }
};

// Treat as RPN stack evaluation
HOST_DEVICE inline float evaluate_resolved_state_equation(const int num_components, const ResolvedStateEquationComponent* components, const float* cuda_tags, const int num_cuda_tags, int& error) {
    error = 0;
    double stack[256];
    int stack_size = 0;
    for (int i = 0; i < num_components; i++) {
        const ResolvedStateEquationComponent* comp = &components[i];
        if (comp->type == RESOLVED_CONSTANT) {
            stack[stack_size] = comp->content.constant;
            stack_size++;
        } else if(comp->type == RESOLVED_OPERATOR) {
            StateOperator op = comp->content.op;
            int num_operands = get_operator_arity(op);
            if (stack_size < num_operands) {
                error = 1;
                printf("Error: Not enough operands for operator %s\n", state_operator_to_string(op));
                return 0.0;
            }
            stack_size -= num_operands;
            stack[stack_size] = evaluate_operator(op, &stack[stack_size]);
            stack_size++;
        } else if (comp->type == RESOLVED_CUDA_TAG) {
            char tag = comp->content.cuda_tag;
            int tag_index = (int)(tag - 'a');
            if (tag_index < 0 || tag_index >= num_cuda_tags) {
                error = 2;
                printf("Error: Invalid CUDA tag index %d for tag '%c'\n", tag_index, tag);
                return 0.0;
            }
            stack[stack_size++] = cuda_tags[tag_index];
        } else {
            error = 3;
            printf("Error: Unknown component type %d\n", comp->type);
            return 0.0;
        }
    }
    if (stack_size != 1) {
        error = 4;
        printf("Error: Stack size after evaluation is %d, expected 1. Equation size is %d.\n", stack_size, num_components);
        return 0.0;
    }
    return stack[0];
}

HOST_DEVICE inline void print_resolved_state_equation(const int num_components, const ResolvedStateEquationComponent* components) {
    for (int i = 0; i < num_components; i++) {
        const ResolvedStateEquationComponent* comp = &components[i];
        if (comp->type == RESOLVED_CONSTANT) {
            printf("%f ", comp->content.constant);
        } else if(comp->type == RESOLVED_OPERATOR) {
            StateOperator op = comp->content.op;
            printf("%s ", state_operator_to_string(op));
        } else if (comp->type == RESOLVED_CUDA_TAG) {
            char tag = comp->content.cuda_tag;
            printf("tag_%d ", (int)(tag));
        } else {
            printf("UNKNOWN_COMPONENT ");
        }
    }
}

