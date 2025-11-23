#include "StateOperators.c"

enum ResolvedStateEquationComponentType {
    CONSTANT,
    OPERATOR,
    CUDA_TAG,
};
struct ResolvedStateEquationComponent {
    UnresolvedStateEquationComponentType type;
    union {
        double constant;
        StateOperator op;
        char cuda_tag;
    };
};
