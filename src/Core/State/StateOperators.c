#pragma once
#include "../../Host_Device_Shared/helpers.h"
#include "../../Host_Device_Shared/shared_precompiler_directives.h"
#include <cstring>

SHARED_FILE_PREFIX

enum StateOperator {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_SIN,
    OP_COS,
    OP_ATAN2,
    OP_EXP,
    OP_SQRT,
    OP_ABS,
    OP_LOG,
    OP_FLOOR,
    OP_CEIL,
    OP_PI,
    OP_E,
    OP_PHI,
    OP_GT,
    OP_LT,
    OP_GTE,
    OP_LTE,
    OP_EQ,
    OP_NEQ,
    OP_SMOOTHLERP,
    OP_LERP,
    OP_LOGISTIC,
    OP_MIN,
    OP_MAX,
};

HOST_DEVICE inline static int get_operator_arity(StateOperator op) {
    switch (op) {
        case OP_ADD:        return 2;
        case OP_SUB:        return 2;
        case OP_MUL:        return 2;
        case OP_DIV:        return 2;
        case OP_POW:        return 2;
        case OP_SIN:        return 1;
        case OP_COS:        return 1;
        case OP_ATAN2:      return 2;
        case OP_EXP:        return 1;
        case OP_SQRT:       return 1;
        case OP_ABS:        return 1;
        case OP_LOG:        return 1;
        case OP_FLOOR:      return 1;
        case OP_CEIL:       return 1;
        case OP_PI:         return 0;
        case OP_E:          return 0;
        case OP_PHI:        return 0;
        case OP_GT:         return 2;
        case OP_LT:         return 2;
        case OP_GTE:        return 2;
        case OP_LTE:        return 2;
        case OP_EQ:         return 2;
        case OP_NEQ:        return 2;
        case OP_SMOOTHLERP: return 3;
        case OP_LERP:       return 3;
        case OP_LOGISTIC:   return 1;
        case OP_MIN:        return 2;
        case OP_MAX:        return 2;
        default:            return -1;
    }
}

HOST_DEVICE inline static float evaluate_operator(StateOperator op, float *a) {
    switch (op) {
        case OP_ADD:        return a[0] + a[1];
        case OP_SUB:        return a[0] - a[1];
        case OP_MUL:        return a[0] * a[1];
        case OP_DIV:        return a[0] / a[1];
        case OP_POW:        return powf(a[0], a[1]);
        case OP_SIN:        return sinf(a[0]);
        case OP_COS:        return cosf(a[0]);
        case OP_ATAN2:      return atan2f(a[0], a[1]);
        case OP_EXP:        return expf(a[0]);
        case OP_SQRT:       return sqrtf(a[0]);
        case OP_ABS:        return fabsf(a[0]);
        case OP_LOG:        return logf(a[0]);
        case OP_FLOOR:      return floorf(a[0]);
        case OP_CEIL:       return ceilf(a[0]);
        case OP_PI:         return 3.141592653589793f;
        case OP_E:          return 2.718281828459045f;
        case OP_PHI:        return 1.618033988749895f;
        case OP_GT:         return a[0] > a[1] ? 1.0f : 0.0f;
        case OP_LT:         return a[0] < a[1] ? 1.0f : 0.0f;
        case OP_GTE:        return a[0] >= a[1] ? 1.0f : 0.0f;
        case OP_LTE:        return a[0] <= a[1] ? 1.0f : 0.0f;
        case OP_EQ:         return fabsf(a[0] - a[1]) < 1e-9 ? 1.0f : 0.0f;
        case OP_NEQ:        return fabsf(a[0] - a[1]) >= 1e-9 ? 1.0f : 0.0f;
        case OP_SMOOTHLERP: return smoothlerp(a[0], a[1], a[2]);
        case OP_LERP:       return a[0] + a[2] * (a[1] - a[0]);
        case OP_LOGISTIC:   return 1.0f / (1.0f + expf(-a[0]));
        case OP_MIN:        return fminf(a[0], a[1]);
        case OP_MAX:        return fmaxf(a[0], a[1]);
                            // Should not reach here, return NaN
        default:            return nanf("");
    }
}

HOST_DEVICE const inline char* state_operator_to_string(StateOperator op){
    switch(op){
        case OP_ADD: return "+";
        case OP_SUB: return "-";
        case OP_MUL: return "*";
        case OP_DIV: return "/";
        case OP_POW: return "^";
        case OP_SIN: return "sin";
        case OP_COS: return "cos";
        case OP_ATAN2: return "atan2";
        case OP_EXP: return "exp";
        case OP_SQRT: return "sqrt";
        case OP_ABS: return "abs";
        case OP_LOG: return "log";
        case OP_FLOOR: return "floor";
        case OP_CEIL: return "ceil";
        case OP_PI: return "pi";
        case OP_E: return "e";
        case OP_PHI: return "phi";
        case OP_GT: return ">";
        case OP_LT: return "<";
        case OP_GTE: return ">=";
        case OP_LTE: return "<=";
        case OP_EQ: return "==";
        case OP_NEQ: return "!==";
        case OP_SMOOTHLERP: return "smoothlerp";
        case OP_LERP: return "lerp";
        case OP_LOGISTIC: return "logistic";
        case OP_MIN: return "min";
        case OP_MAX: return "max";
        default: return "UNKNOWN_OPERATOR";
    }
}

StateOperator inline parse_state_operator(const char* in){
    if(strcmp(in, "+") == 0) return OP_ADD;
    if(strcmp(in, "-") == 0) return OP_SUB;
    if(strcmp(in, "*") == 0) return OP_MUL;
    if(strcmp(in, "/") == 0) return OP_DIV;
    if(strcmp(in, "^") == 0) return OP_POW;
    if(strcmp(in, "sin") == 0) return OP_SIN;
    if(strcmp(in, "cos") == 0) return OP_COS;
    if(strcmp(in, "atan2") == 0) return OP_ATAN2;
    if(strcmp(in, "exp") == 0) return OP_EXP;
    if(strcmp(in, "sqrt") == 0) return OP_SQRT;
    if(strcmp(in, "abs") == 0) return OP_ABS;
    if(strcmp(in, "log") == 0) return OP_LOG;
    if(strcmp(in, "floor") == 0) return OP_FLOOR;
    if(strcmp(in, "ceil") == 0) return OP_CEIL;
    if(strcmp(in, "pi") == 0) return OP_PI;
    if(strcmp(in, "e") == 0) return OP_E;
    if(strcmp(in, "phi") == 0) return OP_PHI;
    if(strcmp(in, ">") == 0) return OP_GT;
    if(strcmp(in, "<") == 0) return OP_LT;
    if(strcmp(in, ">=") == 0) return OP_GTE;
    if(strcmp(in, "<=") == 0) return OP_LTE;
    if(strcmp(in, "==") == 0) return OP_EQ;
    if(strcmp(in, "!==") == 0) return OP_NEQ;
    if(strcmp(in, "smoothlerp") == 0) return OP_SMOOTHLERP;
    if(strcmp(in, "lerp") == 0) return OP_LERP;
    if(strcmp(in, "logistic") == 0) return OP_LOGISTIC;
    if(strcmp(in, "min") == 0) return OP_MIN;
    if(strcmp(in, "max") == 0) return OP_MAX;
    throw runtime_error("Unknown state operator");
    return OP_ADD;
}

SHARED_FILE_SUFFIX
