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
}

StateOperator parse_state_operator(const char* in){
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
    throw runtime_error("Unknown state operator");
}
