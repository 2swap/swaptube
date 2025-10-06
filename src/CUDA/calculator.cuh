#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include "../Host_Device_Shared/helpers.h"

// TODO this include kinda doesn't make sense because we are not actually using
// this calculator on host yet.
#include "../Host_Device_Shared/shared_precompiler_directives.h"

typedef double (*OperatorFunc)(double *a, int n);

typedef struct {
    const char *name;
    OperatorFunc func;
    int numOperands;
} OperatorInfo;

/**
 * Basic calculator for evaluating mathematical expressions. The calculator
 * supports basic arithmetic operators, as well as a few trigonometric functions.
 * 
 * Notation:
 * The calculator parses expressions in Reverse Polish Notation (RPN) also known as 
 * postfix notation. In this notation, the operator follows the operands. This eliminates
 * the need for parentheses as long as the number of operands for each operator is fixed.
 *
 * Supported Operators and Functions:
 * 1. Basic arithmetic:
 *    - '+' : addition (binary operator)
 *    - '-' : subtraction (binary operator)
 *    - '*' : multiplication (binary operator)
 *    - '/' : division (binary operator)
 *    - '^' : power (binary operator)
 * 2. Trigonometric functions:
 *    - 'sin' : sine function (unary operator)
 *    - 'cos' : cosine function (unary operator)
 * 3. Variable:
 *    - 't' : a variable whose value can be passed as an argument to the calculator function.
 * 
 * Examples:
 * 
 * 1. Basic arithmetic:
 *    - "3 4 +"     => 3 + 4 = 7
 *    - "5 2 -"     => 5 - 2 = 3
 *    - "3 4 *"     => 3 * 4 = 12
 *    - "8 4 /"     => 8 / 4 = 2
 *    - "2 3 ^"     => 2^3 = 8
 * 2. Combination of operators:
 *    - "3 4 + 2 *" => (3 + 4) * 2 = 14
 * 3. Using trigonometric functions:
 *    - "pi sin"    => sin(pi) = 0 (assuming the variable 'pi' represents 3.14159...)
 *    - "0 cos"     => cos(0) = 1
 * 4. Using the 't' variable:
 *    - If the expression is "t 2 +", and the value of 't' provided is 3, 
 *      the result will be 3 + 2 = 5
 *
 * Note:
 * - If the expression has invalid operators, or insufficient operands, the function will print 
 *   an error message and return 0.0.
 * - The function assumes that the given expression is space-delimited.
 */

HOST_DEVICE static inline int shared_isdigit(char c) {
    return (c >= '0' && c <= '9');
}

HOST_DEVICE static inline int cuda_ftoa(double val, char *buf, int precision) {
    // Simple GPU-safe float-to-string (no exp notation)
    if (precision < 0) precision = 6;
    int i = 0;
    if (val < 0) {
        buf[i++] = '-';
        val = -val;
    }
    long long int_part = (long long)val;
    double frac = val - (double)int_part;

    // Write integer part (reverse order)
    char tmp[32];
    int len = 0;
    if (int_part == 0) tmp[len++] = '0';
    else {
        while (int_part > 0) {
            tmp[len++] = '0' + (int)(int_part % 10);
            int_part /= 10;
        }
    }
    for (int j = len - 1; j >= 0; --j)
        buf[i++] = tmp[j];

    // Fraction
    if (precision > 0) {
        buf[i++] = '.';
        for (int p = 0; p < precision; ++p) {
            frac *= 10.0;
            int d = (int)frac;
            buf[i++] = '0' + d;
            frac -= d;
        }
    }
    buf[i] = '\0';
    return i;
}

// Helper function that takes a string like "(u) (v)" as well as u and v values, and performs string substitution.
// For example, insert_tags("(u) * (v)", 1.0f, 2.0f) -> "1.0 * 2.0"
// We do not use functions like snprintf since they are not supported on device.
HOST_DEVICE void insert_tags(const char *src, double u_val, double v_val, char *dest, int maxlen) {
    int si = 0, di = 0;

    while (src[si] && di < maxlen - 1) {
        if (src[si] == '(' && src[si+2] == ')' && di < maxlen - 1) {
            char tag = src[si+1];
            char numbuf[64];
            int n = 0;

            if (tag == 'u') {
                n = cuda_ftoa(u_val, numbuf, 6);
            } else if (tag == 'v') {
                n = cuda_ftoa(v_val, numbuf, 6);
            } else {
                // unrecognized tag, copy as-is
                dest[di++] = src[si++];
                continue;
            }

            // Copy the numeric string
            for (int j = 0; j < n && di < maxlen - 1; ++j)
                dest[di++] = numbuf[j];

            si += 3; // skip "(x)"
        } else {
            dest[di++] = src[si++];
        }
    }

    dest[di] = '\0';
}

HOST_DEVICE size_t shared_strcspn(const char *s, const char *reject) {
    const char *p, *r;
    size_t count = 0;

    for (p = s; *p; p++) {
        for (r = reject; *r; r++) {
            if (*p == *r)
                return count;
        }
        count++;
    }
    return count;
}

HOST_DEVICE int shared_sscanf_token(const char *p, char *out, int maxlen) {
    int n = 0;
    while (*p && *p != ' ' && *p != '\t' && n < maxlen - 1) {
        out[n++] = *p++;
    }
    out[n] = '\0';
    return (n > 0) ? 1 : 0;
}

/* ---- Operator Implementations ---- */
HOST_DEVICE static double op_smoothlerp(double *a, int n) { return smoothlerp(a[0], a[1], a[2]); }
HOST_DEVICE static double op_lerp(double *a, int n)  { return a[1] + a[0] * (a[2] - a[1]); }
HOST_DEVICE static double op_add(double *a, int n)   { return a[0] + a[1]; }
HOST_DEVICE static double op_sub(double *a, int n)   { return a[0] - a[1]; }
HOST_DEVICE static double op_mul(double *a, int n)   { return a[0] * a[1]; }
HOST_DEVICE static double op_div(double *a, int n)   { return a[0] / a[1]; }
HOST_DEVICE static double op_pow(double *a, int n)   { return pow(a[0], a[1]); }
HOST_DEVICE static double op_sin(double *a, int n)   { return sin(a[0]); }
HOST_DEVICE static double op_cos(double *a, int n)   { return cos(a[0]); }
HOST_DEVICE static double op_exp(double *a, int n)   { return exp(a[0]); }
HOST_DEVICE static double op_sqrt(double *a, int n)  { return sqrt(a[0]); }
HOST_DEVICE static double op_abs(double *a, int n)   { return fabs(a[0]); }
HOST_DEVICE static double op_log(double *a, int n)   { return log(a[0]); }
HOST_DEVICE static double op_floor(double *a, int n) { return floor(a[0]); }
HOST_DEVICE static double op_ceil(double *a, int n)  { return ceil(a[0]); }
HOST_DEVICE static double op_pi(double *a, int n)    { return M_PI; }
HOST_DEVICE static double op_e(double *a, int n)     { return M_E; }
HOST_DEVICE static double op_phi(double *a, int n)   { return 1.618033988749895; }
HOST_DEVICE static double op_gt(double *a, int n)    { return a[0] > a[1] ? 1.0 : 0.0; }
HOST_DEVICE static double op_lt(double *a, int n)    { return a[0] < a[1] ? 1.0 : 0.0; }
HOST_DEVICE static double op_ge(double *a, int n)    { return a[0] >= a[1] ? 1.0 : 0.0; }
HOST_DEVICE static double op_le(double *a, int n)    { return a[0] <= a[1] ? 1.0 : 0.0; }
HOST_DEVICE static double op_eq(double *a, int n)    { return fabs(a[0] - a[1]) < 1e-9 ? 1.0 : 0.0; }
HOST_DEVICE static double op_neq(double *a, int n)   { return fabs(a[0] - a[1]) >= 1e-9 ? 1.0 : 0.0; }

/* ---- Operator Table ---- */
HOST_DEVICE static const OperatorInfo operators[] = {
    {"smoothlerp", op_smoothlerp, 3},
    {"lerp",       op_lerp,       3},
    {"+",          op_add,        2},
    {"-",          op_sub,        2},
    {"*",          op_mul,        2},
    {"/",          op_div,        2},
    {"^",          op_pow,        2},
    {"sin",        op_sin,        1},
    {"cos",        op_cos,        1},
    {"exp",        op_exp,        1},
    {"sqrt",       op_sqrt,       1},
    {"abs",        op_abs,        1},
    {"log",        op_log,        1},
    {"floor",      op_floor,      1},
    {"ceil",       op_ceil,       1},
    {"pi",         op_pi,         0},
    {"e",          op_e,          0},
    {"phi",        op_phi,        0},
    {">",          op_gt,         2},
    {"<",          op_lt,         2},
    {">=",         op_ge,         2},
    {"<=",         op_le,         2},
    {"==",         op_eq,         2},
    {"!=",         op_neq,        2},
    {NULL,         NULL,          0} // sentinel
};

HOST_DEVICE int shared_strcmp(const char *s1, const char *s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    // difference of unsigned chars (to match standard strcmp semantics)
    return (*(const unsigned char *)s1 - *(const unsigned char *)s2);
}

/* ---- Helper: find operator by token ---- */
HOST_DEVICE static const OperatorInfo* find_operator(const char *token) {
    for (int i = 0; operators[i].name != NULL; ++i) {
        if (shared_strcmp(token, operators[i].name) == 0)
            return &operators[i];
    }
    return NULL;
}

HOST_DEVICE double shared_atof(const char *s) {
    double sign = 1.0, value = 0.0, fraction = 0.0;
    double divisor = 1.0;
    int exponent = 0, exp_sign = 1;
    bool has_fraction = false;

    // skip whitespace
    while (*s == ' ' || *s == '\t') s++;

    // handle sign
    if (*s == '-') { sign = -1.0; s++; }
    else if (*s == '+') { s++; }

    // integer part
    while (*s >= '0' && *s <= '9') {
        value = value * 10.0 + (*s - '0');
        s++;
    }

    // fractional part
    if (*s == '.') {
        has_fraction = true;
        s++;
        while (*s >= '0' && *s <= '9') {
            fraction = fraction * 10.0 + (*s - '0');
            divisor *= 10.0;
            s++;
        }
    }

    double result = value + (has_fraction ? fraction / divisor : 0.0);

    // exponential notation
    if (*s == 'e' || *s == 'E') {
        s++;
        if (*s == '-') { exp_sign = -1; s++; }
        else if (*s == '+') { s++; }
        while (*s >= '0' && *s <= '9') {
            exponent = exponent * 10 + (*s - '0');
            s++;
        }
        double power = 1.0;
        while (exponent--) power *= 10.0;
        result = exp_sign > 0 ? result * power : result / power;
    }

    return sign * result;
}

/* ---- Main Calculator Function ---- */
HOST_DEVICE bool calculator(const char *expression, double *out_result) {
    if (!expression || !out_result) {
        printf("Calculator error: invalid input arguments.\n");
        return false;
    }

    double stack[256];
    int top = 0;

    char token[64];
    const char *p = expression;

    while (shared_sscanf_token(p, token, 64) == 1) {
        // Advance p to next token
        p += shared_strcspn(p, " ");
        while (*p == ' ') p++;

        // Is this a number?
        if (shared_isdigit(token[0]) || token[0] == '.' || 
            (token[0] == '-' && shared_isdigit(token[1]))) {
            if (top >= 256) {
                printf("Calculator error: stack overflow.\n");
                return false;
            }
            stack[top++] = shared_atof(token);
        } else {
            // Lookup operator
            const OperatorInfo *op = find_operator(token);
            if (!op) {
                printf("Calculator error: invalid operator '%s'\n", token);
                return false;
            }

            if (top < op->numOperands) {
                printf("Calculator error: insufficient operands for '%s'\n", token);
                return false;
            }

            double operands[3]; // max operands = 3 (for extendability)
            for (int i = op->numOperands - 1; i >= 0; --i)
                operands[i] = stack[--top];

            double result = op->func(operands, op->numOperands);

            if (top >= 256) {
                printf("Calculator error: stack overflow after operator '%s'\n", token);
                return false;
            }
            stack[top++] = result;
        }
    }

    if (top == 0) {
        printf("Calculator error: empty expression.\n");
        return false;
    }

    if (top > 1) {
        printf("Calculator error: invalid expression (multiple values remain on stack).\n");
        return false;
    }

    *out_result = stack[0];
    return true;
}
