#pragma once

#include <iostream>
#include <stack>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <functional>
#include "inlines.h"

using namespace std;

struct OperatorInfo {
    function<double(vector<double>&)> operator_function;
    int numOperands;
};

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
double calculator(const string& expression) {
    stack<double> stack;
    unordered_map<string, OperatorInfo> operators = {
        {"smoothlerp", {[](vector<double>& operands) { return smoothlerp(operands[0], operands[1], operands[2]); }, 3}},
        {"smoothlerp", {[](vector<double>& operands) { return lerp(operands[0], operands[1], operands[2]); }, 3}},
        {"+", {[](vector<double>& operands) { return operands[0] + operands[1]; }, 2}},
        {"-", {[](vector<double>& operands) { return operands[0] - operands[1]; }, 2}},
        {"*", {[](vector<double>& operands) { return operands[0] * operands[1]; }, 2}},
        {"/", {[](vector<double>& operands) { return operands[0] / operands[1]; }, 2}},
        {"^", {[](vector<double>& operands) { return pow(operands[0], operands[1]); }, 2}},
        {"sin", {[](vector<double>& operands) { return sin(operands[0]); }, 1}},
        {"cos", {[](vector<double>& operands) { return cos(operands[0]); }, 1}},
        {"log", {[](vector<double>& operands) { return log(operands[0]); }, 1}}
    };

    istringstream iss(expression);
    string token;
    while (iss >> token) {
        if (isdigit(token[0]) || token[0] == '.' || (token[0] == '-' && token.size() > 1)) {
            stack.push(stod(token));
        } else {
            auto it = operators.find(token);
            if (it != operators.end()) {
                OperatorInfo& opInfo = it->second;
                if (stack.size() < static_cast<size_t>(opInfo.numOperands)) {
                    throw runtime_error("Insufficient operands for operator: " + token);
                    return 0.0;
                }

                vector<double> operands(opInfo.numOperands);
                for (int i = opInfo.numOperands - 1; i >= 0; --i) {
                    operands[i] = stack.top();
                    stack.pop();
                }

                double result = opInfo.operator_function(operands);
                stack.push(result);
            } else {
                throw runtime_error("Invalid operator: " + token);
                return 0.0;
            }
        }
    }

    return stack.top();
}
