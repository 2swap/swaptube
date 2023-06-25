#include <iostream>
#include <stack>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <functional>

struct OperatorInfo {
    function<double(vector<double>&)> operator_function;
    int numOperands;
};

double calculator(const string& expression, double t) {
    stack<double> stack;
    unordered_map<string, OperatorInfo> operators = {
        {"+", {[](vector<double>& operands) { return operands[0] + operands[1]; }, 2}},
        {"-", {[](vector<double>& operands) { return operands[0] - operands[1]; }, 2}},
        {"*", {[](vector<double>& operands) { return operands[0] * operands[1]; }, 2}},
        {"/", {[](vector<double>& operands) { return operands[0] / operands[1]; }, 2}},
        {"^", {[](vector<double>& operands) { return pow(operands[0], operands[1]); }, 2}},
        {"sin", {[](vector<double>& operands) { return sin(operands[0]); }, 1}},
        {"cos", {[](vector<double>& operands) { return cos(operands[0]); }, 1}}
    };

    istringstream iss(expression);
    string token;
    while (iss >> token) {
        if (isdigit(token[0]) || token[0] == '.') {
            stack.push(stod(token));
        } else if (token == "t") {
            stack.push(t);
        } else {
            auto it = operators.find(token);
            if (it != operators.end()) {
                OperatorInfo& opInfo = it->second;
                if (stack.size() < static_cast<size_t>(opInfo.numOperands)) {
                    cerr << "Insufficient operands for operator: " << token << endl;
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
                cerr << "Invalid operator: " << token << endl;
                return 0.0;
            }
        }
    }

    return stack.top();
}
