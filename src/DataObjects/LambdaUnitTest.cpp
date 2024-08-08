#include <iostream>
#include <cassert>
#include <sstream>
#include "LambdaCalculus.cpp"

using namespace std;

void test_parse_and_reduce(const string& input, const string& expected) {
    shared_ptr<LambdaExpression> expr = parse_lambda_from_string(input);
    cout << "Parsed expression: " << expr->get_string() << endl;
    if (expr->is_reducible()) {
        expr = expr->reduce();
    }
    cout << "Reduced expression: " << expr->get_string() << endl;
    assert(expr->get_string() == expected);
}

int main() {
    // Test 1: Identity function
    test_parse_and_reduce("(\\x. x)", "(\\x. x)");

    // Test 2: Constant function
    test_parse_and_reduce("(\\x. (\\y. x))", "(\\x. (\\y. x))");

    // Test 3: Application of identity function to a variable
    test_parse_and_reduce("((\\x. x) y)", "y");

    // Test 4: Application of a constant function to a variable
    test_parse_and_reduce("((\\x. (\\y. x)) z)", "(\\y. z)");

    // Test 5: Application of a function to another function
    test_parse_and_reduce("((\\x. (x x)) (\\x. (x x)))", "((\\x. (x x)) (\\x. (x x)))"); // Should not reduce further

    // Test 6: Alpha renaming
    test_parse_and_reduce("((\\x. (\\y. (x y))) (\\y. y))", "(\\y. ((\\a. a) y))");

    cout << "All tests completed." << endl;
    return 0;
}
