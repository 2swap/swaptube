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
    // Irreducible Term
    test_parse_and_reduce("(\\x. x)", "(\\x. x)");
    test_parse_and_reduce("(\\x. (\\y. x))", "(\\x. (\\y. x))");

    // Simple substitutions
    test_parse_and_reduce("((\\x. x) y)", "y");
    test_parse_and_reduce("((\\x. (\\y. x)) z)", "(\\y. z)");

    // Omega
    test_parse_and_reduce("((\\x. (x x)) (\\x. (x x)))", "((\\x. (x x)) (\\x. (x x)))"); // Should not reduce further

    //Test production of double bindings
    test_parse_and_reduce("((\\x. (\\y. (x y))) (\\y. y))", "(\\y. ((\\y. y) y))");

    // Test alpha renaming
    test_parse_and_reduce("((\\y. (\\x. (y x))) (\\z. (x z)))", "(\\a. ((\\z. (x z)) a))");

    // Test double-bound variables
    test_parse_and_reduce("((\\x. (x (\\x. x))) (\\z. z))", "((\\z. z) (\\x. x))");
    test_parse_and_reduce("((\\x. ((\\x. x) x)) y)", "((\\x. x) y)");

    // Triply bound variables
    test_parse_and_reduce("(\\x. ((\\x. (x (\\x. x))) (\\z. z)))", "(\\x. ((\\z. z) (\\x. x)))");

    cout << "All tests completed." << endl;
    return 0;
}
