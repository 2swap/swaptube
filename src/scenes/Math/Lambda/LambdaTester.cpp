#include <iostream>
#include <string>
#include <cassert>

#include "LambdaExpression.cpp"

using namespace std;

int main() {
    // Example lambda expressions
    LambdaExpression* expr3 = parse_lambda_from_string("((\\x. (x x)) (\\y. y))");

    // Printing the original expressions
    cout << "Original Expression 3: " << expr3->get_string() << endl;

    while (expr3->is_reducible()) {
        expr3 = expr3->reduce();
        cout << "Reduced Expression: " << expr3->get_string() << endl;
    }

    cout << "Expression in beta normal form!" << endl;

    // Cleanup
    delete expr3;

    return 0;
}
