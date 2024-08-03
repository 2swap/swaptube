#pragma once

#include <string>
#include <stdexcept>
#include <cassert>
#include <stack>

// For failout function
#include "../../../misc/inlines.h"

using namespace std;

bool is_single_letter(const string& str) {
    return str.length() == 1 && isalpha(str[0]);
}

class LambdaVariable;
class LambdaAbstraction;
class LambdaApplication;

class LambdaExpression {
public:
    const string type;
    LambdaExpression* parent;

    LambdaExpression(const string& t, LambdaExpression* p) : type(t), parent(p) {}
    virtual ~LambdaExpression() = default;
    virtual void estrange_children_before_deletion() = 0;
    virtual LambdaExpression* clone() const = 0;

    virtual LambdaExpression* reduce() = 0;
    virtual LambdaExpression* replace(const LambdaVariable& v, const LambdaExpression& e) = 0;
    virtual string get_string() const = 0;
    virtual int parenthetical_depth() const = 0;
    virtual int num_variable_instantiations() const = 0;
    virtual bool is_reducible() const = 0;

    string get_latex() const {
        string str = get_string();
        // Basic conversion to LaTeX format
        for (size_t i = 0; i < str.length(); ++i) {
            if (str[i] == '\\') {
                str.replace(i, 1, "\\lambda ");
            } else if (str[i] == '.') {
                str.replace(i, 1, ".\\ ");
            }
        }
        return str;
    }

    int get_abstraction_depth() const {
        int depth = 0;
        const LambdaExpression* current = this;
        while (current) {
            if (current->type == "Abstraction") {
                depth++;
            }
            current = current->parent;
        }
        return depth;
    }

    class Iterator;

    Pixels draw_lambda_diagram();
};

class LambdaVariable : public LambdaExpression {
private:
    const string varname;
public:
    LambdaVariable(const string& vn, LambdaExpression* p) : LambdaExpression("Variable", p), varname(vn) {
        if(!is_single_letter(vn)){
            failout("Lambda variable was not a single letter!");
        }
    }

    LambdaExpression* clone() const override {
        return new LambdaVariable(varname, parent);
    }

    int parenthetical_depth() const override {
        return 0;
    }

    int num_variable_instantiations() const override {
        return 1;
    }

    string get_string() const override {
        return varname;
    }

    bool is_reducible() const override { return false; }

    LambdaExpression* replace(const LambdaVariable& v, const LambdaExpression& e) override {
        if(varname == v.get_string()) {
            return e.clone();
        } else {
            return this->clone();
        }
    }

    LambdaExpression* reduce() override {
        failout("Reduction was attempted, but no legal reduction was found!");
        return nullptr;
    }

    const LambdaAbstraction* get_bound_abstraction() const;

    void estrange_children_before_deletion() override {
        // LambdaVariable has no children
    }
};

class LambdaAbstraction : public LambdaExpression {
private:
    LambdaVariable* variable;
    LambdaExpression* body;
public:
    LambdaAbstraction(LambdaVariable* v, LambdaExpression* b, LambdaExpression* p) 
        : LambdaExpression("Abstraction", p), variable(v), body(b) { }

    ~LambdaAbstraction() {
        if (variable) delete variable;
        if (body) delete body;
    }

    LambdaExpression* clone() const override {
        return new LambdaAbstraction(static_cast<LambdaVariable*>(variable->clone()), body->clone(), parent);
    }

    string get_string() const override {
        return "(\\" + variable->get_string() + ". " + body->get_string() + ")";
    }

    int parenthetical_depth() const override {
        return 1 + body->parenthetical_depth();
    }

    int num_variable_instantiations() const override {
        return body->num_variable_instantiations();
    }

    bool is_reducible() const override { return body->is_reducible(); }

    LambdaExpression* replace(const LambdaVariable& v, const LambdaExpression& e) override {
        if (variable->get_string() != v.get_string()) {
            LambdaExpression* old_body = body;
            body = body->replace(v, e);
            old_body->estrange_children_before_deletion();
            delete old_body;
        }
        return this;
    }

    LambdaExpression* reduce() override {
        if(body->is_reducible()) {
            LambdaExpression* old_body = body;
            body = body->reduce();
            old_body->estrange_children_before_deletion();
            delete old_body;
        } else {
            failout("Reduction was attempted, but no legal reduction was found!");
        }
        return this;
    }

    const LambdaVariable* get_variable() const {
        return variable;
    }

    LambdaExpression* get_body() const {
        return body;
    }

    void estrange_children_before_deletion() override {
        variable = nullptr;
        body = nullptr;
    }
};

class LambdaApplication : public LambdaExpression {
private:
    LambdaExpression* first;
    LambdaExpression* second;
public:
    LambdaApplication(LambdaExpression* f, LambdaExpression* s, LambdaExpression* p) 
        : LambdaExpression("Application", p), first(f), second(s) { }

    ~LambdaApplication() {
        if (first) delete first;
        if (second) delete second;
    }

    LambdaExpression* clone() const override {
        return new LambdaApplication(first->clone(), second->clone(), parent);
    }

    string get_string() const override {
        return "(" + first->get_string() + " " + second->get_string() + ")";
    }

    int parenthetical_depth() const override {
        return 1 + max(first->parenthetical_depth(), second->parenthetical_depth());
    }

    int num_variable_instantiations() const override {
        return first->num_variable_instantiations() + second->num_variable_instantiations();
    }

    bool is_immediately_reducible() const {
        return first->type == "Abstraction";
    }

    bool is_reducible() const override {
        return is_immediately_reducible() || first->is_reducible() || second->is_reducible();
    }

    LambdaExpression* replace(const LambdaVariable& v, const LambdaExpression& e) override {
        LambdaExpression* old_first = first;
        first = first->replace(v, e);
        old_first->estrange_children_before_deletion();
        delete old_first;

        LambdaExpression* old_second = second;
        second = second->replace(v, e);
        old_second->estrange_children_before_deletion();
        delete old_second;

        return this;
    }

    LambdaExpression* reduce() override {
        if(is_immediately_reducible()) {
            LambdaAbstraction* abs = static_cast<LambdaAbstraction*>(first);
            return abs->get_body()->replace(*abs->get_variable(), *second);
        } else if(first->is_reducible()) {
            LambdaExpression* old_first = first;
            first = first->reduce();
            old_first->estrange_children_before_deletion();
            delete old_first;
            return this;
        }
        else if(second->is_reducible()) {
            LambdaExpression* old_second = second;
            second = second->reduce();
            old_second->estrange_children_before_deletion();
            delete old_second;
            return this;
        }
        else {
            failout("Reduction was attempted, but no legal reduction was found!");
            return nullptr;
        }
    }

    LambdaExpression* get_first() const {
        return first;
    }

    LambdaExpression* get_second() const {
        return second;
    }

    void estrange_children_before_deletion() override {
        first = nullptr;
        second = nullptr;
    }
};

class LambdaExpression::Iterator {
public:
    Iterator(LambdaExpression* root) : current(root) {
        if (current) stack.push(current);
    }

    bool has_next() const {
        return !stack.empty();
    }

    LambdaExpression* next() {
        if (!has_next()) {
            throw out_of_range("Iterator has no more elements.");
        }
        LambdaExpression* current = stack.top();
        stack.pop();
        push_children(current);
        return current;
    }

private:
    std::stack<LambdaExpression*> stack;
    LambdaExpression* current;

    void push_children(LambdaExpression* expr) {
        if (auto* app = dynamic_cast<LambdaApplication*>(expr)) {
            if (app->get_second()) stack.push(app->get_second());
            if (app->get_first()) stack.push(app->get_first());
        } else if (auto* abs = dynamic_cast<LambdaAbstraction*>(expr)) {
            if (abs->get_body()) stack.push(abs->get_body());
            if (abs->get_variable()) stack.push(abs->get_variable());
        }
    }
};

Pixels LambdaExpression::draw_lambda_diagram() {
    int w = num_variable_instantiations() * 4;
    int h = parenthetical_depth() * 2;
    Pixels pix(w + 5, h + 5);
    pix.fill(0xff808080);

    // Pass 1: Draw Abstractions
    Iterator it(this);
    int x = 0;
    while (it.has_next()) {
        LambdaExpression* current = it.next();
        if (current->type == "Variable") {
            x+=4;
        }
        if(current->type == "Abstraction") {
            int abstraction_y = current->get_abstraction_depth();
            int abstraction_w = dynamic_cast<LambdaAbstraction*>(current)->num_variable_instantiations() * 4 - 1;
            pix.fill_rect(x, abstraction_y, abstraction_w, 1, TRANSPARENT_WHITE);
        }
    }

    // Pass 2: Draw Applications

    // Pass 3: Draw Variables
    return pix;
}

const LambdaAbstraction* LambdaVariable::get_bound_abstraction() const {
    LambdaExpression* current = this->parent;
    while (current) {
        if (current->type == "Abstraction") {
            LambdaAbstraction* abstraction = dynamic_cast<LambdaAbstraction*>(current);
            if (abstraction && abstraction->get_variable()->get_string() == varname) {
                return abstraction;
            }
        }
        current = current->parent;
    }
    return nullptr;
}

LambdaExpression* parse_lambda_from_string(const string& input, LambdaExpression* parent = nullptr) {
    //cout << "Parsing '" << input << "'..." << endl;

    if(is_single_letter(input))
        return new LambdaVariable(input, parent);

    assert(input.size() > 3);

    if(input[0             ] == '('  &&
       input[1             ] == '\\' &&
       input[3             ] == '.'  &&
       input[4             ] == ' '  &&
       input[input.size()-1] == ')'){
        LambdaExpression* v = parse_lambda_from_string(string(1, input[2]), parent);
        LambdaExpression* b = parse_lambda_from_string(input.substr(5, input.size() - 6), parent);
        return new LambdaAbstraction(dynamic_cast<LambdaVariable*>(v), b, parent);
    }

    // Parsing application
    if (input[0] == '(' && input[input.size()-1] == ')') {
        // Find the space that separates the two parts of the application
        int level = 0;
        for (size_t i = 1; i < input.size()-1; ++i) {
            if (input[i] == '(') level++;
            if (input[i] == ')') level--;
            if (input[i] == ' ' && level == 0) {
                LambdaExpression* f = parse_lambda_from_string(input.substr(1, i-1), parent);
                LambdaExpression* s = parse_lambda_from_string(input.substr(i+1, input.size()-i-2), parent);
                return new LambdaApplication(f, s, parent);
            }
        }
    }

    // No valid pattern was matched
    failout("Failed to parse string!");
    return nullptr;
}
