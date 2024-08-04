#pragma once

#include <string>
#include <stdexcept>
#include <cassert>
#include <stack>
#include <memory>

// For failout function
#include "../../../misc/inlines.h"

using namespace std;

bool is_single_letter(const string& str) {
    return str.length() == 1 && isalpha(str[0]);
}

class LambdaVariable;
class LambdaBind;
class LambdaAbstraction;
class LambdaApplication;

class LambdaExpression : public enable_shared_from_this<LambdaExpression> {
public:
    const string type;
    shared_ptr<LambdaExpression> parent;

    LambdaExpression(const string& t, shared_ptr<LambdaExpression> p = nullptr) : type(t), parent(p) {}
    virtual shared_ptr<LambdaExpression> clone() const = 0;

    virtual shared_ptr<LambdaExpression> reduce() = 0;
    virtual shared_ptr<LambdaExpression> replace(const LambdaBind& b, const LambdaExpression& e) = 0;
    virtual string get_string() const = 0;
    virtual int parenthetical_depth() const = 0;
    virtual int num_variable_instantiations() const = 0;
    virtual bool is_reducible() const = 0;
    void set_parent(shared_ptr<LambdaExpression> p) {
        parent = p;
    }

    string get_latex() const {
        string str = get_string();
        // Basic conversion to LaTeX format
        for (size_t i = 0; i < str.length(); ++i) {
            if (str[i] == '\\') {
                str.replace(i, 1, "\\lambda ");
            }
        }
        return str;
    }

    int get_type_depth(const string& s) const {
        int depth = 0;
        shared_ptr<const LambdaExpression> current = parent;
        while (current) {
            if (current->type == s) {
                depth++;
            }
            current = current->parent;
        }
        return depth;
    }

    shared_ptr<LambdaExpression> get_nearest_ancestor_application() const {
        shared_ptr<LambdaExpression> current = parent;
        while (current) {
            if (current->type == "Application") {
                return current;
            }
            current = current->parent;
        }
        return nullptr;
    }

    int get_abstraction_depth() const {
        return get_type_depth("Abstraction");
    }

    int get_application_depth() const {
        return get_type_depth("Application");
    }

    class Iterator;

    Pixels draw_lambda_diagram();
};

class LambdaBind : public LambdaExpression {
private:
    const char varname;
public:
    LambdaBind(const char vn, shared_ptr<LambdaExpression> p = nullptr) : LambdaExpression("Bind", p), varname(vn) {
        if(!isalpha(vn)){
            failout("Lambda bind was not a single letter!");
        }
    }

    shared_ptr<LambdaExpression> clone() const override {
        return make_shared<LambdaBind>(varname, parent);
    }

    int parenthetical_depth() const override {
        return 0;
    }

    int num_variable_instantiations() const override {
        return 0;
    }

    string get_string() const override {
        return string() + varname;
    }

    bool is_reducible() const override { return false; }

    shared_ptr<LambdaExpression> replace(const LambdaBind& b, const LambdaExpression& e) override {
        failout("You cannot replace a bind!");
        return nullptr;
    }

    shared_ptr<LambdaExpression> reduce() override {
        failout("You cannot reduce a bind!");
        return nullptr;
    }
};

class LambdaVariable : public LambdaExpression {
private:
    const char varname;
public:
    LambdaVariable(const char vn, shared_ptr<LambdaExpression> p = nullptr) : LambdaExpression("Variable", p), varname(vn) {
        if(!isalpha(vn)){
            failout("Lambda variable was not a single letter!");
        }
    }

    shared_ptr<LambdaExpression> clone() const override {
        return make_shared<LambdaVariable>(varname, parent);
    }

    int parenthetical_depth() const override {
        return 0;
    }

    int num_variable_instantiations() const override {
        return 1;
    }

    string get_string() const override {
        return string() + varname;
    }

    bool is_reducible() const override { return false; }

    shared_ptr<LambdaExpression> replace(const LambdaBind& b, const LambdaExpression& e) override {
        shared_ptr<LambdaExpression> ret;
        if(get_string() == b.get_string()) {
            ret = e.clone();
        } else {
            ret = clone();
        }
        ret->parent = parent;
        return ret;
    }

    shared_ptr<LambdaExpression> reduce() override {
        failout("Reduction was attempted, but no legal reduction was found!");
        return nullptr;
    }

    shared_ptr<const LambdaAbstraction> get_bound_abstraction() const;
};

class LambdaAbstraction : public LambdaExpression {
private:
    shared_ptr<LambdaBind> bind;
    shared_ptr<LambdaExpression> body;
public:
    LambdaAbstraction(shared_ptr<LambdaBind> bi, shared_ptr<LambdaExpression> b, shared_ptr<LambdaExpression> p = nullptr) 
        : LambdaExpression("Abstraction", p), bind(bi), body(b) { }

    shared_ptr<LambdaExpression> clone() const override {
        return make_shared<LambdaAbstraction>(dynamic_pointer_cast<LambdaBind>(bind->clone()), body->clone(), parent);
    }

    string get_string() const override {
        return "(\\" + bind->get_string() + ". " + body->get_string() + ")";
    }

    int parenthetical_depth() const override {
        return 1 + body->parenthetical_depth();
    }

    int num_variable_instantiations() const override {
        return body->num_variable_instantiations();
    }

    bool is_reducible() const override { return body->is_reducible(); }

    shared_ptr<LambdaExpression> replace(const LambdaBind& b, const LambdaExpression& e) override {
        if (bind->get_string() != b.get_string()) {
            shared_ptr<LambdaExpression> old_body = body;
            body = body->replace(b, e);
            body->set_parent(shared_from_this());
        }
        return shared_from_this();
    }

    shared_ptr<LambdaExpression> reduce() override {
        if(body->is_reducible()) {
            shared_ptr<LambdaExpression> old_body = body;
            body = body->reduce();
            body->set_parent(shared_from_this());
        } else {
            failout("Reduction was attempted, but no legal reduction was found!");
        }
        return shared_from_this();
    }

    shared_ptr<LambdaBind> get_bind() const {
        return bind;
    }

    shared_ptr<LambdaExpression> get_body() const {
        return body;
    }
};

class LambdaApplication : public LambdaExpression {
private:
    shared_ptr<LambdaExpression> first;
    shared_ptr<LambdaExpression> second;
public:
    LambdaApplication(shared_ptr<LambdaExpression> f, shared_ptr<LambdaExpression> s, shared_ptr<LambdaExpression> p = nullptr) 
        : LambdaExpression("Application", p), first(f), second(s) { }

    shared_ptr<LambdaExpression> clone() const override {
        return make_shared<LambdaApplication>(first->clone(), second->clone(), parent);
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

    shared_ptr<LambdaExpression> replace(const LambdaBind& b, const LambdaExpression& e) override {
        shared_ptr<LambdaExpression> old_first = first;
        first = first->replace(b, e);
        first->set_parent(shared_from_this());

        shared_ptr<LambdaExpression> old_second = second;
        second = second->replace(b, e);
        second->set_parent(shared_from_this());

        return shared_from_this();
    }

    shared_ptr<LambdaExpression> reduce() override {
        if(is_immediately_reducible()) {
            shared_ptr<LambdaAbstraction> abs = dynamic_pointer_cast<LambdaAbstraction>(first);
            shared_ptr<LambdaExpression> abss_body = abs->get_body();
            shared_ptr<LambdaBind> abss_bind = abs->get_bind();
            shared_ptr<LambdaExpression> ret = abss_body->replace(*abss_bind, *second);
            ret->set_parent(parent);
            return ret;
        } else if(first->is_reducible()) {
            shared_ptr<LambdaExpression> old_first = first;
            first = first->reduce();
            first->set_parent(shared_from_this());
            return shared_from_this();
        }
        else if(second->is_reducible()) {
            shared_ptr<LambdaExpression> old_second = second;
            second = second->reduce();
            second->set_parent(shared_from_this());
            return shared_from_this();
        }
        else {
            failout("Reduction was attempted, but no legal reduction was found!");
            return nullptr;
        }
    }

    shared_ptr<LambdaExpression> get_first() const {
        return first;
    }

    shared_ptr<LambdaExpression> get_second() const {
        return second;
    }
};

class LambdaExpression::Iterator {
public:
    Iterator(shared_ptr<LambdaExpression> root) : current(root) {
        if (current) node_stack.push(current);
    }

    bool has_next() const {
        return !node_stack.empty();
    }

    shared_ptr<LambdaExpression> next() {
        if (!has_next()) {
            throw out_of_range("Iterator has no more elements.");
        }
        shared_ptr<LambdaExpression> current = node_stack.top();
        node_stack.pop();
        push_children(current);
        return current;
    }

private:
    stack<shared_ptr<LambdaExpression>> node_stack;
    shared_ptr<LambdaExpression> current;

    void push_children(shared_ptr<LambdaExpression> expr) {
        if (auto* app = dynamic_cast<LambdaApplication*>(expr.get())) {
            if (app->get_second()) node_stack.push(app->get_second());
            if (app->get_first()) node_stack.push(app->get_first());
        } else if (auto* abs = dynamic_cast<LambdaAbstraction*>(expr.get())) {
            if (abs->get_body()) node_stack.push(abs->get_body());
            if (abs->get_bind()) node_stack.push(abs->get_bind());
        }
    }
};

Pixels LambdaExpression::draw_lambda_diagram() {
    int w = num_variable_instantiations() * 4 - 1;
    int h = parenthetical_depth() * 2;
    Pixels pix(w, h);
    pix.fill(TRANSPARENT_BLACK);

    Iterator it(shared_from_this());
    int x = 0;
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        shared_ptr<LambdaExpression> nearest_ancestor_application = current->get_nearest_ancestor_application();
        int nearest_ancestor_application_y = nearest_ancestor_application != nullptr ? h - nearest_ancestor_application->get_application_depth() * 2 - 2 : h;
        if (current->type == "Variable") {
            int top_y = dynamic_pointer_cast<LambdaVariable>(current)->get_bound_abstraction()->get_abstraction_depth() * 2;
            pix.fill_rect(x+1, top_y, 1, nearest_ancestor_application_y-top_y, OPAQUE_WHITE);
            x+=4;
        }
        if(current->type == "Abstraction") {
            int abstraction_y = current->get_abstraction_depth() * 2;
            int abstraction_w = current->num_variable_instantiations() * 4 - 1;
            pix.fill_rect(x, abstraction_y, abstraction_w, 1, OPAQUE_WHITE);
        }
        if(current->type == "Application") {
            int application_y = h - current->get_application_depth() * 2 - 2;
            int application_w = dynamic_pointer_cast<LambdaApplication>(current)->get_first()->num_variable_instantiations() * 4 + 1;
            pix.fill_rect(x+1, application_y, 1, nearest_ancestor_application_y - application_y, OPAQUE_WHITE);
            pix.fill_rect(x+1, application_y, application_w, 1, OPAQUE_WHITE);
        }
    }

    return pix;
}

shared_ptr<const LambdaAbstraction> LambdaVariable::get_bound_abstraction() const {
    shared_ptr<LambdaExpression> current = this->parent;
    while (current) {
        if (current->type == "Abstraction") {
            shared_ptr<LambdaAbstraction> abstraction = dynamic_pointer_cast<LambdaAbstraction>(current);
            if (abstraction && abstraction->get_bind()->get_string() == get_string()) {
                return abstraction;
            }
        }
        current = current->parent;
    }
    return nullptr;
}

shared_ptr<LambdaExpression> parse_lambda_from_string(const string& input) {
    //cout << "Parsing '" << input << "'..." << endl;

    if (is_single_letter(input))
        return make_shared<LambdaVariable>(input[0]);

    assert(input.size() > 3);

    if(input[0             ] == '('  &&
       input[1             ] == '\\' &&
       input[3             ] == '.'  &&
       input[4             ] == ' '  &&
       input[input.size()-1] == ')'){
        shared_ptr<LambdaBind> bind = make_shared<LambdaBind>(input[2]);
        shared_ptr<LambdaExpression> b = parse_lambda_from_string(input.substr(5, input.size() - 6));
        shared_ptr<LambdaAbstraction> a = make_shared<LambdaAbstraction>(bind, b);
        bind->set_parent(a);
        b->set_parent(a);
        return a;
    }

    // Parsing application
    if (input[0] == '(' && input[input.size() - 1] == ')') {
        int level = 0;
        for (size_t i = 1; i < input.size() - 1; ++i) {
            if (input[i] == '(') level++;
            if (input[i] == ')') level--;
            if (input[i] == ' ' && level == 0) {
                shared_ptr<LambdaExpression> f = parse_lambda_from_string(input.substr(1, i - 1));
                shared_ptr<LambdaExpression> s = parse_lambda_from_string(input.substr(i + 1, input.size() - i - 2));
                shared_ptr<LambdaExpression> a = make_shared<LambdaApplication>(f, s);
                f->set_parent(a);
                s->set_parent(a);
                return a;
            }
        }
    }

    // No valid pattern was matched
    failout("Failed to parse string!");
    return nullptr;
}

