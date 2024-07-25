#pragma once

#include <memory>
#include <string>

//for failout function
#include "../../misc/inlines.h"

using namespace std;

bool is_single_letter(const string& str) {
        return str.length() == 1 && isalpha(str[0]);
}

class LambdaExpression {
public:
    string get_latex() {
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
    virtual unique_ptr<LambdaExpression> reduce() const = 0;
    virtual unique_ptr<LambdaExpression> replace(const LambdaVariable& v, const LambdaExpression& e) const = 0;
    virtual string get_string() const = 0;
    virtual bool is_reducible() const = 0;
    virtual ~LambdaExpression() = default;
    virtual unique_ptr<LambdaExpression> clone() const = 0;
};

class LambdaVariable : public LambdaExpression {
private:
    const string varname;
public:
    LambdaVariable(const string& vn) : varname(vn){
        if(!is_single_letter(vn)){
            failout("Lambda variable was not a single letter!");
        }
    }
    string get_string() const override {
        return varname;
    }
    bool is_reducible() const override { return false; }
    unique_ptr<LambdaExpression> replace(const LambdaVariable& v, const LambdaExpression& e) const override {
        if (varname == v.get_string())
            return e.clone();
        else
            return clone();
    }
    unique_ptr<LambdaExpression> reduce() const override {
        failout("Reduction was attempted, but no legal reduction was found!");
    }
    unique_ptr<LambdaExpression> clone() const override {
        return make_unique<LambdaVariable>(*this);
    }
};

class LambdaAbstraction : public LambdaExpression {
private:
    const LambdaVariable variable;
    const unique_ptr<LambdaExpression> body;
public:
    LambdaAbstraction(const LambdaVariable& v, unique_ptr<LambdaExpression> b) : variable(v), body(move(b)) { }
    string get_string() const override {
        return "(\\" + variable.get_string() + ". " + body->get_string() + ")";
    }
    bool is_reducible() const override { return body->is_reducible(); }
    unique_ptr<LambdaExpression> replace(const LambdaVariable& v, const LambdaExpression& e) const override {
        return make_unique<LambdaAbstraction>(variable, body->replace(v, e));
    }
    unique_ptr<LambdaExpression> reduce() const override {
        if(body->is_reducible())
            return make_unique<LambdaAbstraction>(variable, body->reduce());
        else
            failout("Reduction was attempted, but no legal reduction was found!");
    }
    unique_ptr<LambdaExpression> clone() const override {
        return make_unique<LambdaAbstraction>(variable, body->clone());
    }
};

class LambdaApplication : public LambdaExpression {
private:
    const unique_ptr<LambdaExpression> first;
    const unique_ptr<LambdaExpression> second;
public:
    LambdaApplication(unique_ptr<LambdaExpression> f, unique_ptr<LambdaExpression> s) : first(move(f)), second(move(s)) { }
    string get_string() const override {
        return "(" + first->get_string() + " " + second->get_string() + ")";
    }
    bool is_immediately_reducible() const {
        // Is the first thing an abstraction?
        return dynamic_cast<const LambdaAbstraction*>(first.get()) != nullptr;
    }
    bool is_reducible() const override {
        return is_immediately_reducible() || first->is_reducible() || second->is_reducible();
    }
    unique_ptr<LambdaExpression> replace(const LambdaVariable& v, const LambdaExpression& e) const override {
        return make_unique<LambdaApplication>(first->replace(v, e), second->replace(v, e));
    }
    unique_ptr<LambdaExpression> reduce() const override {
        if(is_immediately_reducible()) {
            const LambdaAbstraction* abs = dynamic_cast<const LambdaAbstraction*>(first.get());
            return abs->get_body().replace(abs->get_variable(), *second);
        } else if(first->is_reducible())
            return make_unique<LambdaApplication>(first->reduce(), second->clone());
        else if(second->is_reducible())
            return make_unique<LambdaApplication>(first->clone(), second->reduce());
        else
            failout("Reduction was attempted, but no legal reduction was found!");
    }
    unique_ptr<LambdaExpression> clone() const override {
        return make_unique<LambdaApplication>(first->clone(), second->clone());
    }
};

unique_ptr<LambdaExpression> parse_lambda_from_string(const string& input){
    if(is_single_letter(input))
        return make_unique<LambdaVariable>(input);

    assert(input.size() > 3);

    if(input[0             ] == '('  &&
       input[1             ] == '\\' &&
       input[3             ] == '.'  &&
       input[input.size()-1] == ')'){
        unique_ptr<LambdaExpression> v = parse_lambda_from_string(string(1, input[2]));
        unique_ptr<LambdaExpression> b = parse_lambda_from_string(input.substr(4, input.size() - 1));
        return make_unique<LambdaAbstraction>(*dynamic_cast<LambdaVariable*>(v.get()), move(b));
    }

    // Parsing application
    if (input[0] == '(' && input[input.size()-1] == ')') {
        // Find the space that separates the two parts of the application
        int level = 0;
        for (size_t i = 1; i < input.size()-1; ++i) {
            if (input[i] == '(') level++;
            if (input[i] == ')') level--;
            if (input[i] == ' ' && level == 0) {
                unique_ptr<LambdaExpression> f = parse_lambda_from_string(input.substr(1, i-1));
                unique_ptr<LambdaExpression> s = parse_lambda_from_string(input.substr(i+1, input.size()-i-2));
                return make_unique<LambdaApplication>(move(f), move(s));
            }
        }
    }

    // No valid pattern was matched
    failout("Failed to parse string!");
}

