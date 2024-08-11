#pragma once

#include <string>
#include <stdexcept>
#include <cassert>
#include <stack>
#include <memory>
#include <unordered_set>

// For failout function
#include "../misc/inlines.h"
#include "../misc/color.cpp"
#include "../misc/pixels.h"
#include "DataObject.cpp"

using namespace std;

class LambdaVariable;
class LambdaAbstraction;
class LambdaApplication;

class LambdaExpression : public DataObject, public enable_shared_from_this<LambdaExpression> {
protected:
    const string type;
    int color;
    shared_ptr<LambdaExpression> parent;
    float x;
    float y;
    float w;
    float h;
    int uid;
public:
    LambdaExpression(const string& t, const int c, shared_ptr<LambdaExpression> p = nullptr, float x = 0, float y = 0, float w = 0, float h = 0, int u = 0) : type(t), color(c), parent(p), x(x), y(y), w(w), h(h), uid(u) {}
    virtual shared_ptr<LambdaExpression> clone() const = 0;

    virtual unordered_set<char> free_variables() const = 0;
    virtual unordered_set<char> all_referenced_variables() const = 0;
    virtual shared_ptr<LambdaExpression> reduce() = 0;
    virtual shared_ptr<LambdaExpression> substitute(const char v, const LambdaExpression& e) = 0;
    virtual void rename(const char o, const char n) = 0;
    virtual string get_string() const = 0;
    virtual string get_latex() const = 0;
    virtual int parenthetical_depth() const = 0;
    virtual int num_variable_instantiations() const = 0;
    virtual bool is_reducible() const = 0;
    virtual void set_color_recursive(const int c) = 0;
    virtual void flush_uid_recursive() = 0;
    virtual void tint_recursive(const int c) = 0;
    int get_uid(){ return uid; }
    int count_reductions(){
        shared_ptr<LambdaExpression> cl = clone();
        for(int i = 0; i < 10000; i++){
            if(!cl->is_reducible())return i;
            cl = cl->reduce();
        }
        failout("count_reductions called on a term which does not reduce to BNF in under 10000 reductions!");
        return -1;
    }
    char get_fresh() const {
        if(parent != nullptr) return parent->get_fresh();
        unordered_set<char> used = all_referenced_variables();
        for(char c = 'a'; c <= 'z'; c++)
            if(used.find(c) == used.end())
                return c;
        failout("No fresh variables left!");
        return 0;
    }
    void set_parent(shared_ptr<LambdaExpression> p) {
        parent = p;
        mark_updated();
    }
    void set_color(const int c) {
        color = c;
        mark_updated();
    }
    string get_type() const {
        return type;
    }
    int get_color() const {
        return color;
    }
    shared_ptr<const LambdaExpression> get_parent() const {
        return parent;
    }

    bool check_children_parents() const;

    int get_type_depth(const string& s) const {
        int depth = 0;
        shared_ptr<const LambdaExpression> current = parent;
        while (current) {
            if (current->get_type() == s) {
                depth++;
            }
            current = current->get_parent();
        }
        return depth;
    }

    shared_ptr<const LambdaExpression> get_nearest_ancestor_application() const {
        shared_ptr<const LambdaExpression> current = parent;
        while (current) {
            if (current->get_type() == "Application") {
                return current;
            }
            current = current->get_parent();
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

    Pixels draw_lambda_diagram(float scale);
    void set_positions();
    void interpolate_positions(shared_ptr<const LambdaExpression> l2, const float weight){
        w = smoothlerp(w, l2->w, weight);
        h = smoothlerp(h, l2->h, weight);
        x = smoothlerp(x, l2->x, weight);
        y = smoothlerp(y, l2->y, weight);
    }
};

class LambdaVariable : public LambdaExpression {
private:
    char varname;
public:
    LambdaVariable(const char vn, const int c, shared_ptr<LambdaExpression> p = nullptr, float x = 0, float y = 0, float w = 0, float h = 0, int u = 0) : LambdaExpression("Variable", c, p, x, y, w, h, u), varname(vn) {
        if(!isalpha(vn)){
            failout("Lambda variable was not a letter!");
        }
    }

    unordered_set<char> all_referenced_variables() const override {
        return unordered_set<char>{varname};
    }

    unordered_set<char> free_variables() const override {
        return unordered_set<char>{varname};
    }

    shared_ptr<LambdaExpression> clone() const override {
        return make_shared<LambdaVariable>(varname, color, parent, x, y, w, h, uid);
    }

    void tint_recursive(const int c) {
        set_color(colorlerp(color, c, 0.5));
        mark_updated();
    }

    void flush_uid_recursive() {
        uid = rand();
    }

    void set_color_recursive(const int c) {
        set_color(c);
        mark_updated();
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

    string get_latex() const {
        return latex_color(color, get_string());
    }

    bool is_reducible() const override { return false; }

    void rename(const char o, const char n) override {
        if(varname == o) {
            varname = n;
        }
        mark_updated();
    }

    shared_ptr<LambdaExpression> substitute(const char v, const LambdaExpression& e) override {
        mark_updated();
        if(varname == v && get_bound_abstraction() == nullptr) { // don't substitute bound variables
            shared_ptr<LambdaExpression> ret = e.clone();
            ret->tint_recursive(color);
            return ret;
        }
        return shared_from_this();
    }

    shared_ptr<LambdaExpression> reduce() override {
        failout("Reduction was attempted, but no legal reduction was found!");
        return nullptr;
    }

    shared_ptr<const LambdaAbstraction> get_bound_abstraction() const;
};

shared_ptr<LambdaExpression> apply   (const shared_ptr<const LambdaExpression> f, const shared_ptr<const LambdaExpression> s, const int c, shared_ptr<LambdaExpression> p, float x, float y, float w, float h, int u);
shared_ptr<LambdaExpression> abstract(const char                               v, const shared_ptr<const LambdaExpression> b, const int c, shared_ptr<LambdaExpression> p, float x, float y, float w, float h, int u);

class LambdaAbstraction : public LambdaExpression {
private:
    char bound_variable;
    shared_ptr<LambdaExpression> body;
public:
    LambdaAbstraction(const char v, shared_ptr<LambdaExpression> b, const int c, shared_ptr<LambdaExpression> p = nullptr, float x = 0, float y = 0, float w = 0, float h = 0, int u = 0)
        : LambdaExpression("Abstraction", c, p, x, y, w, h, u), bound_variable(v), body(b) { }

    unordered_set<char> all_referenced_variables() const override {
        unordered_set<char> all_vars = body->all_referenced_variables();
        all_vars.insert(bound_variable);
        return all_vars;
    }

    unordered_set<char> free_variables() const override {
        unordered_set<char> free_vars = body->free_variables();
        free_vars.erase(bound_variable);
        return free_vars;
    }

    shared_ptr<LambdaExpression> clone() const override {
        shared_ptr<LambdaExpression> b = body->clone();
        return abstract(bound_variable, b, color, parent, x, y, w, h, uid);
    }

    string get_string() const override {
        return string("(\\") + bound_variable + ". " + body->get_string() + ")";
    }

    string get_latex() const {
        return latex_color(color, string("(\\lambda ") + bound_variable + ". ") + body->get_latex() + latex_color(color, ")");
    }

    void tint_recursive(const int c) {
        body->tint_recursive(c);
        set_color(colorlerp(color, c, 0.5));
        mark_updated();
    }

    void flush_uid_recursive() {
        body->flush_uid_recursive();
        uid = rand();
    }

    void set_color_recursive(const int c) {
        body->set_color_recursive(c);
        set_color(c);
        mark_updated();
    }

    int parenthetical_depth() const override {
        return 1 + body->parenthetical_depth();
    }

    int num_variable_instantiations() const override {
        return body->num_variable_instantiations();
    }

    bool is_reducible() const override { return body->is_reducible(); }

    void rename(const char o, const char n) override {
        if(bound_variable == o) {
            bound_variable = n;
        }
        body->rename(o, n);
        mark_updated();
    }

    shared_ptr<LambdaExpression> substitute(const char v, const LambdaExpression& e) override {
        mark_updated();
        if (bound_variable == v) {
            body = body->substitute(v, e);
            body->set_parent(shared_from_this());
            return shared_from_this();
        }

        unordered_set<char> fv = e.free_variables();
        if (fv.find(bound_variable) == fv.end()) {
            body = body->substitute(v, e);
            body->set_parent(shared_from_this());
            return shared_from_this();
        }

        char fresh = get_fresh();
        rename(bound_variable, fresh);
        body = body->substitute(v, e);
        body->set_parent(shared_from_this());
        return shared_from_this();
    }

    shared_ptr<LambdaExpression> reduce() override {
        if(body->is_reducible()) {
            body = body->reduce();
            body->set_parent(shared_from_this());
        } else {
            failout("Reduction was attempted, but no legal reduction was found!");
        }
        mark_updated();
        return shared_from_this();
    }

    char get_bound_variable() const {
        return bound_variable;
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
    LambdaApplication(shared_ptr<LambdaExpression> f, shared_ptr<LambdaExpression> s, const int c, shared_ptr<LambdaExpression> p = nullptr, float x = 0, float y = 0, float w = 0, float h = 0, int u = 0)
        : LambdaExpression("Application", c, p, x, y, w, h, u), first(f), second(s) { }

    unordered_set<char> all_referenced_variables() const override {
        unordered_set<char> all_vars_f = first->all_referenced_variables();
        unordered_set<char> all_vars_s = second->all_referenced_variables();
        for(const char s : all_vars_s){
            all_vars_f.insert(s);
        }
        return all_vars_f;
    }

    unordered_set<char> free_variables() const override {
        unordered_set<char> free_vars_f = first->free_variables();
        unordered_set<char> free_vars_s = second->free_variables();
        for(const char s : free_vars_s){
            free_vars_f.insert(s);
        }
        return free_vars_f;
    }

    shared_ptr<LambdaExpression> clone() const override {
        shared_ptr<LambdaExpression> f = first->clone();
        shared_ptr<LambdaExpression> s = second->clone();
        return apply(f, s, color, parent, x, y, w, h, uid);
    }

    string get_string() const override {
        return "(" + first->get_string() + " " + second->get_string() + ")";
    }

    string get_latex() const {
        return latex_color(color, "(") + first->get_latex() + " " + second->get_latex() + latex_color(color, ")");
    }

    void tint_recursive(const int c) {
        first->tint_recursive(c);
        second->tint_recursive(c);
        set_color(colorlerp(color, c, 0.5));
        mark_updated();
    }

    void flush_uid_recursive() {
        first->flush_uid_recursive();
        second->flush_uid_recursive();
        uid = rand();
    }

    void set_color_recursive(const int c) {
        first->set_color_recursive(c);
        second->set_color_recursive(c);
        set_color(c);
        mark_updated();
    }

    int parenthetical_depth() const override {
        return 1 + max(first->parenthetical_depth(), second->parenthetical_depth());
    }

    int num_variable_instantiations() const override {
        return first->num_variable_instantiations() + second->num_variable_instantiations();
    }

    bool is_immediately_reducible() const {
        return first->get_type() == "Abstraction";
    }

    bool is_reducible() const override {
        return is_immediately_reducible() || first->is_reducible() || second->is_reducible();
    }

    void rename(const char o, const char n) override {
        first->rename(o, n);
        second->rename(o, n);
        mark_updated();
    }

    shared_ptr<LambdaExpression> substitute(const char v, const LambdaExpression& e) override {
        first = first->substitute(v, e);
        first->set_parent(shared_from_this());

        second = second->substitute(v, e);
        second->set_parent(shared_from_this());

        mark_updated();
        return shared_from_this();
    }

    shared_ptr<LambdaExpression> reduce() override {
        mark_updated();
        if(is_immediately_reducible()) {
            shared_ptr<LambdaAbstraction> abs = dynamic_pointer_cast<LambdaAbstraction>(first);
            char abss_variable = abs->get_bound_variable();
            shared_ptr<LambdaExpression> abss_body = abs->get_body();
            abss_body->set_parent(nullptr);
            shared_ptr<LambdaExpression> ret = abss_body->substitute(abss_variable, *second);
            ret->set_parent(parent);
            return ret;
        } else if(first->is_reducible()) {
            first = first->reduce();
            first->set_parent(shared_from_this());
            return shared_from_this();
        }
        else if(second->is_reducible()) {
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
        }
    }
};

shared_ptr<LambdaExpression> apply(const shared_ptr<const LambdaExpression> f, const shared_ptr<const LambdaExpression> s, const int c, shared_ptr<LambdaExpression> p = nullptr, float x = 0, float y = 0, float w = 0, float h = 0, int u = 0){
    shared_ptr<LambdaExpression> nf = f->clone();
    shared_ptr<LambdaExpression> ns = s->clone();
    shared_ptr<LambdaExpression> ret = make_shared<LambdaApplication>(nf, ns, c, p, x, y, w, h, u);
    nf->set_parent(ret);
    ns->set_parent(ret);
    return ret;
}

shared_ptr<LambdaExpression> abstract(const char v, const shared_ptr<const LambdaExpression> b, const int c, shared_ptr<LambdaExpression> p = nullptr, float x = 0, float y = 0, float w = 0, float h = 0, int u = 0){
    shared_ptr<LambdaExpression> nb = b->clone();
    shared_ptr<LambdaExpression> ret = make_shared<LambdaAbstraction>(v, nb, c, p, x, y, w, h, u);
    nb->set_parent(ret);
    return ret;
}

void LambdaExpression::set_positions() {
    if(parent != nullptr) failout("set_positions called on a child expression");
    int bounding_box_h = parenthetical_depth() * 2;

    Iterator it(shared_from_this());
    int iter_x = 0;
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        shared_ptr<const LambdaExpression> nearest_ancestor_application = current->get_nearest_ancestor_application();
        int nearest_ancestor_y = nearest_ancestor_application != nullptr ? bounding_box_h - nearest_ancestor_application->get_application_depth() * 2 - 2 : bounding_box_h;
        if (current->get_type() == "Variable") {
            current->x = iter_x+1;
            current->y = 1 + dynamic_pointer_cast<LambdaVariable>(current)->get_bound_abstraction()->get_abstraction_depth() * 2;
            current->h = nearest_ancestor_y - current->y;
            iter_x+=4;
        }
        if(current->get_type() == "Abstraction") {
            current->y = current->get_abstraction_depth() * 2;
            current->w = current->num_variable_instantiations() * 4 - 1;
            current->x = iter_x;
        }
        if(current->get_type() == "Application") {
            current->x = iter_x+1;
            current->w = dynamic_pointer_cast<LambdaApplication>(current)->get_first()->num_variable_instantiations() * 4 + 1;
            current->y = bounding_box_h - current->get_application_depth() * 2 - 2;
            current->h = nearest_ancestor_y - current->y;
        }
    }
}

Pixels LambdaExpression::draw_lambda_diagram(float scale = 1) {
    if(parent != nullptr) failout("draw_lambda_diagram called on a child expression");
    if(w + h + x + y == 0) failout("Attempted drawing lambda diagram with unset positions!");
    int bounding_box_w = (num_variable_instantiations() * 4 - 1) * scale;
    int bounding_box_h = (parenthetical_depth() * 2) * scale;
    Pixels pix(bounding_box_w, bounding_box_h);
    pix.fill(TRANSPARENT_BLACK);

    for(int i = 0; i < 2; i++){
        Iterator it(shared_from_this());
        while (it.has_next()) {
            shared_ptr<LambdaExpression> current = it.next();
            int color = current->get_color();
            if((i==0)==(geta(color) == 255)) continue;
            if (current->get_type() == "Variable") {
                pix.fill_rect(current->x * scale, current->y * scale, 8, current->h * scale, color);
            }
            if(current->get_type() == "Abstraction") {
                pix.fill_rect(current->x * scale, current->y * scale, current->w * scale, 8, color);
            }
            if(current->get_type() == "Application") {
                pix.fill_rect(current->x * scale, current->y * scale, 8, current->h * scale, color);
                pix.fill_rect(current->x * scale, current->y * scale, current->w * scale, 8, color);
            }
        }
    }
    return pix;
}

shared_ptr<LambdaExpression> get_interpolated(shared_ptr<const LambdaExpression> l1, shared_ptr<LambdaExpression> l2, const float weight){
    shared_ptr<LambdaExpression> ret = l1->clone();
    LambdaExpression::Iterator it_ret(ret);
    while (it_ret.has_next()) {
        shared_ptr<LambdaExpression> current_ret = it_ret.next();
        int uid_ret = current_ret->get_uid();
        LambdaExpression::Iterator it_l2(l2);
        bool found = false;
        while (it_l2.has_next()) {
            shared_ptr<LambdaExpression> current_l2 = it_l2.next();
            int uid_l2 = current_l2->get_uid();
            if(uid_ret == uid_l2){
                current_ret->interpolate_positions(current_l2, weight);
                current_ret->set_color(colorlerp(current_ret->get_color(), current_l2->get_color(), weight));
                found = true;
                break;
            }
        }
        if(!found) current_ret->set_color(colorlerp(current_ret->get_color(), 0, weight));
    }
    return ret;
}

shared_ptr<const LambdaAbstraction> LambdaVariable::get_bound_abstraction() const {
    shared_ptr<const LambdaExpression> current = parent;
    while (current) {
        if (current->get_type() == "Abstraction") {
            shared_ptr<const LambdaAbstraction> abstraction = dynamic_pointer_cast<const LambdaAbstraction>(current);
            if (abstraction && abstraction->get_bound_variable() == varname) {
                return abstraction;
            }
        }
        current = current->get_parent();
    }
    return nullptr;
}

shared_ptr<LambdaExpression> parse_lambda_from_string(const string& input) {
    cout << "Parsing '" << input << "'..." << endl;

    if (is_single_letter(input))
        return make_shared<LambdaVariable>(input[0], OPAQUE_WHITE);

    assert(input.size() > 3);

    if(input[0             ] == '('  &&
       input[1             ] == '\\' &&
       input[3             ] == '.'  &&
       input[4             ] == ' '  &&
       input[input.size()-1] == ')'){
        shared_ptr<LambdaExpression> b = parse_lambda_from_string(input.substr(5, input.size() - 6));
        return abstract(input[2], b, OPAQUE_WHITE);
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
                return apply(f, s, OPAQUE_WHITE);
            }
        }
    }

    // No valid pattern was matched
    failout("Failed to parse string!");
    return nullptr;
}
