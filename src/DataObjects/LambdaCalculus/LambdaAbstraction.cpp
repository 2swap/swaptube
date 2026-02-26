#include "LambdaAbstraction.h"

#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include "LambdaUtils.h"

using std::runtime_error;
using std::max;
using std::dynamic_pointer_cast;

LambdaAbstraction::LambdaAbstraction(const char v, shared_ptr<LambdaExpression> b, const int c, weak_ptr<LambdaExpression> p, float x, float y, float w, float h, int u)
    : LambdaExpression("Abstraction", c, p, x, y, w, h, u), bound_variable(v), body(b) { }

unordered_set<char> LambdaAbstraction::all_referenced_variables() const {
    unordered_set<char> all_vars = body->all_referenced_variables();
    all_vars.insert(bound_variable);
    return all_vars;
}

unordered_set<char> LambdaAbstraction::free_variables() const {
    unordered_set<char> free_vars = body->free_variables();
    free_vars.erase(bound_variable);
    return free_vars;
}

shared_ptr<LambdaExpression> LambdaAbstraction::clone() const {
    return abstract(bound_variable, body, color, parent, x, y, w, h, uid);
}

string LambdaAbstraction::get_string() const {
    return string("(\\") + bound_variable + ". " + body->get_string() + ")";
}

string LambdaAbstraction::get_latex() const {
    return latex_color(color, string("(\\lambda ") + bound_variable + ". ") + body->get_latex() + latex_color(color, ")");
}

void LambdaAbstraction::check_children_parents() const {
    if(body->get_parent() != shared_from_this())
        throw runtime_error("LambdaAbstraction failed child-parent check!");
}

float LambdaAbstraction::get_width_recursive() const {
    return max(x + w, body->get_width_recursive());
}

float LambdaAbstraction::get_height_recursive() const {
    return max(y + h, body->get_height_recursive());
}

void LambdaAbstraction::interpolate_recursive(shared_ptr<const LambdaExpression> l2, const float weight) {
    interpolate_positions(l2, weight);
    shared_ptr<const LambdaExpression> l2_body = dynamic_pointer_cast<const LambdaAbstraction>(l2)->get_body();
    body->interpolate_recursive(l2_body, weight);
    mark_updated();
}

void LambdaAbstraction::tint_recursive(const int c) {
    body->tint_recursive(c);
    set_color(colorlerp(color, c, 0.5));
    mark_updated();
}

void LambdaAbstraction::flush_uid_recursive() {
    body->flush_uid_recursive();
    uid = rand();
}

void LambdaAbstraction::set_color_recursive(const int c) {
    body->set_color_recursive(c);
    set_color(c);
    mark_updated();
}

int LambdaAbstraction::parenthetical_depth() const {
    return 1 + body->parenthetical_depth();
}

int LambdaAbstraction::num_variable_instantiations() const {
    return body->num_variable_instantiations();
}

bool LambdaAbstraction::is_reducible() const { return body->is_reducible(); }

int LambdaAbstraction::count_parallel_reductions() const {
    return body->count_parallel_reductions();
}

void LambdaAbstraction::rename(const char o, const char n) {
    if(bound_variable == o) {
        bound_variable = n;
    }
    body->rename(o, n);
    mark_updated();
}

shared_ptr<LambdaExpression> LambdaAbstraction::substitute(const char v, const LambdaExpression& e) {
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

shared_ptr<LambdaExpression> LambdaAbstraction::reduce() {
    if(body->is_reducible()) {
        body = body->reduce();
        body->set_parent(shared_from_this());
    } else {
        throw runtime_error("Reduction was attempted, but no legal reduction was found!");
    }
    mark_updated();
    return shared_from_this();
}

shared_ptr<LambdaExpression> LambdaAbstraction::specific_reduction(int x) {
    body = body->specific_reduction(x);
    body->set_parent(shared_from_this());
    mark_updated();
    return shared_from_this();
}

char LambdaAbstraction::get_bound_variable() const {
    return bound_variable;
}

shared_ptr<LambdaExpression> LambdaAbstraction::get_body() const {
    return body;
}
