#include "LambdaApplication.h"

#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <string>

#include "LambdaAbstraction.h"
#include "LambdaUtils.h"

using std::shared_ptr;
using std::weak_ptr;
using std::dynamic_pointer_cast;
using std::unordered_set;
using std::string;
using std::max;
using std::runtime_error;
using std::to_string;

LambdaApplication::LambdaApplication(shared_ptr<LambdaExpression> f, shared_ptr<LambdaExpression> s, const int c, weak_ptr<LambdaExpression> p, float x, float y, float w, float h, int u)
    : LambdaExpression("Application", c, p, x, y, w, h, u), first(f), second(s) { }

unordered_set<char> LambdaApplication::all_referenced_variables() const {
    unordered_set<char> all_vars_f = first->all_referenced_variables();
    unordered_set<char> all_vars_s = second->all_referenced_variables();
    for(const char s : all_vars_s){
        all_vars_f.insert(s);
    }
    return all_vars_f;
}

unordered_set<char> LambdaApplication::free_variables() const {
    unordered_set<char> free_vars_f = first->free_variables();
    unordered_set<char> free_vars_s = second->free_variables();
    for(const char s : free_vars_s){
        free_vars_f.insert(s);
    }
    return free_vars_f;
}

shared_ptr<LambdaExpression> LambdaApplication::clone() const {
    return apply(first, second, color, parent, x, y, w, h, uid);
}

string LambdaApplication::get_string() const {
    return "(" + first->get_string() + " " + second->get_string() + ")";
}

string LambdaApplication::get_latex() const {
    return latex_color(color, "(") + first->get_latex() + " " + second->get_latex() + latex_color(color, ")");
}

void LambdaApplication::check_children_parents() const {
    if(first->get_parent() != shared_from_this() || second->get_parent() != shared_from_this())
        throw runtime_error("LambdaApplication failed child-parent check!");
}

float LambdaApplication::get_width_recursive() const {
    return max(x + w, max(first->get_width_recursive(), second->get_width_recursive()));
}

float LambdaApplication::get_height_recursive() const {
    return max(y + h, max(first->get_height_recursive(), second->get_height_recursive()));
}

void LambdaApplication::interpolate_recursive(shared_ptr<const LambdaExpression> l2, const float weight) {
    interpolate_positions(l2, weight);
    shared_ptr<const LambdaExpression> l2_first = dynamic_pointer_cast<const LambdaApplication>(l2)->get_first();
    first->interpolate_recursive(l2_first, weight);
    shared_ptr<const LambdaExpression> l2_second = dynamic_pointer_cast<const LambdaApplication>(l2)->get_second();
    second->interpolate_recursive(l2_second, weight);
    mark_updated();
}

void LambdaApplication::tint_recursive(const int c) {
    first->tint_recursive(c);
    second->tint_recursive(c);
    set_color(colorlerp(color, c, 0.5));
    mark_updated();
}

void LambdaApplication::flush_uid_recursive() {
    first->flush_uid_recursive();
    second->flush_uid_recursive();
    uid = rand();
}

void LambdaApplication::set_color_recursive(const int c) {
    first->set_color_recursive(c);
    second->set_color_recursive(c);
    set_color(c);
    mark_updated();
}

int LambdaApplication::parenthetical_depth() const {
    return 1 + max(first->parenthetical_depth(), second->parenthetical_depth());
}

int LambdaApplication::num_variable_instantiations() const {
    return first->num_variable_instantiations() + second->num_variable_instantiations();
}

bool LambdaApplication::is_immediately_reducible() const {
    return first->get_type() == "Abstraction";
}

bool LambdaApplication::is_reducible() const {
    return is_immediately_reducible() || first->is_reducible() || second->is_reducible();
}

int LambdaApplication::count_parallel_reductions() const {
    int parallel_reductions = (is_immediately_reducible()?1:0) + first->count_parallel_reductions() + second->count_parallel_reductions();
    return parallel_reductions;
}

void LambdaApplication::rename(const char o, const char n) {
    first->rename(o, n);
    second->rename(o, n);
    mark_updated();
}

shared_ptr<LambdaExpression> LambdaApplication::substitute(const char v, const LambdaExpression& e) {
    first = first->substitute(v, e);
    first->set_parent(shared_from_this());

    second = second->substitute(v, e);
    second->set_parent(shared_from_this());

    mark_updated();
    return shared_from_this();
}

shared_ptr<LambdaExpression> LambdaApplication::reduce() {
    mark_updated();
    if(is_immediately_reducible()) {
        shared_ptr<LambdaAbstraction> abs = dynamic_pointer_cast<LambdaAbstraction>(first);
        char abss_variable = abs->get_bound_variable();
        shared_ptr<LambdaExpression> abss_body = abs->get_body();
        abss_body->set_parent(shared_ptr<LambdaExpression>());
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
        throw runtime_error("Reduction was attempted, but no legal reduction was found!");
        return nullptr;
    }
}

shared_ptr<LambdaExpression> LambdaApplication::specific_reduction(int x) {
    mark_updated();
    int x_copy = x;
    if(is_immediately_reducible()) {
        if(x == 0){
            return reduce();
        }
        x--;
    }
    if(first->is_reducible()) {
        int first_reductions = first->count_parallel_reductions();
        if(x < first_reductions){
            first = first->specific_reduction(x);
            first->set_parent(shared_from_this());
            return shared_from_this();
        }
        x-=first_reductions;
    }
    if(second->is_reducible()) {
        int second_reductions = second->count_parallel_reductions();
        if(x < second_reductions){
            second = second->specific_reduction(x);
            second->set_parent(shared_from_this());
            return shared_from_this();
        }
        x-=second_reductions;
    }
    throw runtime_error("Specific reduction was attempted, but no legal reduction was found! " + get_string() + "; x = " + to_string(x_copy) + ".");
    return nullptr;
}

shared_ptr<LambdaExpression> LambdaApplication::get_first() const {
    return first;
}

shared_ptr<LambdaExpression> LambdaApplication::get_second() const {
    return second;
}
