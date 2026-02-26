#include "LambdaVariable.h"
#include <cstdlib>

LambdaVariable::LambdaVariable(const char vn, const int c, std::weak_ptr<LambdaExpression> p, float x, float y, float w, float h, int u) : LambdaExpression("Variable", c, p, x, y, w, h, u), varname(vn) {
    if(!std::isalpha(static_cast<unsigned char>(vn))){
        throw std::runtime_error("Lambda variable was not a letter!");
    }
}

std::unordered_set<char> LambdaVariable::all_referenced_variables() const {
    return std::unordered_set<char>{varname};
}

std::unordered_set<char> LambdaVariable::free_variables() const {
    return std::unordered_set<char>{varname};
}

std::shared_ptr<LambdaExpression> LambdaVariable::clone() const {
    return std::make_shared<LambdaVariable>(varname, color, parent, x, y, w, h, uid);
}

void LambdaVariable::check_children_parents() const {
    // no children
}

float LambdaVariable::get_width_recursive() const {
    return x + w;
}

float LambdaVariable::get_height_recursive() const {
    return y + h;
}

void LambdaVariable::interpolate_recursive(std::shared_ptr<const LambdaExpression> l2, const float weight) {
    interpolate_positions(l2, weight);
    // No children to recurse
    mark_updated();
}

void LambdaVariable::tint_recursive(const int c) {
    set_color(colorlerp(color, c, 0.5));
    mark_updated();
}

void LambdaVariable::flush_uid_recursive() {
    uid = std::rand();
}

void LambdaVariable::set_color_recursive(const int c) {
    set_color(c);
    mark_updated();
}

int LambdaVariable::parenthetical_depth() const {
    return 0;
}

int LambdaVariable::num_variable_instantiations() const {
    return 1;
}

std::string LambdaVariable::get_string() const {
    return std::string(1, varname);
}

std::string LambdaVariable::get_latex() const {
    return latex_color(color, get_string());
}

bool LambdaVariable::is_reducible() const { return false; }

int LambdaVariable::count_parallel_reductions() const {
    return 0;
}

void LambdaVariable::rename(const char o, const char n) {
    if(varname == o) {
        varname = n;
    }
    mark_updated();
}

std::shared_ptr<LambdaExpression> LambdaVariable::substitute(const char v, const LambdaExpression& e) {
    mark_updated();
    if(varname == v && get_bound_abstraction() == nullptr) { // don't substitute bound variables
        std::shared_ptr<LambdaExpression> ret = e.clone();
        ret->tint_recursive(color);
        return ret;
    }
    return shared_from_this();
}

std::shared_ptr<LambdaExpression> LambdaVariable::reduce() {
    throw std::runtime_error("Reduction was attempted, but no legal reduction was found!");
    return nullptr;
}

std::shared_ptr<LambdaExpression> LambdaVariable::specific_reduction(int x) {
    throw std::runtime_error("Reduction was attempted, but no legal reduction was found!");
    return nullptr;
}
