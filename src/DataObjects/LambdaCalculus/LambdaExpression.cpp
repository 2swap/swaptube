#include "LambdaExpression.h"
#include <stdexcept>
#include "LambdaAbstraction.h"
#include "LambdaApplication.h"

LambdaExpression::LambdaExpression(const std::string& t, const int c, std::weak_ptr<LambdaExpression> p, float x, float y, float w, float h, int u)
    : type(t), color(c), parent(p), x(x), y(y), w(w), h(h), uid(u) {}

std::unordered_set<std::shared_ptr<LambdaExpression>> LambdaExpression::get_all_legal_reductions() {
    std::unordered_set<std::shared_ptr<LambdaExpression>> reductions;
    int num_reductions = count_parallel_reductions();
    
    for (int n = 0; n < num_reductions; ++n) {
        // Insert the reduced term into the set
        reductions.insert(clone()->specific_reduction(n));
    }

    return reductions;
}

int LambdaExpression::get_uid(){ return uid; }

int LambdaExpression::count_reductions(){
    std::shared_ptr<LambdaExpression> cl = clone();
    for(int i = 0; i < 10000; i++){
        if(!cl->is_reducible())return i;
        cl = cl->reduce();
    }
    throw std::runtime_error("count_reductions called on a term which does not reduce to BNF in under 10000 reductions!");
    return -1;
}

char LambdaExpression::get_fresh() const {
    if(!parent.expired()) return parent.lock()->get_fresh();
    std::unordered_set<char> used = all_referenced_variables();
    for(char c = 'a'; c <= 'z'; c++)
        if(used.find(c) == used.end())
            return c;
    throw std::runtime_error("No fresh variables left!");
    return 0;
}

void LambdaExpression::set_parent(std::weak_ptr<LambdaExpression> p) {
    parent = p;
    mark_updated();
}

void LambdaExpression::set_color(const int c) {
    color = c;
    mark_updated();
}

std::string LambdaExpression::get_type() const {
    return type;
}

int LambdaExpression::get_color() const {
    return color;
}

std::shared_ptr<const LambdaExpression> LambdaExpression::get_parent() const {
    return parent.lock();
}

int LambdaExpression::get_type_depth(const std::string& s) const {
    int depth = 0;
    std::shared_ptr<const LambdaExpression> current = parent.lock();
    while (current) {
        if (current->get_type() == s) {
            depth++;
        }
        current = current->get_parent();
    }
    return depth;
}

std::shared_ptr<const LambdaExpression> LambdaExpression::get_nearest_ancestor_with_type(const std::string& type) const {
    std::shared_ptr<const LambdaExpression> current = parent.lock();
    while (current) {
        if (current->get_type() == type) {
            return current;
        }
        current = current->get_parent();
    }
    return nullptr;
}

int LambdaExpression::get_abstraction_depth() const {
    return get_type_depth("Abstraction");
}

int LambdaExpression::get_application_depth() const {
    return get_type_depth("Application");
}

void LambdaExpression::interpolate_positions(std::shared_ptr<const LambdaExpression> l2, const float weight){
    w = smoothlerp(w, l2->w, weight);
    h = smoothlerp(h, l2->h, weight);
    x = smoothlerp(x, l2->x, weight);
    y = smoothlerp(y, l2->y, weight);
}

LambdaExpression::Iterator::Iterator(std::shared_ptr<LambdaExpression> root) : current(root) {
    if (current) node_stack.push(current);
}

bool LambdaExpression::Iterator::has_next() const {
    return !node_stack.empty();
}

std::shared_ptr<LambdaExpression> LambdaExpression::Iterator::next() {
    if (!has_next()) {
        throw std::out_of_range("Iterator has no more elements.");
    }
    std::shared_ptr<LambdaExpression> current = node_stack.top();
    node_stack.pop();
    push_children(current);
    return current;
}

void LambdaExpression::Iterator::push_children(std::shared_ptr<LambdaExpression> expr) {
    if (auto* app = dynamic_cast<LambdaApplication*>(expr.get())) {
        if (app->get_second()) node_stack.push(app->get_second());
        if (app->get_first()) node_stack.push(app->get_first());
    } else if (auto* abs = dynamic_cast<LambdaAbstraction*>(expr.get())) {
        if (abs->get_body()) node_stack.push(abs->get_body());
    }
}
