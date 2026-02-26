#pragma once

#include <memory>
#include <unordered_set>
#include <string>
#include "LambdaExpression.h"

using std::shared_ptr;
using std::weak_ptr;
using std::unordered_set;
using std::string;

class LambdaAbstraction : public LambdaExpression {
private:
    char bound_variable;
    shared_ptr<LambdaExpression> body;
public:
    LambdaAbstraction(const char v, shared_ptr<LambdaExpression> b, const int c, weak_ptr<LambdaExpression> p = shared_ptr<LambdaExpression>(), float x = 0, float y = 0, float w = 0, float h = 0, int u = 0);

    unordered_set<char> all_referenced_variables() const override;
    unordered_set<char> free_variables() const override;
    shared_ptr<LambdaExpression> clone() const override;
    string get_string() const override;
    string get_latex() const;
    void check_children_parents() const;
    float get_width_recursive() const;
    float get_height_recursive() const;
    void interpolate_recursive(shared_ptr<const LambdaExpression> l2, const float weight);
    void tint_recursive(const int c);
    void flush_uid_recursive();
    void set_color_recursive(const int c);
    int parenthetical_depth() const override;
    int num_variable_instantiations() const override;
    bool is_reducible() const override;
    int count_parallel_reductions() const override;
    void rename(const char o, const char n) override;
    shared_ptr<LambdaExpression> substitute(const char v, const LambdaExpression& e) override;
    shared_ptr<LambdaExpression> reduce() override;
    shared_ptr<LambdaExpression> specific_reduction(int x) override;
    char get_bound_variable() const;
    shared_ptr<LambdaExpression> get_body() const;
};
