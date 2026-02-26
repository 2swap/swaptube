#pragma once

#include <memory>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <cctype>

#include "LambdaExpression.h"

class LambdaAbstraction;

class LambdaVariable : public LambdaExpression {
private:
    char varname;
public:
    LambdaVariable(const char vn, const int c, std::weak_ptr<LambdaExpression> p = std::shared_ptr<LambdaExpression>(), float x = 0, float y = 0, float w = 0, float h = 0, int u = 0);
    std::unordered_set<char> all_referenced_variables() const override;
    std::unordered_set<char> free_variables() const override;
    std::shared_ptr<LambdaExpression> clone() const override;
    void check_children_parents() const;
    float get_width_recursive() const;
    float get_height_recursive() const;
    void interpolate_recursive(std::shared_ptr<const LambdaExpression> l2, const float weight);
    void tint_recursive(const int c);
    void flush_uid_recursive();
    void set_color_recursive(const int c);
    int parenthetical_depth() const override;
    int num_variable_instantiations() const override;
    std::string get_string() const override;
    std::string get_latex() const;
    bool is_reducible() const override;
    int count_parallel_reductions() const override;
    void rename(const char o, const char n) override;
    std::shared_ptr<LambdaExpression> substitute(const char v, const LambdaExpression& e) override;
    std::shared_ptr<LambdaExpression> reduce() override;
    std::shared_ptr<LambdaExpression> specific_reduction(int x) override;
    std::shared_ptr<const LambdaAbstraction> get_bound_abstraction() const;
};
