#pragma once

#include <string>
#include <memory>
#include <unordered_set>
#include <stack>
#include "../DataObject.h"
#include "../../Core/Pixels.h"

class LambdaExpression : public DataObject, public std::enable_shared_from_this<LambdaExpression> {
protected:
    const std::string type;
    int color;
    std::weak_ptr<LambdaExpression> parent;
    float x;
    float y;
    float w;
    float h;
    int uid;
public:
    LambdaExpression(const std::string& t, const int c, std::weak_ptr<LambdaExpression> p = std::shared_ptr<LambdaExpression>(), float x = 0, float y = 0, float w = 0, float h = 0, int u = 0);
    virtual std::shared_ptr<LambdaExpression> clone() const = 0;

    virtual std::unordered_set<char> free_variables() const = 0;
    virtual std::unordered_set<char> all_referenced_variables() const = 0;
    virtual std::shared_ptr<LambdaExpression> reduce() = 0;
    virtual int count_parallel_reductions() const = 0;
    virtual std::shared_ptr<LambdaExpression> specific_reduction(int x) = 0;
    virtual std::shared_ptr<LambdaExpression> substitute(const char v, const LambdaExpression& e) = 0;
    virtual void rename(const char o, const char n) = 0;
    virtual std::string get_string() const = 0;
    virtual std::string get_latex() const = 0;
    virtual int parenthetical_depth() const = 0;
    virtual int num_variable_instantiations() const = 0;
    virtual float get_width_recursive() const = 0;
    virtual float get_height_recursive() const = 0;
    virtual bool is_reducible() const = 0;
    virtual void set_color_recursive(const int c) = 0;
    virtual void flush_uid_recursive() = 0;
    virtual void tint_recursive(const int c) = 0;
    virtual void interpolate_recursive(std::shared_ptr<const LambdaExpression> l2, const float weight) = 0;
    virtual void check_children_parents() const = 0;
    std::unordered_set<std::shared_ptr<LambdaExpression>> get_all_legal_reductions();
    int get_uid();
    int count_reductions();
    char get_fresh() const;
    void set_parent(std::weak_ptr<LambdaExpression> p);
    void set_color(const int c);
    std::string get_type() const;
    int get_color() const;
    std::shared_ptr<const LambdaExpression> get_parent() const;

    int get_type_depth(const std::string& s) const;

    std::shared_ptr<const LambdaExpression> get_nearest_ancestor_with_type(const std::string& type) const;

    int get_abstraction_depth() const;

    int get_application_depth() const;

    class Iterator;

    Pixels draw_lambda_diagram(float scale);
    void set_positions();
    void interpolate_positions(std::shared_ptr<const LambdaExpression> l2, const float weight);
};

class LambdaExpression::Iterator {
public:
    Iterator(std::shared_ptr<LambdaExpression> root);

    bool has_next() const;

    std::shared_ptr<LambdaExpression> next();

private:
    std::stack<std::shared_ptr<LambdaExpression>> node_stack;
    std::shared_ptr<LambdaExpression> current;

    void push_children(std::shared_ptr<LambdaExpression> expr);
};
