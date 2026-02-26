#pragma once

#include <iomanip>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <regex>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include "TransitionType.h"
#include "GlobalState.h"
#include "UnresolvedStateEquation.h"
#include "ResolvedStateEquationComponent.c"

typedef std::vector<ResolvedStateEquationComponent> ResolvedStateEquation;
using StateQuery = unordered_set<string>;
using StateSet = unordered_map<string, string>;
void state_query_insert_multiple(StateQuery& sq, const StateQuery& additions);

/* StateManager is a DAG (Directed Acyclic Graph) of state assignments
 * used to facilitate frame-by-frame manipulation of state.
 * It is composed of a number of equations which express how to update
 * certain variables in terms of others. Cyclical computations are not
 * permitted, such as "y=x+1" and "x=y", hence the DAG of computational
 * direction. Similarly, if there are equations such as "x=y" and "y=3",
 * StateManager will intentionally run the second equation first.
 */

/* Parenthetical notation hints:
 * <w> : local state variable, such as scene width as fraction of screen
 * [defer_to_parent] : parent state variable, when sharing state between child scenes
 * {t} : global variable, such as time in seconds
 * (u) : A passthrough tag to be interpreted in CUDA, such as manifold colors
 */

struct VariableContents {
    // The numerical value currently stored by this variable
    double value;

    // Is this variable fresh (updated since last control cycle) or stale?
    bool fresh;

    // A list of variable-equation pairs like <"x", "y 5 +"> to represent "x=y+5" (in RPN)
    UnresolvedStateEquation equation;

    // A list of all of the variables which any certain variable depends on.
    // Using "x=y+5" as an example, the key "x"'s value would be a list containing only "y".
    std::list<std::string> local_dependencies;

    VariableContents();
    VariableContents(string eq, double val = 0.0, bool fr = true);
};

class StateReturn {
public:
    StateReturn();
    StateReturn(const unordered_map<string, double>& m);

    void print() const;
    int size() const;

    void set(const string& key, double value);
    bool contains(const string& key) const;
    const double& operator[](const string& key) const;

    bool operator==(const StateReturn& other) const;
    bool operator!=(const StateReturn& other) const;

    unordered_map<string, double> get_map() const;

private:
    unordered_map<string, double> map;
};

class StateManager {
public:
    StateManager();

    bool contains(const string& varname) const;
    void print_state() const;

    StateSet set(const string& variable, const string& equation);
    StateSet set(const StateSet& equations);

    void remove(const string& variable);
    void remove(const unordered_set<string>& equations);

    StateSet transition(TransitionType tt, const string& variable, const string& equation, bool smooth = true);
    StateSet transition(TransitionType tt, const StateSet& equations, bool smooth = true);
    void close_transitions(TransitionType tt);

    void set_parent(StateManager* p, const string& name);

    void evaluate_all();

    string get_equation_string(const string& variable) const;
    double get_local_value(const string& variable) const;

    const void set_subjugated(bool b);
    const void begin_timer(const string& timer_name);

    const StateReturn respond_to_query(const StateQuery& query) const;
    const ResolvedStateEquation get_resolved_equation(const string& variable) const;

private:
    unordered_map<string, VariableContents> variables;
    list<string> last_compute_order;
    unordered_set<string> in_microblock_transition;
    unordered_set<string> in_macroblock_transition;

    StateManager* parent;
    bool subjugated;

    void evaluate_single_variable(const string& variable);
    VariableContents get_local_variable(const string& variable) const;
    double get_parent_value(const string& variable) const;
    void close_all_transitions(unordered_set<string>& in_transition);

    ResolvedStateEquation resolve_state_equation(const UnresolvedStateEquation& ueq) const;
};
