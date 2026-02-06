#pragma once

#include <iomanip>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <regex>
#include <cassert>
#include <iostream>
#include "GlobalState.cpp"
#include "UnresolvedStateEquation.cpp"
#include "ResolvedStateEquationComponent.c"
//#include "../../Host_Device_Shared/helpers.h"

typedef vector<ResolvedStateEquationComponent> ResolvedStateEquation;

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
    list<string> local_dependencies;

    VariableContents()
                : value(0.0), fresh(true), equation(""), local_dependencies() {}

    VariableContents(string eq,
                     double val = 0.0,
                     bool fr = true
                    ) : value(val), fresh(fr), equation(eq), local_dependencies(equation.get_local_dependencies()) {}
};

using StateQuery = unordered_set<string>;
void state_query_insert_multiple(StateQuery& sq, const StateQuery& additions){
    for(const string& s : additions){
        sq.insert(s);
    }
}

using StateSet = unordered_map<string, string>;
class StateReturn {
public:
    StateReturn() {}
    StateReturn(const unordered_map<string, double>& m) : map(m) {}

    void print() const {
        // Print alphabetically
        cout << endl << "STATE vvv" << endl;
        vector<string> keys;
        for (const auto& pair : map) {
            keys.push_back(pair.first);
        }
        sort(keys.begin(), keys.end());
        for (const string& key : keys) {
            cout << key << ": " << map.at(key) << endl;
        }

        cout << "STATE ^^^" << endl << endl;
    }

    int size() const {
        return map.size();
    }

    void set(const string& key, const double value) {
        map[key] = value;
    }

    bool contains(const string& key) const {
        return map.find(key) != map.end();
    }

    const double& operator[](const string& key) const {
        if (!contains(key)) {
            print();
            throw runtime_error("Invalid State Access: " + key);
        }
        return map.at(key);
    }

    // Implement equality operator
    bool operator==(const StateReturn& other) const {
        return map == other.map;
    }

    // Implement inequality operator
    bool operator!=(const StateReturn& other) const {
        return !(*this == other);
    }

    unordered_map<string, double> get_map() const {
        return map;
    }

private:
    unordered_map<string, double> map;
};

class StateManager {
public:
    StateManager() : parent(nullptr), subjugated(false) {}

    bool contains(const string& varname) const {
        return variables.find(varname) != variables.end();
    }

    void print_state() const {
        /* Print out all variable names alphabetically along with their current value and their computation status. */
        // TODO alphabeticalize
        cout << "Variables:" << endl;
        cout << "-----------------------" << endl;
        for (const auto& variable : variables) {
            const VariableContents& vc = variable.second;
            cout << left << setw(32) << variable.first
                 << setw(38) << vc.equation.to_string()
                 << " : " << setw(10) << vc.value
                 << (vc.fresh ? " (Fresh)" : " (Stale)")
                 << " " << &vc
                 << endl;
        }
        cout << "-----------------------" << endl;
    }

    StateSet set(const string& variable, const string& equation) {
        // Validate variable name to contain only letters, numbers, dots, and underscores
        regex valid_variable_regex("^[a-zA-Z0-9._]+$");
        if (!regex_match(variable, valid_variable_regex)) {
            throw runtime_error("Error adding equation to state manager: Variable name '" + variable + "' contains invalid characters.");
        }

        // Check that the variable is not undergoing a transition
        if(in_microblock_transition.find(variable) != in_microblock_transition.end() ||
           in_macroblock_transition.find(variable) != in_macroblock_transition.end()){
            throw runtime_error("Attempted to set variable " + variable + " while it is undergoing a transition!");
        }

        StateSet ret = {};
        if(contains(variable)) ret = { {variable, get_equation_string(variable)} };

        /* When a new component is added, we do not know the
         * correct compute order anymore since there are new equations.
         */
        last_compute_order.clear();
        variables[variable] = VariableContents(equation);

        return ret;
    }

    StateSet set(const StateSet& equations) {
        StateSet ret = {};
        for(auto it = equations.begin(); it != equations.end(); it++){
            StateSet prev = set(it->first, it->second);
            ret.insert(prev.begin(), prev.end());
        }
        return ret;
    }

    void remove(const string& variable) {
        /* When a new component is removed, we do not know the
         * correct compute order anymore.
         */
        last_compute_order.clear();
        variables.erase(variable);
    }
    void remove(const unordered_set<string>& equations) {
        for(const string& varname : equations){
            remove(varname);
        }
    }

    StateSet transition(const TransitionType tt, const string& variable, const string& equation, const bool smooth = true) {
        if(!contains(variable)){
            print_state();
            throw runtime_error("ERROR: Attempted to transition nonexistent variable " + variable + "!\nState printed above.");
        }

        // Nested transitions not supported
        if(in_microblock_transition.find(variable) != in_microblock_transition.end() ||
           in_macroblock_transition.find(variable) != in_macroblock_transition.end()){
            throw runtime_error("Transition added to a variable already in transition: " + variable);
        }

        if(tt != MICRO && tt != MACRO) throw runtime_error("Invalid transition type: " + to_string(tt));

        string eq1 = get_equation_string(variable);

        // No point in doing a noop transition
        if(eq1 != equation) {
            string lerp_both = eq1 + " " + equation + " {" + (tt==MICRO?"micro":"macro") + "block_fraction} " + (smooth?"smooth":"") + "lerp";
            set(variable+".post_transition", equation);
            set(variable, lerp_both);
                 if(tt == MICRO) in_microblock_transition.insert(variable);
            else if(tt == MACRO) in_macroblock_transition.insert(variable);
        }

        return { {variable, eq1} };
    }
    StateSet transition(const TransitionType tt, const StateSet& equations, bool smooth = true) {
        StateSet ret = {};
        for(auto it = equations.begin(); it != equations.end(); it++){
            StateSet prev = transition(tt, it->first, it->second, smooth);
            ret.insert(prev.begin(), prev.end());
        }
        return ret;
    }
    void close_transitions(const TransitionType tt){
        if(tt == MICRO) close_all_transitions(in_microblock_transition);
        if(tt == MACRO) close_all_transitions(in_macroblock_transition);
    }

    void set_parent(StateManager* p, const string& name) {
        if((parent == nullptr) == (p == nullptr))
            throw runtime_error("Parent must change state from set to unset or vice versa. Current: " + to_string(reinterpret_cast<uintptr_t>(parent)) + ", setting to: " + to_string(reinterpret_cast<uintptr_t>(p)) + ". This scene's child identifier was " + name + ".");
        parent = p;
    }

    void evaluate_all() {
        /* Step 1: Iterate through all variables,
         * check that they are fresh from last cycle,
         * and reset it to false. */
        for (auto& variable : variables) {
            if(!variable.second.fresh){
                print_state();
                throw runtime_error("ERROR: variable " + variable.first + " was not fresh when expected!\nState has been printed above.");
            }
            variable.second.fresh = false;
        }

        /* Step 2: Follow the last_iteration_order to set variables in accordance with the DAG. */
        if (!last_compute_order.empty()) {
            for (const string& variable : last_compute_order) {
                evaluate_single_variable(variable);
            }
        }

        /* Step 3: If there is no last_iteration_order, then repeatedly go through the variables,
         * computing any whose dependencies are satisfied, until all are computed.
         * Simultaneously document the order of computation in last_iteration_order. */
        else {
            bool unevaluated_variable_remaining = true;
            while (unevaluated_variable_remaining) {
                bool one_variable_changed = false;
                unevaluated_variable_remaining = false;
                for (const auto& variable : variables) {
                    const string& variable_name = variable.first;
                    const VariableContents& vc = variable.second;

                    // Don't recompute something already computed
                    if (vc.fresh) continue;

                    // Check that all variable dependencies are met
                    bool all_dependencies_met = true;
                    for (const string& dependency : vc.local_dependencies) {
                        if(!contains(dependency)){
                            print_state();
                            throw runtime_error("error: attempted to access variable " + dependency
                                    + " during evaluation of " + variable_name + " := "
                                    + vc.equation.to_string() + ", but it does not exist in this StateManager!"
                                    + "\nstate has been printed above.");
                        }
                        const VariableContents& dep_vc = variables.at(dependency);
                        if (!dep_vc.fresh) {
                            all_dependencies_met = false;
                            break;
                        }
                    }
                    if (!all_dependencies_met) {
                        unevaluated_variable_remaining = true;
                        continue;
                    }

                    one_variable_changed = true;
                    evaluate_single_variable(variable_name);
                    last_compute_order.push_back(variable_name); // Document the order of computation
                }
                /* Each iteration we expect a variable to be computed at the least.
                 * If that doesn't happen, that means it isn't really a DAG. */
                if(!one_variable_changed && unevaluated_variable_remaining){
                    print_state();
                    throw runtime_error("error: variable dependency graph appears not to be a DAG! State has been printed above.");
                }
            }
        }
    }

    string get_equation_string(const string& variable) const {
        try {
            return get_local_variable(variable).equation.to_string();
        } catch (const runtime_error& e) {
            throw runtime_error("ERROR: Attempted to read equation string of variable " + variable + " but failed. " + e.what());
        }
    }

    double get_local_value(const string& variable) const {
        try {
            VariableContents vc = get_local_variable(variable);

            // We should never ever read from a stale variable.
            if(!vc.fresh){
                print_state();
                throw runtime_error("ERROR: Attempted to read stale variable " + variable + "!\nState has been printed above.");
            }

            return vc.value;
        } catch (const runtime_error& e) {
            throw runtime_error("ERROR: Attempted to read value of variable " + variable + " but failed. " + e.what());
        }
    }

    const void set_subjugated(bool b) {
        subjugated = b;
    }

    const void begin_timer(const string& timer_name) {
        set(timer_name, "{t} " + to_string(global_state["t"]) + " -");
    }

    const StateReturn respond_to_query(const StateQuery& query) const {
        if(subjugated){
            if (parent == nullptr)
                throw runtime_error("A StateManager was queried while marked as subjugated despite not having a parent.");
            return parent->respond_to_query(query);
        }
        StateReturn result;
        for (const auto& varname : query) {
            if(contains(varname)){
                result.set(varname, get_local_value(varname));
            } else if (global_state.find(varname) != global_state.end()){
                result.set(varname, global_state.at(varname));
            } else {
                print_state();
                throw runtime_error("ERROR: Attempted to get state for queried variable " + varname + " but it does not exist locally or globally!\nState has been printed above.");
            }
        }
        return result;
    }

    const ResolvedStateEquation get_resolved_equation(const string& variable) const {
        return resolve_state_equation(get_local_variable(variable).equation);
    }

private:
    // A list of variables and their relevant data
    unordered_map<string, VariableContents> variables;

    // A list of variables in the order they were computed the last time around
    list<string> last_compute_order;

    // A list of all variable names which are currently undergoing transitions
    unordered_set<string> in_microblock_transition;
    unordered_set<string> in_macroblock_transition;

    StateManager* parent = nullptr;
    // When a state manager is "subjugated" to a parent, that means it
    // strictly defers to the parent's state without considering its
    // own variables first.
    bool subjugated = false;

    void evaluate_single_variable(const string& variable) {
        VariableContents& vc = variables.at(variable);
        assert(!vc.fresh);
        vc.fresh = true;
        ResolvedStateEquation rse = resolve_state_equation(vc.equation);
        try {
            int error = 0;
            float blank_cuda_tags[8] = {1.20f, -.19f, 90.0f, 1.0f, 2.0f, 3.0f, -4.0f, 5.0f};
            vc.value = evaluate_resolved_state_equation(rse.size(), rse.data(), blank_cuda_tags, 8, error);
            if(error != 0) {
                throw runtime_error("Error code " + to_string(error) + " returned from evaluate_resolved_state_equation.");
            }
        } catch (const runtime_error& e) {
            print_state();
            cout << "Resolved equation:" << endl;
            print_resolved_state_equation(rse.size(), rse.data());
            throw runtime_error("Error evaluating equation for variable " + variable + "\n" + e.what() + "\nEquation: " + vc.equation.to_string() + "\nState has been printed above.");
        }
    }

    VariableContents get_local_variable(const string& variable) const {
        if(variables.find(variable) != variables.end()){
            return variables.at(variable);
        }
        print_state();
        throw runtime_error("ERROR: Attempted to access variable " + variable + " without it existing!\nState has been printed above.");
    }

    double get_parent_value(const string& variable) const {
        if(parent == nullptr)
            throw runtime_error("Parent was a nullptr while looking for [" + variable + "]");
        return parent->get_local_value(variable);
    }

    void close_all_transitions(unordered_set<string>& in_transition){
        unordered_set<string> in_transition_copy = in_transition;
        in_transition.clear();
        for(string varname : in_transition_copy){
            set(varname, get_equation_string(varname + ".post_transition"));
            VariableContents& vc = variables.at(varname);
            vc.value = get_local_value(varname + ".post_transition");
            remove(varname + ".post_transition");
        }
    }

    ResolvedStateEquation resolve_state_equation(const UnresolvedStateEquation& ueq) const {
        ResolvedStateEquation rse;
        for(const UnresolvedStateEquationComponent& comp : ueq.components){
            ResolvedStateEquationComponent rsec;
            switch (comp.type){
                case UNRESOLVED_CONSTANT:
                    rsec.type = RESOLVED_CONSTANT;
                    rsec.content.constant = comp.content.constant;
                    break;
                case UNRESOLVED_LOCAL_VARIABLE:
                    rsec.type = RESOLVED_CONSTANT;
                    try {
                        rsec.content.constant = get_local_value(comp.content.variable_name);
                    } catch (const runtime_error& e) {
                        throw runtime_error("ERROR: Attempted to access variable [" + comp.content.variable_name + "] during resolution. " + e.what());
                    }
                    break;
                case UNRESOLVED_PARENT_VARIABLE:
                    rsec.type = RESOLVED_CONSTANT;
                    try {
                        rsec.content.constant = get_parent_value(comp.content.variable_name);
                    } catch (const runtime_error& e) {
                        throw runtime_error("ERROR: Attempted to access parent variable [" + comp.content.variable_name + "] during resolution. " + e.what());
                    }
                    break;
                case UNRESOLVED_GLOBAL_VARIABLE:
                    rsec.type = RESOLVED_CONSTANT;
                    if(global_state.find(comp.content.variable_name) == global_state.end())
                        throw runtime_error("ERROR: Attempted to access nonexistent global variable {" + comp.content.variable_name + "} during resolution.");
                    rsec.content.constant = global_state.at(comp.content.variable_name);
                    break;
                case UNRESOLVED_CUDA_TAG:
                    rsec.type = RESOLVED_CUDA_TAG;
                    rsec.content.cuda_tag = comp.content.cuda_tag;
                    if(rsec.content.cuda_tag >= 10 || rsec.content.cuda_tag < 0)
                        throw runtime_error("ERROR: Invalid CUDA tag index " + to_string(rsec.content.cuda_tag) + " during resolution.");
                    break;
                case UNRESOLVED_OPERATOR:
                    rsec.type = RESOLVED_OPERATOR;
                    rsec.content.op = comp.content.op;
                    if(get_operator_arity(comp.content.op) == -1)
                        throw runtime_error("Invalid operator found during resolution.");
                    break;
                default:
                    throw runtime_error("Invalid UnresolvedStateEquationComponent type during resolution.");
            }
            rse.push_back(rsec);
        }
        return rse;
    }
};
