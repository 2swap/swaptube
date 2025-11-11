#pragma once

#include <iomanip>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <regex>
#include <cassert>
#include <iostream>
#include "calculator.h"
#include "../Host_Device_Shared/helpers.h"

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
    string equation;

    // A list of all of the variables which any certain variable depends on.
    // Using "x=y+5" as an example, the key "x"'s value would be a list containing only "y".
    list<string> dependencies;

    VariableContents()
                : value(0.0), fresh(true), equation(""), dependencies() {}

    VariableContents(string eq,
                     double val = 0.0,
                     bool fr = true
                    ) : value(val), fresh(fr), equation(eq), dependencies() {}
};

using StateQuery = unordered_set<string>;
void state_query_insert_multiple(StateQuery& sq, const StateQuery& additions){
    for(const string& s : additions){
        sq.insert(s);
    }
}

using StateSet = unordered_map<string, string>;
class State {
public:
    State() {}
    State(const unordered_map<string, double>& m) : map(m) {}

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
    bool operator==(const State& other) const {
        return map == other.map;
    }

    // Implement inequality operator
    bool operator!=(const State& other) const {
        return !(*this == other);
    }

    unordered_map<string, double> get_map() const {
        return map;
    }

private:
    unordered_map<string, double> map;
};

static unordered_map<string, double> global_state{
    {"frame_number", 0},
    {"t", 0},
    {"macroblock_number", 0},
    {"microblock_number", 0},
    {"macroblock_fraction", 0},
    {"microblock_fraction", 0},
    {"VIDEO_WIDTH", VIDEO_WIDTH},
    {"VIDEO_HEIGHT", VIDEO_HEIGHT},
};
void print_global_state(){
    cout << endl << endl << "=====GLOBAL STATE=====" << endl;
    for(const auto& pair : global_state){
        cout << pair.first << ": " << pair.second << endl;
    }
    cout << "======================" << endl << endl;
}
double get_global_state(string key){
    const auto& pair = global_state.find(key);
    if(pair == global_state.end()){
        // I used to error on this always. However, there is a general pattern in which
        // we give some scene "A" the ability to write to global state and give
        // scene "B"'s state manager a dependency on that variable. Depending on
        // the order in which the scenes are rendered, we get indeterminate behavior.
        // We could fix this by enforcing a render order, but I don't think
        // the complexity is, or ever will be worth it.
        // This zero-default behavior fixes the problem for the first frame,
        // after which scene "A" will have published to global state.
        if (global_state["microblock_fraction"] != 0) {
            print_global_state();
            throw runtime_error("global state access failed on element " + key);
        }
        return 0;
    }
    return pair->second;
}

class StateManager {
public:
    StateManager() : parent(nullptr), subjugated(false) {}

    /* Accessors */
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
                 << setw(38) << (vc.equation == ""?"":" := " + vc.equation)
                 << " : " << setw(10) << vc.value
                 << (vc.fresh ? " (Fresh)" : " (Stale)")
                 << " " << &vc
                 << endl;
        }
        cout << "-----------------------" << endl;
    }


    /* Modifiers */
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
        if(contains(variable)) ret = { {variable, get_equation(variable)} };

        /* When a new component is added, we do not know the
         * correct compute order anymore since there are new equations.
         */
        last_compute_order.clear();
        variables[variable] = VariableContents(equation);

        // Parse equation to find dependencies and add them to the list
        size_t pos = 0;
        while ((pos = equation.find("<", pos)) != string::npos) {
            size_t end_pos = equation.find(">", pos);
            if (end_pos != string::npos) {
                string dependency = equation.substr(pos + 1, end_pos - pos - 1);
                variables[variable].dependencies.push_back(dependency);
                pos = end_pos + 1;
            } else break;
        }

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

    StateSet transition(const TransitionType tt, const string& variable, const string& equation) {
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

        string eq1 = get_equation(variable);

        // No point in doing a noop transition
        if(eq1 != equation) {
            string lerp_both = eq1 + " " + equation + " {" + (tt==MICRO?"micro":"macro") + "block_fraction} smoothlerp";
            set(variable+".post_transition", equation);
            set(variable, lerp_both);
                 if(tt == MICRO) in_microblock_transition.insert(variable);
            else if(tt == MACRO) in_macroblock_transition.insert(variable);
        }

        return { {variable, eq1} };
    }
    StateSet transition(const TransitionType tt, const StateSet& equations) {
        StateSet ret = {};
        for(auto it = equations.begin(); it != equations.end(); it++){
            StateSet prev = transition(tt, it->first, it->second);
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
                    for (const string& dependency : vc.dependencies) {
                        if(!contains(dependency)){
                            print_state();
                            throw runtime_error("error: attempted to access variable " + dependency
                                    + " during evaluation of " + variable_name + " := "
                                    + vc.equation + ", but it does not exist in this StateManager!"
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

    string get_equation(const string& variable) const {
        return get_variable(variable).equation;
    }

    string get_equation_with_tags(const string& variable) const {
        // This is called by Scenes which read directly from state and hand the equation
        // to the CUDA device. The tags are then inserted on-device with thread-specific values
        // which are passed into the tags.
        string equation = get_equation(variable);
        equation = insert_local_state_dependencies(equation);
        equation = insert_parent_state_dependencies(equation);
        equation = insert_global_state_dependencies(equation);
        return equation;
    }

    double get_value(const string& variable) const {
        VariableContents vc = get_variable(variable);

        // We should never ever read from a stale variable.
        if(!vc.fresh){
            print_state();
            throw runtime_error("ERROR: Attempted to read stale variable " + variable + "!\nState has been printed above.");
        }

        return vc.value;
    }

    const void set_subjugated(bool b) {
        subjugated = b;
    }

    const void begin_timer(const string& timer_name) {
        set(timer_name, "{t} " + to_string(global_state["t"]) + " -");
    }

    const State get_state(const StateQuery& query) const {
        if(subjugated){
            if (parent == nullptr)
                throw runtime_error("A StateManager was queried while marked as subjugated despite not having a parent.");
            return parent->get_state(query);
        }
        State result;
        for (const auto& varname : query) {
            result.set(varname, get_value(varname));
        }
        return result;
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

    string insert_equation_dependencies(string equation) const {
        equation = insert_tag_dependencies(equation);
        equation = insert_local_state_dependencies(equation);
        equation = insert_parent_state_dependencies(equation);
        equation = insert_global_state_dependencies(equation);
        return equation;
    }

    string insert_tag_dependencies(string equation) const {
        // Replace all instances of "(...)" with unicode value of first character in parentheses
        size_t pos = 0;
        while ((pos = equation.find('(')) != string::npos) {
            size_t end_pos = equation.find(')', pos);
            if (end_pos != string::npos) {
                equation.replace(pos, end_pos - pos + 1, to_string(static_cast<int>(equation[pos + 1])));
            } else {
                throw runtime_error("Mismatched parentheses in equation: " + equation);
            }
        }

        return equation;
    }

    // Logic to replace substrings in angle brackets <> with the local variable's value
    // Take a string like "<variable_that_equals_7> 5 +" and return "7 5 +"
    string insert_local_state_dependencies(string equation) const {
        size_t start_pos = equation.find('<');
        while (start_pos != string::npos) {
            size_t end_pos = equation.find('>', start_pos);
            if (end_pos != string::npos) {
                string content = equation.substr(start_pos + 1, end_pos - start_pos - 1);
                if(!contains(content)){
                    print_state();
                    throw runtime_error("ERROR: Attempted to access variable " + content + " during local state dependency insertion, but it does not exist!\nState has been printed above.");
                }
                string value = to_string(get_value(content));
                equation.replace(start_pos, end_pos - start_pos + 1, value);
                // Update start_pos to search for the next occurrence
                start_pos = equation.find('<', start_pos + value.length());
            } else {
                throw runtime_error("Mismatched angle brackets in equation: " + equation);
            }
        }
        return equation;
    }

    // Logic to replace substrings in square brackets [] with the parent's value
    string insert_parent_state_dependencies(string equation) const {
        size_t start_pos = equation.find('[');
        while (start_pos != string::npos) {
            size_t end_pos = equation.find(']', start_pos);
            if (end_pos != string::npos) {
                string content = equation.substr(start_pos + 1, end_pos - start_pos - 1);
                string value_from_parent = to_string(get_value_from_parent(content));
                equation.replace(start_pos, end_pos - start_pos + 1, value_from_parent);
                // Update start_pos to search for the next occurrence
                start_pos = equation.find('[', start_pos + value_from_parent.length());
            } else {
                throw runtime_error("Mismatched square brackets in equation: " + equation);
            }
        }
        return equation;
    }

    // Logic to replace substrings in curly braces {} with the global data-published value
    string insert_global_state_dependencies(string equation) const {
        size_t start_pos = equation.find('{');
        while (start_pos != string::npos) {
            size_t end_pos = equation.find('}', start_pos);
            if (end_pos != string::npos) {
                string content = equation.substr(start_pos + 1, end_pos - start_pos - 1);
                string value = to_string(get_global_state(content));
                equation.replace(start_pos, end_pos - start_pos + 1, value);
                // Update start_pos to search for the next occurrence
                start_pos = equation.find('{', start_pos + value.length());
            } else {
                throw runtime_error("Mismatched curly braces in equation: " + equation);
            }
        }

        return equation;
    }

    void evaluate_single_variable(const string& variable) {
        VariableContents& vc = variables.at(variable);
        assert(!vc.fresh);
        vc.fresh = true;
        string scrubbed_equation = insert_equation_dependencies(vc.equation);
        try {
            vc.value = calculator(scrubbed_equation);
        } catch (const runtime_error& e) {
            print_state();
            throw runtime_error("Error evaluating equation for variable " + variable + "\n" + e.what() + "\nScrubbed equation: " + scrubbed_equation + "\nState has been printed above.");
        }
    }

    VariableContents get_variable(const string& variable) const {
        if(variables.find(variable) != variables.end()){
            return variables.at(variable);
        }
        if(global_state.find(variable) != global_state.end()){
            return VariableContents("", get_global_state(variable), true);
        }
        print_state();
        print_global_state();
        throw runtime_error("ERROR: Attempted to access variable " + variable + " without it existing!\nState has been printed above.");
    }

    double get_value_from_parent(const string& variable) const {
        if(parent == nullptr)
            throw runtime_error("Parent was a nullptr while looking for [" + variable + "]");
        return parent->get_value(variable);
    }

    void close_all_transitions(unordered_set<string>& in_transition){
        unordered_set<string> in_transition_copy = in_transition;
        in_transition.clear();
        for(string varname : in_transition_copy){
            set(varname, get_equation(varname + ".post_transition"));
            VariableContents& vc = variables.at(varname);
            vc.value = get_value(varname + ".post_transition");
            remove(varname + ".post_transition");
        }
    }

};
