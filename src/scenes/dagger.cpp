#pragma once

#include <iomanip>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <cassert>
#include <iostream>
#include "calculator.cpp"
#include "../misc/inlines.h"

using namespace std;

/* Dagger, short for Directed Acyclic Grapher, is a class
 * intended to facilitate frame-by-frame manipulation of state.
 * It is composed of a number of equations which express how to update
 * certain variables in terms of others. Cyclical computations are not
 * permitted, such as "y=x+1" and "x=y", hence the DAG of computational
 * direction. Similarly, if there are equations such as "x=y" and "y=3",
 * Dagger will intentionally run the second equation first.
 */

struct VariableContents {
    // The numerical value currently stored by this variable
    double value;

    // Is this variable fresh (updated since last control cycle) or stale?
    bool fresh;

    // Special variables are not updated in accordance with their equation and are expected
    // to be modified elsewhere. For example, the time variable <t> is special.
    bool special;

    // A list of variable-equation pairs like <"x", "y 5 +"> to represent "x=y+5" (in RPN)
    string equation;

    // A list of all of the variables which any certain variable depends on.
    // Using "x=y+5" as an example, the key "x"'s value would be a list containing only "y".
    list<string> dependencies;

    VariableContents()
                : value(0.0), fresh(true), special(false), equation(""), dependencies() {}

    VariableContents(string eq,
                     double val = 0.0,
                     bool fr = true,
                     bool spec = false
                    ) : equation(eq), value(val), fresh(fr), special(spec), dependencies() {}
};

class Dagger {
public:
    Dagger() {
        variables["t"] = VariableContents("", 0, true, true);
        variables["frame_number"] = VariableContents("", -1, true, true);
        variables["audio_segment_number"] = VariableContents("", -1, true, true);
        variables["transition_fraction"] = VariableContents("", 0, true, true);
    }

    /* Accessors */
    bool contains(const string& varname) const {
        return variables.find(varname) != variables.end();
    }
    double operator [](const string v) const {return get_value(v);}
    double get_with_default(const string v, double deflt) const {
        if (contains(v)) {
            return get_value(v);
        }
        return deflt;
    }
    void print_state() const {
        /* Print out all variable names along with their current value and their computation status. */
        cout << "Variables:" << endl;
        cout << "-----------------------" << endl;
        for (const auto& variable : variables) {
            const VariableContents& vc = variable.second;
            cout << (vc.special?"*":" ")
                 << left << setw(32) << variable.first
                 << setw(38) << (vc.equation == ""?"":" := " + vc.equation)
                 << " : " << setw(10) << vc.value
                 << (vc.fresh ? " (Fresh)" : " (Stale)")
                 << " " << &vc
                 << endl;
        }
        cout << "-----------------------" << endl;
    }


    /* Modifiers */
    void add_equation(string variable, string equation) {
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
    }
    void add_transition(string variable, string equation) {

        // No point in doing a noop transition
        if(get_equation(variable) == equation) return;

        // Nested transitions not supported
        if(in_transition.find(variable) != in_transition.end()){
            failout("Transition added to a variable already in transition: " + variable);
        }

        in_transition.insert(variable);
        string eq1 = get_equation(variable);
        string eq2 = equation;
        string lerp_both = "<" + variable + ".pre_transition> <" + variable + ".post_transition> <transition_fraction> smoothlerp";
        add_equation(variable+".pre_transition", eq1);
        add_equation(variable+".post_transition", eq2);
        add_equation(variable, lerp_both);
    }
    void remove_equation(string variable) {
        /* When a new component is removed, we do not know the
         * correct compute order anymore.
         */
        last_compute_order.clear();
        variables.erase(variable);
    }

    /* Bulk Modifiers. Naive. One per modifier. */
    void add_transitions(std::unordered_map<std::string, std::string> equations) {
        for(auto it = equations.begin(); it != equations.end(); it++){
            add_transition(it->first, it->second);
        }
    }
    void add_equations(std::unordered_map<std::string, std::string> equations) {
        for(auto it = equations.begin(); it != equations.end(); it++){
            add_equation(it->first, it->second);
        }
    }
    void remove_equations(std::unordered_map<std::string, std::string> equations) {
        for(auto it = equations.begin(); it != equations.end(); it++){
            remove_equation(it->first);
        }
    }

    void close_all_transitions(){
        for(string varname : in_transition){
            add_equation(varname, get_equation(varname + ".post_transition"));
            VariableContents& vc = variables.at(varname);
            vc.value = get_value(varname + ".post_transition");
            remove_equation(varname + ".post_transition");
            remove_equation(varname + ".pre_transition");
        }
        in_transition.clear();
    }

    void set_special(const string& varname, double value){
        // If the variable doesn't exist yet, make it
        if(!contains(varname)){
            last_compute_order.clear();
            variables[varname] = VariableContents("", value, true, true);
            return;
        }

        //Just call get_variable because it performs some checks for existence.
        get_variable(varname);

        VariableContents& vc = variables.at(varname);

        // If this isnt a special variable, this value will soon be stomped on.
        // It wouldnt make sense to use this function in other cases.
        assert(vc.special);

        vc.value = value;
    }

    void evaluate_all() {
        /* Step 1: Iterate through all variables,
         * check that they are fresh from last cycle,
         * and reset it to false. */
        for (auto& variable : variables) {
            if(!variable.second.fresh){
                print_state();
                failout("ERROR: variable " + variable.first + " was not fresh when expected!\nState has been printed above.");
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
                            failout("error: attempted to access variable " + dependency
                                    + " during evaluation of " + variable_name + " := "
                                    + vc.equation + "!\nstate has been printed above.");
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
                    failout("error: variable dependency graph appears not to be a DAG! State has been printed above.");
                }
            }
        }
    }

private:
    // A list of variables and their relevant data
    unordered_map<string, VariableContents> variables;

    // A list of variables in the order they were computed the last time around
    list<string> last_compute_order;

    // A list of all variable names which are currently undergoing transitions
    unordered_set<string> in_transition;

    // Take a string like "<variable_that_equals_7> 5 +" and return "7 5 +"
    string insert_equation_dependencies(string variable, string equation) const {
        for (const string& dependency : variables.at(variable).dependencies) {
            const VariableContents& vc = variables.at(dependency);
            // Make sure that the dependency is already computed.
            assert(vc.fresh);
            string replaced_substring = "<" + dependency + ">";
            size_t pos = equation.find(replaced_substring);
            while (pos != string::npos) {
                equation.replace(pos, replaced_substring.length(), to_string(vc.value));
                // Update pos to search for the next occurrence
                pos = equation.find(replaced_substring, pos + to_string(vc.value).length());
            }
        }
        return equation;
    }

    void evaluate_single_variable(const string& variable) {
        VariableContents& vc = variables.at(variable);
        assert(!vc.fresh);
        vc.fresh = true;
        if(vc.special) return;
        string scrubbed_equation = insert_equation_dependencies(variable, vc.equation);
        vc.value = calculator(scrubbed_equation);
    }

    VariableContents get_variable(string variable) const {
        if(variables.find(variable) == variables.end()){
            print_state();
            failout("ERROR: Attempted to access slate variable " + variable + " without it existing!\nState has been printed above.");
        }
        return variables.at(variable);
    }

    string get_equation(string variable) const {
        return get_variable(variable).equation;
    }

    double get_value(string variable) const {
        VariableContents vc = get_variable(variable);

        // We should never ever read from a stale variable.
        assert(vc.fresh);

        return vc.value;
    }
};

// Global dagger
Dagger dag;

void test_dagger() {
    cout << "Testing dagger" << endl;
    // Construct a Dagger object
    Dagger dagger;
    assert(dagger["t"] == 0);

    // Add equations
    dagger.add_equation("x", "5"); // x = 5
    dagger.add_equation("y", "10"); // y = 10
    dagger.add_equation("z", "<x> <y> +"); // z = x + y
    dagger.set_special("t", 420);
    dagger.evaluate_all();

    // Validate initial values
    assert(dagger["x"] == 5.0);
    assert(dagger["y"] == 10.0);
    assert(dagger["z"] == 15.0);
    assert(dagger["t"] == 420);

    // Modify equations
    dagger.add_equation("x", "7"); // x = 7
    dagger.add_equation("y", "20"); // y = 20
    dagger.add_equation("z", "<x> <y> +"); // z = x + y
    dagger.set_special("t", 69);
    dagger.evaluate_all();

    // Validate updated values
    assert(dagger["x"] == 7.0);
    assert(dagger["y"] == 20.0);
    assert(dagger["z"] == 27.0);
    assert(dagger["t"] == 69);
}
