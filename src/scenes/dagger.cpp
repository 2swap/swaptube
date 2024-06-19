#pragma once

#include <string>
#include <unordered_map>
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
    double value;
    bool state;
    bool special;
    VariableContents(double val = 0.0, bool st = true, bool spec = false) : value(val), state(st), special(spec) {}
};

class Dagger {
public:
    Dagger() {
        variables["t"] = VariableContents(0, true, true);
    }

    void remove_equation(string variable) {
        variables.erase(variable);
        dependencies.erase(variable);
        equations.erase(variable);
    }
    double operator [](const string v) const {return get(v);}
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

    void add_equation(string variable, string equation) {
        /* When a new component is added, we do not know the
         * correct compute order anymore since there are new equations.
         */
        last_compute_order.clear();
        equations[variable] = equation;
        variables[variable] = VariableContents();

        // Parse equation to find dependencies and add them to the list
        dependencies[variable] = {};
        size_t pos = 0;
        while ((pos = equation.find("<", pos)) != string::npos) {
            size_t end_pos = equation.find(">", pos);
            if (end_pos != string::npos) {
                string dependency = equation.substr(pos + 1, end_pos - pos - 1);
                dependencies[variable].push_back(dependency);
                pos = end_pos + 1;
            } else break;
        }
    }

    double get(string variable) const {
        if(variables.find(variable) == variables.end()){
            print_state();
            failout("ERROR: Attempted to read slate variable " + variable + " without it existing!\nState has been printed above.");
        }
        assert(variables.at(variable).state);
        return variables.at(variable).value;
    }

    /* Print out all variable names along with their current value and their computation status. */
    void print_state() const {
        cout << "Variables:" << endl;
        cout << "-----------------------" << endl;
        for (const auto& variable : variables) {
            const VariableContents& vc = variable.second;
            cout << variable.first << " : " << vc.value << (vc.state ? " (Fresh)" : " (Stale)") << endl;
        }
        cout << "-----------------------" << endl;
    }

    void evaluate_all() {
        /* Step 0: Increment timer. */
        variables["t"].value+=1.0/VIDEO_FRAMERATE;
        
        /* Step 1: Iterate through all variables,
         * check that their "state" is true from last cycle,
         * and reset it to false. */
        for (auto& variable : variables) {
            assert(variable.second.state);
            variable.second.state = false;
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
                    if (vc.state) continue;

                    // Check that all variable dependencies are met
                    bool all_dependencies_met = true;
                    for (const string& dependency : dependencies[variable_name]) {
                        const VariableContents& dep_vc = variables.at(dependency);
                        if (!dep_vc.state) {
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
                assert(one_variable_changed || !unevaluated_variable_remaining);
            }
        }
    }

private:
    // A list of variable-equation pairs like <"x", "y 5 +"> to represent "x=y+5" (in RPN)
    unordered_map<string, string> equations;

    // A list of all of the variables which any certain variable depends on.
    // Using "x=y+5" as an example, the key "x"'s value would be a list containing only "y".
    unordered_map<string, list<string>> dependencies;

    // A list of variables and their values (along with whether they have been computed this cycle).
    unordered_map<string, VariableContents> variables;

    // A list of variables in the order they were computed the last time around
    list<string> last_compute_order;

    // Take a string like "<variable_that_equals_7> 5 +" and return "7 5 +"
    string insert_equation_dependencies(string variable, string equation) const {
        for (const string& dependency : dependencies.at(variable)) {
            const VariableContents& vc = variables.at(dependency);
            // Make sure that the dependency is already computed.
            assert(vc.state);
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
        assert(!vc.state);
        vc.state = true;
        if(vc.special) return;
        string scrubbed_equation = insert_equation_dependencies(variable, equations.at(variable));
        vc.value = calculator(scrubbed_equation);
    }
};

// Global dagger
Dagger dag;

void test_dagger() {
    cout << "Testing dagger" << endl;
    // Construct a Dagger object
    Dagger dagger;
    assert(dagger.get("t") == 0);

    // Add equations
    dagger.add_equation("x", "5"); // x = 5
    dagger.add_equation("y", "10"); // y = 10
    dagger.add_equation("z", "<x> <y> +"); // z = x + y
    dagger.evaluate_all();
    dagger.print_state();

    // Validate initial values
    assert(dagger.get("x") == 5.0);
    assert(dagger.get("y") == 10.0);
    assert(dagger.get("z") == 15.0);
    assert(dagger.get("t") == 1.0/VIDEO_FRAMERATE);

    // Modify equations
    dagger.add_equation("x", "7"); // x = 7
    dagger.add_equation("y", "20"); // y = 20
    dagger.add_equation("z", "<x> <y> +"); // z = x + y
    dagger.evaluate_all();
    dagger.print_state();

    // Validate updated values
    assert(dagger.get("x") == 7.0);
    assert(dagger.get("y") == 20.0);
    assert(dagger.get("z") == 27.0);
    assert(dagger.get("t") == 2.0/VIDEO_FRAMERATE);
}
