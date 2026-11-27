// Accept keyboard input in the terminal to manually control state.
// This allows you to quickly prototype a scene's various state inputs without re-rendering each time.
//
// The syntax is as so:
// my_state_value:{t} 1 +
// Commands must strictly start with a state value name, a colon, and then a valid state equation.

#include "StateTester.h"
#include <exception>
#include <iostream>
#include <string>
#include <unistd.h>
#include "../../Host_Device_Shared/vec.h"
#include "../Pixels.h"
#include "../../IO/LivePlayer.h"

bool parse_command(const string& command, string& state_value, string& operation)
{
    size_t colon_pos = command.find(':');
    if (colon_pos == string::npos)
    {
        cerr << "Invalid command format. Expected 'state_value:operation'." << endl;
        return false;
    }

    state_value = command.substr(0, colon_pos);
    operation = command.substr(colon_pos + 1);
    return true;
}

void open_ui(Scene& scene) {
    // For instance, renders started by ui/ui.py have nowhere to show the prompt.
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        cout << endl << "State Tester UI skipped: not running in a terminal." << endl;
        return;
    }

    cout << endl << "State Tester UI" << endl;
    cout << "Enter commands in the format: state_value:operation" << endl;
    cout << "Type 'exit' to keep the desired state value and continue rendering the video." << endl;
    cout << "Type 'print_state' to print the entire local state of the scene." << endl;

    LivePlayer lp(scene.get_width_height());
    StateManager last_good = scene.manager;

    bool redraw = true;
    while (true)
    {
        if (redraw) {
            try {
                scene.update();
                uint32_t* gpu_ptr = scene.query();
                last_good = scene.manager;
                if (!lp.accept_frame(gpu_ptr, true)) {
                    cout << "Player closed, leaving the State Tester UI." << endl;
                    break;
                }
            } catch (const exception& e) {
                // A failed evaluation leaves the manager half-evaluated, so put it back as it was.
                cerr << "Reverting: " << e.what() << endl;
                scene.manager = last_good;
            }
        }
        redraw = false; // Don't let a new picture hog up the screen unless something changed

        cout << endl << "> " << flush;
        string input;
        if (!getline(cin, input) || input == "exit")
            break;
        if (input.empty())
            continue;
        if (input == "print_state") {
            scene.manager.print_state();
            continue;
        }

        string state_value, operation;
        if (!parse_command(input, state_value, operation))
            continue;
        cout << "State Value: " << state_value << ", Operation: " << operation << endl;
        try {
            scene.manager.set(state_value, operation);
            redraw = true;
        } catch (const exception& e) {
            cerr << "Reverting: " << e.what() << endl;
            scene.manager = last_good;
        }
    }
}
