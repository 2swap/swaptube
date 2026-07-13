// Accept keyboard input in the terminal to manually control state.
// This allows you to quickly prototype a scene's various state inputs without re-rendering each time.
//
// The syntax is as so:
// my_state_value:{t} 1 +
// Commands must strictly start with a state value name, a colon, and then a valid state equation.

#include "StateTester.h"
#include <iostream>
#include <string>
#include <cstdio>
#include "../../Host_Device_Shared/vec.h"
#include "../Pixels.h"

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

void parse_command(const string& command, string& state_value, string& operation)
{
    size_t colon_pos = command.find(':');
    if (colon_pos == string::npos)
    {
        cerr << "Invalid command format. Expected 'state_value:operation'." << endl;
        return;
    }

    state_value = command.substr(0, colon_pos);
    operation = command.substr(colon_pos + 1);
}

void open_ui(Scene& scene) {
    cout << endl << "State Tester UI" << endl;
    cout << "Enter commands in the format: state_value:operation" << endl;
    cout << "Type 'exit' to keep the desired state value and continue rendering the video." << endl;
    cout << "Type 'print_state' to print the entire local state of the scene." << endl;

    ivec2 dimensions = scene.get_width_height();
    string ffplay_cmd_str = "ffplay -f rawvideo -pixel_format argb -video_size " + to_string(dimensions.x) + "x" + to_string(dimensions.y) + " -";
    cout << "Running command: " << ffplay_cmd_str << endl;
    const char* ffplay_cmd = ffplay_cmd_str.c_str();

    FILE* pipe = popen(ffplay_cmd, "w");

    string input;
    bool skip_render = false;
    while (true)
    {
        if (!skip_render) {
            uint32_t* gpu_ptr = scene.query();
            Pixels pix(dimensions);
            cuda_copy_pixels_to_host(pix.pixels.data(), pix.pixels.size(), gpu_ptr);

            pix.print_to_terminal();

            fwrite(pix.pixels.data(),
                sizeof(int32_t),
                pix.pixels.size(),
                pipe);

            fflush(pipe);
        }
        skip_render = false;

        cout << "> ";
        getline(cin, input);

        if (input == "exit")
            break;
        else if (input == "print_state") {
            scene.manager.print_state();
            skip_render = true; // We don't want a new picture to hog up the screen
        } else {
            string state_value, operation;
            parse_command(input, state_value, operation);

            // Here you would apply the operation to the state_value
            // For demonstration purposes, we'll just print them out
            cout << "State Value: " << state_value << ", Operation: " << operation << endl;

            scene.manager.set(state_value, operation);
        }
    }
}
