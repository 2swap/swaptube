// This is a GUI for a runtime application which renders videos.
// It has a few TMUX-style panels.
// Use ANSI codes to move the cursor around and draw the panels.

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sys/ioctl.h>
#include <unistd.h>
#include "../misc/StateManager.cpp" // Defines State
#include "../misc/pixels.h" // Defines Pixels

using namespace std;

// Basic Window base class
class Window {
public:
    int x = 0; // x position (column, 1-based for ANSI)
    int y = 0; // y position (row, 1-based for ANSI)
    int w = 0;
    int h = 0;
    virtual void print() = 0;
    virtual ~Window() {}

protected:
    // Move cursor to this window's top-left position
    void move_cursor() {
        cout << "\033[" << y << ";" << x << "H";
    }
};

// Global variable for terminal dimensions
struct winsize get_terminal_size() {
    struct winsize wsz;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsz);
    return wsz;
}

// Shows the latest frame. Remembers a Pixels object (the latest frame) as data.
class FrameWindow : public Window {
public:
    Pixels pix;

    void update(const Pixels* newpix) {
        pix = *newpix;
        print();
    }

};

// Draws a graph of State variables. Remembers a State as data.
class StateWindow : public Window {
public:
    State s; // Note: State is a wrapper around unordered_map<string, double>;

    void update(const State& newstate) {
        s = newstate;
    }

    void print() override {
        move_cursor();

        // TODO
    }
};

// Shows the timeline status. Remembers basic statistics about timeline history.
class TimelineWindow : public Window {
public:
    int num_frames_rendered = 0;
    double num_seconds_rendered = 0.0;
    int num_microblocks_completed_this_macroblock = 0;
    int num_microblocks_total_this_macroblock = 0;
    int total_frames_this_microblock = 0;
    int completed_frames_this_microblock = 0;

    void update(int frames_rendered, double seconds_rendered,
                int microblocks_completed, int microblocks_total, int completed_frames_micro, int total_frames_micro) {
        num_frames_rendered = frames_rendered;
        num_seconds_rendered = seconds_rendered;
        num_microblocks_completed_this_macroblock = microblocks_completed;
        num_microblocks_total_this_macroblock = microblocks_total;
        total_frames_this_microblock = total_frames_micro;
        completed_frames_this_microblock = completed_frames_micro;
        print();
    }

    void print() override {
        move_cursor();

        // Print stats lines inside box
        vector<string> lines;
        lines.push_back("Total Seconds: " + to_string(num_seconds_rendered));
        lines.push_back("Total Frames: " + to_string(num_frames_rendered));

        // Construct microblocks progress bar or numeric representation
        string microblocks_str;
        if (num_microblocks_total_this_macroblock <= w - 23) {
            // sufficient space: show progress bar
            microblocks_str = string(num_microblocks_completed_this_macroblock, '#') + string(num_microblocks_total_this_macroblock - num_microblocks_completed_this_macroblock, '.');
        } else {
            // not enough space: show "completed/total"
            microblocks_str = to_string(num_microblocks_completed_this_macroblock) + "/" + to_string(num_microblocks_total_this_macroblock);
        }
        lines.push_back("Microblocks this macroblock: " + microblocks_str);

        // Construct frames progress bar or numeric representation
        string frames_str;
        if (total_frames_this_microblock <= w - 18) {
            // sufficient space: show progress bar
            frames_str = string(completed_frames_this_microblock, '#') + string(total_frames_this_microblock - completed_frames_this_microblock, '.');
        } else {
            // not enough space: show "completed/total"
            frames_str = to_string(completed_frames_this_microblock) + "/" + to_string(total_frames_this_microblock);
        }
        lines.push_back("Frames this microblock: " + frames_str);

        int line_count = (int)lines.size();
        for(int i=0; i<h; ++i) {
            string content = (i < line_count) ? lines[i] : "";
            if ((int)content.length() > w) content = content.substr(0, w);
            cout << content << string(w - content.length(), ' ') << endl;
        }
    }
};

// Shows the output logs. Use this in place of stdout. Remembers ??? as data.
class LogWindow : public Window {
public:
    vector<string> logs;

    void log(const string& log) {
        logs.push_back(log);
    }

    void append(const string& partial) {
        logs[logs.size()-1] += partial;
        print();
    }

    void print() override {
        move_cursor();

        int start_log = max(0, (int)logs.size() - h);
        int end_log = (int)logs.size();

        for(int i=0; i<h; ++i) {
            string content = "";
            int idx = start_log + i;
            if (idx < end_log) content = logs[idx];
            if ((int)content.length() > w) content = content.substr(0, w);
            cout << content << string(w - content.length(), ' ') << endl;
        }
    }
};

// Global variables for terminal size and frame height to coordinate window sizes
int termWidth = 80;
int termHeight = 24;

// GUI that manages, and sets width and height, and positions (x,y), of all windows
class Gui {
public:

    // Should be the largest frame. Should match the resolution defined by VIDEO_WIDTH and VIDEO_HEIGHT, should fill the terminal width and be placed at the top.
    FrameWindow frame_window;

    // This should take up the left half of the remaining space under FrameWindow.
    StateWindow state_window;

    // Takes up the top half of the remaining bottom right section.
    TimelineWindow timeline_window;

    // This should take up the bottom half of the bottom right section.
    LogWindow log_window;

    // Update dimensions and positions based on current terminal size
    void update_sizes() {
        struct winsize wsz = get_terminal_size();
        termWidth = wsz.ws_col;
        termHeight = wsz.ws_row;

        {
            double imageAspect = static_cast<double>(VIDEO_WIDTH) / VIDEO_HEIGHT;

            // Set FrameWindow size and position
            frame_window.x = 1;
            frame_window.y = 1;
            frame_window.w = termWidth;
            //frame_window.w = min(frame_window.pix.w, termWidth);

        }

        // Timeline and Log windows take right half width and split bottom height half and half
        int bottom_height = termHeight - frame_window.h;
        int right_width = termWidth - state_window.w;
        timeline_window.x = state_window.w + 1;
        timeline_window.y = frame_window.h + 1;
        timeline_window.w = termWidth;
        timeline_window.h = bottom_height / 2;

        log_window.x = state_window.w + 1;
        log_window.y = frame_window.h + 1 + timeline_window.h;
        log_window.w = right_width;
        log_window.h = bottom_height - timeline_window.h; // adjust for odd number

    }

    // Print all windows in the layout
    void print_all() {
        // Update sizes and positions
        update_sizes();

        frame_window.print();
        //state_window.print();
        timeline_window.print();
        log_window.print();

        // Move cursor to bottom right after drawing
        cout << "\033[" << termHeight << ";" << termWidth << "H" << flush;
    }
};

Gui GUI;

