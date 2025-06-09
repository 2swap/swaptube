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

    // ---------------------------------------------------------------------
    // This function outputs the image to the terminal using Unicode half-block
    // characters (▀). Each printed character cell represents two rows of supersampled
    // image data. The top half of the block uses the foreground color and the bottom
    // half uses the background color.
    // 
    // We compute the sampling regions by mapping the terminal grid to the source image
    // and then average over each corresponding rectangle.
    void print() override {
        // Move cursor to this window's top-left
        move_cursor();


        int sample_height = h * 2;

        // For each terminal character cell, we determine the corresponding region in the image.
        for (int ycell = 0; ycell < h; ++ycell) {
            if(!FOR_REAL && ycell == 1) {
                cout << "In Smoketest Mode. No video is being rendered." << endl;
                continue;
            }
            for (int xcell = 0; xcell < w; ++xcell) {
                // Determine the horizontal region that maps to this terminal column.
                int x0 = xcell * pix.w / w;
                int x1 = (xcell + 1) * pix.w / w;

                // For the top half of the character cell:
                int top_y0 = (2 * ycell) * pix.h / sample_height;
                int top_y1 = (2 * ycell + 1) * pix.h / sample_height;

                // For the bottom half of the character cell:
                int bot_y0 = (2 * ycell + 1) * pix.h / sample_height;
                int bot_y1 = (2 * ycell + 2) * pix.h / sample_height;

                int a_top = 0, r_top = 0, g_top = 0, b_top = 0;
                int a_bot = 0, r_bot = 0, g_bot = 0, b_bot = 0;

                // Supersample (average) over the regions.
                pix.get_average_color(x0, top_y0, x1, top_y1, a_top, r_top, g_top, b_top);
                pix.get_average_color(x0, bot_y0, x1, bot_y1, a_bot, r_bot, g_bot, b_bot);

                double alpha_top = a_top / 255.;
                double one_minus_alpha_top = 1-alpha_top;
                r_top = r_top * alpha_top + getr(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_top;
                g_top = g_top * alpha_top + getg(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_top;
                b_top = b_top * alpha_top + getb(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_top;

                double alpha_bot = a_bot / 255.;
                double one_minus_alpha_bot = 1-alpha_bot;
                r_bot = r_bot * alpha_bot + getr(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_bot;
                g_bot = g_bot * alpha_bot + getg(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_bot;
                b_bot = b_bot * alpha_bot + getb(VIDEO_BACKGROUND_COLOR) * one_minus_alpha_bot;

                // Use ANSI true-color escape sequences:
                //  - Set foreground to the average top color.
                //  - Set background to the average bottom color.
                // Then print the Unicode upper half block (▀), which renders the top half in
                // the foreground color and the bottom half in the background color.
                cout << "\033[38;2;" << r_top << ";" << g_top << ";" << b_top << "m"
                     << "\033[48;2;" << r_bot << ";" << g_bot << ";" << b_bot << "m"
                     << "\u2580";
            }
            cout << endl;
        }
        // Reset at the end.
        cout << "\033[0m" << endl;
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

            // Determine the effective vertical resolution (in image pixels) that we wish to sample.
            // We multiply by 2 because each printed line represents two image rows.
            int sample_height = static_cast<int>(frame_window.w / imageAspect);

            // Implicitly ensure sample_height is taken as even.
            // Assume a half-block is square
            frame_window.h = sample_height / 2;
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

