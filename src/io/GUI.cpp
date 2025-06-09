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
            // Reset colors at the end of each line.
            cout << "\033[0m" << endl;
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

        // Draw border
        if(w < 2 || h < 2) return; // need minimum size
        cout << "+" << string(w-2, '-') << "+" << endl;

        int rows_for_graph = h-3;
        if(rows_for_graph <= 0) {
            // Fill empty lines
            for(int i=0; i<h-2; ++i){
                cout << "|" << string(w-2,' ') << "|" << endl;
            }
            cout << "+" << string(w-2, '-') << "+" << endl;
            return;
        }

        int n_vars = (int)s.size();
        if(n_vars == 0) {
            for(int i=0; i<rows_for_graph; ++i){
                cout << "|" << string(w-2,' ') << "|" << endl;
            }
            cout << "+" << string(w-2, '-') << "+" << endl;
            return;
        }

        // Extract values for graphing: show all vars
        int name_width = 10;
        double max_val = numeric_limits<double>::lowest();
        double min_val = numeric_limits<double>::max();
        unordered_map<string, double> map = s.get_map();
        for(auto& p : map) {
            if(p.second > max_val) max_val = p.second;
            if(p.second < min_val) min_val = p.second;
        }
        double val_range = max_val - min_val;
        if(val_range < 1e-6) val_range = 1.0; // Avoid zero division

        int max_bar_len = w - name_width - 4;
        if (max_bar_len < 1) max_bar_len = 1;

        int lines_printed = 0;
        for (auto p : map) {
            if(lines_printed >= rows_for_graph) break;
            // Format name (truncate/pad)
            string name = p.first;
            if ((int)name.size() > name_width) name = name.substr(0, name_width);
            else if ((int)name.size() < name_width) name += string(name_width - name.size(), ' ');

            int bar_len = static_cast<int>((p.second - min_val) / val_range * max_bar_len);
            if (bar_len < 0) bar_len = 0;
            if (bar_len > max_bar_len) bar_len = max_bar_len;

            // Print line: | name bar |
            cout << "| " << name << " ";
            cout << string(bar_len, '#') << string(max_bar_len - bar_len, ' ');
            cout << "|" << endl;
            lines_printed++;
        }

        // Fill leftover lines with empty space inside box
        for(int i = lines_printed; i < rows_for_graph; ++i) {
            cout << "|" << string(w-2, ' ') << "|" << endl;
        }

        // Bottom border
        cout << "+" << string(w-2, '-') << "+" << endl;
    }
};

// Shows the timeline status. Remembers basic statistics about timeline history.
class TimelineWindow : public Window {
public:
    int num_macroblocks_completed = 0;
    int num_frames_rendered = 0;
    double num_seconds_rendered = 0.0;
    int num_microblocks_completed_this_macroblock = 0;
    int num_microblocks_total_this_macroblock = 0;
    int total_frames_this_microblock = 0;
    int completed_frames_this_microblock = 0;

    void update(int macroblocks_completed, int frames_rendered, double seconds_rendered,
                int microblocks_completed, int microblocks_total, int completed_frames_micro, int total_frames_micro) {
        num_macroblocks_completed = macroblocks_completed;
        num_frames_rendered = frames_rendered;
        num_seconds_rendered = seconds_rendered;
        num_microblocks_completed_this_macroblock = microblocks_completed;
        num_microblocks_total_this_macroblock = microblocks_total;
        total_frames_this_microblock = total_frames_micro;
        completed_frames_this_microblock = completed_frames_micro;
    }

    void print() override {
        move_cursor();

        if(w < 2 || h < 2) return; // need minimum size
        // Draw border
        cout << "+" << string(w-2, '-') << "+" << endl;

        // Print stats lines inside box
        vector<string> lines;
        lines.push_back("Macroblocks: " + to_string(num_macroblocks_completed));
        lines.push_back("Total Seconds: " + to_string(num_seconds_rendered));
        lines.push_back("Total Frames: " + to_string(num_frames_rendered));
        lines.push_back("Microblocks this macroblock: " + string(num_microblocks_completed_this_macroblock, '#') + string(num_microblocks_total_this_macroblock - num_microblocks_completed_this_macroblock, '.'));
        lines.push_back("Frames this microblock: " + string(completed_frames_this_microblock, '#') + string(total_frames_this_microblock - completed_frames_this_microblock, '.'));

        int line_count = (int)lines.size();
        int max_lines = h - 2;
        for(int i=0; i<max_lines; ++i) {
            string content = (i < line_count) ? lines[i] : "";
            if ((int)content.length() > w-2) content = content.substr(0, w-2);
            cout << "|" << content << string(w-2 - content.length(), ' ') << "|" << endl;
        }
        cout << "+" << string(w-2, '-') << "+" << endl;
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
    }

    void print() override {
        move_cursor();

        if(w < 2 || h < 2) return; // need minimum size
        // Draw border
        cout << "+" << string(w-2, '-') << "+" << endl;

        int max_lines = h - 2;
        int start_log = max(0, (int)logs.size() - max_lines);
        int end_log = (int)logs.size();

        for(int i=0; i<max_lines; ++i) {
            string content = "";
            int idx = start_log + i;
            if (idx < end_log) content = logs[idx];
            if ((int)content.length() > w-2) content = content.substr(0, w-2);
            cout << "|" << content << string(w-2 - content.length(), ' ') << "|" << endl;
        }
        cout << "+" << string(w-2, '-') << "+" << endl;
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
            double imageAspect = static_cast<double>(frame_window.pix.w) / frame_window.pix.h;

            // Set FrameWindow size and position
            frame_window.x = 1;
            frame_window.y = 1;
            frame_window.w = min(frame_window.pix.w, termWidth);

            // Determine the effective vertical resolution (in image pixels) that we wish to sample.
            // We multiply by 2 because each printed line represents two image rows.
            int sample_height = static_cast<int>(frame_window.w / imageAspect);

            // Implicitly ensure sample_height is taken as even.
            // Assume a half-block is square
            frame_window.h = sample_height / 2;
        }

        // StateWindow takes remaining height (below frame window), full left half
        int bottom_height = termHeight - frame_window.h;
        state_window.x = 1;
        state_window.y = frame_window.h + 1;
        state_window.w = termWidth / 2;
        state_window.h = bottom_height;

        // Timeline and Log windows take right half width and split bottom height half and half
        int right_width = termWidth - state_window.w;
        timeline_window.x = state_window.w + 1;
        timeline_window.y = frame_window.h + 1;
        timeline_window.w = right_width;
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

        // Print the frame window at top left
        frame_window.print();

        // Print state window below frame left half
        state_window.print();

        // Print timeline right top half
        timeline_window.print();

        // Print logs right bottom half
        log_window.print();

        // Move cursor to bottom right after drawing
        cout << "\033[" << termHeight << ";" << termWidth << "H" << flush;
    }
};

Gui GUI;

