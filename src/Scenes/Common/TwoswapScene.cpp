#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class TwoswapScene : public Scene {
public:
    TwoswapScene(const double width = 1, const double height = 1) : Scene(width, height) {
        ScalingParams sp(pix.w*.7, pix.h);
        saved_pix = Pixels(width, height);
        Pixels twoswap_pix = latex_to_pix(latex_text("2swap"), sp);
        saved_pix.fill_ellipse(pix.w/4, pix.h/2, pix.w/14, pix.w/14, OPAQUE_WHITE);
        double yval = (pix.h-twoswap_pix.h)/2+pix.w/48;
        saved_pix.overwrite(twoswap_pix, pix.w/4+pix.w/14+pix.w/48, yval);
        ScalingParams sp2(pix.w*.4, pix.h*.2);
        Pixels swaptube_pix = latex_to_pix(" \\normalsize" + latex_text("\\textit{Rendered with love, using SwapTube}") + "\\\\\\\\" + "\\tiny" + latex_text("Commit Hash: " + swaptube_commit_hash()), sp2);
        saved_pix.overlay(swaptube_pix, pix.h*.03, pix.h*.03, .4);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }
    void mark_data_unchanged() override { }
    void change_data() override {}
    bool check_if_data_changed() const override { return false; }
    void draw() override{pix = saved_pix;}
    void on_end_transition() override{}

    string swaptube_commit_hash() {
        const char* command = "git rev-parse HEAD";
        array<char, 128> buffer;
        string result;

        // Open pipe to file
        unique_ptr<FILE, decltype(&pclose)> pipe(popen(command, "r"), pclose);
        if (!pipe) {
            throw runtime_error("popen() failed!");
        }

        // Read the output of the git command
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }

        // Remove any trailing newline characters
        if (!result.empty() && result.back() == '\n') {
            result.pop_back();
        }

        return result;
    }

private:
    Pixels saved_pix;
};
