#pragma once

#include "../../IO/VisualMedia.cpp"
#include "../Scene.cpp"

class TwoswapScene : public Scene {
public:
    TwoswapScene(const double width = 1, const double height = 1) : Scene(width, height) {
        manager.set({"circle_opacity", "1"}, {"swaptube_opacity", ".4"});
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"circle_opacity", "swaptube_opacity"};
    }
    void mark_data_unchanged() override { }
    void change_data() override {}
    bool check_if_data_changed() const override { return false; }
    void draw() override{
        ScalingParams sp(pix.w*.7, pix.h);
        Pixels twoswap_pix = latex_to_pix("\\text{2swap}", sp);
        pix.fill_ellipse(pix.w/4, pix.h/2, pix.w/14, pix.w/14, colorlerp(TRANSPARENT_BLACK, OPAQUE_WHITE, state["circle_opacity"]));
        double yval = (pix.h-twoswap_pix.h)/2+pix.w/48;
        pix.overwrite(twoswap_pix, pix.w/4+pix.w/14+pix.w/48, yval);
        if(state["swaptube_opacity"] > 0.01){
            ScalingParams sp2(pix.w*.4, pix.h*.2);
            Pixels swaptube_pix = latex_to_pix("\\normalsize\\textbf{Animated with love, using SwapTube}\\\\\\\\\\tiny\\text{Commit Hash: " + swaptube_commit_hash() + "}", sp2);
            pix.overlay(swaptube_pix, pix.h*.03, pix.h*.03, state["swaptube_opacity"]);
        }
    }

    string swaptube_commit_hash() {
        const char* command = "git rev-parse HEAD";
        array<char, 128> buffer;
        string result;

        // Open pipe to file
        unique_ptr<FILE, int(*)(FILE*)> pipe(popen(command, "r"), pclose);
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
};
