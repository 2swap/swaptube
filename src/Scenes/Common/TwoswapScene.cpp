#pragma once

#include <array>
#include <random>
#include "../../IO/VisualMedia.cpp"
#include "../Math/MandelbrotScene.cpp"

void stripey_effect(Pixels& in, Pixels& out, const float amount) {
    vector<double> stripe_shift_multipliers = {-.5, 1.5, -.5, .5, -1.5, .5};
    // One substripe per scanline for some extra noise
    vector<double> substripe_shift_multipliers(in.h);
    // For approximately one of every 5 lines, put either -1 or 1, otherwise 0
    for(int y = 0; y < in.h; y++) {
        double r = static_cast<double>(rand()) / RAND_MAX;;
        if(r < 0.3)
            substripe_shift_multipliers[y] = (rand() % 9)/4. - 1.0;
        else
            substripe_shift_multipliers[y] = 0.0;
    }
    out = Pixels(in.w, in.h);
    for(int y = 0; y < in.h; y++) {
        for(int x = 0; x < in.w; x++) {
            int stripe_number = y * 20 / in.h;
            double shift_amount = stripe_shift_multipliers[stripe_number % stripe_shift_multipliers.size()] * amount * in.w;
            double subshift_amount = substripe_shift_multipliers[y] * in.w * 0.002;
            int col  = in.get_pixel_carefully(x + shift_amount + subshift_amount, y) & 0xff00ff00;
                col |= in.get_pixel_carefully(x - shift_amount + subshift_amount, y) & 0xffff0000;
                col |= in.get_pixel_carefully(x                + subshift_amount, y) & 0xff0000ff;
            out.set_pixel_carelessly(x, y, col);
        }
    }
}

class TwoswapScene : public MandelbrotScene {
public:
    TwoswapScene(const double width = 1, const double height = 1) : MandelbrotScene(width, height) {
        manager.set({
            {"swaptube_opacity", "1"},
            {"2swap_effect_completion", "0"},
            {"6884_effect_completion", "0"},
        });

        manager.begin_timer("init_time");
        manager.set({
            {"max_iterations", "500"},
            {"seed_z_r", "-0.16775000017"},
            {"seed_z_i", "-0.09744791666668"},
            {"seed_x_r", "2"},
            {"seed_x_i", "0"},
            {"seed_c_r", "-0.1985 <init_time> .0003 * +"},
            {"seed_c_i", "-0.6705"},
            {"zoom", "2.7"},
            {"pixel_param_z", "1"},
            {"pixel_param_x", "0"},
            {"pixel_param_c", "0"},
            {"phase_shift", "{t} 35 +"},
        });
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = MandelbrotScene::populate_state_query();
        state_query_insert_multiple(sq, {"swaptube_opacity", "2swap_effect_completion", "6884_effect_completion"});
        return sq;
    }

    void draw() override{
        MandelbrotScene::draw();

        { // 2swap logo effect
            Pixels foreground_pix(pix.w, pix.h * .3);

            ScalingParams sp(pix.w * .6, pix.h * .4);
            Pixels twoswap_pix = latex_to_pix("\\text{2swap}", sp);
            foreground_pix.fill_ellipse(pix.w/3, foreground_pix.h/2, pix.w/20, pix.w/20, OPAQUE_WHITE);
            double yval = (foreground_pix.h-twoswap_pix.h)/2+pix.w/96;
            foreground_pix.overwrite(twoswap_pix, pix.w/3+pix.w/20+pix.w/96, yval);

            Pixels stripey_pix;
            stripey_effect(foreground_pix, stripey_pix, 1-state["2swap_effect_completion"]);

            Pixels rotated_pix;
            stripey_pix.rotate_arbitrary_angle(-.2, rotated_pix);

            cuda_overlay(pix.pixels.data(), pix.w, pix.h,
                rotated_pix.pixels.data(), rotated_pix.w, rotated_pix.h,
                (pix.w-rotated_pix.w)/2, (pix.h-rotated_pix.h)/2 - pix.h*.04, state["2swap_effect_completion"]);
        }

        { // 6884 logo effect
            Pixels foreground_pix(pix.w, pix.h * .2);

            Pixels image;
            png_to_pix(image, "musicnote");

            Pixels scaled;
            image.scale_to_bounding_box(pix.w, pix.h * .15, scaled);

            ScalingParams sp(pix.w * .25, pix.h * .25);
            Pixels twoswap_pix = latex_to_pix("\\text{6884}", sp);
            cuda_overlay(foreground_pix.pixels.data(), foreground_pix.w, foreground_pix.h,
                scaled.pixels.data(), scaled.w, scaled.h,
                pix.w*.4, (foreground_pix.h-scaled.h)/2, 1.0f);
            double yval = (foreground_pix.h-twoswap_pix.h)/2;
            foreground_pix.overwrite(twoswap_pix, pix.w*.4 + scaled.w+pix.w/96, yval);

            Pixels stripey_pix;
            stripey_effect(foreground_pix, stripey_pix, 1-state["6884_effect_completion"]);

            Pixels rotated_pix;
            stripey_pix.rotate_arbitrary_angle(-.2, rotated_pix);

            cuda_overlay(pix.pixels.data(), pix.w, pix.h,
                rotated_pix.pixels.data(), rotated_pix.w, rotated_pix.h,
                (pix.w-rotated_pix.w)/2, (pix.h-rotated_pix.h)/2 + pix.h*.14, state["6884_effect_completion"]);
        }

        if(state["swaptube_opacity"] > 0.01){
            ScalingParams sp2(pix.w*.23, pix.h*.2);
            Pixels swaptube_pix = latex_to_pix("\\normalsize\\textbf{Made with love, using SwapTube}\\\\\\\\\\ \\text{Commit Hash: " + swaptube_commit_hash() + "}", sp2);
            pix.overlay(swaptube_pix, pix.w*.98 - swaptube_pix.w, pix.h*.03, state["swaptube_opacity"]);
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

        // Truncate to first 7 characters for brevity
        if (result.length() > 7) {
            result = result.substr(0, 7);
        }

        return result;
    }
};
