#include "TwoswapScene.h"

#include <array>
#include <random>
#include <cstdio>
#include <stdexcept>

void stripey_effect(Pixels& in, Pixels& out, const float amount) {
    vector<double> stripe_shift_multipliers = {-.3, 1.5, -.7, 1, -1, .3, -1.5, .7, -1, 1};
    // One substripe per scanline for some extra noise
    vector<double> substripe_shift_multipliers(in.wh.y);
    for(int y = 0; y < in.wh.y; y++) {
        substripe_shift_multipliers[y] = static_cast<double>(rand()) / RAND_MAX;
    }
    out = Pixels(in.wh.x, in.wh.y);
    for(int y = 0; y < in.wh.y; y++) {
        for(int x = 0; x < in.wh.x; x++) {
            int stripe_number = y * 50 / in.wh.y;
            double shift_amount = stripe_shift_multipliers[stripe_number % stripe_shift_multipliers.size()] * amount * in.wh.x;
            double tangent = tan(substripe_shift_multipliers[y] * 3.14159);
            tangent = 4 * sqrt(abs(tangent));
            tangent = clamp(-tangent, -20.0, 20.0);
            double subshift_amount = in.wh.x * 0.0002 * tangent;
            int col  = in.get_pixel_carefully(x + shift_amount + subshift_amount, y) & 0xff00ff7f;
                col |= in.get_pixel_carefully(x - shift_amount + subshift_amount, y) & 0xffff0080;
            out.set_pixel_carelessly(x, y, col);
        }
    }
}

TwoswapScene::TwoswapScene(const vec2& dimensions) : MandelbrotScene(dimensions) {
    manager.set({
        //{"swaptube_opacity", "1"},
        {"2swap_effect_completion", "0"},
        {"6884_effect_completion", "0"},
        {"swaptube_effect_completion", "0"},
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
        {"zoom", "2.8"},
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
        {"phase_shift", "<init_time> 37 +"},
    });
}

const StateQuery TwoswapScene::populate_state_query() const {
    StateQuery sq = MandelbrotScene::populate_state_query();
    state_query_insert_multiple(sq, {/*"swaptube_opacity", */"2swap_effect_completion", "6884_effect_completion", "swaptube_effect_completion"});
    return sq;
}

void TwoswapScene::draw(){
    MandelbrotScene::draw();

    double twoswapness = state["2swap_effect_completion"];
    double seefness = state["6884_effect_completion"];
    double swaptubeness = state["swaptube_effect_completion"];

    double whole_x_shift = 0;
    double whole_y_shift = pix.wh.x * .04;

    if (twoswapness > 0.01) { // 2swap logo effect
        Pixels foreground_pix(pix.wh.x, pix.wh.y * .3);

        ScalingParams sp(pix.wh.x * .6, pix.wh.y * .4);
        Pixels twoswap_pix = latex_to_pix("\\text{2swap}", sp);
        foreground_pix.fill_ellipse(pix.wh.x/3, foreground_pix.wh.y/2, pix.wh.x/20, pix.wh.y/20, OPAQUE_WHITE);
        double yval = (foreground_pix.wh.y-twoswap_pix.wh.y)/2+pix.wh.x/96;
        foreground_pix.overwrite(twoswap_pix, pix.wh.x/3+pix.wh.x/20+pix.wh.x/96, yval);

        Pixels stripey_pix;
        stripey_effect(foreground_pix, stripey_pix, 1-twoswapness);

        pix.overlay_gpu_with_rotation(stripey_pix,
            (pix.wh.x-stripey_pix.wh.x)/2 - pix.wh.x*.04 + whole_x_shift,
            (pix.wh.y-stripey_pix.wh.y)/2 - pix.wh.y*.04 + whole_y_shift,
            twoswapness * .6, -.2
        );
    }

    if (seefness > 0.01) { // 6884 logo effect
        Pixels foreground_pix(pix.wh.x, pix.wh.y * .2);

        Pixels image;
        png_to_pix(image, "musicnote");

        Pixels scaled;
        image.scale_to_bounding_box(pix.wh.x, pix.wh.y * .14, scaled);

        ScalingParams sp(pix.wh.x * .25, pix.wh.y * .25);
        Pixels seef_pix = latex_to_pix("\\text{6884}", sp);
        foreground_pix.overlay_gpu(scaled, pix.wh.x*.4, (foreground_pix.wh.y-scaled.wh.y)/2 + scaled.wh.y*.1, 1.0f);
        double yval = (foreground_pix.wh.y-seef_pix.wh.y)/2;
        foreground_pix.overwrite(seef_pix, pix.wh.x*.4 + scaled.wh.x+pix.wh.x/96, yval);

        Pixels stripey_pix;
        stripey_effect(foreground_pix, stripey_pix, 1-seefness);

        pix.overlay_gpu_with_rotation(stripey_pix,
            (pix.wh.x-stripey_pix.wh.x)/2 - pix.wh.x*.029 + whole_x_shift,
            (pix.wh.y-stripey_pix.wh.y)/2 + pix.wh.y*.110 + whole_y_shift,
            seefness * .6, -.2
        );
    }

    if (swaptubeness > 0.01) { // SwapTube logo effect
        double height = pix.wh.y * .14;
        ScalingParams sp2(pix.wh.x * .32, height);
        Pixels swaptube_pix_small_box = latex_to_pix("\\normalsize\\textbf{Made with love, using SwapTube}\\\\\\\\\\ \\text{\\quad Commit Hash: " + swaptube_commit_hash() + "}", sp2);
        Pixels swaptube_pix = Pixels(pix.wh.x, height);
        swaptube_pix.overwrite(swaptube_pix_small_box, (pix.wh.x - swaptube_pix_small_box.wh.x)/2, (height - swaptube_pix_small_box.wh.y)/2);

        Pixels stripey_pix;
        stripey_effect(swaptube_pix, stripey_pix, 1-swaptubeness);

        pix.overlay_gpu_with_rotation(stripey_pix,
            (pix.wh.x-stripey_pix.wh.x)/2 + pix.wh.x*.03 + whole_x_shift,
            (pix.wh.y-stripey_pix.wh.y)/2 - pix.wh.y*.15 + whole_y_shift,
            swaptubeness * .6, -.2
        );
    }

    if(state["swaptube_opacity"] > 0.01){
        ScalingParams sp2(pix.wh.x*.23, pix.wh.y*.14);
        Pixels swaptube_pix = latex_to_pix("\\normalsize\\textbf{Made with love, using SwapTube}\\\\\\\\\\ \\text{Commit Hash: " + swaptube_commit_hash() + "}", sp2);
        pix.overlay_gpu(swaptube_pix, pix.wh.x*.98 - swaptube_pix.wh.x, pix.wh.y*.03, state["swaptube_opacity"]);
    }
}

std::string TwoswapScene::swaptube_commit_hash() {
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

    // Truncate for brevity
    int target_length = 20;
    if (result.length() > target_length) {
        result = result.substr(0, target_length);
    }

    return result;
}
