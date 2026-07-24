#include "TwoswapScene.h"

#include <array>
#include <random>
#include <cstdio>
#include <stdexcept>

extern "C" void cuda_overlay(
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle);
extern "C" uint32_t* cuda_alloc_pixels_on_device(int size);
extern "C" void cuda_copy_pixels_to_device(uint32_t* h_pixels, int size, uint32_t* d_pixels);

TwoswapScene::TwoswapScene(const vec2& dimensions) : MandelbrotScene(dimensions) {
    manager.set({
        {"swaptube_opacity", "1"},
        {"2swap_effect_completion", "0"},
        {"6884_effect_completion", "0"},
        {"swaptube_effect_completion", "0"},
    });

    manager.begin_timer("init_time");
    manager.set({
        {"max_iterations", "500"},
        {"seed_z_r", "-0.18775000017"},
        {"seed_z_i", "-0.09744791666668"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"seed_c_r", "-0.1985 <init_time> .0003 * +"},
        {"seed_c_i", "-0.6705"},
        {"zoom", "2.7"},
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

    const vec2 whole_shift(0, get_width() * .04);

    const ivec2 wh = get_width_height();

    if (twoswapness > 0.01) { // 2swap logo effect
        if (!twoswap) {
            twoswap_wh = floor(wh * vec2(1, .3));

            Pixels foreground_pix(twoswap_wh);

            ScalingParams sp(wh * vec2(.6, .4));
            Pixels twoswap_pix = latex_to_pix("\\text{2swap}", sp);
            foreground_pix.fill_circle(ivec2(get_width()/3, foreground_pix.wh.y/2), get_width()/23, OPAQUE_WHITE);
            double yval = (foreground_pix.wh.y-twoswap_pix.wh.y)/2+get_width()/96;
            foreground_pix.overwrite(twoswap_pix, ivec2(get_width()/3+get_width()/23+get_width()/96, yval));

            int foreground_size = foreground_pix.wh.x * foreground_pix.wh.y;
            twoswap = cuda_alloc_pixels_on_device(foreground_size);
            cuda_copy_pixels_to_device(foreground_pix.pixels.data(), foreground_size, twoswap);
        }

        cuda_overlay(gpu_pix->get_ptr(), wh,
            twoswap, twoswap_wh,
            (wh-twoswap_wh)/2 - wh*.04 + whole_shift,
            twoswapness * .6, -.2
        );
    }

    if (seefness > 0.01) { // 6884 logo effect
        if (!seef) {
            seef_wh = floor(get_width() * vec2(1, .2));

            Pixels foreground_pix(seef_wh);

            Pixels image;
            png_to_pix(image, "../musicnote");

            Pixels scaled;
            image.scale_to_bounding_box(wh * vec2(1, .135), scaled);

            foreground_pix.overlay_cpu(scaled, vec2(get_width()*.44, (foreground_pix.wh.y)/2 + scaled.wh.y*.05), 1.0f);

            ScalingParams sp(wh * .25);
            Pixels seef_pix = latex_to_pix("\\text{6884}", sp);
            double yval = (foreground_pix.wh.y-seef_pix.wh.y)/2;
            foreground_pix.overwrite(seef_pix, ivec2(get_width()*.4 + scaled.wh.x+get_width()/96, yval));

            int foreground_size = foreground_pix.wh.x * foreground_pix.wh.y;
            seef = cuda_alloc_pixels_on_device(foreground_size);
            cuda_copy_pixels_to_device(foreground_pix.pixels.data(), foreground_size, seef);
        }

        cuda_overlay(gpu_pix->get_ptr(), wh,
            seef, seef_wh,
            (get_width()-seef_wh)/2 - wh*vec2(.029, .285) + whole_shift,
            seefness * .6, -.2
        );
    }

    if (swaptubeness > 0.01) { // SwapTube logo effect
        if (!swaptube) {
            swaptube_wh = wh;

            vec2 size = wh * vec2(1, .14);
            ScalingParams sp2(wh * vec2(.32, .14));
            Pixels swaptube_pix_small_box = latex_to_pix("\\normalsize\\textbf{Made with love, using SwapTube}\\\\\\\\\\ \\text{\\quad Commit Hash: " + swaptube_commit_hash() + "}", sp2);
            Pixels swaptube_pix = Pixels(wh);
            swaptube_pix.overwrite(swaptube_pix_small_box, (size - swaptube_pix_small_box.wh)/2);

            int foreground_size = swaptube_pix.wh.x * swaptube_pix.wh.y;
            swaptube = cuda_alloc_pixels_on_device(foreground_size);
            cuda_copy_pixels_to_device(swaptube_pix.pixels.data(), foreground_size, swaptube);
        }

        cuda_overlay(gpu_pix->get_ptr(), wh,
            swaptube, swaptube_wh,
            (wh-swaptube_wh)/2 + get_width_height()*vec2(.08, .28) + whole_shift,
            swaptubeness * .6, -.2
        );
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
