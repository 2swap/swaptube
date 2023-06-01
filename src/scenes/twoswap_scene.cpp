#pragma once

using json = nlohmann::json;

class TwoswapScene : public Scene {
public:
    TwoswapScene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
};

TwoswapScene::TwoswapScene(const json& config, const json& contents) : Scene(config, contents) {}

Pixels TwoswapScene::query(int& frames_left) {
    frames_left = scene_duration_frames - time;
    time++;

    Pixels twoswap_pix = eqn_to_pix(latex_text("2swap"), 8);

    pix.fill(0);
    pix.fill_ellipse(pix.w/3, pix.h/2, pix.w/12, pix.w/12, 0xffffff);
    pix.copy(twoswap_pix, pix.w/3+pix.w/12-40, (pix.h-twoswap_pix.h)/2+20, 1, 1);

    return pix;
}