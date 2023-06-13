#pragma once

using json = nlohmann::json;

class TwoswapScene : public Scene {
public:
    TwoswapScene(const json& config, const json& contents);
    Pixels query(int& frames_left) override;
    Scene* createScene(const json& config, const json& scene) override {
        return new TwoswapScene(config, scene);
    }
};

TwoswapScene::TwoswapScene(const json& config, const json& contents) : Scene(config, contents) {
    Pixels twoswap_pix = eqn_to_pix(latex_text("2swap"), pix.w/160);

    pix.fill(BLACK);
    pix.fill_ellipse(pix.w/3, pix.h/2, pix.w/12, pix.w/12, WHITE);
    pix.copy(twoswap_pix, pix.w/3+pix.w/12+pix.w/32, (pix.h-twoswap_pix.h)/2+pix.w/32, 1);
}

Pixels TwoswapScene::query(int& frames_left) {
    frames_left = scene_duration_frames - time;
    time++;

    return pix;
}