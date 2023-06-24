#pragma once

#include "scene.h"
using json = nlohmann::json;

class TwoswapScene : public Scene {
public:
    TwoswapScene(const json& config, const json& contents, MovieWriter& writer);
    Pixels query(int& frames_left) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter& writer) override {
        return new TwoswapScene(config, scene, writer);
    }
};

TwoswapScene::TwoswapScene(const json& config, const json& contents, MovieWriter& writer) : Scene(config, contents, writer) {
    double duration_seconds = 0;
    if(contents.find("audio") != contents.end())
        duration_seconds = writer.add_audio_get_length(contents["audio"].get<string>());
    else{
        duration_seconds = contents["duration_seconds"].get<double>();
        writer.add_silence(duration_seconds);
    }
    scene_duration_frames = duration_seconds * framerate;

    Pixels twoswap_pix = eqn_to_pix(latex_text("2swap"), pix.w/160);

    pix.fill(BLACK);
    pix.fill_ellipse(pix.w/3, pix.h/2, pix.w/12, pix.w/12, WHITE);
    pix.copy(twoswap_pix, pix.w/3+pix.w/12+pix.w/32, (pix.h-twoswap_pix.h)/2+pix.w/32, 1);
    frontload_audio(contents, writer);
}

Pixels TwoswapScene::query(int& frames_left) {
    frames_left = scene_duration_frames - time;

    Pixels ret(pix.w, pix.h);
    ret.fill(BLACK);
    ret.copy(pix, 0, 0, fifo_curve(time / static_cast<double>(scene_duration_frames)));
    time++;

    return ret;
}