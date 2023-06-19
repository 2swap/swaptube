#pragma once

using json = nlohmann::json;

class TwoswapScene : public Scene {
public:
    TwoswapScene(const json& config, const json& contents, MovieWriter& writer);
    Pixels query(int& frames_left) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter& writer) override {
        return new TwoswapScene(config, scene, writer);
    }
};

TwoswapScene::TwoswapScene(const json& config, const json& contents, MovieWriter& writer) : Scene(config, contents) {
    if(contents.find("audio") != contents.end())
        scene_duration_frames = writer.add_audio_get_length(contents["audio"].get<string>());
    else{
        scene_duration_frames = contents["duration_seconds"].get<int>();
        writer.add_silence(scene_duration_frames);
    }
    scene_duration_frames *= framerate;

    Pixels twoswap_pix = eqn_to_pix(latex_text("2swap"), pix.w/160);

    pix.fill(BLACK);
    pix.fill_ellipse(pix.w/3, pix.h/2, pix.w/12, pix.w/12, WHITE);
    pix.copy(twoswap_pix, pix.w/3+pix.w/12+pix.w/32, (pix.h-twoswap_pix.h)/2+pix.w/32, 1);
}

Pixels TwoswapScene::query(int& frames_left) {
    frames_left = scene_duration_frames - time;

    Pixels ret(pix.w, pix.h);
    ret.fill(BLACK);
    ret.copy(pix, 0, 0, fifo_curve(time / static_cast<double>(scene_duration_frames), time/static_cast<double>(framerate), frames_left/static_cast<double>(framerate)));
    time++;

    return ret;
}