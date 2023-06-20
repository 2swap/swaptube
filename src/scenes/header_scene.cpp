#pragma once

#include "scene.h"
using json = nlohmann::json;

class HeaderScene : public Scene {
public:
    HeaderScene(const json& config, const json& contents, MovieWriter& writer);
    Pixels query(int& frames_left) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter& writer) override {
        return new HeaderScene(config, scene, writer);
    }
};

HeaderScene::HeaderScene(const json& config, const json& contents, MovieWriter& writer) : Scene(config, contents) {
    double duration_seconds = 0;
    if(contents.find("audio") != contents.end())
        duration_seconds = writer.add_audio_get_length(contents["audio"].get<string>());
    else{
        duration_seconds = contents["duration_seconds"].get<int>();
        writer.add_silence(duration_seconds);
    }
    scene_duration_frames = duration_seconds * framerate;

    string header = contents["header"].get<string>();
    string subheader = contents["subheader"].get<string>();
    Pixels header_pix = eqn_to_pix(latex_text(header), pix.w / 640 + 1);
    Pixels subheader_pix = eqn_to_pix(latex_text(subheader), pix.w / 640);

    pix.fill(BLACK);
    pix.copy(header_pix, (pix.w - header_pix.w)/2, pix.h/2-100, 1);
    pix.copy(subheader_pix, (pix.w - subheader_pix.w)/2, pix.h/2+50, 1);
}

Pixels HeaderScene::query(int& frames_left) {
    frames_left = scene_duration_frames - time;

    Pixels ret(pix.w, pix.h);
    ret.fill(BLACK);
    ret.copy(pix, 0, 0, fifo_curve(time / static_cast<double>(scene_duration_frames)));
    time++;

    return ret;
}