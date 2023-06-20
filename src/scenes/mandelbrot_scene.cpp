#pragma once

#include "scene.h"
#include "Mandelbrot/mandelbrot_renderer.cpp"
using json = nlohmann::json;

class MandelbrotScene : public Scene {
public:
    MandelbrotScene(const json& config, const json& contents, MovieWriter& writer);
    Pixels query(int& frames_left) override;
    Scene* createScene(const json& config, const json& scene, MovieWriter& writer) override {
        return new MandelbrotScene(config, scene, writer);
    }

private:
    MandelbrotRenderer mr;
};

MandelbrotScene::MandelbrotScene(const json& config, const json& contents, MovieWriter& writer) : Scene(config, contents) {
    double duration_seconds = 0;
    if(contents.find("audio") != contents.end())
        duration_seconds = writer.add_audio_get_length(contents["audio"].get<string>());
    else{
        duration_seconds = contents["duration_seconds"].get<int>();
        writer.add_silence(duration_seconds);
    }
    scene_duration_frames = duration_seconds * framerate;

    Complex center(contents["center"]["real"].get<double>(), contents["center"]["imag"].get<double>());
    Complex current_zoom(contents["current_zoom"]["real"].get<double>(), contents["current_zoom"]["imag"].get<double>());
    Complex zoom_multiplier(contents["zoom_multiplier"]["real"].get<double>(), contents["zoom_multiplier"]["imag"].get<double>());
    Complex z(contents["z"]["real"].get<double>(), contents["z"]["imag"].get<double>());
    Complex x(contents["x"]["real"].get<double>(), contents["x"]["imag"].get<double>());
    Complex c(contents["c"]["real"].get<double>(), contents["c"]["imag"].get<double>());

    string paramValue = contents["WhichParameterization"].get<string>();
    MandelbrotRenderer::WhichParameterization wp = MandelbrotRenderer::C;
    if (paramValue == "Z") {
        wp = MandelbrotRenderer::Z;
    } else if (paramValue == "X") {
        wp = MandelbrotRenderer::X;
    }

    mr = MandelbrotRenderer(z, x, c, wp, center, current_zoom, zoom_multiplier);
}

Pixels MandelbrotScene::query(int& frames_left) {
    frames_left = scene_duration_frames - time;
    time++;

    double duration_frames = contents["duration_seconds"].get<int>() * framerate;
    mr.edge_detect_render(pix);
    return pix;
}