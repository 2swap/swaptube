#include "TwoswapScene.h"
#include "../Media/LatexScene.h"
#include <cstdio>

// Get the current git commit hash
static string get_git_hash() {
    string hash;
    FILE* pipe = popen("git rev-parse HEAD", "r");
    if (!pipe) return "unknown";
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        hash += buffer;
    }
    pclose(pipe);
    // Remove any trailing newline characters
    hash.erase(hash.find_last_not_of("\n\r") + 1);
    if (hash.empty()) return "unknown";

    // Truncate for brevity
    return hash.substr(0, 20);
}

TwoswapScene::TwoswapScene(const vec2& delta) : CompositeScene() {
    shared_ptr<LatexScene> logo = make_shared<LatexScene>("\\text{2swap}", vec2(.7, .7));
    add_scene(logo, "logo", vec2(.6, .55) + delta);

    shared_ptr<LatexScene> hash = make_shared<LatexScene>("\\text{Animated with love, using Swaptube!} \\\\\\\\ \\text{Commit Hash: " + get_git_hash() + "}", vec2(.4, .1));
    add_scene(hash, "hash", vec2(.18, .05));

    shared_ptr<LatexScene> music = make_shared<LatexScene>("\\text{Music by 6884}", vec2(.4, .4));
    add_scene(music, "music", vec2(.56, .7) + delta);
    manager.set("music_opacity", "0");
    manager.set("music.opacity", "<music_opacity>");
}
