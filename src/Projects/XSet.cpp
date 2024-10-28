using namespace std;
#include <string>
const string project_name = "XSet";
const int width_base = 640;
const int height_base = 360;
const float mult = 2;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/StateSliderScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"
#include "../Scenes/Common/ExposedPixelsScene.cpp"

void intro() {
    CompositeScene cs;
    MandelbrotScene ms;
    cs.add_scene(&ms, "ms");
    unordered_map<string,string> init = {
        {"zoom_r", "2 <zoom_exp> ^"},
        {"zoom_exp", "0"},
        {"zoom_i", "0"},
        {"max_iterations", "100 2 <zoom_exp> -3 / ^ *"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"gradation", "1"},
        {"side_panel", "0"},
        {"point_path_r", "0"},
        {"point_path_i", "0"},
        {"point_path_length", "0"},
    };
    ms.state_manager.set(unordered_map<string,string>{
        {"zoom_r", "[zoom_r]"},
        {"zoom_i", "[zoom_i]"},
        {"max_iterations", "[max_iterations]"},
        {"seed_z_r", "[seed_z_r]"},
        {"seed_z_i", "[seed_z_i]"},
        {"seed_x_r", "[seed_x_r]"},
        {"seed_x_i", "[seed_x_i]"},
        {"seed_c_r", "[seed_c_r]"},
        {"seed_c_i", "[seed_c_i]"},
        {"pixel_param_z", "[pixel_param_z]"},
        {"pixel_param_x", "[pixel_param_x]"},
        {"pixel_param_c", "[pixel_param_c]"},
        {"gradation", "[gradation]"},
        {"side_panel", "[side_panel]"},
        {"point_path_length", "[point_path_length]"},
        {"point_path_r", "[point_path_r]"},
        {"point_path_i", "[point_path_i]"},
    });
    cs.state_manager.set(init);

    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-0.743643887037151"},
        {"seed_c_i", "0.131825904205330"},
    });
    cs.inject_audio_and_render(AudioSegment("This is the mandelbrot set."));
    // Zoom in on an interesting spot with a minibrot
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "-9"},
    });
    cs.inject_audio_and_render(AudioSegment("Known for its beauty at all resolutions and scales,"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-0.743643887037151"},
        {"seed_c_i", "0.15"},
    });
    cs.inject_audio_and_render(AudioSegment("and its stunning self-similarity,"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("it is the cornerstone example of a fractal."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("What's more, if we just change our perspective, or, rather, spin our viewpanel through the fourth dimension,"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "-1"},
    });
    cs.inject_audio_and_render(AudioSegment("we get a similar fractal called a Julia set."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-.5"},
        {"seed_c_i", ".6"},
    });
    cs.inject_audio_and_render(AudioSegment("The Mandelbrot sets and the Julia sets are in planes orthogonal to each other in a certain 4d-space."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-.47"},
        {"seed_c_i", "-.65"},
        {"zoom_exp", "1"},
        {"gradation", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("This much is well-documented. And don't worry, I'll explain how all of that works in a sec."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "0"},
        {"pixel_param_x", "1"},
        {"pixel_param_c", "0"},
        {"seed_x_r", "2"},
    });
    cs.inject_audio_and_render(AudioSegment("But before that, I wanna twist our perspective once more, into yet another orthogonal 2d-plane."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-.2"},
        {"seed_c_i", "-.4"},
        {"seed_z_r", ".3"},
        {"seed_z_i", ".2"},
    });
    cs.inject_audio_and_render(AudioSegment("This is another natural extension of the Mandelbrot Set which I found."));
    cs.state_manager.microblock_transition(init);
    cs.inject_audio_and_render(AudioSegment("But, let's take it from the start."));
    cs.inject_audio_and_render(AudioSegment("The original mandelbrot set."));
    cs.state_manager.set(unordered_map<string,string>{
        {"ms.opacity", "1"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"ms.opacity", ".2"},
    });
    PngScene comp_axes("complex_axes");
    cs.add_scene(&comp_axes, "comp_axes");
    cs.state_manager.set(unordered_map<string,string>{
        {"comp_axes.opacity", "0"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"comp_axes.opacity", "1"},
    });
    cs.inject_audio_and_render(AudioSegment("It lives in the complex plane."));
    LatexScene eqn("z_{" + latex_text("next") + "} = z^2 + c", 0.7, 1, .5);
    cs.add_scene_fade_in(&eqn, "eqn");
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"comp_axes.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("The equation which generates it is surprisingly simple."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-0.082953220453125015264"},
        {"seed_c_i", "-0.966181199195312500001"},
        {"zoom_exp", "<t> 50 - -6 /"},
    });
    eqn.begin_latex_transition(latex_color(0xffff0000, "z_{" + latex_text("next") + "}") + " = " + latex_color(0xffff0000, "z") + "^2 + c");
    LatexScene vals("\\begin{tabular}{p{4cm}|p{4cm}} & \\\\\\\\ \\hline z_0 & \\\\\\\\ z_1 & \\\\\\\\ z_2 & \\\\\\\\ z_3 & \\\\\\\\ z_4 & \\\\\\\\ z_5 & \\\\\\\\ z_6 & \\end{tabular}", 0.8, 0.5, 0.4);
    cs.add_scene_fade_in(&vals, "vals", 0.25, 0.2);
    cs.inject_audio_and_render(AudioSegment("It tells us how to update this variable Z."));
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & \\\\\\\\ z_1 & \\\\\\\\ z_2 & \\\\\\\\ z_3 & \\\\\\\\ z_4 & \\\\\\\\ z_5 & \\\\\\\\ z_6 & \\end{tabular}");
    cs.inject_audio(AudioSegment("Let's let C be 1 just for now. We'll try other values in a sec."), 4);
    eqn.begin_latex_transition(latex_color(0xffffffff, "z") + "^2 + " + latex_color(0xff008888, "c"));
    cs.render();
    eqn.begin_latex_transition("z^2 + 1");
    cs.render();
    cs.render();
    cs.render();
    eqn.begin_latex_transition("0^2 + 1");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & 0 \\\\\\\\ z_1 & \\\\\\\\ z_2 & \\\\\\\\ z_3 & \\\\\\\\ z_4 & \\\\\\\\ z_5 & \\\\\\\\ z_6 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("And let's let Z start at 0."));
    eqn.begin_latex_transition("0 + 1");
    cs.inject_audio_and_render(AudioSegment("We do some arithmetic,"));
    eqn.begin_latex_transition("1");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & 0 \\\\\\\\ z_1 & 1\\\\\\\\ z_2 & \\\\\\\\ z_3 & \\\\\\\\ z_4 & \\\\\\\\ z_5 & \\\\\\\\ z_6 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("and we find the updated value of Z."));
    cs.inject_audio_and_render(AudioSegment("We then take that updated value,"));
    eqn.begin_latex_transition("1^2 + 1");
    cs.inject_audio_and_render(AudioSegment("plug it into the equation,"));
    eqn.begin_latex_transition("2");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & 0 \\\\\\\\ z_1 & 1 \\\\\\\\ z_2 & 2 \\\\\\\\ z_3 & \\\\\\\\ z_4 & \\\\\\\\ z_5 & \\\\\\\\ z_6 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("and do the math again."));
    eqn.begin_latex_transition("2^2 + 1");
    cs.inject_audio(AudioSegment("We want to keep repeating this over and over."), 2);
    cs.render();
    eqn.begin_latex_transition("4 + 1");
    cs.render();
    eqn.begin_latex_transition("5");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & 0 \\\\\\\\ z_1 & 1\\\\\\\\ z_2 & 2 \\\\\\\\ z_3 & 5 \\\\\\\\ z_4 & \\\\\\\\ z_5 & \\\\\\\\ z_6 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("Just square, and add one."));
    eqn.begin_latex_transition("5^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("25 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("26");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & 0 \\\\\\\\ z_1 & 1\\\\\\\\ z_2 & 2 \\\\\\\\ z_3 & 5 \\\\\\\\ z_4 & 26 \\\\\\\\ z_5 & \\\\\\\\ z_6 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("Square, and add one."));
    eqn.begin_latex_transition("26^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("676 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & 0 \\\\\\\\ z_1 & 1\\\\\\\\ z_2 & 2 \\\\\\\\ z_3 & 5 \\\\\\\\ z_4 & 26 \\\\\\\\ z_5 & 677 \\\\\\\\ z_6 & \\end{tabular}");
    eqn.begin_latex_transition("677");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("677^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("458229 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} & \\textbf{c=1} \\\\\\\\ \\hline z_0 & 0 \\\\\\\\ z_1 & 1\\\\\\\\ z_2 & 2 \\\\\\\\ z_3 & 5 \\\\\\\\ z_4 & 26 \\\\\\\\ z_5 & 677 \\\\\\\\ z_6 & 458330 \\end{tabular}");
    eqn.begin_latex_transition("458330");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("458330^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("In our case, Z is blowing up towards infinity."));
    eqn.begin_latex_transition("z^2 + c");
    cs.inject_audio_and_render(AudioSegment("Let's try that again."));
    eqn.begin_latex_transition("z^2 - 1");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} \\\\\\\\ \\hline z_0 & 0 & \\\\\\\\ z_1 & 1 & \\\\\\\\ z_2 & 2 & \\\\\\\\ z_3 & 5 & \\\\\\\\ z_4 & 26 & \\\\\\\\ z_5 & 677 & \\\\\\\\ z_6 & 458330 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("This time, let's let C be negative 1 instead of positive 1."));
    eqn.begin_latex_transition("0^2 - 1");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} \\\\\\\\ \\hline z_0 & 0 & 0 \\\\\\\\ z_1 & 1 & \\\\\\\\ z_2 & 2 & \\\\\\\\ z_3 & 5 & \\\\\\\\ z_4 & 26 & \\\\\\\\ z_5 & 677 & \\\\\\\\ z_6 & 458330 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("Plug in 0 to start off Z,"));
    eqn.begin_latex_transition("0 - 1");
    cs.inject_audio_and_render(AudioSegment("iterate,"));
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} \\\\\\\\ \\hline z_0 & 0 & 0 \\\\\\\\ z_1 & 1 & -1 \\\\\\\\ z_2 & 2 & \\\\\\\\ z_3 & 5 & \\\\\\\\ z_4 & 26 & \\\\\\\\ z_5 & 677 & \\\\\\\\ z_6 & 458330 & \\end{tabular}");
    eqn.begin_latex_transition("-1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("(-1)^2 - 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("1 - 1");
    cs.inject_audio_and_render(AudioSegment(1));
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} \\\\\\\\ \\hline z_0 & 0 & 0 \\\\\\\\ z_1 & 1 & -1 \\\\\\\\ z_2 & 2 & 0 \\\\\\\\ z_3 & 5 & \\\\\\\\ z_4 & 26 & \\\\\\\\ z_5 & 677 & \\\\\\\\ z_6 & 458330 & \\end{tabular}");
    eqn.begin_latex_transition("0");
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio(AudioSegment("and this time, we fall into a cyclic trap."), 4);
    eqn.begin_latex_transition("0^2 - 1");
    cs.render();
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} \\\\\\\\ \\hline z_0 & 0 & 0 \\\\\\\\ z_1 & 1 & -1 \\\\\\\\ z_2 & 2 & 0 \\\\\\\\ z_3 & 5 & -1 \\\\\\\\ z_4 & 26 & \\\\\\\\ z_5 & 677 & \\\\\\\\ z_6 & 458330 & \\end{tabular}");
    eqn.begin_latex_transition("-1");
    cs.render();
    eqn.begin_latex_transition("(-1)^2 - 1");
    cs.render();
    eqn.begin_latex_transition("0");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} \\\\\\\\ \\hline z_0 & 0 & 0 \\\\\\\\ z_1 & 1 & -1 \\\\\\\\ z_2 & 2 & 0 \\\\\\\\ z_3 & 5 & -1 \\\\\\\\ z_4 & 26 & 0 \\\\\\\\ z_5 & 677 & -1 \\\\\\\\ z_6 & 458330 & 0 \\end{tabular}");
    cs.render();
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("z^2 + c");
    cs.inject_audio_and_render(AudioSegment("Remember, we're working with Complex numbers here, not just real numbers."));
    eqn.begin_latex_transition("z^2 + i");
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} & \\textbf{c=i} \\\\\\\\ \\hline z_0 & 0 & 0 & 0 & \\\\\\\\ z_1 & 1 & -1 & \\\\\\\\ z_2 & 2 & 0 & \\\\\\\\ z_3 & 5 & -1 & \\\\\\\\ z_4 & 26 & 0 & \\\\\\\\ z_5 & 677 & -1 & \\\\\\\\ z_6 & 458330 & 0 & \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("C can be a number like i."));
    cs.inject_audio(AudioSegment("Running our simulation again, in this case it ends up stuck as well."), 8);
    eqn.begin_latex_transition("0^2 + i");
    cs.render();
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} & \\textbf{c=i} \\\\\\\\ \\hline z_0 & 0 & 0 & 0 & \\\\\\\\ z_1 & 1 & -1 & i & \\\\\\\\ z_2 & 2 & 0 & \\\\\\\\ z_3 & 5 & -1 & \\\\\\\\ z_4 & 26 & 0 & \\\\\\\\ z_5 & 677 & -1 & \\\\\\\\ z_6 & 458330 & 0 & \\end{tabular}");
    eqn.begin_latex_transition("i");
    cs.render();
    eqn.begin_latex_transition("i^2 + i");
    cs.render();
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} & \\textbf{c=i} \\\\\\\\ \\hline z_0 & 0 & 0 & 0 \\\\\\\\ z_1 & 1 & -1 & i \\\\\\\\ z_2 & 2 & 0 & -1+i \\\\\\\\ z_3 & 5 & -1 & \\\\\\\\ z_4 & 26 & 0 & \\\\\\\\ z_5 & 677 & -1 & \\\\\\\\ z_6 & 458330 & 0 & \\end{tabular}");
    eqn.begin_latex_transition("-1+i");
    cs.render();
    eqn.begin_latex_transition("(-1+i)^2 + i");
    cs.render();
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} & \\textbf{c=i} \\\\\\\\ \\hline z_0 & 0 & 0 & 0 \\\\\\\\ z_1 & 1 & -1 & i \\\\\\\\ z_2 & 2 & 0 & -1+i \\\\\\\\ z_3 & 5 & -1 & -i \\\\\\\\ z_4 & 26 & 0 & \\\\\\\\ z_5 & 677 & -1 & \\\\\\\\ z_6 & 458330 & 0 & \\end{tabular}");
    eqn.begin_latex_transition("-i");
    cs.render();
    eqn.begin_latex_transition("(-i)^2 + i");
    cs.render();
    vals.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}|p{4cm}|p{4cm}} & \\textbf{c=1} & \\textbf{c=-1} & \\textbf{c=i} \\\\\\\\ \\hline z_0 & 0 & 0 & 0 \\\\\\\\ z_1 & 1 & -1 & i \\\\\\\\ z_2 & 2 & 0 & -1+i \\\\\\\\ z_3 & 5 & -1 & -i \\\\\\\\ z_4 & 26 & 0 & -1+i \\\\\\\\ z_5 & 677 & -1 & -i \\\\\\\\ z_6 & 458330 & 0 & -1+i \\end{tabular}");
    eqn.begin_latex_transition("-1+i");
    cs.render();
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"eqn.opacity", "0"},
        {"vals.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("The differences in behavior here are the key."));
    LatexScene table("\\begin{tabular}{p{4cm}|p{4cm}} \\textbf{z stays bounded} & \\textbf{z explodes} \\end{tabular}", 0.7, 1, 1);
    cs.add_scene_fade_in(&table, "table");
    cs.inject_audio(AudioSegment("Depending on what value of C we choose, sometimes Z will stay close to zero,"), 2);
    table.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} \\textbf{z stays bounded} & \\textbf{z explodes} \\\\\\\\ \\hline -1, i & \\\\\\\\ \\end{tabular}");
    cs.render();
    cs.render();
    table.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} \\textbf{z stays bounded} & \\textbf{z explodes} \\\\\\\\ \\hline -1, i & 1 \\\\\\\\ \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("and sometimes it won't."));
    cs.fade_out_all_scenes();
    cs.inject_audio_and_render(AudioSegment("That is the difference between a point inside the mandelbrot set, and outside it."));
    cs.remove_all_scenes();
    ExposedPixelsScene eps;
    int gray = 0x88444444;
    eps.exposed_pixels.fill(gray);
    cs.add_scene(&comp_axes, "comp_axes");
    cs.add_scene(&eps, "eps");
    cs.state_manager.set(init);
    cs.state_manager.set(unordered_map<string,string>{
        {"eps.opacity", "0"},
        {"comp_axes.opacity", "0"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"eps.opacity", "1"},
        {"comp_axes.opacity", "1"},
    });
    cs.inject_audio_and_render(AudioSegment("If we make a plot of the complex plane,"));
    cs.inject_audio_and_render(AudioSegment("and for each value of C, we run this little simulation,"));
    double r = VIDEO_WIDTH/500.;
    eps.exposed_pixels.fill_circle(VIDEO_WIDTH/2.+VIDEO_HEIGHT/4., VIDEO_HEIGHT/2., 3*r, OPAQUE_BLACK);
    cs.inject_audio_and_render(AudioSegment("we'll paint points black if they blow up,"));
    eps.exposed_pixels.fill_circle(VIDEO_WIDTH/2., VIDEO_HEIGHT/4., 3*r, OPAQUE_WHITE);
    eps.exposed_pixels.fill_circle(VIDEO_WIDTH/2.-VIDEO_HEIGHT/4., VIDEO_HEIGHT/2., 3*r, OPAQUE_WHITE);
    cs.inject_audio_and_render(AudioSegment("and white if they don't."));
    int num_microblocks = 1000;
    cs.add_scene(&ms, "ms");
    cs.state_manager.set(unordered_map<string,string>{
        {"ms.opacity", "0"},
    });
    Pixels* queried = nullptr;
    ms.query(queried);
    cs.inject_audio(AudioSegment("Do that for all the points..."), num_microblocks);
    for(int i = 0; i < num_microblocks; i++) {
        for(int j = 0; j < 50; j++){
            int point = rand()%(VIDEO_WIDTH*VIDEO_HEIGHT);
            while(eps.exposed_pixels.get_pixel(point%VIDEO_WIDTH, point/VIDEO_WIDTH) != gray){
                point = (point+7841)%(VIDEO_WIDTH*VIDEO_HEIGHT); // 7841 is an arbitrary prime
            }
            int x = point%VIDEO_WIDTH;
            int y = point/VIDEO_WIDTH;
            int col = queried->get_pixel(x, y);
            eps.exposed_pixels.fill_circle(x, y, r, col==OPAQUE_WHITE?OPAQUE_WHITE:OPAQUE_BLACK);
        }
        cs.render();
    }

    cs.inject_audio_and_render(AudioSegment("That gives us the characteristic shape of the Mandelbrot set."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"ms.opacity", "1"},
    });
    cs.inject_audio_and_render(AudioSegment("We can additionally add color to show how long it takes for the number to blow up."));
    cs.remove_scene(&eps);
    cs.remove_scene(&comp_axes);
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = z^2 + c");
    cs.inject_audio_and_render(AudioSegment("And that's how you get these pretty pictures."));
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = z^2 + c");
    cs.add_scene_fade_in(&eqn, "eqn");
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"ms.opacity", "0.2"},
    });
    cs.inject_audio_and_render(AudioSegment("But let's look back at our equation."));
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = ?^2 + " + latex_color(0xff008888, "?"));
    cs.inject_audio_and_render(AudioSegment("We not only had the free choice of what C can be, which is what we're plotting on screen,"));
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = "+latex_color(0xffff8888, "?")+"^2 + ?");
    cs.inject_audio_and_render(AudioSegment("but also the option to choose a starting value of Z."));
    cs.inject_audio_and_render(AudioSegment("Before, we always started Z at 0..."));
    StateSliderScene sszr("[seed_z_r]", "z_r", -1, 1, .3, .07);
    StateSliderScene sszi("[seed_z_i]", "z_i", -1, 1, .3, .07);
    cs.add_scene(&sszr, "sszr", 0.1666, 0.85); 
    cs.add_scene(&sszi, "sszi", 0.1666, 0.95); 
    cs.state_manager.set(unordered_map<string,string>{
        {"sszr.opacity", "0"},
        {"sszi.opacity", "<sszr.opacity>"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-.5"},
        {"ms.opacity", "1"},
        {"eqn.opacity", "0"},
        {"sszr.opacity", "0.5 <pixel_param_z> <pixel_param_z> 0.3 * * -"},
    });
    cs.inject_audio_and_render(AudioSegment("But let's start at 1 and see what happens."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "-1"},
        {"seed_z_r", "1"},
    });
    cs.inject_audio_and_render(AudioSegment("Woah! That's a totally different fractal."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_z_r", "<t> sin 2 /"},
        {"seed_z_i", "<t> cos 2 /"},
    });
    cs.inject_audio_and_render(AudioSegment("We can play around with different values of Z all day, just like with C."));
    cs.inject_audio_and_render(AudioSegment("Which makes you think..."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "0"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("we started by fixing Z at zero,"));
    cs.inject_audio_and_render(AudioSegment("and making C vary as a function of the pixel."));
    PngScene parameterized("parameterized", 0.5, 0.2);
    cs.add_scene(&parameterized, "parameterized", 0.1666, 1.2); 
    StateSliderScene sscr("[seed_c_r]", "c_r", -1, 1, .3, .07);
    StateSliderScene ssci("[seed_c_i]", "c_i", -1, 1, .3, .07);
    cs.add_scene(&sscr, "sscr", 0.8333, 0.85); 
    cs.add_scene(&ssci, "ssci", 0.8333, 0.95); 
    cs.send_to_front("parameterized");
    cs.state_manager.set(unordered_map<string,string>{
        {"sscr.opacity", "0"},
        {"ssci.opacity", "<sscr.opacity>"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"sscr.opacity", "0.5 <pixel_param_c> <pixel_param_c> 0.3 * * -"},
        {"parameterized.y", ".9"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("What if we fix C, and make Z start where the pixel is?"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "<t> sin .75 *"},
        {"seed_c_i", "<t> cos .75 *"},
    });
    cs.inject_audio_and_render(AudioSegment("We get a Julia set!"));
    cs.inject_audio_and_render(AudioSegment("So, we have 2 variables which we use to seed the iterated procedure."));
    cs.inject_audio_and_render(AudioSegment("We get to choose C, and what Z should start at."));
    cs.inject_audio_and_render(AudioSegment("Since each of these has a real and imaginary component, that's 4 degrees of choice- a 4D shape."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-1"},
        {"seed_c_i", "-.1"},
        {"parameterized.x", ".8333 <pixel_param_c> * .1666 <pixel_param_z> * .5 <pixel_param_x> * + +"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
    });
    cs.inject_audio_and_render(AudioSegment("Parameterizing the pixels of the screen as C yields a Mandelbrot set,"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("whereas parameterizing Z yields a Julia set."));
    cs.inject_audio_and_render(AudioSegment("Now, since we are looking at the space of Z right now, we can also watch what happens to Z over time throughout our simulation."));
    cs.state_manager.set(unordered_map<string,string>{
        {"point_path_length", "0.5"},
        {"point_path_r", "0.25"},
        {"point_path_i", "0.6"},
    });
    cs.inject_audio_and_render(AudioSegment("Let's let this be our starting point."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"point_path_length", "1.5"},
    });
    cs.inject_audio_and_render(AudioSegment("Iterate the function once, and we get this new point."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"point_path_length", "8"},
    });
    cs.inject_audio_and_render(AudioSegment("Iterate again, and again, and again..."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"point_path_length", "50"},
    });
    cs.inject_audio_and_render(AudioSegment("We can see how Z changes across iterations."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"point_path_r", "<t> sin .75 *"},
        {"point_path_i", "<t> cos .75 *"},
    });
    cs.inject_audio_and_render(AudioSegment(2));
    cs.inject_audio_and_render(AudioSegment("Moving the point around a little,"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"point_path_r", "1.1"},
        {"point_path_i", ".05"},
    });
    cs.inject_audio(AudioSegment("you can tell the difference between when it converges inside the set,"), 2);
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"point_path_r", ".7"},
        {"point_path_i", ".3"},
    });
    cs.inject_audio(AudioSegment("and when it explodes off to infinity."), 2);
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"point_path_r", "-.15 <t> 0.9 * cos 100 / +"},
        {"point_path_i", "-.6 <t> 1.2 * sin 100 / +"},
    });
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = z^2 + c");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = z^2 + c");
    cs.inject_audio_and_render(AudioSegment("When we start near the border, it dances around for a bit and then decides to explode."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"ms.opacity", "0.2"},
        {"eqn.opacity", "1"},
        {"point_path_length", "0"},
    });
    cs.inject_audio(AudioSegment("Looking back at the original equation, there's a knob we've left unturned."), 2);
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = z^" + latex_color(0xff008888, "?") + " + c");
    cs.render();
    eqn.begin_latex_transition("z_{" + latex_text("next") + "} = z^3 + c");
    cs.render();
    cs.inject_audio_and_render(AudioSegment("This exponent- what if, instead of squaring Z, we cube it?"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_x_r", "3"},
        {"ms.opacity", "1"},
        {"eqn.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("Here's what we get."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-.3"},
        {"seed_c_i", "-.4"},
    });
    cs.inject_audio_and_render(AudioSegment("This is a third-degree Julia set!"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-.58"},
        {"seed_c_i", "-.27"},
        {"seed_z_r", "<t> 5 / sin 2 /"},
        {"seed_z_i", "<t> 5 / cos 2 /"},
    });
    cs.inject_audio_and_render(AudioSegment(2));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "-0.5"},
    });
    cs.inject_audio_and_render(AudioSegment(2));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_c_r", "0"},
        {"seed_c_i", "-.3"},
    });
    cs.inject_audio_and_render(AudioSegment("We can look at its mandelbrot counterpart too."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "<t> 5 / sin 2 /"},
        {"seed_c_i", "<t> 5 / cos 2 /"},
        {"zoom_exp", "-1"},
    });
    cs.inject_audio_and_render(AudioSegment(2));
    cs.inject_audio_and_render(AudioSegment(2));
    StateSliderScene ssxr("[seed_x_r]", "x_r", 0, 4, .3, .07);
    StateSliderScene ssxi("[seed_x_i]", "x_i", 0, 4, .3, .07);
    cs.add_scene_fade_in(&ssxr, "ssxr", 0.5, 0.85); 
    cs.inject_audio_and_render(AudioSegment("We can experiment with the exponent all day."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "0"},
        {"seed_x_r", "4"},
        {"ssxr.opacity", "0.5 <pixel_param_x> <pixel_param_x> 0.3 * * -"},
        {"seed_c_r", "0.1"},
        {"seed_c_i", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("Here's an exponent of 4."));
    cs.inject_audio_and_render(AudioSegment(1));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_x_r", "3.5"},
    });
    cs.inject_audio_and_render(AudioSegment("We can even make it a non-integer."));
    cs.inject_audio_and_render(AudioSegment(1));
    cs.add_scene_fade_in(&ssxi, "ssxi", 0.5, 0.95); 
    cs.send_to_front("parameterized");
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_x_r", "2"},
        {"seed_x_i", ".1"},
        {"gradation", "0"},
        {"seed_z_r", "0.5"},
        {"seed_z_i", "0.5"},
    });
    cs.inject_audio_and_render(AudioSegment("Heck, we can even make it a complex number too!"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "<t> 5 / sin 2 /"},
        {"seed_c_i", "<t> 5 / cos 2 /"},
        {"zoom_exp", "-1"},
    });
    cs.inject_audio_and_render(AudioSegment(1));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "0.1"},
        {"seed_c_i", "0"},
        {"ssxi.opacity", "<ssxr.opacity>"},
        {"seed_x_r", "2"},
        {"seed_x_i", "1"},
        {"zoom_exp", "0"},
    });
    cs.inject_audio_and_render(AudioSegment(2));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_x_r", "0"},
        {"seed_x_i", "2"},
    });
    cs.inject_audio_and_render(AudioSegment(1));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_x_r", "0"},
        {"seed_x_i", "3"},
    });
    cs.inject_audio_and_render(AudioSegment("With a purely imaginary exponent, you get these alien tendrily figures."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "0.5"},
        {"seed_c_i", "-.5"},
        {"zoom_exp", "-3"},
    });
    cs.inject_audio_and_render(AudioSegment("To be clear- this is still fundamentally a mandelbrot set! All I've done is change the exponent."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_z_r", "0.2"},
        {"seed_z_i", "0.5"},
    });
    cs.inject_audio_and_render(AudioSegment("So, in some sense, I guess we really had 3 knobs to turn after all!"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "0"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("And that, to me, raises another question."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "0"},
        {"pixel_param_x", "1"},
        {"pixel_param_c", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("What if we now change the pixel to represent the exponent?"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_x_r", "2"},
        {"seed_x_i", "0.5"},
    });
    cs.inject_audio_and_render(AudioSegment("So, this is the third orthogonal plane in a six-dimensional shape. I call it the X-Set."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "1"},
        {"seed_c_i", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("We are now fixing Z and C, and letting X vary with the pixel."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"side_panel", "1"},
        {"parameterized.opacity", "0"},
        {"sszr.opacity", "0"},
        {"sscr.opacity", "0"},
        {"ssxr.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("I'll add panels on the right which parameterize each of the 3 variables."));
    LatexScene lsz(latex_color(0xffff88ff, "z"), 1, 0.15, 0.15);
    LatexScene lsx(latex_color(0xffff88ff, "x"), 1, 0.15, 0.15);
    LatexScene lsc(latex_color(0xffff88ff, "c"), 1, 0.15, 0.15);
    cs.inject_audio(AudioSegment("These are the 3 orthogonal 2-D planes corresponding to Z, X, and C."), 3);
    cs.add_scene(&lsz, "lsz", 0.95, 0.29);
    cs.render();
    cs.add_scene(&lsx, "lsx", 0.95, 0.62);
    cs.render();
    cs.add_scene(&lsc, "lsc", 0.95, 0.96);
    cs.render();
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"max_iterations", "200"},
        {"seed_c_r", "<t> 5 / sin 2 *"},
        {"seed_c_r", "<t> 6 / cos 2 *"},
        {"seed_x_r", "<t> 6.5 / sin 2.5 * 2 +"},
        {"seed_x_r", "<t> 7.5 / sin 2.5 * 2 +"},
        {"seed_z_r", "<t> 5.5 / cos 2 *"},
        {"seed_z_r", "<t> 7 / cos 2 *"},
        {"zoom_exp", "2"},
    });
    cs.inject_audio_and_render(AudioSegment("Here's a tour of the X-Set, by moving the origin around in 6-space. Enjoy!"));
    cs.inject_audio_and_render(AudioSegment(40));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"max_iterations", "400"},
        {"seed_c_r", "-1.74572428"},
        {"seed_c_i", "-0.00564798"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"zoom_exp", "0"},
        {"gradation", "1"},
        {"lsz.opacity", "0"},
        {"lsx.opacity", "0"},
        {"lsc.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("There's one more thing which I thought was too cool not to share. Spinning back into Mandelbrot territory,"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "-15.5"},
    });
    cs.inject_audio_and_render(AudioSegment("if we zoom into the armpit of this miniature mandelbrot..."));
    cs.inject_audio_and_render(AudioSegment("It's a mini Julia set, inside the Mandelbrot set!"));
    cs.inject_audio_and_render(AudioSegment("Now, burn this shape into your mind and watch..."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "-3"},
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("...as I spin into Julia land..."));
    cs.inject_audio_and_render(AudioSegment("it turns out to be a copy of the Julia set for the C value which we are zoomed in on."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "0"},
        {"gradation", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("The takeaway here is that these fractals are not only self-similar, but also self-similar among one-another."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "0 8 / 11 + 2.2 * sin 1.5 +"},
        {"seed_c_i", "0 8 / 11 + 2.4 * cos 1.5 +"},
        {"seed_z_r", "0 8 / 11 + 2.6 * sin 1.5 *"},
        {"seed_z_i", "0 8 / 11 + 2.8 * cos 1.5 *"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "1"},
        {"pixel_param_c", "0"},
        {"seed_x_r", "1.45"},
        {"seed_x_i", "0.4"},
        {"max_iterations", "100"},
        {"zoom_exp", "-3"},
    });
    cs.inject_audio_and_render(AudioSegment("I don't know what I was expecting when I parameterized the Mandelbrot exponent, but it sure wasn't _more julia spirals and mandelbrot bulbs_!"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "2 8 / 11 + 2.2 * sin 1.5 +"},
        {"seed_c_i", "2 8 / 11 + 2.4 * cos 1.5 +"},
        {"seed_z_r", "2 8 / 11 + 2.6 * sin 1.5 *"},
        {"seed_z_i", "2 8 / 11 + 2.8 * cos 1.5 *"},
    });
    cs.inject_audio_and_render(AudioSegment("This degree of cross-parameter self-similarity totally took me by surprise."));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
        {"seed_z_r", "3.55"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
        {"side_panel", "0"},
        {"zoom_exp", "1"},
        {"gradation", "1"},
        {"lsz.opacity", "0"},
        {"lsx.opacity", "0"},
        {"lsc.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment(2));
    TwoswapScene tss;
    tss.state_manager.set(unordered_map<string,string>{
        {"circle_opacity", "0"},
    });
    cs.add_scene(&tss, "tss");
    cs.state_manager.set(unordered_map<string,string>{
        {"tss.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("I hope you enjoyed!"));
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "<t> sin .75 *"},
        {"seed_c_i", "<t> cos .75 *"},
        {"tss.opacity", "1"},
    });
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio(AudioSegment("This has been 2swap."), 4);
    cs.render();
    cs.render();
    cs.render();
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
    });
    cs.render();
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment(1));
}

int main() {
    Timer timer;
    PRINT_TO_TERMINAL = true;
    try {
        intro();
    }
    catch(std::exception& e) {
        cout << "EXCEPTION CAUGHT IN RUNTIME:" << endl;
        cout << e.what() << endl;
    }
    return 0;
}
