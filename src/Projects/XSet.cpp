using namespace std;
#include <string>
const string project_name = "XSet";
const int width_base = 640;
const int height_base = 360;
const float mult = 1;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;

#include "../io/writer.cpp"
#include "../misc/Timer.cpp"

#include "../Scenes/Common/CompositeScene.cpp"
#include "../Scenes/Media/LatexScene.cpp"
#include "../Scenes/Media/PngScene.cpp"
#include "../Scenes/Math/MandelbrotScene.cpp"
#include "../Scenes/Common/TwoswapScene.cpp"

void intro() {
    FOR_REAL = false;
    MandelbrotScene ms;
    unordered_map<string,string> init = {
        {"zoom_r", "2 <zoom_exp> ^"},
        {"zoom_exp", "0"},
        {"zoom_i", "0"},
        {"max_iterations", "100"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
        {"pixel_param_z", "0"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "1"},
        {"side_panel", "0"},
    };
    ms.state_manager.set(init);

    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"seed_c_r", "-.4621603"},
        {"seed_c_i", "-.5823998"},
    });
    ms.inject_audio_and_render(AudioSegment("This is the mandelbrot set."));
    // Zoom in on an interesting spot with a minibrot
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"max_iterations", "600"},
        {"zoom_exp", "-9"},
    });
    ms.inject_audio_and_render(AudioSegment("Known for its beauty at all resolutions and scales,"));
    ms.inject_audio_and_render(AudioSegment("and stunning self-similarity,"));
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"zoom_exp", "0"},
    });
    ms.inject_audio_and_render(AudioSegment("it is the cornerstone example of a fractal."));
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "1"},
        {"pixel_param_x", "0"},
        {"pixel_param_c", "0"},
    });
    ms.inject_audio_and_render(AudioSegment("What's more, if we just change our perspective, or, rather, spin our viewpanel through the fourth dimension,"));
    ms.inject_audio_and_render(AudioSegment("we get a similar fractal called a Julia set."));
    ms.inject_audio_and_render(AudioSegment("The Mandelbrot sets and the Julia sets are in planes orthogonal to each other in a certain 4d-space."));
    ms.inject_audio_and_render(AudioSegment("This much is well-documented. And don't worry, I'll explain how all of that works in a sec."));
    ms.state_manager.microblock_transition(unordered_map<string,string>{
        {"pixel_param_z", "0"},
        {"pixel_param_x", "1"},
        {"pixel_param_c", "0"},
    });
    ms.inject_audio_and_render(AudioSegment("But before that, I wanna twist our perspective once more, into yet another orthogonal 2d-plane."));
    ms.inject_audio_and_render(AudioSegment("This is another natural extension of the Mandelbrot Set which I found."));
    ms.state_manager.microblock_transition(init);
    ms.inject_audio_and_render(AudioSegment("But, let's take it from the start."));
    FOR_REAL = true;
    ms.inject_audio_and_render(AudioSegment("This is the mandelbrot set."));
    CompositeScene cs;
    cs.add_scene(&ms, "ms");
    cs.state_manager.set(unordered_map<string,string>{
        {"ms.opacity", "1"},
    });
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"ms.opacity", ".2"},
    });
    PngScene comp_axes("complex_axes");
    cs.add_scene_fade_in(&comp_axes, "comp_axes");
    cs.inject_audio_and_render(AudioSegment("It lives in the complex plane."));
    LatexScene eqn("z_{" + latex_text("next") + "} = z^2 + c", 0.7, 1, .5);
    cs.add_scene_fade_in(&eqn, "eqn");
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"comp_axes.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("The equation which generates it is surprisingly simple."));
    cs.inject_audio_and_render(AudioSegment("Here it is."));
    eqn.begin_latex_transition(latex_color(0xffff0000, "z_{next}") + " = " + latex_color(0xffff0000, "z") + "^2 + c");
    cs.inject_audio_and_render(AudioSegment("It's telling us the way to update this variable Z."));
    cs.inject_audio(AudioSegment("Let's let this variable C be 1 just for now. We'll try other values in a sec."), 4);
    eqn.begin_latex_transition("z^2 + " + latex_color(0xff008888, "c"));
    cs.render();
    eqn.begin_latex_transition("z^2 + 1");
    cs.render();
    cs.render();
    cs.render();
    eqn.begin_latex_transition("0^2 + 1");
    cs.inject_audio_and_render(AudioSegment("And let's let Z start at 0."));
    eqn.begin_latex_transition("0 + 1");
    cs.inject_audio_and_render(AudioSegment("We do some arithmetic,"));
    eqn.begin_latex_transition("1");
    cs.inject_audio_and_render(AudioSegment("and we find the updated value of Z."));
    cs.inject_audio_and_render(AudioSegment("We then take that updated value,"));
    eqn.begin_latex_transition("1^2 + 1");
    cs.inject_audio_and_render(AudioSegment("plug it into the equation,"));
    eqn.begin_latex_transition("2");
    cs.inject_audio_and_render(AudioSegment("and do the math again."));
    eqn.begin_latex_transition("2^2 + 1");
    cs.inject_audio(AudioSegment("We want to keep repeating this over and over."), 2);
    cs.render();
    eqn.begin_latex_transition("4 + 1");
    cs.render();
    eqn.begin_latex_transition("5");
    cs.inject_audio_and_render(AudioSegment("Just square, and add one."));
    eqn.begin_latex_transition("5^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("25 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("26");
    cs.inject_audio_and_render(AudioSegment("Square, and add one."));
    eqn.begin_latex_transition("26^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("676 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("677");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("677^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("458229 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("458330");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("458330^2 + 1");
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("In our case, Z is blowing up towards infinity."));
    eqn.begin_latex_transition("z^2 + c");
    cs.inject_audio_and_render(AudioSegment("Let's try that again."));
    eqn.begin_latex_transition("z^2 - 1");
    cs.inject_audio_and_render(AudioSegment("This time, let's let C be negative 1 instead of positive 1."));
    eqn.begin_latex_transition("0^2 - 1");
    cs.inject_audio_and_render(AudioSegment("Plug in 0 to start off Z,"));
    eqn.begin_latex_transition("0 - 1");
    cs.inject_audio_and_render(AudioSegment("iterate,"));
    eqn.begin_latex_transition("-1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("(-1)^2 - 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("1 - 1");
    cs.inject_audio_and_render(AudioSegment(1));
    eqn.begin_latex_transition("0");
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio(AudioSegment("and this time, we fall into a cyclic trap."), 4);
    eqn.begin_latex_transition("0^2 - 1");
    cs.render();
    eqn.begin_latex_transition("-1");
    cs.render();
    eqn.begin_latex_transition("(-1)^2 - 1");
    cs.render();
    eqn.begin_latex_transition("0");
    cs.render();
    eqn.begin_latex_transition("z^2 + c");
    cs.inject_audio_and_render(AudioSegment(1));
    cs.inject_audio_and_render(AudioSegment("Remember, we're working with Complex numbers here, not just real numbers."));
    eqn.begin_latex_transition("z^2 + i");
    cs.inject_audio_and_render(AudioSegment("C can be a number like i."));
    cs.inject_audio(AudioSegment("Running our simulation again, in this case it ends up stuck as well."), 8);
    eqn.begin_latex_transition("0^2 + i");
    cs.render();
    eqn.begin_latex_transition("i");
    cs.render();
    eqn.begin_latex_transition("i^2 + i");
    cs.render();
    eqn.begin_latex_transition("-1+i");
    cs.render();
    eqn.begin_latex_transition("(-1+i)^2 + i");
    cs.render();
    eqn.begin_latex_transition("i");
    cs.render();
    eqn.begin_latex_transition("i^2 + i");
    cs.render();
    eqn.begin_latex_transition("-1+i");
    cs.render();
    cs.state_manager.microblock_transition(unordered_map<string,string>{
        {"eqn.opacity", "0"},
    });
    cs.inject_audio_and_render(AudioSegment("The differences in behavior here are the key."));
    LatexScene table("\\begin{tabular}{p{4cm}|p{4cm}} \\textbf{z stays bounded} & \\textbf{z explodes} \\end{tabular}", 0.7, 1, 1);
    cs.add_scene_fade_in(&table, "table");
    cs.inject_audio(AudioSegment("Depending on what value of C we choose, sometimes Z will stay close to zero,"), 2);
    table.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} \\textbf{z stays bounded} & \\textbf{z explodes} \\\\\\\\ \\hline -1 & \\\\\\\\ \\end{tabular}");
    cs.render();
    table.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} \\textbf{z stays bounded} & \\textbf{z explodes} \\\\\\\\ \\hline -1 & 1 \\\\\\\\ \\end{tabular}");
    cs.render();
    table.begin_latex_transition("\\begin{tabular}{p{4cm}|p{4cm}} \\textbf{z stays bounded} & \\textbf{z explodes} \\\\\\\\ \\hline -1 & 1 \\\\\\\\ i & \\\\\\\\ \\end{tabular}");
    cs.inject_audio_and_render(AudioSegment("and sometimes it won't."));
    return;
    cs.inject_audio_and_render(AudioSegment("That is the difference between a point inside the mandelbrot set, and outside it."));
    cs.inject_audio_and_render(AudioSegment("If we make a plot of the complex plane,"));
    cs.inject_audio_and_render(AudioSegment("and for each value of C, we run this little simulation,"));
    cs.inject_audio_and_render(AudioSegment("we'll paint points white if they blow up,"));
    cs.inject_audio_and_render(AudioSegment("and black if they don't."));
    cs.inject_audio_and_render(AudioSegment("That gives us the characteristic shape of the Mandelbrot set."));
    cs.inject_audio_and_render(AudioSegment("We can additionally add color to show how long it takes for the number to blow up."));
    cs.inject_audio_and_render(AudioSegment("And that's how you get these pretty pictures."));
    cs.inject_audio_and_render(AudioSegment("But let's look back at our equation."));
    cs.inject_audio_and_render(AudioSegment("We not only had the free choice of what C can be,"));
    cs.inject_audio_and_render(AudioSegment("but also the option to choose a starting value of Z."));
    cs.inject_audio_and_render(AudioSegment("Let's change that starting Z from 0 to 1 and see what happens."));
    cs.inject_audio_and_render(AudioSegment("Woah! That's a totally different set."));
    cs.inject_audio_and_render(AudioSegment("We can play around with different values of Z all day, just like with C."));
    cs.inject_audio_and_render(AudioSegment("Which makes you think- we started by fixing Z at zero, and making C vary as a function of the pixel."));
    cs.inject_audio_and_render(AudioSegment("What if we fix C at zero, and make Z start where the pixel is?"));
    cs.inject_audio_and_render(AudioSegment("We get a Julia set!"));
    cs.inject_audio_and_render(AudioSegment("So, we have 2 variables which we can pick at the start of the simulation."));
    cs.inject_audio_and_render(AudioSegment("We get to choose C, and we get to pick what Z should start at."));
    cs.inject_audio_and_render(AudioSegment("Since each of these lives in the complex plane, that's 4 degrees of choice- a 4D shape."));
    cs.inject_audio_and_render(AudioSegment("Parameterizing the pixels of the screen as either Z or C yields a Julia set or Mandelbrot set respectively."));
    cs.inject_audio_and_render(AudioSegment("Now, since we are looking at the space of Z right now, we can also watch what happens when to Z over time throughout our simulation."));
    cs.inject_audio_and_render(AudioSegment("Let's let this be our starting point."));
    cs.inject_audio_and_render(AudioSegment("Iterate the function once, and we get this new point."));
    cs.inject_audio_and_render(AudioSegment("Iterate again, and again, and again..."));
    cs.inject_audio_and_render(AudioSegment("We can see how Z changes over time."));
    cs.inject_audio_and_render(AudioSegment("Moving the point around a little,"));
    cs.inject_audio_and_render(AudioSegment("you can tell the difference between when it converges inside the set,"));
    cs.inject_audio_and_render(AudioSegment("and when it explodes off to infinity."));
    cs.inject_audio_and_render(AudioSegment("When you are near the border, it dances around for a bit and then decides to explode."));
    cs.inject_audio_and_render(AudioSegment("Looking back at the original equation, there's a knob we've left unturned."));
    cs.inject_audio_and_render(AudioSegment("This exponent- what if, instead of squaring Z, we cube it?"));
    cs.inject_audio_and_render(AudioSegment("Here's what we get."));
    cs.inject_audio_and_render(AudioSegment("We can experiment with the exponent all day too."));
    cs.inject_audio_and_render(AudioSegment("We can even make it a non-integer."));
    cs.inject_audio_and_render(AudioSegment("Heck, we can even make it be a complex number too!"));
    cs.inject_audio_and_render(AudioSegment("So, in some sense, I guess we really had 3 knobs to turn!"));
    cs.inject_audio_and_render(AudioSegment("And that, to me, raises another question."));
    cs.inject_audio_and_render(AudioSegment("What if we now change the pixel to represent the exponent?"));
    cs.inject_audio_and_render(AudioSegment("Here goes..."));
    cs.inject_audio_and_render(AudioSegment("So, this is the third orthogonal plane in a six-dimensional shape."));
    cs.inject_audio_and_render(AudioSegment("We are now fixing Z and C, and letting X vary with the pixel."));
    cs.inject_audio_and_render(AudioSegment("I'll add panels on the right which parameterize each of the 3 variables, and we'll move the origin around in 3-space."));
    cs.inject_audio_and_render(AudioSegment("Wild!"));
    cs.inject_audio_and_render(AudioSegment("This has been 2swap."));
}

int main() {
    Timer timer;
    FOR_REAL = true;
    PRINT_TO_TERMINAL = true;
    intro();
    return 0;
}

