using namespace std;
#include <string>
const string project_name = "LatexDemo";
#include "../io/PathManager.cpp"

const int width_base = 640;
const int height_base = 360;
const int mult = 1;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;
#include "../io/writer.cpp"

#include "../scenes/Media/latex_scene.cpp"
#include "../misc/Timer.cpp"
void render_video(){
    PRINT_TO_TERMINAL = false;
    LatexScene latex(      "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline \\end{tabular}", .5);
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline\\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline ... & ... \\\\\\\\ \\hline \\text{Total} & 4,531,985,219,092 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.inject_audio_and_render(AudioSegment(1));
}

void render_latex_demo(){
    PRINT_TO_TERMINAL = false;
    
    // Initial LaTeX content: Introduction to LaTeX
    LatexScene latex("\\text{Welcome to the LaTeX Transition Demo!}", .5);
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to a brief introduction
    latex.begin_transition("\\text{In this demo, we'll explore various LaTeX transitions.}");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to a simple equation with explanation
    latex.begin_transition("\\text{Let's start with Einstein's famous equation:} \\ E = mc^2");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to more context about the equation
    latex.begin_transition("\\text{This equation,} \\ E = mc^2 \\text{, shows the relationship between mass and energy.}");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to another simple equation
    latex.begin_transition("\\text{Next, consider the Pythagorean theorem:} \\ a^2 + b^2 = c^2");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to a more complex equation involving integrals
    latex.begin_transition("\\text{Integrals are important in calculus:} \\ \\int_{a}^{b} f(x) \\, dx");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to matrices with context
    latex.begin_transition("\\text{Matrices are fundamental in linear algebra:} \\ \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to summation notation
    latex.begin_transition("\\text{Summation notation represents series:} \\ \\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Transition to a custom equation
    latex.begin_transition("\\text{Finally, consider the Fourier transform:} \\ F(x) = \\int_{-\\infty}^{\\infty} f(t) \\, e^{-2\\pi i t x} \\, dt");
    latex.inject_audio_and_render(AudioSegment(1));
    
    // Final message
    latex.begin_transition("\\text{This concludes the LaTeX Transition Demo. Thank you!}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.inject_audio_and_render(AudioSegment(1));
}

int main() {
    Timer timer;
    render_latex_demo();
    timer.stop_timer();
    return 0;
}
