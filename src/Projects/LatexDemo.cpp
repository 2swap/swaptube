using namespace std;
#include <string>
const string project_name = "LatexDemo";
#include "../io/PathManager.cpp"

const int width_base = 640;
const int height_base = 360;
const int mult = 2;

// PROJECT GLOBALS
const int VIDEO_WIDTH = width_base * mult;
const int VIDEO_HEIGHT = height_base * mult;
const int VIDEO_FRAMERATE = 30;
#include "../io/writer.cpp"

#include "../Scenes/Media/LatexScene.cpp"
#include "../misc/Timer.cpp"
void render_video(){
    PRINT_TO_TERMINAL = false;
    LatexScene latex(      "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline \\end{tabular}", 0.5);
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline\\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline ... & ... \\\\\\\\ \\hline \\text{Total} & 4,531,985,219,092 \\\\\\\\ \\hline \\end{tabular}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.inject_audio_and_render(AudioSegment(1));
}

void render_latex_demo(){
    PRINT_TO_TERMINAL = false;
    
    // Initial LaTeX content: Introduction to LaTeX
    LatexScene latex("abc", 0.5);
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("a=bc");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.inject_audio_and_render(AudioSegment(1));
    latex.begin_latex_transition("a=\\frac{b}{c}");
    latex.inject_audio_and_render(AudioSegment(1));
    latex.inject_audio_and_render(AudioSegment(1));
    //latex.begin_latex_transition("abcdef");
    //latex.inject_audio_and_render(AudioSegment(1));
    //latex.inject_audio_and_render(AudioSegment(1));
    //latex.begin_latex_transition("\\text{abcdef}");
    //latex.inject_audio_and_render(AudioSegment(1));
    //latex.inject_audio_and_render(AudioSegment(1));
}

int main() {
    Timer timer;
    render_latex_demo();
    return 0;
}
