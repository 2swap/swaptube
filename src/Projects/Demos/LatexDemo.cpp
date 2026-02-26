#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Common/CompositeScene.h"

void render_video(){
    /*
    LatexScene latex("\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline \\end{tabular}", 0.5);
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    latex.begin_latex_transition(MICRO, "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline \\end{tabular}");
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    latex.begin_latex_transition(MICRO, "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline \\end{tabular}");
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    latex.begin_latex_transition(MICRO, "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline\\end{tabular}");
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    latex.begin_latex_transition(MICRO, "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline \\end{tabular}");
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    latex.begin_latex_transition(MICRO, "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline \\end{tabular}");
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    latex.begin_latex_transition(MICRO, "\\begin{tabular}{|c|c|} \\hline \\textbf{Depth} & \\textbf{Nodes} \\\\\\\\ \\hline 0 & 1 \\\\\\\\ \\hline 1 & 7 \\\\\\\\ \\hline 2 & 49 \\\\\\\\ \\hline 3 & 238 \\\\\\\\ \\hline 4 & 1120 \\\\\\\\ \\hline ... & ... \\\\\\\\ \\hline \\text{Total} & 4,531,985,219,092 \\\\\\\\ \\hline \\end{tabular}");
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();
    */

    /*
    shared_ptr<LatexScene> latex = make_shared<LatexScene>("abc", 0.5);
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "a=bc");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "a=\\frac{b}{c}");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "abcdef");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->jump_latex("jumped");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    CompositeScene cs;
    cs.add_scene(latex, "latex");

    latex->jump_latex("jumped2");
    stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();
    latex->jump_latex("jumped3");
    stage_macroblock(SilenceBlock(2), 1);
    cs.render_microblock();
    */

    /*
    shared_ptr<LatexScene> latex = make_shared<LatexScene>("\\frac{1}{x}=\\frac{x}{y}=\\frac{y}{2}", 0.5);
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    // Animate solution
    latex->begin_latex_transition(MICRO, "1 \\cdot 2 = x \\cdot x");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "2 = x^2");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "x^2 = 2");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "x = \\sqrt{2}");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();
    */

    shared_ptr<LatexScene> latex = make_shared<LatexScene>("\\frac{1}{x}=\\frac{x}{y}=\\frac{y}{2}", 0.5);
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "\\frac{1}{x}=\\frac{x}{y} \\qquad \\frac{x}{y}=\\frac{y}{2}");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    // Animate solution
    latex->begin_latex_transition(MICRO, "1 \\cdot y = x \\cdot x \\qquad \\frac{x}{y} = \\frac{y}{2}");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "y = x^2 \\qquad \\frac{x}{y} = \\frac{y}{2}");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "y = x^2 \\qquad 2 \\cdot x = y \\cdot y");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "y = x^2 \\qquad 2x = y^2");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "y = x^2 \\qquad y^2 = 2x");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "(x^2)^2 = 2x");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "x^4 = 2x");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "x^3 = 2");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();

    latex->begin_latex_transition(MICRO, "x = \\sqrt[3]{2}");
    stage_macroblock(SilenceBlock(2), 2);
    latex->render_microblock();
    latex->render_microblock();
}
