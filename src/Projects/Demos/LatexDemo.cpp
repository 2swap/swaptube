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
    return;
    */

    vector<string> latex_strings = {"abc", "a^bc", "abbc", "abc", "a=\\frac{b}{c}", "abcdef", "deee", "ddde"};
    //vector<string> latex_strings = {"x^2+6x+5=0", "x^2+6x=-5", "x^2+6x+9=4", "(x+3)^2=4", "x+3=\\pm2", "x=-3\\pm2", "x=-1,\\,-5"};
    //vector<string> latex_strings = {"\\text{This is some English text.}", "\\text{This is some more English text.}", "\\text{This is even more English text.}"};
    //vector<string> latex_strings = {"aaa", "a^aa", "a^{aa}", "a^{a^a}", "a^{a^{a^a}}", "a^{a^{a^{a^a}}}", "a^{a^{a^{a^{a^a}}}}", "a^{a^{a^{a^{a^{a^a}}}}}", "a^{a^{a^{a^{a^{a^{a^a}}}}}}", "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"};

    LatexScene latex(latex_strings[0], 0.6);
    stage_macroblock(SilenceBlock(1), 1);
    latex.render_microblock();

    for (int i = 1; i < latex_strings.size(); i++){
        const string& latex_str = latex_strings[i];
        latex.begin_latex_transition(MICRO, latex_str);
        stage_macroblock(SilenceBlock(1), 2);
        latex.render_microblock();
        latex.render_microblock();
    }
}
