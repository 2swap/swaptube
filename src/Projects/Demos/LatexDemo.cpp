#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Common/CompositeScene.h"

void render_video(){

    vector<vector<string>> transitions;
    transitions.push_back(vector<string>{"abc", "a^bc", "abbc", "abc", "a=\\frac{b}{c}", "abcdef", "deee", "ddde"});
    transitions.push_back(vector<string>{"ax^2 + bx + c = 0", "x^2 + \\frac{b}{a}x + \\frac{c}{a} = 0"});
    transitions.push_back(vector<string>{"x^2+6x+5=0", "x^2+6x=-5", "x^2+6x+9=4", "(x+3)^2=4", "x+3=\\pm2", "x=-3\\pm2", "x=-1,\\,-5"});
    transitions.push_back(vector<string>{"\\text{This is some English text.}", "\\text{This is some more English text.}", "\\text{This is even more English text.}"});
    transitions.push_back(vector<string>{"aaa", "a^aa", "a^{aa}", "a^{a^a}", "a^{a^{a^a}}", "a^{a^{a^{a^a}}}", "a^{a^{a^{a^{a^a}}}}", "a^{a^{a^{a^{a^{a^a}}}}}", "a^{a^{a^{a^{a^{a^{a^a}}}}}}", "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"});
    transitions.push_back(vector<string>{"(ij)^2 = -1", "ijij = -1", "iji = j", "ij = -ji"});

    for (vector<string> latex_strings : transitions) {
        LatexScene latex(latex_strings[0]);
        stage_macroblock(SilenceBlock(1));
        latex.render_microblock();
        for (int i = 1; i < latex_strings.size(); i++){
            const string& latex_str = latex_strings[i];
            latex.begin_latex_transition(MICRO, latex_str);
            stage_macroblock(SilenceBlock(1));
            latex.render_microblock();
            latex.render_microblock();
        }
    }
}
