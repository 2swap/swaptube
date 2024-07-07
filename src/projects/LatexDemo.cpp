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
