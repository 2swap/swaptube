# LatexScene File Support

LatexScene now supports reading LaTeX from external files, eliminating the need for escaped backslashes in your C++ code.

## Usage

### Old way (inline LaTeX with escaped backslashes):
```cpp
LatexScene latex("a=\\frac{b}{c}", 0.5);
latex.begin_latex_transition(MICRO, "\\begin{tabular}{|c|c|} \\hline \\textbf{A} & \\textbf{B} \\\\\\\\ \\hline \\end{tabular}");
```

### New way (LaTeX from file):
1. Create a file in `media/<ProjectName>/latex/myformula.tex`:
```latex
a=\frac{b}{c}
```

2. Reference the filename in your code:
```cpp
LatexScene latex("myformula.tex", 0.5);
latex.begin_latex_transition(MICRO, "mytable.tex");
```

## Benefits
- No more double/triple/quadruple escaped backslashes
- Easier to edit complex LaTeX formulas
- Better syntax highlighting in your text editor
- Cleaner, more readable C++ code

## Backward Compatibility
The old inline method still works! If a file doesn't exist with the given name, LatexScene will treat the input as literal LaTeX code. This means all existing code continues to work without changes.

## Example Files
See `media/LatexDemo/latex/` for example `.tex` files.