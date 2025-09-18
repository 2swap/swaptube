# contents, in rough order of how much you probably care
    - `StateManager.cpp`: This is at the heart of what makes swaptube tick. The magic tool that lets you perform sick transitions. Lots of comments, check them out.
    - `pixels.h`: A class representing buffered image data.
    - `inlines.h`: Random helper functions.
    - `color.cpp`: Random helper functions, but about color.
    - `Main.cpp`: The main entry point for the application. Just some boilerplate.
    - `calculator.cpp`: This is an RPN string calculator used by the StateManager, which you interface with whenever you define curves in the DAG.
    - `Timer.cpp`: For timing the render
