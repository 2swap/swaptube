# Source Folder Structure

- **./src/**
  - **./src/scenes/**: A scene is an object which the user creates in a project file. This folder holds a bunch of definitions of different types of scenes, such as the connect 4 scene, latex scene, 3d scene, etc.
  - **./src/audio_video/**: This is where all the interfacing with ffmpeg and video encoding happens, as well as where the subtitle creation and audio management lives.
  - **./src/projects/**: Your project goes here! Some samples are provided, although I make no guarantee that they won't have regressed. I do not maintain these, they should be assumed broken by default.
    - `.active_project.tmp`: The go.sh script will copy your project to this path during compilation. I'm probably gonna change this pattern soon.
  - **./src/misc/**: Contains miscellaneous utility code.
    - `Timer.cpp`: For timing the render
    - `calculator.cpp`: This is an RPN string calculator used by the StateManager, which you interface with whenever you define curves in the DAG.
    - `StateManager.cpp`: This is the magic tool that lets you perform sick transitions. Lots of comments, check them out.
    - `inlines.h`: Random helper functions.
    - `visual_media.cpp`: This is what will read SVGs and PNGs and turn them into `pixels`.
    - `pixels.h`: A class representing buffered image data.
  - `Main.cpp`: The main entry point for the application.
