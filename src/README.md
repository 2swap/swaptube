# Source Folder Structure

- **./src/**
  - **./src/Scenes/**: A scene is an object which the user creates in a project file. This folder holds a bunch of definitions of different types of scenes, such as the connect 4 scene, latex scene, 3d scene, etc.
  - **./src/CUDA/**: A list of one-off files with clear interface boundaries that solve specific subproblems using CUDA.
  - **./src/io/**: This is where all the interfacing with ffmpeg and video encoding happens, as well as where the subtitle creation and audio management lives.
  - **./src/Projects/**: Your project goes here! Some samples are provided, although I make no guarantee that they won't have regressed. I do not maintain these, they should be assumed broken by default. The go.sh script will copy your project to `.active_project.tmp` before running.
  - **./src/misc/**: Contains some core swaptube code, most importantly the program entry point, the `pixels` definition, and the StateManager.
