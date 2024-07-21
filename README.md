# SwapTube

This is the repository I use to render my YouTube videos on [2swap's YouTube Channel](https://www.youtube.com/@twoswap).

## Overview

SwapTube is built on top of FFMPEG, but most of the functionalities above video and audio encoding are custom-written. The project does not use any fancy 3D graphics libraries, with a few exceptions for particular functionalities.

## Setup
### External Dependencies

The following external dependencies are required for specific functionalities within the project. These dependencies must be installed if you want to use the related features.

| Dependency | What functionality is it needed for? | Used In | How It Is Used | Installation |
|------------|---------|---------|----------------|--------------|
| CMake | Everything | go.sh script | Compiles the project | `sudo apt install cmake` |
| FFMPEG and associated development libraries | Everything | audio_video folder | Encoding and processing video and audio streams | `sudo apt install ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libpostproc-dev libswresample-dev` |
| gnuplot | Debug plot generation | DebugPlot.h | Data dumped in out/ is rendered to a PNG | `sudo apt install gnuplot` |
| GLM | Graphs and 3D Graphics | 3d_scene.cpp, Graph.cpp | Vectors and quaternions to represent and rotate objects in space | `sudo apt install libglm-dev` |
| MicroTeX | In-Video LaTeX | visual_media.cpp | Converts LaTeX equations into SVG files for rendering | Manual setup from MicroTeX repository |
| RSVG | In-Video LaTeX | visual_media.cpp | Loads and renders SVG files into pixel data | `sudo apt install librsvg2-dev` |
| Cairo | In-Video LaTeX | visual_media.cpp | Renders SVG files onto Cairo surfaces and converts them to pixel data | `sudo apt install libcairo2-dev` |
| Eigen | Rendering Complex Polynomials | complex_plot_scene.cpp | Finding zeros to render plots of complex-valued functions | `sudo apt install libeigen3-dev` |
| LibPNG | PNG scenes | visual_media.cpp | Reads PNG files and converts them to pixel data | `sudo apt install libpng-dev` |

### Further Setup
You should record an audio file at `<repo_base>/media/testaudio.mp3`. The program will look at this at startup to make conclusions about your audio recording environment. If it is missing, the program will gracefully fail out with a kind error message.

## How to Run

When you have created a project file in `projects/yourprojectname.cpp`, you can compile and run the whole project by executing:

```bash
./go.sh yourprojectname
```

This script will handle the compilation and execution of your project file.

Sure, here's an additional section that documents the repository structure based on the provided information:

## Repository Structure

### Top-Level Files and Folders

- **./out/**: Contains the output files (videos and corresponding subtitle files) generated by swaptube.
  - `Some_Project.mp4`: An example output video file.
  - `Some_Project.srt`: An example subtitle file for the video.

- **./media/**: Stores input media files used by the project. This includes script recordings and PNGs. Generated LaTeX is stored in /out/, not /media/.
  - `Some_Project/`: Put media for your project here.
    - `record_list.tsv`: This will be generated by the program after rendering your project, and is read by the record_audios.sh script so that you can record your script easily in bulk.
  - `testaudio.mp3`: An audio file used during initialization to learn about your microphone.

- **./build/**: Contains various files and directories created during the build process, such as CMake cache, object files, and build scripts, but most importantly, the compiled binary. Caches and miscellaneous data products may also be dumped here, for example discovered connect 4 steady states and graphs.

- **record_audios.sh**: Reads the record_list.tsv file and permits you to quickly record all of the audio files for your video script.

- **go.sh**: The program entry point!

### Source Folder

- **./src/**
  - **./src/scenes/**: An atomic unit of animation is called a scene. This folder holds a bunch of definitions of different types of scenes, such as the connect 4 scene, latex scene, 3d scene, etc.
  - **./src/audio_video/**: This is where all the interfacing with ffmpeg and video encoding happens, as well as where the subtitle creation and audio management lives.
  - **./src/projects/**: Your project goes here! Some samples are provided, although I make no claim that they won't be regressed silently. 
    - `.active_project.tmp`: The go.sh script will copy your project to this path during compilation. I'm probably gonna change this pattern soon.
  - **./src/misc/**: Contains miscellaneous utility code and libraries.
    - `Timer.cpp`: For timing the render
    - `calculator.cpp`: This is an RPN string calculator used by the `dagger`, which you interface with whenever you define curves in the DAG.
    - `dagger.cpp`: This is the magic tool that lets you perform sick transitions. Lots of comments, check them out.
    - `inlines.h`: Random helper functions.
    - `visual_media.cpp`: This is what will read SVGs and PNGs and turn them into `pixels`.
    - `pixels.h`: A class representing buffered image data.
  - `main.cpp`: The main entry point for the application.

## Contact

For any questions or issues, please feel free to contact me via my [YouTube Channel](https://www.youtube.com/@twoswap).
