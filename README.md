# SwapTube

This is the repository I use to render [my YouTube videos](https://www.youtube.com/@twoswap).

## Overview

SwapTube is built on FFMPEG, but most of the functionalities above the layer of video and audio encoding are custom-written. The project does not use any fancy graphics libraries, with a few exceptions for particular functionalities.

## Setup
### External Dependencies

The following external dependencies are required for specific functionalities within the project. These dependencies must be installed if you want to use the related features.

| Item | What functionality is it needed for? | Used Where? | Used How? | Sample Ubuntu Installation |
|------------|---------|---------|----------------|--------------|
| CMake | Everything | go.sh script | Compiles the project | `sudo apt install cmake` |
| FFMPEG 5.0 or higher, and associated development libraries | Everything | audio_video folder | Encoding and processing video and audio streams | `sudo apt install ffmpeg libswscale-dev libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev` Note: compiling ffmpeg from source, it will likely be compiled with support for extra features detected on your system, which are not baked into my CMake config. I suggest installing a precompiled binary. |
| CUDA | Accelerating simulations and video rendering | Anything that references the CUDA src dir (most computationally expensive scenes) | Various | Hardware-dependent |
| gnuplot | Debug plot generation | DebugPlot.h | Data dumped in out/ is rendered to a PNG | `sudo apt install gnuplot` |
| GLM | Graphs and 3D Graphics | 3d_scene.cpp, Graph.cpp | Vectors and quaternions to represent and rotate objects in space | `sudo apt install libglm-dev` |
| MicroTeX | In-Video LaTeX, LatexScene | visual_media.cpp | Converts LaTeX equations into SVG files for rendering | Instructions are here: https://github.com/NanoMichael/MicroTeX/ You should install MicroTeX in MicroTeX-master alongside the swaptube checkout. Instructions will be printed if not found. |
| RSVG and GLib | In-Video LaTeX | visual_media.cpp | Loads and renders SVG files into pixel data | `sudo apt install librsvg2-dev libglib2.0-dev` |
| Cairo | In-Video LaTeX | visual_media.cpp | Renders SVG files onto Cairo surfaces and converts them to pixel data | `sudo apt install libcairo2-dev` |
| Eigen | Rendering Complex Polynomials | complex_plot_scene.cpp | Finding zeros to render plots of complex-valued functions | `sudo apt install libeigen3-dev` |
| LibPNG | PNG scenes | visual_media.cpp | Reads PNG files and converts them to pixel data | `sudo apt install libpng-dev` |

## How to Run

When you have created a project file in `projects/yourprojectname.cpp`, you can compile and run the whole project by executing:

```bash
./go.sh yourprojectname 640 360
```

Swaptube defaults to a framerate of 30 FPS and a sample rate of 48000 Hz. If you need to change these for whatever reason, they are specified in `go.sh` and `record_audios.py`.

## Repository Structure

### Top-Level Files and Folders

- **./src/**: Source folder structure is documented in the readme inside of it.

- **./out/**: Contains the output files (videos, corresponding subtitle files, data tables, and gnuplots) generated by swaptube.
  - Each subfolder corresponds to a project, and under that project, each render is stored in a separate folder named by timestamp.

- **./media/**: Stores input media files used by the project. This includes script recordings, generated LaTeX, source MP4s, and source PNGs.
  - You should not ever need to manually modify anything here, with the exception of placing source PNGs and MP4s. Audio should be recorded using `record_audios.py` after rendering your project.
  - `Some_Project/`: Put media for your project here.
    - `record_list.tsv`: This will be generated by the program after rendering your project, and is read by the `record_audios.py` script so that you can record your script easily in bulk.

- **./build/**: Contains various files and directories created during the build process, such as CMake cache, object files, and build scripts, but most importantly, the compiled binary. Caches and miscellaneous data products may also be dumped here, for example discovered connect 4 steady states and graphs.

- **record_audios.py**: Reads the record_list.tsv file and permits you to quickly record all of the audio files for your video script.

- **go.sh**: The program entry point!

## Design Philosophy

### Time Control
Swaptube uses a 2-layer time organization system. At the highest level, the video is divided into Macroblocks, which can be thought of as atomic units of audio. Macroblocks are divided into Microblocks, which are represent atomic time units controlling visual transformations.
Such division permits the user to define a video with an in-line script, such that SwapTube will do all time management and the user does not need to manually time each segment of video.
Furthermore, this permits native transitions: since a transition occurs over either a Macroblock or Microblock, Swaptube knows the duration of time over which the transition occurs, and can manage that transition automatically through State.

#### Macroblocks
There are a few types of macroblocks: FileBlocks, SilenceBlocks, GeneratedBlocks, etc. FileBlocks are defined by a filepath to an audio file inside the media folder.
SilenceBlocks are defined by a duration in seconds, and GeneratedBlocks are defined by a buffered array of audio samples generated in the project file.
A macroblock can be created using `yourscene.stage_macroblock(FileBlock("youraudio_no_file_extension"), 2);` which stages the macroblock to contain 2 microblocks.

#### Microblocks
After a Macroblock has been staged with `n` microblocks, the project file will render each microblock by calling `yourscene.render_microblock();`. Be sure to call this function `n` times, or else SwapTube will failout.

#### Smoketesting
In order to ensure that BOTH your time control is defined correctly (the appropriate number of microblocks are rendered) and that the project file does not crash due to a runtime error in the project file definition WITHOUT potentially kicking off a multi-hour render, you can run `./go.sh MyProjectName 640 360 -s`. Nothing will be rendered and no DataObjects will be manipulated, and the width and height of the video will be ignored. Smoketesting also updates the record_list.tsv file, so you can record your audio script after smoketesting without performing a full render.

### Scenes, State, and Data
The data structure that a single frame is rendered as a function of has three parts, roughly split up to differences in their nature:
- **Scene**: The Scene is the object which is constructed by the user in the project file. It fundamentally defines **what** is rendered. For example, a MandelbrotScene is responsible for rendering Mandelbrot Sets.
- **State**: State can be thought of as any numerical information used by the Scene to render a particular frame. This controls things such as the opacity of certain objects, or, following the Mandelbrot example, the zoom level of the Mandelbrot set. All scenes have a StateManager, and when the user whishes to modify the scene's state, they can do so by calling functions on the StateManager. Usually these will be `set` and `transition` function calls. Since State uniquely contains numerical information, swaptube will handle all the clean transitions of state.
- **Data**: Data is the non-numerical stateful information which is remembered by the Scene. A good example is the LambdaScene, which draws a Tromp Lambda Diagram, and stores as data that particular lambda expression. This type of information is non-numerical, and cannot be naively interpolated for a transition, so it must be kept in a DataObject with an interface defined in the Scene and DataObject.

