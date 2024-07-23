class PathManager {
    const string project_name;
    const string unique_timestamp;
    const string repo_root;
    const string output_directory;
    const string media_directory;
    const string this_run_output_directory;
    const string this_run_media_directory;
    const string plots_folder;
    const string data_folder;
    const string video_output;
    const string subtitle_output;
    const string testaudio_path;
    const string latex_folder;

    PathManager(const string& proj)
        : project_name(proj),
        unique_timestamp(get_timestamp()),
        repo_root("../"),
        output_directory(repo_root + "out/"),
        media_directory(repo_root + "media/"),
        this_run_output_directory(output_directory + unique_timestamp + "/"),
        this_run_media_directory(output_directory + unique_timestamp + "/"),
        plots_folder(this_run_output_directory + "plots/"),
        data_folder(this_run_output_directory + "data/"),
        video_output(this_run_output_directory + project_name + ".mp4"),
        subtitle_output(this_run_output_directory + project_name + ".srt"),
        latex_directory(this_run_media_directory + "latex/")
    { }
};
