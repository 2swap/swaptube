#pragma once
using namespace std;
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <ctime>
string get_timestamp(){
    // Get current time
    auto now = chrono::system_clock::now();
    time_t now_time_t = chrono::system_clock::to_time_t(now);
    tm now_tm = *localtime(&now_time_t);
    // Format time to append to folder name
    stringstream ss;
    ss << put_time(&now_tm, "%Y%m%d_%H%M%S");
    return ss.str();
}

#include <filesystem>
#include "../misc/inlines.h"
void ensure_dir_exists(const string& path) {
    if (filesystem::exists(path)) {
        //cout << "Directory " << path << " already exists, not creating." << endl;
        return;
    } else {
        if (filesystem::create_directories(path)) {
            //cout << "Directory " << path << " created successfully." << endl;
            return;
        } else {
            throw runtime_error("Failed to create dir " + path + ".");
        }
    }
}

class PathManager {
public:
    const string project_name;
    const string unique_timestamp;
    // Directories
    const string repo_root;
    const string output_dir;
    const string media_dir;
    const string this_project_output_dir;
    const string this_run_output_dir;
    const string this_project_media_dir;
    const string plots_dir;
    const string data_dir;
    const string latex_dir;
    //Fles
    const string video_output;
    const string subtitle_output;
    const string record_list_path;
    const string testaudio_path;

    PathManager(const string& proj)
        :
        project_name(proj),
        unique_timestamp(get_timestamp()),
        repo_root("../"),
        output_dir(repo_root + "out/"),
        media_dir(repo_root + "media/"),
        this_project_output_dir(output_dir + project_name + "/"),
        this_run_output_dir(this_project_output_dir + unique_timestamp + "/"),
        this_project_media_dir(media_dir + project_name + "/"),
        plots_dir(this_run_output_dir + "plots/"),
        data_dir(this_run_output_dir + "data/"),
        latex_dir(this_project_media_dir + "latex/"),
        video_output(this_run_output_dir + project_name + ".mp4"),
        subtitle_output(this_run_output_dir + project_name + ".srt"),
        record_list_path(this_project_media_dir + "record_list.tsv"),
        testaudio_path(media_dir + "testaudio.mp3")
    {
        // for the directories (as opposed to files) create them if they dont exist.
        ensure_dir_exists(repo_root              );
        ensure_dir_exists(output_dir             );
        ensure_dir_exists(media_dir              );
        ensure_dir_exists(this_project_output_dir);
        ensure_dir_exists(this_run_output_dir    );
        ensure_dir_exists(this_project_media_dir );
        ensure_dir_exists(plots_dir              );
        ensure_dir_exists(data_dir               );
        ensure_dir_exists(latex_dir              );
    }
};

// Global Path Manager
PathManager PATH_MANAGER(project_name);
