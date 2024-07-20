#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

class DebugPlot {
public:
    // Constructor that initializes the plot name and file path
    DebugPlot(const string& plot_name) 
        : plot_name_(plot_name),
          data_file_path_("../out/" + plot_name + ".dat"),
          plot_file_path_("../out/" + plot_name + ".png") 
    {
        data_file_.open(data_file_path_, ios::out | ios::trunc);
        if (!data_file_.is_open()) {
            throw runtime_error("Failed to open data file: " + data_file_path_);
        }
    }

    // Destructor to close the file
    ~DebugPlot() {
        if (data_file_.is_open()) {
            data_file_.close();
        }
        generate_plot();
    }

    // Function to add a data point to the file
    void add_datapoint(double value) {
        if (data_file_.is_open()) {
            data_file_ << value << endl;
        } else {
            cerr << "Data file is not open. Cannot add data point." << endl;
        }
    }

    // Function to generate the plot using Gnuplot
    void generate_plot() {
        string gnuplot_command = "gnuplot -e \"set terminal png; set output '" + plot_file_path_ + "'; plot '" + data_file_path_ + "' with lines\"";
        int result = system(gnuplot_command.c_str());
        if (result != 0) {
            cerr << "Failed to generate plot with Gnuplot. Command: " << gnuplot_command << endl;
        }
    }

private:
    const string plot_name_;
    const string data_file_path_;
    const string plot_file_path_;
    ofstream data_file_;
};

