#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cassert>

class DebugPlot {
public:
    // Constructor that initializes the plot name, data series names, and file paths
    DebugPlot(const string& plot_name, const string& data_dir, const string& plots_dir, const vector<string>& series_names) 
        : plot_name_(plot_name),
          data_file_path_("io_out/data/" + plot_name + ".dat"),
          plot_file_path_("io_out/plots/" + plot_name + ".png"),
          series_names_(series_names),
          num_series_(series_names.size())
    { init(); }

    void init(){
        cout << "Initializing DebugPlot: " << plot_name_ << endl;
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
    void add_datapoint(const vector<double>& values) {
        if (data_file_.is_open()) {
            if (values.size() != num_series_) {
                cout << "Error: Number of values does not match the number of series." << endl;
                return;
            }
            for (size_t i = 0; i < values.size(); ++i) {
                data_file_ << values[i];
                if (i < values.size() - 1) {
                    data_file_ << "\t";
                }
            }
            data_file_ << endl;
        } else {
            cout << "Data file is not open. Cannot add data point." << endl;
            return;
        }
        data_has_been_added = true;
    }
    void add_datapoint(double value){
        assert(num_series_ == 1);
        add_datapoint(vector<double>{value});
    }

    // Function to generate the plot using Gnuplot
    void generate_plot() {
        if(!data_has_been_added) return;
        string gnuplot_command = "gnuplot -e \"set terminal png size 1920,1080; set output '" + plot_file_path_ + "'; ";
        gnuplot_command += "set ylabel 'Values'; ";
        gnuplot_command += "plot ";
        for (size_t i = 0; i < num_series_; ++i) {
            gnuplot_command += "'" + data_file_path_ + "' using 0:" + to_string(i + 1) + " with lines title '" + series_names_[i] + "'";
            if (i < num_series_ - 1) {
                gnuplot_command += ", ";
            }
        }
        gnuplot_command += "\"";
        int result = system(gnuplot_command.c_str());
        if (result != 0) {
            cout << "Failed to generate plot with Gnuplot. Command: " << gnuplot_command << endl;
        }
    }

private:
    bool data_has_been_added = false;
    const string plot_name_;
    const string data_file_path_;
    const string plot_file_path_;
    const vector<string> series_names_;
    const size_t num_series_;
    ofstream data_file_;
};
