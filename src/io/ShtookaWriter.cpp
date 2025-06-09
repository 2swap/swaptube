#pragma once

using namespace std;


class ShtookaWriter {
private:
    ofstream shtooka_file;
public:
    ShtookaWriter() {
        shtooka_file.open(PATH_MANAGER.record_list_path);
        if (!shtooka_file.is_open()) throw runtime_error("Error opening recorder list: " + PATH_MANAGER.record_list_path);
    }

    void add_shtooka_entry(const string& filename, const string& text) {
        if (!shtooka_file.is_open()) throw runtime_error("Shtooka file is not open. Cannot add entry.");
        shtooka_file << filename << "\t" << text << "\n";
    }

    ~ShtookaWriter() {
        if (shtooka_file.is_open()) shtooka_file.close();
    }
};
