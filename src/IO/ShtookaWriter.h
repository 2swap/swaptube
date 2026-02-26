#pragma once

#include <string>
#include <fstream>
#include <stdexcept>

using namespace std;

class ShtookaWriter {
private:
    ofstream shtooka_file;
public:
    ShtookaWriter();
    void add_shtooka_entry(const string& filename, const string& text);
    ~ShtookaWriter();
};
