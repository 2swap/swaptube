#include "ShtookaWriter.h"

ShtookaWriter::ShtookaWriter() {
    const string record_list_path = "io_in/record_list.tsv";
    shtooka_file.open(record_list_path);
    if (!shtooka_file.is_open()) throw runtime_error("Error opening recorder list: " + record_list_path);
}

void ShtookaWriter::add_shtooka_entry(const string& filename, const string& text) {
    if (!shtooka_file.is_open())
        throw runtime_error("Shtooka file is not open. Cannot add entry.");
    shtooka_file << filename << "\t" << text << "\n";
}

ShtookaWriter::~ShtookaWriter() {
    if (shtooka_file.is_open()) shtooka_file.close();
}
