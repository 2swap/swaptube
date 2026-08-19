#include "TimelinePlan.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace std;

namespace {
ifstream plan_input;
ofstream plan_output;
string plan_path;
size_t next_entry = 0;
bool recording = false;
optional<int> open_count;
optional<int> declared_count;
string open_blurb;

string entry_description(const size_t index) {
    return "macroblock " + to_string(index + 1);
}

void finish_recording_entry() {
    if (!open_count) return;
    if (*open_count <= 0 && !declared_count) {
        plan_output.close();
        throw runtime_error("Smoketest found no render_microblock() calls for "
            + entry_description(next_entry) + ": " + open_blurb);
    }
    if (declared_count && *declared_count != *open_count) {
        cout << "WARNING: " << entry_description(next_entry) << " declares "
             << *declared_count << " microblocks, but smoketest observed " << *open_count
             << ": " << open_blurb << ". The render will use the declared count." << endl;
    }
    plan_output << declared_count.value_or(*open_count) << '\n';
    if (!plan_output) throw runtime_error("Could not write microblock plan: " + plan_path);
    next_entry++;
    open_count.reset();
    declared_count.reset();
    open_blurb.clear();
}
}

void initialize_timeline_plan(const string& path, const bool record) {
    plan_input.close();
    plan_input.clear();
    plan_output.close();
    plan_output.clear();
    if (path.empty()) throw invalid_argument("Timeline plan path cannot be empty");

    plan_path = path;
    next_entry = 0;
    open_count.reset();
    declared_count.reset();
    open_blurb.clear();
    recording = record;

    if (recording) {
        plan_output.open(plan_path, ios::trunc);
        if (!plan_output) throw runtime_error("Could not create timeline plan output file: " + plan_path);
    } else {
        plan_input.open(plan_path);
        if (!plan_input) throw runtime_error("Could not open microblock plan input file: " + plan_path);
    }
}

bool is_recording_microblock_plan() {
    return recording;
}

int begin_macroblock_plan_entry(const string& blurb, const optional<int> declared) {
    if (recording) {
        finish_recording_entry();
        open_count = 0;
        declared_count = declared;
        open_blurb = blurb;
        return declared.value_or(0);
    }

    optional<int> recorded_count;
    int entry_count;
    if (plan_input >> entry_count) {
        if (entry_count <= 0) {
            plan_input.close();
            throw runtime_error("Invalid non-positive microblock count in " + plan_path);
        }
        recorded_count = entry_count;
        next_entry++;
    } else if (!plan_input.eof()) {
        plan_input.close();
        throw runtime_error("Invalid microblock plan: " + plan_path);
    }

    // A declared count overrides the smoketest, so the render can deliberately differ.
    if (declared) return *declared;
    if (!recorded_count) {
        plan_input.close();
        throw runtime_error("Microblock plan ended before "
            + entry_description(next_entry) + ": " + blurb);
    }
    return *recorded_count;
}

void record_planned_microblock() {
    if (!open_count) {
        throw runtime_error("render_microblock() was called without an active macroblock during smoketest");
    }
    (*open_count)++;
}

void finalize_timeline_plan() {
    if (recording) {
        finish_recording_entry();
        plan_output.close();
        if (!plan_output) throw runtime_error("Could not finish microblock plan: " + plan_path);
        return;
    }

    int unused_count;
    if (plan_input >> unused_count) {
        plan_input.close();
        throw runtime_error("Microblock plan contains unused macroblock entries");
    }
    if (!plan_input.eof()) {
        plan_input.close();
        throw runtime_error("Invalid microblock plan: " + plan_path);
    }
    plan_input.close();
}
