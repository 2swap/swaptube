#include "Smoketest.h"

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>

using namespace std;

namespace {
bool SMOKETEST = false; // Overall smoketest flag

struct MacroblockRange { int first; int end; };
optional<MacroblockRange> RANGE; // When unset, every macroblock is in range
bool IN_RANGE = true;
}

bool is_smoketest() { return SMOKETEST; }
void set_smoketest(bool smoketest) { SMOKETEST = smoketest; }
bool macroblock_in_range() { return IN_RANGE; }
bool rendering_on() { return IN_RANGE && !SMOKETEST; }

void parse_macroblock_range(const string& arg) {
    if (arg == "all") return;
    int first, end;
    char extra;
    if (sscanf(arg.c_str(), "%d-%d%c", &first, &end, &extra) != 2) {
        throw runtime_error("Invalid macroblock range argument: " + arg + " (expected all or N-M)");
    }
    if (first < 0 || end <= first) {
        throw runtime_error("Invalid macroblock range: [" + to_string(first) + ", " + to_string(end) + ")");
    }
    RANGE = MacroblockRange{first, end};
    IN_RANGE = false;
}

void enter_macroblock(const int macroblock_index) {
    if (!RANGE) return;
    IN_RANGE = RANGE->first <= macroblock_index && macroblock_index < RANGE->end;
}
