#include "Smoketest.h"

namespace {
bool FOR_REAL = true; // Flag exposed to the project definition to disable sections of video
bool SMOKETEST = false; // Overall smoketest flag
}

bool is_smoketest() { return SMOKETEST; }
bool is_for_real() { return FOR_REAL; }
void set_smoketest(bool smoketest) { SMOKETEST = smoketest; }
void set_for_real(bool for_real) { FOR_REAL = for_real; }
bool rendering_on() { return FOR_REAL && !SMOKETEST; }
