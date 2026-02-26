#include "Smoketest.h"

// This file contains a number of singleton static variables
// which indicate whether we are running a smoketest.

bool is_smoketest() { return SMOKETEST; }
bool is_for_real() { return FOR_REAL; }
void set_smoketest(bool smoketest) { SMOKETEST = smoketest; }
void set_for_real(bool for_real) { FOR_REAL = for_real; }
bool rendering_on() { return FOR_REAL && !SMOKETEST; }
