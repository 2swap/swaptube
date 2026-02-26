#pragma once
// This file contains a number of singleton static variables
// which indicate whether we are running a smoketest.

static bool FOR_REAL = true; // Flag exposed to the project definition to disable sections of video
static bool SMOKETEST= false;// Overall smoketest flag

bool is_smoketest();
bool is_for_real();
void set_smoketest(bool smoketest);
void set_for_real(bool for_real);
bool rendering_on();
