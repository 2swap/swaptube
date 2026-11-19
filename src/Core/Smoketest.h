#pragma once
// Accessors for the process-wide flags which indicate whether we are running a smoketest,
// and whether the project has flagged the current section to be rendered.

bool is_smoketest();
bool is_for_real();
void set_smoketest(bool smoketest);
void set_for_real(bool for_real);
bool rendering_on();
