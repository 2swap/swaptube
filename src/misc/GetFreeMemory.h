#pragma once

#include <sys/sysinfo.h>
#include <stdexcept>
#include <iostream>

int long get_free_memory() {
    struct sysinfo memInfo;

     if (sysinfo(&memInfo) != 0) {
         perror("sysinfo");
         throw runtime_error("Unable to call sysinfo to determine system memory");
     }

     // Free memory in mb
     double free_memory = static_cast<double>(memInfo.freeram) * memInfo.mem_unit / square(1024);

     return free_memory;
}
