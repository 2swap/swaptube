#include <string>
#include <unordered_set>
#include <sstream>
#include <cstdio>
#include <stdexcept>
#include "HashableString.h"
#include "../IO/IoHelpers.h"

using namespace std;

class PacmanPackage : public HashableString {
public:
    PacmanPackage(const string& str) : HashableString(str) {}

    bool is_solution() override {
        // Run pacman to see if the package is installed
        string cmd = "pacman -Qi " + representation + " > /dev/null 2>&1";
        int result = system(cmd.c_str());
        return result == 0;
    }

    unordered_set<GenericBoard*> get_children() override {
        unordered_set<GenericBoard*> children;
        // Run pacman to get the dependencies of the package
        string cmd = "pacman -Qi " + representation + " | grep 'Depends On' | cut -d ':' -f 2";
        FILE* pipe = portable_popen(cmd.c_str(), "r");
        if (!pipe) {
            throw runtime_error("pacman error!");
        }
        char buffer[128];
        string result = "";
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL) {
                result += buffer;
            }
        }
        portable_pclose(pipe);

        // Parse the dependencies from the result string
        stringstream ss(result);
        string pkg;
        while (ss >> pkg) {
            if (pkg != "None") {
                // Create a new PacmanPackage instance for each dependency
                children.insert(new PacmanPackage(pkg));
            }
        }
        return children;
    }
};
