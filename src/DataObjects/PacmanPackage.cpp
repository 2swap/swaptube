#include <string>
#include <unordered_set>
#include <sstream>
#include <cstdio>
#include <stdexcept>
#include "HashableString.h"

using namespace std;

class PacmanPackage : public HashableString {
public:
    PacmanPackage(const string& str) : HashableString(str) {}

    bool is_solution() override {
        // Build the pacman command
        string command = "pacman -Qi " + representation;
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            throw runtime_error("Failed to open pipe for pacman command.");
        }

        bool explicitlyInstalled = false;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            string line(buffer);
            // Look for the "Install Reason" line
            if (line.find("Install Reason") != string::npos) {
                if (line.find("Explicitly installed") != string::npos) {
                    explicitlyInstalled = true;
                }
                break; // Found the field; no need to continue
            }
        }
        pclose(pipe);
        return explicitlyInstalled;
    }

    unordered_set<GenericBoard*> get_children() override {
        unordered_set<GenericBoard*> children;
        // Build the pacman command
        string command = "pacman -Qi " + representation;
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            throw runtime_error("Failed to open pipe for pacman command.");
        }

        char buffer[4096];
        bool readingDeps = false;
        string dependencies;

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            string line(buffer);
            // If we haven't yet started collecting dependencies,
            // look for the "Required By" header.
            if (!readingDeps) {
                if (line.find("Required By") != string::npos) {
                    readingDeps = true;
                    size_t colonPos = line.find(":");
                    if (colonPos != string::npos) {
                        dependencies += line.substr(colonPos + 1);
                    }
                }
            } else {
                // Once in dependency collection mode, check if the line is indented.
                if (!line.empty() && isspace(line[0])) {
                    dependencies += " " + line;
                } else {
                    // If the line is not indented, the dependency block has ended.
                    break;
                }
            }
        }
        pclose(pipe);

        // Split the accumulated dependencies string into individual package names.
        istringstream iss(dependencies);
        string pkg;
        while (iss >> pkg) {
            if (pkg != "None") {
                // Create a new PacmanPackage instance for each dependency
                children.insert(new PacmanPackage(pkg));
            }
        }
        return children;
    }
};

