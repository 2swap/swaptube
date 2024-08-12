#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include "../../misc/json.hpp"
using json = nlohmann::json;

class CacheManager {
public:
    CacheManager(const std::string& filename) : filename_(filename) {ReadCache();}

    // Read cache from a JSON file on disk
    bool ReadCache() {
        std::ifstream file(filename_);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open cache file for reading." << std::endl;
            return false;
        }

        try {
            file >> cache_;
            file.close();
            //std::cout << "Cache read from file: " << filename_ << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error reading cache from file: " << e.what() << std::endl;
            file.close();
            return false;
        }
    }

    // Write cache to a JSON file on disk
    bool WriteCache() {
        std::ofstream file(filename_);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open cache file for writing." << std::endl;
            return false;
        }

        try {
            file << cache_.dump(4); // Indented output with 4 spaces
            file.close();
            //std::cout << "Cache written to file: " << filename_ << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error writing cache to file: " << e.what() << std::endl;
            file.close();
            return false;
        }
    }

    // Add or update an entry in the cache
    void AddOrUpdateEntry(double hash, const std::string& position, int suggestedMove) {
        // Convert the double hash to a string with full precision
        std::ostringstream oss;
        oss << std::setprecision(16) << hash;
        std::string hashString = oss.str();

        json entry;
        entry["position"] = position;
        entry["suggestedMove"] = suggestedMove;

        cache_[hashString] = entry;
    }

    // Get the entry associated with a hash from the cache
    json GetEntry(double hash) {
        // Convert the double hash to a string with full precision
        std::ostringstream oss;
        oss << std::setprecision(16) << hash;
        std::string hashString = oss.str();

        if (cache_.count(hashString) > 0) {
            cout << "Getting entry " << hashString << endl;
            return cache_[hashString];
        } else {
            cout << "didnt find entry " << hashString << endl;
            return json::object(); // Entry not found in cache
        }
    }

    // Get the suggested move for a given hash if it exists in the cache
    int GetSuggestedMoveIfExists(double hash) {
        json entry = GetEntry(hash);
        if (!entry.empty() && entry.find("suggestedMove") != entry.end()) {
            return entry["suggestedMove"];
        } else {
            return -1; // Entry not found or no suggested move in cache
        }
    }

private:
    std::string filename_;
    json cache_;
};

CacheManager movecache("movecache.txt");
