#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

class FhourstonesCache {
public:
    FhourstonesCache(const string& filename);

    string hash_to_json_str(double hash);

    ~FhourstonesCache();

    // Read cache from a JSON file on disk
    bool ReadCache();

    // Write cache to a JSON file on disk
    bool WriteCache();

    // Add or update an entry in the cache
    void AddOrUpdateEntry(double hash, double reverse_hash, const string& position, const string& winner);

    // Get the cached entry for a given hash, considering reverse-hash symmetry
    bool GetEntryIfExists(double hash, double reverse_hash, string& winner);

private:
    // Increment the delta and write to cache if threshold is exceeded
    void increment_delta();

    // Get the cached entry for a single hash
    bool GetEntryIfExists_half(double hash, string& winner);

    bool delete_entry(double hash);

    json GetEntry(double hash);

    string filename_;
    json cache_;
    int delta = 0;
};

// Singleton global instance of CacheManager
FhourstonesCache& get_fhourstonescache();
