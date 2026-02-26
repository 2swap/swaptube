#pragma once

#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace std;

class CacheManager {
public:
    CacheManager(const string& filename, int w, int h);
    string hash_to_json_str(double hash);
    ~CacheManager();

    // Read cache from a JSON file on disk
    bool ReadCache();

    // Write cache to a JSON file on disk
    bool WriteCache();

    // Add or update an entry in the cache
    void AddOrUpdateEntry(double hash, double reverse_hash, const std::string& position, int suggestedMove);

    // Add or update an entry given a steadystate
    void AddOrUpdateEntry(double hash, double reverse_hash, const string& position, const string& steadystate);

    // Get the suggested move for a given hash if it exists in the cache
    bool GetSuggestedMoveIfExists(double hash, double reverse_hash, int& move, string& ss);

private:
    string filename_;
    int c4_width;
    int c4_height;

    // Increment the delta and write to cache if threshold is exceeded
    void increment_delta();

    // Get the suggested move for a given hash if it exists in the cache
    bool GetSuggestedMoveIfExists_half(double hash, string& rep, int& move, string& ss);

    bool delete_entry(double hash);

    json GetEntry(double hash);

    json cache_;
    int delta;
};

// Singleton global instance of CacheManager
CacheManager& get_movecache();
