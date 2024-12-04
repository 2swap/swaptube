#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include "../../misc/json.hpp"
using json = nlohmann::json;

class FhourstonesCache {
public:
    FhourstonesCache(const string& filename) : filename_(filename) { ReadCache(); }

    string hash_to_json_str(double hash) {
        ostringstream oss;
        oss << setprecision(16) << hash;
        return oss.str();
    }

    ~FhourstonesCache() {
        WriteCache();
    }

    // Read cache from a JSON file on disk
    bool ReadCache() {
        ifstream file(filename_);
        if (!file.is_open()) {
            cerr << "Error: Could not open cache file for reading." << endl;
            return false;
        }

        try {
            file >> cache_;
            return true;
        } catch (const exception& e) {
            cerr << "Error reading cache from file: " << e.what() << endl;
            return false;
        }
    }

    // Write cache to a JSON file on disk
    bool WriteCache() {
        ofstream file(filename_);
        if (!file.is_open()) {
            cerr << "Error: Could not open cache file for writing." << endl;
            return false;
        }

        try {
            file << cache_.dump(0); // unindented output with 0 spaces
            cout << "Cache written to file: " << filename_ << endl;
            return true;
        } catch (const exception& e) {
            cerr << "Error writing cache to file: " << e.what() << endl;
            return false;
        }
    }

    // Add or update an entry in the cache
    void AddOrUpdateEntry(double hash, double reverse_hash, const string& position, const string& winner) {
        if (winner != "RED" && winner != "YELLOW" && winner != "TIE") {
            throw runtime_error("Invalid winner value: " + winner);
        }
        delete_entry(reverse_hash);

        json entry;
        entry["rep"] = position;
        entry["winner"] = winner;

        cache_[hash_to_json_str(hash)] = entry;
        increment_delta();
    }

    // Get the cached entry for a given hash, considering reverse-hash symmetry
    bool GetEntryIfExists(double hash, double reverse_hash, string& winner) {
        string forward_winner, reverse_winner;
        bool found_forward = GetEntryIfExists_half(hash, forward_winner);
        bool found_reverse = GetEntryIfExists_half(reverse_hash, reverse_winner);

        // Resolve potential conflicts in symmetric entries
        if (found_forward && found_reverse && hash != reverse_hash) {
            if (forward_winner != reverse_winner) {
                cerr << "Conflict detected between symmetric entries. Eliminating both." << endl;
                delete_entry(hash);
                delete_entry(reverse_hash);
                return false;
            }
            // Consistency across symmetric entries; return forward hash as primary
            winner = forward_winner;
        } else if (found_forward) {
            winner = forward_winner;
        } else if (found_reverse) {
            winner = reverse_winner;
        } else {
            return false;
        }

        return true;
    }

private:
    // Increment the delta and write to cache if threshold is exceeded
    void increment_delta() {
        delta++;
        if (delta > 10000) {
            WriteCache();
            delta = 0;
        }
    }

    // Get the cached entry for a single hash
    bool GetEntryIfExists_half(double hash, string& winner) {
        json entry = GetEntry(hash);
        if (!entry.empty()) {
            winner = entry["winner"];
            return true;
        }
        return false; // Entry not found or no suggested move in cache
    }

    bool delete_entry(double hash) {
        string hashKey = hash_to_json_str(hash);
        if (cache_.contains(hashKey)) {
            cache_.erase(hashKey);
            return true; // Successfully deleted
        }
        return false; // Entry not found
    }

    json GetEntry(double hash) {
        string hashString = hash_to_json_str(hash);
        if (cache_.count(hashString) > 0) {
            return cache_[hashString];
        }
        return json::object(); // Entry not found in cache
    }

    string filename_;
    json cache_;
    int delta = 0;
};

FhourstonesCache fhourstonesCache("fhourstonescache.txt");
