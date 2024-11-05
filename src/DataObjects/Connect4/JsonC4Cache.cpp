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
    CacheManager(const string& filename) : filename_(filename) {ReadCache();}
    string hash_to_json_str(double hash) {
        ostringstream oss;
        oss << setprecision(16) << hash;
        return oss.str();
    }

    ~CacheManager(){
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
            file.close();
            //cout << "Cache read from file: " << filename_ << endl;
            return true;
        } catch (const exception& e) {
            cerr << "Error reading cache from file: " << e.what() << endl;
            file.close();
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
            file << cache_.dump(4); // Indented output with 4 spaces
            file.close();
            //cout << "Cache written to file: " << filename_ << endl;
            return true;
        } catch (const exception& e) {
            cerr << "Error writing cache to file: " << e.what() << endl;
            file.close();
            return false;
        }
    }

    // Add or update an entry in the cache
    void AddOrUpdateEntry(double hash, const string& position, int suggestedMove) {
        json entry;
        entry["position"] = position;
        entry["suggestedMove"] = suggestedMove;

        cache_[hash_to_json_str(hash)] = entry;
    }

    // Get the suggested move for a given hash if it exists in the cache
    int GetSuggestedMoveIfExists(double hash, double reverse_hash) {
        int ret = GetSuggestedMoveIfExists_half(hash);
        int rev = GetSuggestedMoveIfExists_half(reverse_hash);
        if(rev != -1) rev = 8 - rev;
        if(ret != -1 && rev != -1 && rev != ret) {
            throw runtime_error("Double source of truth on board with hash " + to_string(hash) + " and reverse-hash " + to_string(reverse_hash));
        }
        if(ret != -1) return ret;
        else if(rev != -1) return rev;
        return -1;
    }

private:
    // Get the suggested move for a given hash if it exists in the cache
    int GetSuggestedMoveIfExists_half(double hash) {
        json entry = GetEntry(hash);
        if (!entry.empty() && entry.find("suggestedMove") != entry.end()) {
            cout << "Got entry " << entry["suggestedMove"] << " for hash " << hash << endl;
            return entry["suggestedMove"];
        } else {
            return -1; // Entry not found or no suggested move in cache
        }
    }

    // Get the entry associated with a hash from the cache
    json GetEntry(double hash) {
        string hashString = hash_to_json_str(hash);

        if (cache_.count(hashString) > 0) {
            cout << "Getting entry " << hashString << endl;
            return cache_[hashString];
        } else {
            cout << "didnt find entry " << hashString << endl;
            return json::object(); // Entry not found in cache
        }
    }

    string filename_;
    json cache_;
};

CacheManager movecache("movecache.txt");
