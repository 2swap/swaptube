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
    CacheManager(const string& filename) : filename_(filename) { ReadCache(); }

    string hash_to_json_str(double hash) {
        ostringstream oss;
        oss << setprecision(17) << hash;
        return oss.str();
    }

    ~CacheManager() {
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
        if(delta == 0) return true;
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
    void AddOrUpdateEntry(double hash, double reverse_hash, const string& position, int suggestedMove) {
        if(suggestedMove < 1 || suggestedMove > C4_WIDTH) {
            throw runtime_error("Illegal cache move write: " + to_string(suggestedMove));
        }
        delete_entry(reverse_hash);

        json entry;
        entry["rep"] = position;
        entry["move"] = suggestedMove;

        cache_[hash_to_json_str(hash)] = entry;
        increment_delta();
    }

    // Add or update an entry given a steadystate
    void AddOrUpdateEntry(double hash, double reverse_hash, const string& position, const string& steadystate) {
        json entry;
        if(steadystate.size() != C4_WIDTH * C4_HEIGHT) throw runtime_error("Illegal cache steadystate write: " + steadystate);
        delete_entry(reverse_hash);
        entry["rep"] = position;
        entry["ss"] = steadystate;

        cache_[hash_to_json_str(hash)] = entry;
        increment_delta();
    }

    // Get the suggested move for a given hash if it exists in the cache
    bool GetSuggestedMoveIfExists(double hash, double reverse_hash, int& move, string& ss) {
        int ret_move = -1;
        int rev_move = -1;
        move = -1;
        string ret_ss = "";
        string rev_ss = "";
        ss = "";
        string ret_rep = "";
        string rev_rep = "";
        bool ret = GetSuggestedMoveIfExists_half(        hash, ret_rep, ret_move, ret_ss);
        bool rev = GetSuggestedMoveIfExists_half(reverse_hash, rev_rep, rev_move, rev_ss);

        if(ret && rev && hash != reverse_hash) {
            cout << "====== Stumbled upon double source of truth. Eliminating..." << endl;
            cout << hash << endl;
            cout << ret_rep << endl;
            cout << ret_move << endl;
            cout << ret_ss << endl;
            cout << reverse_hash << endl;
            cout << rev_rep << endl;
            cout << rev_move << endl;
            cout << rev_ss << endl;
            cout << "======" << endl;
            if(ret_ss == "" && rev_ss == "" && ret_move != 8-rev_move) {
                delete_entry(reverse_hash);
                delete_entry(hash);
            }
            else if(ret_ss == "" && rev_ss == "" && ret_move == 8-rev_move) delete_entry(reverse_hash);
            else if(ret_ss != "" && rev_ss != "") delete_entry(reverse_hash);
            else if(ret_ss == "" && rev_ss != "") delete_entry(hash);
            else if(ret_ss != "" && rev_ss == "") delete_entry(reverse_hash);
            else throw runtime_error("This should be unreachable");

            // Now that we have cleaned up the dupe truth, we should be able to call the function and expect success
            return GetSuggestedMoveIfExists(hash, reverse_hash, move, ss);
        }

        // Nothing found
        if(!ret && !rev) return false;

        move = ret ? ret_move : 8-rev_move;
        ss   = ret ? ret_ss   : (rev_ss == "" ? "" : reverse_ss(rev_ss));
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

    // Get the suggested move for a given hash if it exists in the cache
    bool GetSuggestedMoveIfExists_half(double hash, string& rep, int& move, string& ss) {
        json entry = GetEntry(hash);
        if (!entry.empty()) {
            rep = entry["rep"];
            move = entry.value("move", -1);
            ss = entry.value("ss", "");
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

CacheManager movecache("movecache.txt");
