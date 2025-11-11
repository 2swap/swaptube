#pragma once

#include <string>
#include <unordered_map>

class HashableString : public GenericBoard {
public:
    HashableString(const string& str) : GenericBoard(str) {}

    double type_specific_hash() override {
        std::hash<string> hasher;
        return static_cast<double>(hasher(representation));
    }
};
