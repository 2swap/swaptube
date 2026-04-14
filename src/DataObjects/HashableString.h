#pragma once

#include <string>
#include "GenericBoard.h"

class HashableString : public GenericBoard {
public:
    HashableString(const std::string& str);
    double type_specific_hash() override;
    void tick(const StateReturn& state);
    bool is_solution() override;
    std::unordered_set<GenericBoard*> get_children() override;
};
