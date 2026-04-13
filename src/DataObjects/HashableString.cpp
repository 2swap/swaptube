#include "HashableString.h"
#include <functional>

HashableString::HashableString(const std::string& str) : GenericBoard(str) {}

double HashableString::type_specific_hash() {
    std::hash<std::string> hasher;
    return static_cast<double>(hasher(representation));
}

bool HashableString::is_solution() {
    return false;
}

std::unordered_set<GenericBoard*> HashableString::get_children() {
    return {};
}
