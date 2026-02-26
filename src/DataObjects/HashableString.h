#pragma once

#include <string>
#include "GenericBoard.h"

class HashableString : public GenericBoard {
public:
    HashableString(const std::string& str);
    double type_specific_hash() override;
};
