#pragma once

#include <vector>
#include <cstring>
#include <cmath>
#include <set>

class GenericBoard{
public:
    virtual void print() const = 0;
    virtual bool is_solution() = 0;
    virtual double board_specific_hash() const = 0;
    virtual int symmetry_class() = 0;
    virtual double reverse_hash() = 0;

    double get_hash() {
        if(hash != 0)
            return hash;
        hash = board_specific_hash();
        return hash;
    }

protected:
    double hash = 0;
};
