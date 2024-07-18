#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <bitset>
#include <cassert>
#include "VectorOperations.cpp"

class PieceSet {
public:
    PieceSet(unsigned long int bit_representation) : bit_representation(bit_representation), size(set_size(bit_representation)) {}

    PieceSet(const vector<int>& effect) : bit_representation(set_bit_representation(effect)), size(set_size(set_bit_representation(effect))) {}

    void print() const {
        cout << "PieceSet bit representation: " << bitset<64>(bit_representation) << endl;
    }

    size_t get_size() const {
        return size;
    }

    unsigned long int get_bit_representation() const {
        return bit_representation;
    }

private:
    const size_t size;
    const unsigned long int bit_representation;

    unsigned long int set_bit_representation(const vector<int>& effect) const {
        unsigned long int bits = 0u;
        for (size_t i = 0; i < effect.size(); ++i) {
            if (effect[i] != i) {
                bits |= 1ul << i;
            }
        }
        return bits;
    }

    int set_size(unsigned long int bit_representation) const {
        return bitset<64>(bit_representation).count();
    }
};

PieceSet get_intersection(const PieceSet& ps1, const PieceSet& ps2) {
    return PieceSet(ps1.get_bit_representation() & ps2.get_bit_representation());
}
void test_get_intersection() {
    PieceSet ps1(0b1010);
    PieceSet ps2(0b1100);
    PieceSet expected(0b1000);

    PieceSet result = get_intersection(ps1, ps2);
    assert(result.get_bit_representation() == expected.get_bit_representation());
}

bool does_intersect(const PieceSet& ps1, const PieceSet& ps2) {
    return get_intersection(ps1, ps2).get_bit_representation() != 0ul;
}
void test_does_intersect() {
    PieceSet ps1(0b1010);
    PieceSet ps2(0b1100);
    PieceSet ps3(0b0010);

    assert(does_intersect(ps1, ps2) == true);
    assert(does_intersect(ps1, ps3) == true);
    assert(does_intersect(ps2, ps3) == false);
}

bool is_equal(const PieceSet& ps1, const PieceSet& ps2) {
    return ps1.get_bit_representation() == ps2.get_bit_representation();
}
void test_is_equal() {
    PieceSet ps1(0b1010);
    PieceSet ps2(0b1010);
    PieceSet ps3(0b1110);

    assert(is_equal(ps1, ps2) == true);
    assert(is_equal(ps1, ps3) == false);
}

bool is_subset_of(const PieceSet& ps1, const PieceSet& ps2) {
    return is_equal(get_intersection(ps1, ps2), ps1);
}
void test_is_subset_of() {
    PieceSet ps1(0b1010);
    PieceSet ps2(0b1110);
    PieceSet ps3(0b0010);

    assert(is_subset_of(ps1, ps2) == true);
    assert(is_subset_of(ps3, ps1) == true);
    assert(is_subset_of(ps2, ps1) == false);
}

PieceSet image(const PieceSet& ps, const vector<int>& effect) {
    unsigned long int permuted_bits = 0ul;
    unsigned long int bit_representation = ps.get_bit_representation();
    for (size_t i = 0; i < effect.size(); ++i) {
        if (bit_representation & (1ul << effect[i])) {
            permuted_bits |= 1ul << i;
        }
    }
    return PieceSet(permuted_bits);
}
void test_image() {
    PieceSet ps(0b0011);
    std::vector<int> effect = {0, 3, 1, 2};
    PieceSet expected(0b0101); // should be the same after permutation

    PieceSet result = image(ps, effect);
    assert(result.get_bit_representation() == expected.get_bit_representation());
}

PieceSet preimage(const PieceSet& ps, const vector<int>& effect) {
    return image(ps, ~effect);
}
void test_preimage() {
    PieceSet ps(0b0011); // bits: 0011
    std::vector<int> effect = {1, 0, 3, 2}; // swap positions 0 and 1, and 2 and 3
    std::vector<int> inverse_effect = {1, 0, 3, 2}; // inverse effect should be the same for a simple swap
    PieceSet expected(0b0011); // should be the same after inverse permutation

    PieceSet result = preimage(ps, inverse_effect);
    assert(result.get_bit_representation() == expected.get_bit_representation());
}

PieceSet get_full_pieceset(int size){
    unsigned long int bits = 0u;
    for(int i = 0; i < size; i++){
        bits |= 1ul << i;
    }
    return PieceSet(bits);
}
void test_get_full_pieceset() {
    PieceSet expected(0b1111); // bits: 1111 for size 4

    PieceSet result = get_full_pieceset(4);
    assert(result.get_bit_representation() == expected.get_bit_representation());
}

void pieceset_unit_tests() {
    test_get_intersection();
    test_does_intersect();
    test_is_equal();
    test_is_subset_of();
    test_image();
    test_preimage();
    test_get_full_pieceset();

    std::cout << "All tests passed!" << std::endl;
}
