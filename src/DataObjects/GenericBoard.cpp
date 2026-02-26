#include "GenericBoard.h"

GenericBoard::GenericBoard(const string& s) : representation(s), hash(0), reverse_hash(0), reverse_hash_2(0), highlight_type(-1) {}
GenericBoard::GenericBoard() : representation(""), hash(0), reverse_hash(0), reverse_hash_2(0), highlight_type(-1) {}

std::unordered_set<double> GenericBoard::get_children_hashes() {
    std::unordered_set<GenericBoard*> kids = get_children();
    std::unordered_set<double> ret;
    for(GenericBoard* kid : kids){
        ret.insert(kid->get_hash());
        delete kid;
    }
    return ret;
}

void GenericBoard::reset_hashes() { hash = 0; reverse_hash = 0; reverse_hash_2 = 0; }
double GenericBoard::get_hash() {
    if(hash == 0)
        hash = type_specific_hash();
    return hash;
}
double GenericBoard::get_reverse_hash() {
    if(reverse_hash == 0)
        reverse_hash = type_specific_reverse_hash();
    return reverse_hash;
}
double GenericBoard::get_reverse_hash_2() {
    if(reverse_hash_2 == 0)
        reverse_hash_2 = type_specific_reverse_hash_2();
    return reverse_hash_2;
}

int GenericBoard::get_highlight_type() {
    if(highlight_type == -1)
        highlight_type = is_solution() ? 1 : 0;
    return highlight_type;
}

json GenericBoard::get_data() const {
    return json();
}

std::shared_ptr<Scene> GenericBoard::make_scene() const {
    return std::make_shared<LatexScene>(representation, 1);
}

// Which mirror side this node is on, if applicable.
int GenericBoard::which_side() const { return 0; }

double GenericBoard::type_specific_reverse_hash() {return -3.1415;}
double GenericBoard::type_specific_reverse_hash_2() {return -3.1415;}
