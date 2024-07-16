#pragma once

#include "C4Board.h"
#include "SteadyState.cpp"
#include "JsonC4Cache.cpp"

Graph<C4Board>* graph_to_check_if_points_are_in = NULL;

std::array<std::string, C4_HEIGHT> ss_list = {
    " @22@| ",
    " 2112| ",
    " 1221| ",
    " 2112| ",
    " 2121|1",
    " 211211"
};
SteadyState ss_simple_weak(ss_list);

C4Board::C4Board(const C4Board& other) {
    // Copy the representation
    representation = other.representation;

    // Copy the bitboards
    red_bitboard = other.red_bitboard;
    yellow_bitboard = other.yellow_bitboard;
}

C4Board::C4Board(std::string representation) {
    fill_board_from_string(representation);
}

//this is in 0-indexed, upside down x and y land
int bitboard_at(Bitboard bb, int x, int y){
    return (bb >> x >> y*(C4_WIDTH+1))&1UL;
}

int C4Board::piece_code_at(int x, int y) const {
    return bitboard_at(red_bitboard, x, y) + (2*bitboard_at(yellow_bitboard, x, y));
}

void C4Board::print() const {
    std::cout << representation << std::endl;
    for(int y = 0; y < C4_HEIGHT; y++) {
        for(int x = 0; x < C4_WIDTH; x++) {
            std::cout << disk_col(piece_code_at(x, y)) << " ";
        }
        std::cout << std::endl;
    }
}

bool C4Board::is_legal(int x) const {
    return (((red_bitboard|yellow_bitboard) >> (x-1)) & 1) == 0;
}

int C4Board::random_legal_move() const {
    std::vector<int> legal_columns;

    for (int x = 1; x <= C4_WIDTH; ++x) {
        if(is_legal(x))
            legal_columns.push_back(x);
    }

    if (legal_columns.empty()) {
        return -1; // No legal columns available
    }

    int random_index = rand() % legal_columns.size();
    return legal_columns[random_index];
}

C4Result C4Board::who_won() const {
    const int v = C4_WIDTH;
    const int w = C4_WIDTH + 1; // there is a space of padding on the right of the bitboard
    const int x = C4_WIDTH + 2; // since otherwise horizontal wins would wrap walls
    if((yellow_bitboard|red_bitboard) == 140185576636287ul) // this wont be resilient to other board sizes...
        return TIE;

    for (int i = 0; i < 2; i++){
        const Bitboard b = i==0?red_bitboard:yellow_bitboard;
        if( (b & (b>>1) & (b>>(2*1)) & (b>>(3*1)))
         || (b & (b>>w) & (b>>(2*w)) & (b>>(3*w)))
         || (b & (b>>v) & (b>>(2*v)) & (b>>(3*v)))
         || (b & (b>>x) & (b>>(2*x)) & (b>>(3*x))) )
            return i==0?RED:YELLOW;
    }

    return INCOMPLETE;
}

bool C4Board::is_solution() {
    if(has_steady_state) return true;
    C4Result winner = who_won();
    return winner == RED || winner == YELLOW;
}

/*double unsignedLongToDouble(Bitboard value) {
    // Use type punning to reinterpret the bits of 'value' as a double
    double result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double C4Board::board_specific_hash() const {
    return unsignedLongToDouble(red_bitboard) + unsignedLongToDouble(yellow_bitboard)*2;
}*/

double C4Board::board_specific_hash() const {
    double a = 1;
    double hash_in_progress = 0;
    for (int y = 0; y < C4_HEIGHT; y++) {
        for (int x = 0; x < C4_WIDTH; x++) {
            hash_in_progress += a * piece_code_at(x, y);
            a *= 1.021813947;
        }
    }
    return hash_in_progress;
}

double C4Board::board_specific_reverse_hash() const {
    double a = 1;
    double hash_in_progress = 0;
    for (int y = 0; y < C4_HEIGHT; y++) {
        for (int x = 0; x < C4_WIDTH; x++) {
            hash_in_progress += a * piece_code_at(C4_WIDTH-1-x, y);
            a *= 1.021813947;
        }
    }
    return hash_in_progress;
}

void C4Board::fill_board_from_string(const std::string& rep)
{
    // Iterate through the moves and fill the board
    for (int i = 0; i < rep.size(); i++) {
        play_piece(rep[i]-'0');
    }
}

void C4Board::play_piece(int piece){
    if(piece < 0) {print(); std::cout << "uh oh " << representation << " " << piece << std::endl; exit(1);}
    if(hash != 0) {print(); std::cout << "oops " << representation << " " << piece << std::endl; exit(1);}
    if(piece > 0){
        if(!is_legal(piece)) {print(); std::cout << "gah " << representation << " " << piece << std::endl; exit(1);}
        int x = piece - 1; // convert from 1index to 0
        int y = 0;
        Bitboard bb = (red_bitboard | yellow_bitboard);
        for (y = C4_HEIGHT - 1; y >= 0; y--) {
            if (bitboard_at(bb, x, y) == 0) {
                Bitboard piece = 1UL << x << (y*(C4_WIDTH+1));
                if(is_reds_turn()) red_bitboard += piece;
                else yellow_bitboard += piece;
                break;
            }
        }
        representation+=std::to_string(piece);
    } else {
        representation+='0';
    }
}

C4Board C4Board::child(int piece) const{
    C4Board new_board(*this);
    new_board.hash = 0;
    new_board.play_piece(piece);
    return new_board;
}

C4Result C4Board::who_is_winning(int& work, bool verbose) {
    C4Result cachedResult;
    auto it = cache.find(representation);
    if (it != cache.end()) {
        if(verbose)std::cout << "Using cached result..." << std::endl;
        return it->second;
    }

    char command[150];
    std::sprintf(command, "echo %s | ~/Unduhan/Fhourstones/SearchGame", representation.c_str());
    if(verbose)std::cout << "Calling fhourstones on " << command << "... ";
    FILE* pipe = popen(command, "r");
    if (!pipe) {
        std::cout << "fhourstones error!" << std::endl;
        C4Board c4(representation);
        std::cout << std::setprecision (15) << c4.get_hash() << std::endl;
        exit(1);
    }
    char buffer[4096];
    std::string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 4096, pipe) != NULL) {
            result += buffer;
        }
    }
    pclose(pipe);

    C4Result gameResult;
    size_t workPos = result.find("work = ");
    if (workPos != std::string::npos) {
        work = std::stoi(result.substr(workPos + 7, result.find('\n', workPos) - workPos - 7));
    } else {
        work = -1; // Set work to a default value if not found
    }

    if (result.find("(=)") != std::string::npos) {
        if(verbose)std::cout << "Tie!" << std::endl;
        gameResult = TIE;
    } else if ((result.find("(+)") != std::string::npos) == is_reds_turn()) {
        if(verbose)std::cout << "Red!" << std::endl;
        gameResult = RED;
    } else {
        if(verbose)std::cout << "Yellow!" << std::endl;
        gameResult = YELLOW;
    }

    cache[representation] = gameResult;
    return gameResult;
}

void C4Board::add_all_winning_fhourstones(std::unordered_set<C4Board*>& neighbors){
    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            int work = -1;
            if(moved.who_is_winning(work) == RED){
                neighbors.insert(new C4Board(moved));
            }
        }
    }
}

int C4Board::get_instant_win() const{
    for (int x = 1; x <= C4_WIDTH; ++x){
        if(!is_legal(x)) continue;
        C4Result whowon = child(x).who_won();
        if(whowon == RED || whowon == YELLOW)
            return x;
    }
    return -1;
}

int C4Board::get_blocking_move() const{
    for (int x = 1; x <= C4_WIDTH; ++x){
        if(!is_legal(x)) continue;
        C4Result whowon = child(0).child(x).who_won();
        if(whowon == RED || whowon == YELLOW)
            return x;
    }
    return -1;
}

int C4Board::get_best_winning_fhourstones() {
    int lowest_work = INT_MAX;
    int lowest_work_move = -1;

    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            int work = -1;
            C4Result winner = moved.who_is_winning(work);
            if (winner == RED && work < lowest_work) {
                lowest_work = work;
                lowest_work_move = i;
            }
        }
    }
    return lowest_work_move;
}

std::vector<int> C4Board::get_winning_moves() const{
    std::vector<int> ret;
    for (int x = 1; x <= C4_WIDTH; x++) {
        if (is_legal(x)) {
            C4Board moved = child(x);
            int work = -1;
            C4Result winner = moved.who_is_winning(work);
            if (winner == RED) {
                ret.push_back(x);
            }
        }
    }
    return ret;
}

bool C4Board::is_reds_turn() const{
    return representation.size() % 2 == 0;
}

int C4Board::burst() const{
    int wm = get_instant_win();
    if(wm != -1){
        //std::cout << representation<<wm << " added for instawin" << std::endl;
        return wm;
    }

    std::vector<int> winning_columns = get_winning_moves();
    if (winning_columns.size() == 0){
        std::string representation2(representation);
        while(representation2 != ""){
            C4Board c4(representation2);
            std::cout << representation2 << ": " << std::setprecision (15) << c4.get_hash() << std::endl;
            representation2 = representation2.substr(0, representation2.size()-1);
        }
        std::cout << "Burst winning columns error!" << std::endl;
        exit(1);
    }

    // Add things already in the graph!
    //drop a red piece in each column and see if it is in the graph
    for (int i = 0; i < winning_columns.size(); ++i) {
        int x = winning_columns[i];
        if(graph_to_check_if_points_are_in->node_exists(child(x).get_hash())) {
            //std::cout << representation <<x<< " added since it is in the graph already" << std::endl;
            return x;
        }
    }

    // Next Priority: Test for easy steadystates!
    //drop a red piece in each column and see if it can make a steadystate
    int attempt = 200;
    for(int j = 0; j < 7; j++){
        for (int i = 0; i < winning_columns.size(); ++i) {
            int x = winning_columns[i];
            SteadyState ss;
            if(find_steady_state(child(x).representation, attempt, ss, true)){
                //std::cout << representation<<x << " added since a steadystate was found" << std::endl;
                return x;
            }
        }
        attempt *= 2;
    }

    // Recurse!
    for (int i = 0; i < winning_columns.size(); ++i) {
        int x = winning_columns[i];
        C4Board forcing = child(x);
        int bm = forcing.get_blocking_move();
        if(bm != -1){
            int recurse = forcing.child(bm).burst();
            if(recurse != -1) {
                std::cout << forcing.representation<<bm << " added recursively" << std::endl;
                return x;
            }
        }
    }

    return -1; // no easy line found... casework will be necessary :(
}

int C4Board::get_human_winning_fhourstones() {
    int ret = movecache.GetSuggestedMoveIfExists(get_hash());
    if(ret != -1) return ret;

    int b = burst();
    if(b != -1){
        //std::cout << representation <<b<< " added by burst" << std::endl;
        movecache.AddOrUpdateEntry(get_hash(), representation, b);
        return b;
    }

    std::vector<int> winning_columns = get_winning_moves();
    if (winning_columns.size() == 1) {
        // Single winning column
        char wc = winning_columns[0];
        //std::cout << representation<<wc << " added as the only winning move" << std::endl;
        movecache.AddOrUpdateEntry(get_hash(), representation, wc);
        return wc;
    } else if (winning_columns.size() == 0){
        //std::cout << "Get human winning fhourstones error!" << std::endl;
        exit(1);
    }

    int ret2 = movecache.GetSuggestedMoveIfExists(get_hash());
    if(ret2 != -1) return ret2;

    print();

    //std::cout << representation << " (" << get_hash() << ") has multiple winning columns. Please select one:" << std::endl;
    for (size_t i = 0; i < winning_columns.size(); i++) {
        std::cout << "Column " << winning_columns[i] << std::endl;
    }

    int choice;
    do {
        std::cout << "Enter your choice: ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cout << "ERROR -- You did not enter an integer" << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    } while (std::find(winning_columns.begin(), winning_columns.end(), choice) == winning_columns.end());

    movecache.AddOrUpdateEntry(get_hash(), representation, choice);
    movecache.WriteCache();
    return choice;
}

void C4Board::add_best_winning_fhourstones(std::unordered_set<C4Board*>& neighbors) {
    C4Board moved = child(get_human_winning_fhourstones());
    //std::cout << moved.representation << " added since it was selected" << std::endl;
    neighbors.insert(new C4Board(moved));
}

void C4Board::add_all_legal_children(std::unordered_set<C4Board*>& neighbors){
    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            neighbors.insert(new C4Board(moved));
        }
    }
}

void C4Board::add_all_good_children(std::unordered_set<C4Board*>& neighbors){
    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            // Check the move isn't giga dumb
            if(moved.get_instant_win() == -1){
                //std::cout << representation << ": attempting " << i << ". Result: " << moved.representation << " added since it is not dumb" << std::endl;
                neighbors.insert(new C4Board(moved));
            }
        }
    }
}

json C4Board::get_data() const {
    json data;  // Create a JSON object

    // Add the 'blurb' string to the JSON object
    data["blurb"] = blurb;

    // Create a nested JSON array for the 2D array steadystate.steadystate
    json steadystate_array;

    for (const auto& row : steadystate.steadystate) {
        json row_array;
        for (const auto& value : row) {
            char c = has_steady_state?value:' ';
            row_array.push_back(c);
        }
        steadystate_array.push_back(row_array);
    }

    // Add the nested array to the JSON object
    data["steadystate"] = steadystate_array;

    return data;
}

void C4Board::add_only_child_steady_state(const SteadyState& ss, std::unordered_set<C4Board*>& neighbors){
    int x = ss.query_steady_state(*this);
    C4Board moved = child(x);
    neighbors.insert(new C4Board(moved));
}

std::unordered_set<C4Board*> C4Board::get_children(){
    std::unordered_set<C4Board*> neighbors;

    if (is_solution()) {
        return neighbors;
    }

    switch (c4_branch_mode){
        case MANUAL:
        case FULL:
            add_all_legal_children(neighbors);
            break;
        case UNION_WEAK:
            add_all_winning_fhourstones(neighbors);
            break;
        case SIMPLE_WEAK:
            if(is_reds_turn()){
                add_only_child_steady_state(ss_simple_weak, neighbors);
            }else{
                add_all_legal_children(neighbors);
            }
            break;
        case TRIM_STEADY_STATES:
            if(!is_reds_turn()){ // if it's yellow's move
                int bm = get_blocking_move();
                if(bm != -1 && child(bm).get_instant_win() != -1) break; // if i cant stop an insta win
                SteadyState ss;
                bool found = find_steady_state(representation, 1, ss, true);
                if(found){
                    has_steady_state = true;
                    steadystate = ss;
                    //std::cout << "found a steady state!" << std::endl;
                    break;
                }
                else{
                    //std::cout << "Adding children as yellow!" << std::endl;
                    add_all_good_children(neighbors);
                }
            }else{ // red's move
                //std::cout << "Making a good move as red." << std::endl;
                add_best_winning_fhourstones(neighbors);
            }
            break;
    }
    return neighbors;
}

std::unordered_set<double> C4Board::get_children_hashes(){
    if(!children_hashes_initialized) {
        std::unordered_set<C4Board*> children = get_children();
        for (const auto& child : children) {
            children_hashes.insert(child->get_hash());
        }
        children_hashes_initialized = true;
    }
    return children_hashes;
}
