#pragma once

#include "C4Board.h"
#include "SteadyState.cpp"
#include "JsonC4Cache.cpp"
#include <string>

Graph<C4Board>* graph_to_check_if_points_are_in = NULL;

C4Board::C4Board(const C4Board& other) {
    // Copy the representation
    representation = other.representation;

    // Copy the bitboards
    red_bitboard = other.red_bitboard;
    yellow_bitboard = other.yellow_bitboard;

    steadystate = other.steadystate;
}

C4Board::C4Board() { }

C4Board::C4Board(string representation) {
    fill_board_from_string(representation);
}

C4Board::C4Board(string representation, shared_ptr<SteadyState> ss) : steadystate(ss) {
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
    cout << representation << endl;
    for(int y = 0; y < C4_HEIGHT; y++) {
        for(int x = 0; x < C4_WIDTH; x++) {
            cout << disk_col(piece_code_at(x, y)) << " ";
        }
        cout << endl;
    }
}

bool C4Board::is_legal(int x) const {
    return (((red_bitboard|yellow_bitboard) >> (x-1)) & 1) == 0;
}

int C4Board::random_legal_move() const {
    vector<int> legal_columns;

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
    memcpy(&result, &value, sizeof(result));
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

void C4Board::fill_board_from_string(const string& rep)
{
    // Iterate through the moves and fill the board
    for (int i = 0; i < rep.size(); i++) {
        play_piece(rep[i]-'0');
    }
}

void C4Board::play_piece(int piece){
    if(piece < 0) {
        print();
        steadystate->print();
        failout("Attempted playing a piece in an illegal column. Representation: " + representation + ", piece: " + to_string(piece));
    }
    if(hash != 0) {print(); cout << "oops " << representation << " " << piece << endl; exit(1);}
    if(piece > 0){
        if(!is_legal(piece)) {print(); cout << "gah " << representation << " " << piece << endl; exit(1);}
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
        representation+=to_string(piece);
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
        if(verbose)cout << "Using cached result..." << endl;
        return it->second;
    }

    char command[150];
    sprintf(command, "echo %s | ~/Unduhan/Fhourstones/SearchGame", representation.c_str());
    if(verbose)cout << "Calling fhourstones on " << command << "... ";
    FILE* pipe = popen(command, "r");
    if (!pipe) {
        C4Board c4(representation);
        cout << setprecision (15) << c4.get_hash() << endl;
        failout("fhourstones error!");
    }
    char buffer[4096];
    string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 4096, pipe) != NULL) {
            result += buffer;
        }
    }
    pclose(pipe);

    C4Result gameResult;
    size_t workPos = result.find("work = ");
    if (workPos != string::npos) {
        work = stoi(result.substr(workPos + 7, result.find('\n', workPos) - workPos - 7));
    } else {
        work = -1; // Set work to a default value if not found
    }

    if (result.find("(=)") != string::npos) {
        if(verbose)cout << "Tie!" << endl;
        gameResult = TIE;
    } else if ((result.find("(+)") != string::npos) == is_reds_turn()) {
        if(verbose)cout << "Red!" << endl;
        gameResult = RED;
    } else {
        if(verbose)cout << "Yellow!" << endl;
        gameResult = YELLOW;
    }

    cache[representation] = gameResult;
    return gameResult;
}

void C4Board::add_all_winning_fhourstones(unordered_set<C4Board*>& neighbors){
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

vector<int> C4Board::get_winning_moves() const{
    vector<int> ret;
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
        //cout << representation<<wm << " added for instawin" << endl;
        return wm;
    }

    vector<int> winning_columns = get_winning_moves();
    if (winning_columns.size() == 0){
        string representation2(representation);
        while(representation2 != ""){
            C4Board c4(representation2);
            cout << representation2 << ": " << setprecision (15) << c4.get_hash() << endl;
            representation2 = representation2.substr(0, representation2.size()-1);
        }
        cout << "Burst winning columns error!" << endl;
        exit(1);
    }

    // Add things already in the graph!
    //drop a red piece in each column and see if it is in the graph
    for (int i = 0; i < winning_columns.size(); ++i) {
        int x = winning_columns[i];
        if(graph_to_check_if_points_are_in->node_exists(child(x).get_hash())) {
            //cout << representation <<x<< " added since it is in the graph already" << endl;
            return x;
        }
    }

    // Next Priority: Test for easy steadystates!
    //drop a red piece in each column and see if it can make a steadystate
    int attempt = 200;
    for(int j = 0; j < 7; j++){
        for (int i = 0; i < winning_columns.size(); ++i) {
            int x = winning_columns[i];
            C4Board xth_child = child(x);
            xth_child.print();
            shared_ptr<SteadyState> ss = find_steady_state(xth_child.representation, attempt);
            if(ss != nullptr){
                //cout << representation<<x << " added since a steadystate was found" << endl;
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
                cout << forcing.representation<<bm << " added recursively" << endl;
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
        //cout << representation <<b<< " added by burst" << endl;
        movecache.AddOrUpdateEntry(get_hash(), representation, b);
        return b;
    }

    vector<int> winning_columns = get_winning_moves();
    if (winning_columns.size() == 1) {
        // Single winning column
        char wc = winning_columns[0];
        //cout << representation<<wc << " added as the only winning move" << endl;
        movecache.AddOrUpdateEntry(get_hash(), representation, wc);
        return wc;
    } else if (winning_columns.size() == 0){
        //cout << "Get human winning fhourstones error!" << endl;
        exit(1);
    }

    int ret2 = movecache.GetSuggestedMoveIfExists(get_hash());
    if(ret2 != -1) return ret2;

    print();

    //cout << representation << " (" << get_hash() << ") has multiple winning columns. Please select one:" << endl;
    for (size_t i = 0; i < winning_columns.size(); i++) {
        cout << "Column " << winning_columns[i] << endl;
    }

    int choice;
    do {
        cout << "Enter your choice: ";
        cin >> choice;
        if (cin.fail()) {
            cout << "ERROR -- You did not enter an integer" << endl;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }
    } while (find(winning_columns.begin(), winning_columns.end(), choice) == winning_columns.end());

    movecache.AddOrUpdateEntry(get_hash(), representation, choice);
    movecache.WriteCache();
    return choice;
}

void C4Board::add_all_legal_children(unordered_set<C4Board*>& neighbors){
    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            neighbors.insert(new C4Board(moved));
        }
    }
}

void C4Board::add_all_good_children(unordered_set<C4Board*>& neighbors){
    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            // Check the move isn't giga dumb
            if(moved.get_instant_win() == -1){
                //cout << representation << ": attempting " << i << ". Result: " << moved.representation << " added since it is not dumb" << endl;
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

    for (const auto& row : steadystate->steadystate) {
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

void C4Board::add_only_child_steady_state(unordered_set<C4Board*>& neighbors){
    if(steadystate == nullptr)
        failout("Querying a null steadystate!");
    int x = steadystate->query_steady_state(*this);
    C4Board moved = child(x);
    neighbors.insert(new C4Board(moved));
}

unordered_set<C4Board*> C4Board::get_children(){
    unordered_set<C4Board*> neighbors;

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
                add_only_child_steady_state(neighbors);
            }else{
                add_all_legal_children(neighbors);
            }
            break;
        case TRIM_STEADY_STATES:
            if(!is_reds_turn()){ // if it's yellow's move
                int bm = get_blocking_move();
                if(bm != -1 && child(bm).get_instant_win() != -1) break; // if i cant stop an insta win
                shared_ptr<SteadyState> ss = find_steady_state(representation, 1);
                if(ss != nullptr){
                    has_steady_state = true;
                    steadystate = ss;
                    //cout << "found a steady state!" << endl;
                    break;
                }
                else{
                    //cout << "Adding children as yellow!" << endl;
                    add_all_good_children(neighbors);
                }
            }else{ // red's move
                C4Board moved = child(get_human_winning_fhourstones());
                //cout << moved.representation << " added since it was selected" << endl;
                neighbors.insert(new C4Board(moved));
            }
            break;
    }
    return neighbors;
}

unordered_set<double> C4Board::get_children_hashes(){
    if(!children_hashes_initialized) {
        unordered_set<C4Board*> children = get_children();
        for (const auto& child : children) {
            children_hashes.insert(child->get_hash());
        }
        children_hashes_initialized = true;
    }
    return children_hashes;
}

string replerp(const string& b1, const string& b2, double w) {
    if (b1.find(b2) == 0 || b2.find(b1) == 0) {
        // One string begins with the other
        int range = b2.size() - b1.size();
        return (b1.size() > b2.size() ? b1 : b2).substr(0, round(b1.size() + w * range));
    } else {
        // Neither string begins with the other
        int common_prefix_len = 0;
        while (b1[common_prefix_len] == b2[common_prefix_len]) {
            common_prefix_len++;
        }

        string common_prefix = b1.substr(0, common_prefix_len);
        if(w<0.5) return replerp(b1, common_prefix, w*2);
        else return replerp(common_prefix, b2, (w-.5)*2);
    }
}

C4Board c4lerp(C4Board b1, C4Board b2, double w){
    string representation = replerp(b1.representation, b2.representation, smoother2(w));
    C4Board transition(representation);
    return transition;
}

// Unit test for replerp function
void replerp_ut() {
    string b1 = "12345";
    string b2 = "1";
    bool pass = true;
    pass &= replerp(b1, b2, 0.) == "12345";
    pass &= replerp(b1, b2, 0.25) == "1234";
    pass &= replerp(b1, b2, 0.5) == "123";
    pass &= replerp(b1, b2, 0.75) == "12";
    pass &= replerp(b1, b2, 1.) == "1";

    pass &= replerp(b2, b1, 0.) == "1";
    pass &= replerp(b2, b1, 0.25) == "12";
    pass &= replerp(b2, b1, 0.5) == "123";
    pass &= replerp(b2, b1, 0.75) == "1234";
    pass &= replerp(b2, b1, 1.) == "12345";
    if (!pass) {
        cout << "replerp_ut - Case 1: Failed." << endl;
        exit(1);
    }

    string b3 = "abc";
    string b4 = "de";
    pass = true;
    pass &= replerp(b3, b4, 0.) == "abc";
    pass &= replerp(b3, b4, 0.2) == "ab";
    pass &= replerp(b3, b4, 0.4) == "a";
    pass &= replerp(b3, b4, 0.6) == "";
    pass &= replerp(b3, b4, 0.8) == "d";
    pass &= replerp(b3, b4, 1.) == "de";

    pass &= replerp(b4, b3, 0.) == "de";
    pass &= replerp(b4, b3, 0.2) == "d";
    pass &= replerp(b4, b3, 0.4) == "";
    pass &= replerp(b4, b3, 0.6) == "a";
    pass &= replerp(b4, b3, 0.8) == "ab";
    pass &= replerp(b4, b3, 1.) == "abc";
    if (!pass) {
        cout << "replerp_ut - Case 2: Failed." << endl;
        exit(1);
    }
}
