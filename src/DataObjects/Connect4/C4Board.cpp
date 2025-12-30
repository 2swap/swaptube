#pragma once

#include "C4Board.h"
#include "SteadyState.cpp"
#include "JsonC4Cache.cpp"
#include "FhourstonesCache.cpp"
#include "../Graph.cpp"
#include <string>
#include <vector>
#include <algorithm>

Graph* graph_to_check_if_points_are_in = NULL;

C4Board::C4Board() { }

C4Board::C4Board(const string& rep) {
    fill_board_from_string(rep);
}

C4Board::C4Board(const string& rep, shared_ptr<SteadyState> ss) : steadystate(ss) {
    fill_board_from_string(rep);
}

// Returns a bitboard with bits set for all winning discs
Bitboard C4Board::winning_discs() const {
    Bitboard ret = 0ul;
    int vert_slide = C4_WIDTH + 1;
    int diag_slide = C4_WIDTH;
    int back_slide = C4_WIDTH + 2;
    for(int i = 0; i < 2; i++){
        const Bitboard b = i==0?red_bitboard:yellow_bitboard;
        Bitboard win_bits = 0ul;
        Bitboard temp;

        // Horizontal
        temp = b & (b>>1);
        temp &= (temp>>2);
        ret |= temp | (temp<<1) | (temp<<2) | (temp<<3);

        // Vertical
        temp = b & (b>>vert_slide);
        temp &= (temp>>(2*vert_slide));
        ret |= temp | (temp<<vert_slide) | (temp<<(2*vert_slide)) | (temp<<(3*vert_slide));

        // Diagonal /
        temp = b & (b>>diag_slide);
        temp &= (temp>>(2*diag_slide));
        ret |= temp | (temp<<diag_slide) | (temp<<(2*diag_slide)) | (temp<<(3*diag_slide));

        // Diagonal backslash
        temp = b & (b>>back_slide);
        temp &= (temp>>(2*back_slide));
        ret |= temp | (temp<<back_slide) | (temp<<(2*back_slide)) | (temp<<(3*back_slide));
    }
    return ret;
}

// Returns 0 if the spot is empty, 1 if red, 2 if yellow
int C4Board::piece_code_at(int x, int y) const {
    return bitboard_at(red_bitboard, x, y) + (2*bitboard_at(yellow_bitboard, x, y));
}

string C4Board::reverse_representation() const {
    string result;
    for (char ch : representation) {
        int num = ch - '0';
        int reversedNum = 8 - num;
        result += to_string(reversedNum);
    }
    return result;
}

void C4Board::print() {
    cout << setprecision(17) << representation << " (" << get_hash() << ")" << endl;
    for(int y = 0; y < C4_HEIGHT; y++) {
        for(int x = 0; x < C4_WIDTH; x++) {
            cout << disk_col(piece_code_at(x, y)) << " ";
        }
        cout << endl;
    }
}

Bitboard C4Board::legal_moves() const {
    return downset(red_bitboard | yellow_bitboard);
}

bool C4Board::is_legal(int x) const {
    return (((red_bitboard|yellow_bitboard) >> (x-1)) & 1ul) == 0ul;
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

    for (int i = 0; i < 2; i++){
        const Bitboard b = i==0?red_bitboard:yellow_bitboard;
        if( (b & (b>>1) & (b>>(2*1)) & (b>>(3*1)))
         || (b & (b>>w) & (b>>(2*w)) & (b>>(3*w)))
         || (b & (b>>v) & (b>>(2*v)) & (b>>(3*v)))
         || (b & (b>>x) & (b>>(2*x)) & (b>>(3*x))) )
            return i==0?RED:YELLOW;
    }

    if((yellow_bitboard|red_bitboard) == 140185576636287ul) // this wont be resilient to other board sizes...
        return TIE;

    return INCOMPLETE;
}

bool C4Board::is_solution() {
    if(has_steady_state) return true;
    C4Result winner = who_won();
    return winner == RED || winner == YELLOW;
}

double C4Board::type_specific_reverse_hash() {
    return C4Board(reverse_representation()).get_hash();
}


double C4Board::type_specific_hash() {
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

void C4Board::fill_board_from_string(const string& rep) {
    // Iterate through the moves and fill the board
    for (int i = 0; i < rep.size(); i++) {
        play_piece(rep[i]-'0');
    }
}

void C4Board::play_piece(int piece){
    if(piece < 0) {
        print();
        throw runtime_error("Attempted playing a piece in an illegal column. Representation: " + representation + ", piece: " + to_string(piece));
    }
    if(piece > 0){
        if(!is_legal(piece)) {
            print();
            throw runtime_error("Tried playing illegal piece " + representation + " " + to_string(piece));
        }
        int x = piece - 1; // convert from 1index to 0
        Bitboard p = legal_moves() & make_column(x);
        if(is_reds_turn()) red_bitboard += p;
        else yellow_bitboard += p;
    }
    representation+=to_string(piece);
    reset_hashes();
}

C4Board C4Board::child(int piece) const{
    C4Board new_board(*this);
    new_board.play_piece(piece);
    return new_board;
}

C4Result C4Board::who_is_winning(int& work, bool verbose) {

    string winner;
    if (fhourstonesCache.GetEntryIfExists(get_hash(), get_reverse_hash(), winner)) {
        if (verbose) cout << "Using cached result..." << endl;

        // Convert winner string to C4Result
        if (winner == "TIE") return TIE;
        else if (winner == "RED") return RED;
        else if (winner == "YELLOW") return YELLOW;

        throw runtime_error("Invalid winner value in fhourstonescache: " + winner);
    }

    if(verbose) cout << "Calling fhourstones on " << representation << endl;

    // If not found in cache, compute using fhourstones
    char command[150];
    sprintf(command, "echo %s | ~/Fhourstones/SearchGame", representation.c_str());
    if (verbose) cout << "Calling fhourstones on " << command << "... ";
    FILE* pipe = popen(command, "r");
    if (!pipe) {
        C4Board c4(representation);
        cout << setprecision(17) << c4.get_hash() << endl;
        throw runtime_error("fhourstones error!");
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
        gameResult = TIE;
        winner = "TIE";
    } else if ((result.find("(+)") != string::npos) == is_reds_turn()) {
        gameResult = RED;
        winner = "RED";
    } else {
        gameResult = YELLOW;
        winner = "YELLOW";
    }

    // Store result in persistent cache
    fhourstonesCache.AddOrUpdateEntry(get_hash(), get_reverse_hash(), representation, winner);

    return gameResult;
}

void C4Board::add_all_winning_fhourstones(unordered_set<GenericBoard*>& neighbors) const {
    vector<GenericBoard*> children;

    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            int work = -1;
            if (moved.who_is_winning(work) == RED) {
                children.push_back(new C4Board(moved));
            }
        }
    }

    // Use the helper function to sort by minimum hash and insert
    insert_sorted_children_by_min_hash(children, neighbors);
}

void C4Board::add_random_winning_fhourstones(unordered_set<GenericBoard*>& neighbors) const {
    cout << "Adding random winning fhourstones for " << representation << endl;
    vector<GenericBoard*> children;

    int iter_count = 0;
    for(int i = 1; i < 8; i++) {
        if(!is_legal(i)) continue;
        C4Board moved = child(i);
        int work = -1;
        if (moved.who_is_winning(work) == RED) {
            children.push_back(new C4Board(moved));
            break;
        }
        if(iter_count > 1000) throw runtime_error("No winning moves found after 1000 tries! " + representation);
    }

    insert_sorted_children_by_min_hash(children, neighbors);
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
    return child(0).get_instant_win();
}

int C4Board::get_best_winning_fhourstones() const {
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

// Is symmetry broken first on the left or right?
// Return -1 if left, 1 if right, 0 if not broken
int C4Board::which_side() const {
    for (char ch : representation) {
        int num = ch - '0';
        if (num < 4) return -1;
        if (num > 4) return 1;
    }
    return 0;
}

int C4Board::search_nply_id(const int depth, const vector<int>& order_in, vector<int>& order_out) const {
    vector<int> ordering_last = order_in;
    for(int i = 0; i <= depth; i+=2) {
        cout << "Attempting " << i << "-ply search for steadystates..." << endl;
        int nou = 0;
        vector<int> ordering_this;
        int ret = search_nply(i, nou, true, ordering_last, ordering_this);
        ordering_last = ordering_this;
        if(ret != -1)
            return ret;
    }
    order_out = ordering_last;
    return -1;
}

int C4Board::search_nply(const int depth, int& num_ordered_unfound, bool verbose, vector<int> ordering_in, vector<int>& ordering_out) const {
    ordering_out.clear();
    if (depth == 0) verbose = false;
    if (depth % 2 == 1 || depth < 0) throw runtime_error("Invalid search depth");
    int giw = get_instant_win();
    if(giw != -1) {
        ordering_out = get_winning_moves();
        num_ordered_unfound = 0;
        return giw;
    }

    if(ordering_in.size() == 0) ordering_in = get_winning_moves();

    // Base case
    if (depth == 0) {
        bool fncall = find_steady_state(representation, nullptr) != nullptr;
        ordering_out = ordering_in;
        num_ordered_unfound = fncall ? 0 : 1;
        // Return value is only used by top level caller, doesn't matter
        return -1;
    }

    num_ordered_unfound = 10000000;
    int best_col = -1;

    // Vector to store (column, nou_child) pairs
    vector<pair<int, int>> column_nou_pairs;

    for (const int i : ordering_in) {
        if (verbose) cout << "Column " << i << ": " << flush;
        int nou_child = 0;
        bool give_up = false;
        C4Board childi = child(i); // Red plays
        for (int j = 1; j <= C4_WIDTH; j++) {
            if (give_up            ) { if (verbose) cout << "?" << flush; continue; }
            if (!childi.is_legal(j)) { if (verbose) cout << "." << flush; continue; }
            C4Board grandchild = childi.child(j); // Yellow plays
            int nou_grandchild = 0;
            bool node_in_graph = graph_to_check_if_points_are_in->node_exists(grandchild.get_hash()) || graph_to_check_if_points_are_in->node_exists(grandchild.get_reverse_hash());
            bool no_steadystate_yet = true;
            if(depth != 2) no_steadystate_yet = find_steady_state(grandchild.representation, nullptr) == nullptr;
            if(!node_in_graph && no_steadystate_yet) {
                vector<int> ignore_out;
                grandchild.search_nply(depth - 2, nou_grandchild, false, vector<int>(), ignore_out);
                nou_child += nou_grandchild;
                if (nou_child >= 1)
                    give_up = true;
            }
            if (verbose) cout << (nou_grandchild == 0 ? "T" : "F") << flush;
        }
        if (verbose) cout << " | " << nou_child << endl;

        // Store the (column, nou_child) pair
        column_nou_pairs.push_back({i, nou_child});

        if (nou_child < num_ordered_unfound || (nou_child == num_ordered_unfound && square(4 - i) < square(4 - best_col))) {
            num_ordered_unfound = nou_child;
            best_col = i;
        }
        if (num_ordered_unfound == 0) return i;
    }

    // Sort columns based on their nou_child values, ascending order
    std::sort(column_nou_pairs.begin(), column_nou_pairs.end(),
              [](const pair<int, int>& a, const pair<int, int>& b) {
                  return a.second < b.second;
              });

    // Populate ordering with sorted columns
    for (const auto& pair : column_nou_pairs) {
        ordering_out.push_back(pair.first);
    }

    return -1;
}

int C4Board::get_human_winning_fhourstones() {
    shared_ptr<SteadyState> ss = find_cached_steady_state(C4Board(representation));
    if(ss != nullptr)
        return -1;

    int wm = get_instant_win();
    if(wm != -1) {
        return wm;
    }
    int bm = get_blocking_move();
    if(bm != -1) {
        return bm;
    }

    // Optional speedup which will naively assume that if no steadystate was found on a prior run, none exists.
    const bool SKIP_UNFOUND_STEADYSTATES = true;
    if(SKIP_UNFOUND_STEADYSTATES || representation.size() < 5){
        int move = -1;
        string ss = "";
        bool found = movecache.GetSuggestedMoveIfExists(get_hash(), get_reverse_hash(), move, ss);
        if(ss != "") throw runtime_error("Cached steady state found in ghwf, but should have been caught before entry");
        if(move > 0) return move;
    }

    vector<int> winning_columns = get_winning_moves();
    if (winning_columns.size() == 0)
        throw runtime_error("Burst winning columns error! Possibly wrong move in movecache. " + representation);

    // Add things already in the graph!
    //drop a red piece in each column and see if it is in the graph
    for (int i = 0; i < winning_columns.size(); ++i) {
        int x = winning_columns[i];
        C4Board child_x = child(x);
        if(graph_to_check_if_points_are_in->node_exists(child_x.get_hash())) {
            //cout << representation <<x<< " added since it is in the graph already" << endl;
            return x;
        }
    }

    if(find_steady_state(representation, nullptr, false, true, 80, 100) != nullptr)
        return -1;
    if(find_steady_state(representation, nullptr, false, true, 80, 100) != nullptr)
        return -1;
    if(find_steady_state(representation, nullptr, false, true, 80, 100) != nullptr)
        return -1;

    if (winning_columns.size() == 1) {
        char wc = winning_columns[0];
        return wc;
    } else if (winning_columns.size() == 0){
        throw runtime_error("Get human winning fhourstones error!");
    }

    const bool BACKTRACK = !SKIP_UNFOUND_STEADYSTATES;
    if(BACKTRACK){
        vector<int> order_out;
        int snp = search_nply_id(8, winning_columns, order_out);
        if(snp > 0) return snp;
    }

    ss = find_cached_steady_state(C4Board(representation));
    if(ss != nullptr)
        return -1;

    // Check Cache
    {
        int move = -1;
        string ss = "";
        int ret = movecache.GetSuggestedMoveIfExists(get_hash(), get_reverse_hash(), move, ss);
        if(move > 0) return move;
    }

    print();

    cout << representation << " (" << get_hash() << ") has multiple winning columns. Please select one:" << endl;
    for (const int i : winning_columns) {
        cout << i << " : Column " << i << endl;
    }
    cout << "-1: continue searching for steadystates." << endl;
    cout << "-2: 2-ply search." << endl;
    cout << "-4: 4-ply search." << endl;
    cout << "-6: 6-ply search." << endl;

    int choice = 0;
    do {
        cout << "Enter your choice: ";
        cin >> choice;
        if (cin.fail()) {
            cout << "ERROR -- You did not enter an integer" << endl;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
        }
        if (choice == -1) return get_human_winning_fhourstones();
        else if (choice < -1 && (-choice) % 2 == 0) {
            vector<int> order_out;
            int snp = search_nply_id(-choice, winning_columns, order_out);
            if(snp > 0) return snp;
        }
    } while (find(winning_columns.begin(), winning_columns.end(), choice) == winning_columns.end());

    movecache.WriteCache();
    return choice;
}

void C4Board::insert_sorted_children_by_min_hash(vector<GenericBoard*>& children, unordered_set<GenericBoard*>& neighbors) const {
    // Temporary vector to store pairs of C4Board* and their minimum hash values
    vector<pair<GenericBoard*, size_t>> children_with_hashes;

    // Populate the vector with each child board and its minimum hash
    for (GenericBoard* child : children) {
        size_t hash_value = min(child->get_hash(), child->get_reverse_hash());
        children_with_hashes.push_back({child, hash_value});
    }

    // Sort the vector based on the minimum hash value
    sort(children_with_hashes.begin(), children_with_hashes.end(),
              [](const pair<GenericBoard*, size_t>& a, const pair<GenericBoard*, size_t>& b) {
                  return a.second < b.second;
              });

    // Insert the sorted boards into the neighbors set
    for (const auto& pair : children_with_hashes) {
        neighbors.insert(pair.first);
    }
}

void C4Board::add_all_legal_children(unordered_set<GenericBoard*>& neighbors) const {
    vector<GenericBoard*> children;

    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            children.push_back(new C4Board(moved));
        }
    }

    // Use the helper function to sort by minimum hash and insert
    insert_sorted_children_by_min_hash(children, neighbors);
}

void C4Board::add_all_good_children(unordered_set<GenericBoard*>& neighbors) const {
    vector<GenericBoard*> children;

    for (int i = 1; i <= C4_WIDTH; i++) {
        if (is_legal(i)) {
            C4Board moved = child(i);
            if (moved.get_instant_win() == -1) {
                children.push_back(new C4Board(moved));
            }
        }
    }

    // Use the helper function to sort by minimum hash and insert
    insert_sorted_children_by_min_hash(children, neighbors);
}

json C4Board::get_data() const {
    json data;  // Create a JSON object

    // Create a nested JSON array for the 2D array steadystate.steadystate
    json steadystate_array;

    for (int y = 0; y < C4_HEIGHT; ++y) {
        json row_array;
        for (int x = 0; x < C4_WIDTH; ++x) {
            char c = has_steady_state?steadystate->get_char(x, y):' ';
            row_array.push_back(c);
        }
        steadystate_array.push_back(row_array);
    }

    // Add the nested array to the JSON object
    data["ss"] = steadystate_array;

    return data;
}

void C4Board::add_only_child_steady_state(unordered_set<GenericBoard*>& neighbors) const {
    if(steadystate == nullptr)
        throw runtime_error("Querying a null steadystate!");
    int x = steadystate->query_steady_state(*this);
    C4Board moved = child(x);
    neighbors.insert(new C4Board(moved));
}

unordered_set<GenericBoard*> C4Board::get_children(){
    unordered_set<GenericBoard*> neighbors;

    if (is_solution()) {
        return neighbors;
    }

    switch (c4_branch_mode){
        case MANUAL:
            break;
        case FULL:
            add_all_legal_children(neighbors);
            break;
        case LEFTMOST_LOWEST_2: {
            int last_lowest_column = -1;
            for (int i = 0; i < (is_reds_turn() ? 2 : 1); i++) {
                int lowest_column = -1;
                int min_height = 100;
                for (int col = 1; col <= C4_WIDTH; col++) {
                    if(col == last_lowest_column) continue;
                    int height = 0;
                    for (int y = 0; y < C4_HEIGHT; y++) {
                        if (piece_code_at(col - 1, y) != 0) {
                            height++;
                        }
                    }
                    if(is_legal(col) && height < min_height) {
                        min_height = height;
                        lowest_column = col;
                    }
                }
                neighbors.insert(new C4Board(child(lowest_column)));
                last_lowest_column = lowest_column;
            }
            break;
        }
        case RANDOM_WEAK:
            if(is_reds_turn())
                add_random_winning_fhourstones(neighbors);
            else
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
            if(is_reds_turn()){
                shared_ptr<SteadyState> ss = find_cached_steady_state(C4Board(representation));
                if(ss != nullptr){
                    has_steady_state = true;
                    steadystate = ss;
                    break;
                }
                int hwf = get_human_winning_fhourstones();
                if(hwf == -1) break;
                else movecache.AddOrUpdateEntry(get_hash(), get_reverse_hash(), representation, hwf);
                C4Board moved = child(hwf);
                neighbors.insert(new C4Board(moved));
            } else { // yellow's move
                add_all_good_children(neighbors);
            }
            break;
    }
    return neighbors;
}

unordered_set<double> C4Board::get_children_hashes() {
    if(!children_hashes_initialized) {
        unordered_set<GenericBoard*> children = get_children();
        for (const auto& child : children) {
            children_hashes.insert(child->get_hash());
        }
        children_hashes_initialized = true;
    }
    return children_hashes;
}
