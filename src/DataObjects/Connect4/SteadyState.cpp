#pragma once

#include <list>
#include <unordered_set>
#include "SteadyState.h"
#include <random>
#include <cassert>
#include <array>
#include "C4Board.h"
#include "JsonC4Cache.cpp"

const bool VISION = true;

vector<char> replacement_chars = {'+', '=', '-'};
vector<char> all_instructions = {'+', '=', '-', '|', ' ', '@', '!'};

void SteadyState::set_char_bitboard(const Bitboard point, char c){
    Bitboard notpoint = ~point;

    bitboard_miai &= notpoint;
    bitboard_claimeven &= notpoint;
    bitboard_claimodd &= notpoint;
    bitboard_plus &= notpoint;
    bitboard_equal &= notpoint;
    bitboard_minus &= notpoint;
    bitboard_red &= notpoint;
    bitboard_yellow &= notpoint;
    bitboard_urgent &= notpoint;

    switch(c) {
        case '@':
            bitboard_miai |= point;
            break;
        case ' ':
        case '.':
            bitboard_claimeven |= point;
            break;
        case '|':
            bitboard_claimodd |= point;
            break;
        case '+':
            bitboard_plus |= point;
            break;
        case '=':
            bitboard_equal |= point;
            break;
        case '-':
            bitboard_minus |= point;
            break;
        case '1':
            bitboard_red |= point;
            break;
        case '2':
            bitboard_yellow |= point;
            break;
        case '!':
            bitboard_urgent |= point;
            break;
        default:
            throw runtime_error(string("Invalid SteadyState::set_char character: '") + c + "'");
    }
}

void SteadyState::set_char(int x, int y, char c){
    Bitboard point = make_point(x, y);
    set_char_bitboard(point, c);
}

// Method to populate char** array from array of strings
void SteadyState::populate_char_array(const array<string, C4_HEIGHT>& source) {
    for (int y = 0; y < C4_HEIGHT; ++y) {
        for(int x = 0; x < C4_WIDTH; ++x) {
            set_char(x, y, source[y][x]);
        }
    }
}

char SteadyState::get_char(const int x, const int y) const {
    Bitboard point = make_point(x, y);
    char ret = 0;
         if((bitboard_miai       & point) != 0ul) ret='@';
    else if((bitboard_claimeven  & point) != 0ul) ret=' ';
    else if((bitboard_claimodd   & point) != 0ul) ret='|';
    else if((bitboard_plus       & point) != 0ul) ret='+';
    else if((bitboard_equal      & point) != 0ul) ret='=';
    else if((bitboard_minus      & point) != 0ul) ret='-';
    else if((bitboard_red        & point) != 0ul) ret='1';
    else if((bitboard_yellow     & point) != 0ul) ret='2';
    else if((bitboard_urgent     & point) != 0ul) ret='!';
    else if((frame               & point) != 0ul) ret='F';
    if(ret == 0) {
        throw runtime_error("Steadystate was unset when queried!");
    }
    return ret;
}

SteadyState::SteadyState(const array<string, C4_HEIGHT>& chars) {
    // Initialize the character array with the provided strings
    populate_char_array(chars);
}

int SteadyState::query_steady_state(const C4Board& board) const {
    // Given a board, use the steady state to determine where to play.
    // Return the x position of the 1-indexed row to play in.

    Bitboard moveset = board.legal_moves();

    // If there is only one legal move, play it.
    /*if(is_power_of_two(moveset)) {
        for(int x = 1; x < C4_WIDTH + 1; x++) {
            if(board.is_legal(x)) return x;
        }
    }*/
    
    if(VISION) {
        int wm = board.get_instant_win();
        if(wm != -1) return wm;
        int bm = board.get_blocking_move();
        if(bm != -1) return bm;
    }

    // Construct priority list
    Bitboard miai_moveset = moveset & bitboard_miai;
    if (!is_power_of_two(miai_moveset)) miai_moveset = 0ul;
    const Bitboard claims_moveset = (odd_rows & bitboard_claimodd) | (even_rows & bitboard_claimeven);
    const Bitboard bitboards[] = {bitboard_urgent, miai_moveset, claims_moveset, bitboard_plus, bitboard_equal, bitboard_minus};

    const int num_bitboards = sizeof(bitboards) / sizeof(bitboards[0]);

    for (int n = 0; n < num_bitboards; n++) {
        const Bitboard this_priority_moveset = bitboards[n] & moveset;
        if (this_priority_moveset) {
            // Case of two equal priorities
            if (!is_power_of_two(this_priority_moveset) && this_priority_moveset != 0) return -6;
            for (int x = 0; x < C4_WIDTH; x++) if(make_column(x) & this_priority_moveset) return x+1;
            throw runtime_error("Failed to find a priority marking when one was expected.");
        }
    }

    // No instruction was provided.
    return -4;
}

// TODO this is not performant and does not really make use of bitboards
void SteadyState::drop(const int x, const char c){
    int y = C4_HEIGHT-1;
    for(y; y >= 0; y--){
        const char here = get_char(x, y);
        if(here != c && (here == ' ' || here == '|'))
            break;
    }
    if(y>=0) set_char(x, y, c);
}

// TODO this is not performant and does not really make use of bitboards
void SteadyState::mutate() {
    int r = rand()%11;

    /*if(r < 1) {
        // Select a random x coordinate
        int x = rand()%C4_WIDTH;
        // Find the number of rows that are occupied by 1s and 2s
        int y = C4_HEIGHT-1;
        for(y; y >= 0; y--) {
            char c_here = get_char(x, y);
            if(c_here != '1' && c_here != '2') break;
        }
        // Select an available y coordinate
        if(y == C4_HEIGHT-1) {
            return;
        }
        y = rand() % (C4_HEIGHT - y) + y;
        set_char(x, y, all_instructions[rand()%all_instructions.size()]);
    }*/
    if(r<2){
        // flush all miai
        char c = '@';
        for(int y = 0; y < C4_HEIGHT; y++){
            for(int x = 0; x < C4_WIDTH; x++){
                if(get_char(x, y) == c){
                    set_char(x, y, replacement_chars[rand()%replacement_chars.size()]);
                }
            }
        }
        if(r<1){
            //drop a miai pair
            int giveupcount = 0;
            for(int i = 0; i < 2; i++){
                int x = rand()%C4_WIDTH;
                int y = C4_HEIGHT-1;
                for(y; y >= 0; y--) {
                    char c_here = get_char(x, y);
                    if(c_here != '1' && c_here != '2') break;
                }
                if(y>=0 && !is_miai(get_char(x, y))) set_char(x, y, c);
                else {
                    i--;
                    giveupcount++;
                }
                if(giveupcount > 10) break;
            }
        }
    }

    else if (r < 3){
        drop(rand()%C4_WIDTH, replacement_chars[rand()%replacement_chars.size()]);
    }
    else if (r < 4){
        drop(rand()%C4_WIDTH, ' ');
    }
    else if (r < 5){
        drop(rand()%C4_WIDTH, '|');
    }


    else if (r < 6){
        int x = rand()%C4_WIDTH;
        for(int i = 0; i < 6; i++){
            drop(x, '=');
        }
    }
    else if(r<7){
        // flush all urgents
        for(int y = 0; y < C4_HEIGHT; y++){
            for(int x = 0; x < C4_WIDTH; x++){
                if(get_char(x, y) == '!'){
                    set_char(x, y, replacement_chars[rand()%replacement_chars.size()]);
                }
            }
        }
        // drop an urgent
        int giveupcount = 0;
        while (true) {
            int x = rand()%C4_WIDTH;
            int y = C4_HEIGHT-1;
            for(y; y >= 0; y--) {
                char c_here = get_char(x, y);
                if(c_here != '1' && c_here != '2') break;
            }
            if(y>=0 && !is_miai(get_char(x, y))) {
                set_char(x, y, '!');
                break;
            }
            giveupcount++;
            if(giveupcount > 10) break;
        }
    }

    else {
        int x = rand()%C4_WIDTH;
        int ct = 0;
        char claim = r==9?'|':' ';
        for (int y = 0; y < C4_HEIGHT; ++y) {
            char c = get_char(x, y);
            if(c != '1' && c != '2' && !is_miai(c)){
                ct++;
                set_char(x, y, claim);
            }
        }
        if((claim=='|') == (ct%2==0)){
            drop(x, replacement_chars[rand()%replacement_chars.size()]);
        }
    }
}

void SteadyState::print() const {
    for(int y = 0; y < C4_HEIGHT; y++) {
        for(int x = 0; x < C4_WIDTH; x++) {
            char c = get_char(x, y);
            if(c == '1' or c == '2')
                cout << disk_col(c-'0') << " ";
            else
                cout << c << " ";
        }
        cout << endl;
    }
    cout << endl;
}

SteadyState create_empty_steady_state(const C4Board& b) {
    array<string, C4_HEIGHT> chars; // Should contain C4_HEIGHT strings, each with length C4_WIDTH, full of spaces.
    for (int y = 0; y < C4_HEIGHT; ++y)
        chars[y] = string(C4_WIDTH, ' ');

    for (int y = 0; y < C4_HEIGHT; ++y) {
        for (int x = 0; x < C4_WIDTH; ++x) {
            int pc = b.piece_code_at(x, y);
            char c = ' ';
                 if (pc == 1) c = '1';
            else if (pc == 2) c = '2';
            chars[y][x] = c;
        }
    }
    SteadyState ss(chars);
    return ss;
}

SteadyState create_random_steady_state(const C4Board& b) {
    SteadyState ss = create_empty_steady_state(b);
    ss.mutate();
    ss.mutate();
    ss.mutate();
    ss.mutate();
    ss.mutate();
    ss.mutate();
    return ss;
}

C4Result SteadyState::play_one_game(const C4Board& b) const {
    C4Board board = b;
    C4Result winner = INCOMPLETE;
    while (true) {
        if(board.is_reds_turn()){
            // Query the steady state and play the determined disk
            int columnToPlay = query_steady_state(board);
            if (columnToPlay < 0) return YELLOW;
            else if (columnToPlay >= 1 && columnToPlay <= 7) board.play_piece(columnToPlay);
            winner = board.who_won();
            if(winner != INCOMPLETE) return winner;
        }

        if(!board.is_reds_turn()){
            int randomColumn = board.get_instant_win();
            if (randomColumn == -1 && VISION) randomColumn = board.get_blocking_move();
            if (randomColumn == -1) randomColumn = board.random_legal_move();
            board.play_piece(randomColumn);
            winner = board.who_won();
            if(board.who_won() != INCOMPLETE) return winner;
        }
    }
}

bool SteadyState::check_matches_board(const C4Board& b) const {
    // Check if the representation matches
    for(int y = 0; y < C4_HEIGHT; y++) {
        for(int x = 0; x < C4_WIDTH; x++) {
            char c = get_char(x, y);
            int piece_code = b.piece_code_at(x, y);
            if(piece_code == 1 && c != '1') return false;
            if(piece_code == 2 && c != '2') return false;
            if(piece_code == 0 && (c == '1' || c == '2')) return false;
        }
    }
    return true;
}

bool SteadyState::check_no_illegal_characters() const {
    // Check if the representation contains any illegal characters
    for(int y = 0; y < C4_HEIGHT; y++) {
        for(int x = 0; x < C4_WIDTH; x++) {
            char c = get_char(x, y);
            if(c != '1' && c != '2' && c != ' ' && c != '|' && c != '+' && c != '=' && c != '-' && c != '@' && c != '!') {
                return false;
            }
        }
    }
    return true;
}

bool SteadyState::validate(C4Board b, bool verbose = false) {
    if(!check_matches_board(b)) {
        if(verbose) cout << "Steady state does not match board representation." << endl;
        return false;
    }
    if(!check_no_illegal_characters()) {
        if(verbose) cout << "Steady state contains illegal characters." << endl;
        return false;
    }
    unordered_set<double> wins_cache;
    bool validated = validate_recursive_call(b, wins_cache, verbose);
    if(validated){
        movecache.AddOrUpdateEntry(b.get_hash(), b.get_reverse_hash(), b.representation, to_string());
        //cout << "Steady state validated on " << wins_cache.size() << " non-leaves." << endl;
    }
    return validated;
}

// Given a steady state and a board position, make sure that steady state solves that position
bool SteadyState::validate_recursive_call(C4Board b, unordered_set<double>& wins_cache, bool verbose = false) {
    if(wins_cache.find(b.get_hash()) != wins_cache.end()) return true;

    if(b.is_reds_turn()){
        // Query the steady state and play the determined disk
        int columnToPlay = query_steady_state(b.representation);
        C4Board child = b;
        if (columnToPlay >= 1 && columnToPlay <= 7) child.play_piece(columnToPlay);
        else {
            if(verbose) {
                cout << "Validation failed because Red failed to select a move." << endl;
                b.print();
            }
            return false;
        }
        if(child.who_won() == RED) return true;
        else if(!validate_recursive_call(child, wins_cache, verbose)) { return false; }
    } else {
        for (int i = 1; i <= 7; i++){
            if(!b.is_legal(i)) continue;
            C4Board child = b.child(i);
            C4Result who_won = child.who_won();
            if(who_won != INCOMPLETE) {
                if(verbose) {
                    cout << "Validation failed because " << (who_won == TIE ? "tie" : "Yellow won") << endl;
                    b.print();
                }
                return false;
            }
            if(!validate_recursive_call(child, wins_cache, verbose)) { return false; }
        }
    }

    wins_cache.insert(b.get_hash());

    return true;
}

bool SteadyState::check_ss_matches_board(C4Board b){
    return b.yellow_bitboard == bitboard_yellow && b.red_bitboard == bitboard_red;
}

shared_ptr<SteadyState> find_cached_steady_state(C4Board b) {
    // First check the movecache.
    {
        int move = -1;
        string ss = "";
        bool found = movecache.GetSuggestedMoveIfExists(b.get_hash(), b.get_reverse_hash(), move, ss);
        if(ss.size() == C4_WIDTH*C4_HEIGHT){
            return make_shared<SteadyState>(make_steady_state_from_string(ss));
        }
    }

    {
        int move = -1;
        string ss = "";
        bool found = movecache.GetSuggestedMoveIfExists(b.get_hash(), b.get_reverse_hash(), move, ss);
        if(ss.size() == C4_WIDTH*C4_HEIGHT){
            return make_shared<SteadyState>(make_steady_state_from_string(ss));
        }
    }

    // If no cached state was found, return nullptr
    return nullptr;
}

shared_ptr<SteadyState> modify_child_suggestion(const shared_ptr<SteadyState> parent, const C4Board& b) {
    SteadyState child = create_empty_steady_state(b);
    bool miai_mode = rand() % 10 == 0;
    for(int y = 0; y < C4_HEIGHT; y++) {
        for(int x = 0; x < C4_WIDTH; x++) {
            char c = parent->get_char(x, y);
            int piece_code = b.piece_code_at(x, y);
            if(piece_code == 1 && c != '1') {
                child.set_char(x, y, '1');
            } else if(piece_code == 2 && c != '2') {
                child.set_char(x, y, '2');
            } else if(piece_code == 0 && (c == '1' || c == '2')) {
                child.set_char(x, y, miai_mode ? '@' : vector<char>{' ', '|', '+', '=', '-', '@', '!'}[rand() % 7]);
            } else {
                child.set_char(x, y, c);
            }
        }
    }
    return make_shared<SteadyState>(child);
}

shared_ptr<SteadyState> find_steady_state(const string& representation, const shared_ptr<SteadyState> suggestion, bool verbose = false, bool read_from_cache = true, int pool = 40, int generations = 50) {
    if(pool < 3) throw runtime_error("Pool size too small! Must be at least 3 for propagation strategy.");
    if(verbose) cout << "Finding for a steady state of " << representation << "..." << endl;
    if(representation.size() % 2 == 1)
        throw runtime_error("Steady state requested on board which is yellow-to-move!");
    C4Board board(representation);

    // Check if a cached steady state file exists and read from it
    shared_ptr<SteadyState> cached = find_cached_steady_state(board);
    if(read_from_cache && cached != nullptr) {
        if(verbose) cout << "Found cached steady state, returning it..." << endl;
        return cached;
    }
    if(verbose) cout << "No cached steady state found, proceeding with search..." << endl;

    C4Board copy = board;

    // Spawn a pool of random steady states
    vector<SteadyState> steady_states;
    vector<int> win_counts(pool, 0); // Track max consecutive wins for each agent
    for (int i = 0; i < pool; ++i) {
        if(suggestion != nullptr) {
            steady_states.push_back(*modify_child_suggestion(suggestion, copy));
            if(rand()%2==0) steady_states.back().mutate(); // Mutate the suggestion to introduce slight diversity
        } else {
            steady_states.push_back(create_random_steady_state(copy));
        }
    }
    if(verbose) cout << "Initial pool of " << pool << " agents created." << endl;

    for(int generation = 0; generation < generations; generation++) {
        if(verbose) cout << "Generation " << generation << " in progress..." << endl;

        // Play each agent in the pool until it fails
        for (int i = 0; i < pool; ++i) {
            int consecutive_wins = 0;
            while(consecutive_wins < 1000) {
                C4Result result = steady_states[i].play_one_game(copy);
                if(result == RED) {
                    consecutive_wins++;
                } else {
                    break; // The agent failed to maintain a win streak
                }
            }

            // Store the max consecutive wins for this agent
            win_counts[i] = consecutive_wins;

            // Check for correctness if an agent has reached the target of 500 consecutive wins
            if (consecutive_wins >= 1000) {
                if(verbose) cout << "Attempting validation for a potential steady state..." << endl;
                if(steady_states[i].validate(copy)) {
                    return make_shared<SteadyState>(steady_states[i]);
                }
            }
        }

        // Rank agents based on their max consecutive wins
        vector<int> indices(pool);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](int a, int b) { return win_counts[a] > win_counts[b]; });
        if(verbose) cout << "Generation " << generation << " completed. Top agent max consecutive wins: " << win_counts[indices[0]] << endl;

        // Define the sizes for each third of the pool
        int num_parents = pool / 4;
        int num_new_agents = pool / 4;
        int num_copies = pool - num_new_agents;

        // Fill the next third with copies of the best third
        for (int i = 0; i < num_copies; ++i) {
            int parent_idx = indices[i % num_parents]; // Select a top agent to copy from
            steady_states[indices[i]] = steady_states[parent_idx]; // Copy a top agent
            steady_states[indices[i]].mutate(); // Mutate to introduce slight diversity
            win_counts[indices[i]] = 0; // Reset win count for copied agents
        }

        // Fill the final third with new random agents
        for (int i = num_copies; i < pool; ++i) {
            steady_states[indices[i]] = create_random_steady_state(copy); // Generate a new random agent
            win_counts[indices[i]] = 0; // Reset win count for new agents
        }

    }
    return nullptr; // Return null if no steady state was found within the given generations
}

void run_test(const string& board_str, const int expected, const SteadyState& ss) {
    C4Board board(board_str);
    int actual = ss.query_steady_state(board);
    if(actual != expected) {
        board.print();
        ss.print();
        cout << ("SteadyState unit test failed! Expected " + to_string(expected) + " but got " + to_string(actual)) << endl;
        exit(0);
    }
}

void steady_state_unit_tests_problem_1() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        "       ",
        "       ",
        " 11  ++",
        " 12  ==",
        "221  --",
        "212  @@"
    };
    SteadyState ss(ss_list);
    int actual = -1;

    run_test("23333221212"   , 2, ss);
    run_test("23333221211"   , 1, ss);
    run_test("23333221216"   , 7, ss);
    run_test("23333221217"   , 6, ss);
    run_test("23333221213"   , 3, ss);
    run_test("23333221214"   , 4, ss);
    run_test("2333322121676" , 6, ss);
    run_test("23333221217677", 7, ss);
}

void steady_state_unit_tests_problem_2() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        "   |@  ",
        "   |1  ",
        "  1|1  ",
        "  2|2  ",
        "  2|12@",
        "  21211"
    };
    SteadyState ss(ss_list);
    int actual = -1;


    run_test("43667555535337", 5, ss);
    run_test("43667555535335", 7, ss);
    run_test("43667555535334", 4, ss);
    run_test("43667555535332", 2, ss);
}

void steady_state_unit_tests_problem_6() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        "  |2   ",
        "  |1   ",
        " @|2@  ",
        " 1|11  ",
        " 2|21  ",
        " 2112  "
    };
    SteadyState ss(ss_list);
    int actual = -1;


    run_test("44444435525221"    , 1, ss);
    run_test("44444435525222"    , 5, ss);
    run_test("44444435525223"    , 3, ss);
    run_test("44444435525225"    , 2, ss);
    run_test("44444435525226"    , 6, ss);
    run_test("44444435525227"    , 7, ss);
    run_test("444444355252252116", 6, ss);


    string s = "";
    C4Board b("4444443552522");
    for(int i = 0; i < 1000; i++){
        assert(ss.play_one_game(b) == RED);
    }
}

void steady_state_unit_tests_problem_8() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        " |     ",
        " |     ",
        " | |2- ",
        " | |1@ ",
        "2|+@1!=",
        "2-21112"
    };
    SteadyState ss(ss_list);


    run_test("4153515567", 6, ss);
}

void steady_state_unit_tests_problem_7() {
    shared_ptr<SteadyState> ss = find_steady_state("444442222666662477777762", nullptr, true, false, 100, 1000);
    if(ss == nullptr) cout << "No ss found" << endl;
    else ss->print();
/*
    ss = find_steady_state("4444415666676222243325", true);
    if(ss == nullptr) cout << "No ss found" << endl;
    else ss->print();

    ss = find_steady_state("4444415666622226215574267713", true);
    if(ss == nullptr) cout << "No ss found" << endl;
    else ss->print();
    */
}

void steady_state_unit_tests(){
    cout << "Steady State Unit Tests..." << endl;
    steady_state_unit_tests_problem_1();
    steady_state_unit_tests_problem_2();
    steady_state_unit_tests_problem_6();
    steady_state_unit_tests_problem_8();
    cout << "Steady State Unit Tests Passed" << endl;
}
