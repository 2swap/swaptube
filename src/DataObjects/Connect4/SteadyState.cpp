#pragma once

#include <list>
#include <unordered_set>
#include "SteadyState.h"
#include <random>
#include <cassert>
#include <array>
#include "C4Board.h"

const bool VISION = true;

vector<char> replacement_chars = {'+', '=', '-'};

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

    if(VISION) {
        int wm = board.get_instant_win();
        if(wm != -1) return wm;
        int bm = board.get_blocking_move();
        if(bm != -1) return bm;
    }

    Bitboard moveset = board.legal_moves();

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
    int r = rand()%10;

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

SteadyState create_random_steady_state(const C4Board& b) {
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
    ss.mutate();
    return ss;
}

C4Result SteadyState::play_one_game(const C4Board& b) const {
    C4Board board = b;
    C4Result winner = INCOMPLETE;
    while (true) {
        // Yellow's Turn
        int randomColumn = board.get_instant_win();
        if (randomColumn == -1 && VISION) randomColumn = board.get_blocking_move();
        if (randomColumn == -1) randomColumn = board.random_legal_move();
        board.play_piece(randomColumn);
        winner = board.who_won();
        if(board.who_won() != INCOMPLETE) return winner;

        // Red's Turn
        // Query the steady state and play the determined disk
        int columnToPlay = query_steady_state(board);
        if (columnToPlay < 0) return YELLOW;
        else if (columnToPlay >= 1 && columnToPlay <= 7) board.play_piece(columnToPlay);
        winner = board.who_won();
        if(winner != INCOMPLETE) return winner;
    }
}

// Given a steady state and a board position, make sure that steady state solves that position
bool SteadyState::validate_steady_state(const C4Board& b, int& branches_searched) {
    for (int i = 1; i <= 7; i++){
        if(!b.is_legal(i)) continue;
        branches_searched++;
        C4Board child = b.child(i);
        if(child.who_won() != INCOMPLETE) return false;

        // Query the steady state and play the determined disk
        int columnToPlay = query_steady_state(child.representation);

        if (columnToPlay <= 0) return false;
        else if (columnToPlay >= 1 && columnToPlay <= 7) child.play_piece(columnToPlay);
        C4Result winner = child.who_won();
        if(winner == RED) continue;
        if(!validate_steady_state(child, branches_searched)) return false;
    }

    return true;
}

shared_ptr<SteadyState> find_cached_steady_state(double hash, double reverse_hash, string& cache_filename) {
    // Array of the two hash values to iterate over
    array<double, 2> hashes = {hash, reverse_hash};
    
    for (int i = 0; i < 2; ++i) {
        // Convert the current hash to a string with high precision
        ostringstream ss_hash_stream;
        ss_hash_stream << fixed << setprecision(numeric_limits<double>::max_digits10) << hashes[i];
        string ss_hash = ss_hash_stream.str();

        // Create the cache filename based on the hash
        cache_filename = "steady_states/" + ss_hash + ".ss";
        
        // Attempt to open the cache file
        if (ifstream(cache_filename)) {
            // Use the current hash index to determine if this is a reverse hash (index 1)
            shared_ptr<SteadyState> ss = make_shared<SteadyState>(read_from_file(cache_filename, i == 1));
            //ss->print();
            // cout << "Loaded cached steady state from file " << cache_filename << endl;
            return ss;
        }
    }
    
    // If no cached state was found, return nullptr
    return nullptr;
}

shared_ptr<SteadyState> find_steady_state(const string& representation, bool verbose = true) {
    int num_games = 3000;
    if(verbose) cout << "Searching for a steady state..." << endl;
    if(representation.size() % 2 == 1)
        throw runtime_error("Steady state requested on board which is yellow-to-move!");
    C4Board board(representation);

    // Check if a cached steady state file exists and read from it
    string cache_filename = "";
    shared_ptr<SteadyState> cached = find_cached_steady_state(board.get_hash(), board.reverse_hash(), cache_filename);
    if(cached != nullptr) return cached;

    C4Board copy = board;

    vector<SteadyState> steady_states;
    int best = 1;

    // Generate a lot of random steady states
    for (int i = 0; i < 100; ++i) {
        steady_states.push_back(create_random_steady_state(copy));
    }
    int games_played = 0;

    while(true){
        int consecutive_wins = 0;
        int idx = rand()%steady_states.size(); // Randomly select a steady state
        while (true) {
            C4Result col;
            col = steady_states[idx].play_one_game(copy);
            games_played++;

            bool eliminate_agent = false;
            if (col != RED) {
                eliminate_agent = true;
            } else {
                if(consecutive_wins > 500){
                    int how_many_branches = 0;
                    if(verbose) cout << "Attempting validation..." << endl;
                    if(steady_states[idx].validate_steady_state(copy, how_many_branches)) {
                        if(verbose) cout << "Steady state found after " << games_played << " games." << endl;
                        if(verbose) cout << "Steady state validated on " << how_many_branches << " branches." << endl;
                        shared_ptr<SteadyState> ss = make_shared<SteadyState>(steady_states[idx]);
                        if(verbose) ss->print();
                        ss->write_to_file(cache_filename);
                        return ss;
                    } else {
                        eliminate_agent = true;
                        consecutive_wins = 0; // if it fails validation we dont want it to reproduce
                    }
                }
                consecutive_wins++;
            }

            if(eliminate_agent){
                int n = 2.0*sqrt(consecutive_wins + .1);
                for (int i = 0; i < n; ++i) {
                    int random_idx = rand() % steady_states.size();
                    for(int y = 0; y < C4_HEIGHT; y++){
                        for(int x = 0; x < C4_WIDTH; x++){
                            steady_states[random_idx].set_char(x, y, steady_states[idx].get_char(x, y));
                        }
                    }
                    steady_states[random_idx].mutate();
                }
                if(games_played>num_games) return nullptr;
                steady_states[idx].mutate();
                if(best < consecutive_wins){
                    best = consecutive_wins;
                    games_played = 0;
                }
                break;
            }
        }
    }
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

void steady_state_unit_tests_problem_7() {
    shared_ptr<SteadyState> ss = find_steady_state("4444415666676222243325", true);
    if(ss == nullptr) cout << "No ss found" << endl;
    else ss->print();

    ss = find_steady_state("4444415666622226215574267713", true);
    if(ss == nullptr) cout << "No ss found" << endl;
    else ss->print();
}

void steady_state_unit_tests(){
    cout << "Steady State Unit Tests..." << endl;
    steady_state_unit_tests_problem_1();
    steady_state_unit_tests_problem_2();
    steady_state_unit_tests_problem_6();
    steady_state_unit_tests_problem_7();
    cout << "Steady State Unit Tests Passed" << endl;
}
