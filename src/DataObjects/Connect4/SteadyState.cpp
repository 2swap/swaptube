#pragma once

#include <list>
#include <unordered_set>
#include "SteadyState.h"
#include <random>
#include <cassert>
#include "C4Board.h"

// Method to populate char** array from array of strings
void populate_char_array(const array<string, C4_HEIGHT>& source, char dest[C4_HEIGHT][C4_WIDTH]);

vector<char> replacement_chars = {'+', '=', '-'};

void SteadyState::set_char(int x, int y, char c){
    steadystate[y][x] = c;
}

char SteadyState::get_char(int x, int y) const {
    return steadystate[y][x];
}

SteadyState::SteadyState() {
    // Initialize the character array with empty cells
    for (int row = 0; row < C4_HEIGHT; ++row) {
        for (int col = 0; col < C4_WIDTH; ++col) {
            steadystate[row][col] = ' ';
        }
    }
}

SteadyState::SteadyState(const array<string, C4_HEIGHT>& chars) {
    // Initialize the character array with empty cells
    for (int row = 0; row < C4_HEIGHT; ++row) {
        for (int col = 0; col < C4_WIDTH; ++col) {
            steadystate[row][col] = ' ';
        }
    }
    // Initialize the character array with the provided strings
    populate_char_array(chars, steadystate);
}

int SteadyState::query_steady_state(const C4Board board) const {
    int b[C4_HEIGHT][C4_WIDTH];
    for (int y = 0; y < C4_HEIGHT; ++y) {
        for (int x = 0; x < C4_WIDTH; ++x) {
            b[y][x] = board.piece_code_at(x, y);
        }
    }
    // Given a board, use the steady state to determine where to play.
    // Return the x position of the row to play in.

    //Bitboard moveset = board.legal_moves();
    // First Priority: Obey Miai

    int num_hash = 0;
    int hash_move = -10;
    int num_atp = 0;
    int atp_move = -10;
    for (int x = 0; x < C4_WIDTH; ++x) {
        for (int y = 0; y < C4_HEIGHT; ++y) {
            if(b[y][x] != 0) break;
            if (steadystate[y][x]=='@') {
                num_atp++;
                atp_move = x+1;
            } else if (steadystate[y][x]=='#') {
                num_hash++;
                hash_move = x+1;
            }
        }
    }
    if(num_atp == 1){
        return atp_move;
    }
    if(num_hash == 1){
        return hash_move;
    }

    // Second Priority: Claimeven and Claimodd
    // First, check there aren't 2 available claimparities
    int return_x = -1;
    vector<char> priorities(C4_WIDTH, 'x');
    for (int x = 0; x < C4_WIDTH; ++x) {
        bool even = true;
        for (int y = C4_HEIGHT - 1; y >= 0; --y) {
            even = !even;
            if (b[y][x] == 0) {
                char ss = steadystate[y][x];
                priorities[x] = ss;
                if ((ss == ' ' && even) || (ss == '|' && !even)){
                    if(return_x == -1){
                        return_x = x;
                    }
                    else
                        return -5;
                        //if(rand()%2==1) return_x = x;
                }
                break;
            }
        }
    }
    if(return_x != -1) return return_x+1;

    // Third Priority: Priority Markings
    int x = -1;
    for (char c : priority_list) {
        auto it = find(priorities.begin(), priorities.end(), c);
        if (it != priorities.end()) {
            auto next_it = find(next(it), priorities.end(), c);
            if (next_it == priorities.end()) {
                x = static_cast<int>(distance(priorities.begin(), it));
                break;
            } else {
                // Case of two equal priorities
                return -6;
            }
        }
    }

    int y = -1;
    for (int i = 0; i < C4_HEIGHT; ++i) {
        if (b[i][x] == 0) {
            y = i;
            break;
        }
    }

    if (y == -1 || x == -1) {
        return -4;
    }

    return x+1;
}

void SteadyState::drop(int x, char c){
    int y = C4_HEIGHT-1;
    for(y; y >= 0; y--){
        char here = steadystate[y][x];
        if(here != c && (here == ' ' || here == '|')){
            break;
        }
    }
    if(y>=0) steadystate[y][x] = c;
}

void SteadyState::mutate() {
    int r = rand()%10;

    if(r<2){
        //drop a miai pair
        char c = '@';
        for(int y = 0; y < C4_WIDTH; y++){
            for(int x = 0; x < C4_WIDTH; x++){
                if(steadystate[y][x] == c){
                    steadystate[y][x] = replacement_chars[rand()%replacement_chars.size()];
                }
            }
        }
        for(int i = 0; i < 2; i++){
            int x = rand()%C4_WIDTH;
            int y = C4_HEIGHT-1;
            for(y; y >= 0; y--)
                if(steadystate[y][x] != '1' && steadystate[y][x] != '2'){
                    break;
                }
            if(y>=0 && !is_miai(steadystate[y][x])) steadystate[y][x] = c;
            else i--;
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


    else {
        int x = rand()%C4_WIDTH;
        int ct = 0;
        char claim = r==9?'|':' ';
        for (int y = 0; y < C4_HEIGHT; ++y) {
            char c = steadystate[y][x];
            if(c != '1' && c != '2' && !is_miai(c)){
                ct++;
                steadystate[y][x] = claim;
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
            char c = steadystate[y][x];
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
    SteadyState steadyState;

    for (int y = 0; y < C4_HEIGHT; ++y) {
        for (int x = 0; x < C4_WIDTH; ++x) {
            int pc = b.piece_code_at(x, y);
            if (pc == 1) {
                steadyState.steadystate[y][x] = '1';
            } else if (pc == 2) {
                steadyState.steadystate[y][x] = '2';
            } else {
                steadyState.steadystate[y][x] = ' ';
            }
        }
    }

    steadyState.mutate();

    return steadyState;
}

C4Result SteadyState::play_one_game(const string& boardString) const {
    assert(boardString.size() % 2 == 1);
    C4Board board(boardString);

    C4Result winner = INCOMPLETE;
    while (true) {
        int randomColumn = board.random_legal_move();
        board.play_piece(randomColumn);
        winner = board.who_won();
        if(board.who_won() != INCOMPLETE) return winner;

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
        if(child.who_won() == RED) return true;
        if(!validate_steady_state(child, branches_searched)) return false;
    }

    return true;
}

shared_ptr<SteadyState> find_steady_state(const string& rep, int num_games) {
    //cout << "Searching for a steady state..." << endl;

    C4Board b(rep);

    // Convert the hash to a string
    ostringstream ss_hash_stream;
    ss_hash_stream << fixed << setprecision(numeric_limits<double>::max_digits10) << b.get_hash();
    string ss_hash = ss_hash_stream.str();

    // Check if a cached steady state file exists and read from it
    string cachedFilename = "steady_states/" + ss_hash + ".ss";
    if (ifstream(cachedFilename)) {
        shared_ptr<SteadyState> ss = make_shared<SteadyState>();
        ss->read_from_file(cachedFilename);
        ss->print();
        cout << "Loaded cached steady state from file " << cachedFilename << endl;
        int branches_searched = 0;
        if(!ss->validate_steady_state(b, branches_searched)){
            failout("Saved steadystate was caught losing!");
        }
        cout << "Steadystate validated over " << branches_searched << " branches!" << endl;
        return ss;
    }

    vector<SteadyState> steady_states;
    int best = 1;

    // Generate a lot of random steady states
    for (int i = 0; i < 100; ++i) {
        steady_states.push_back(create_random_steady_state(b));
    }
    int games_played = 0;

    while(true){
        int consecutive_wins = 0;
        int idx = rand()%steady_states.size(); // Randomly select a steady state
        while (true) {
            C4Result col;
            col = steady_states[idx].play_one_game(rep);
            games_played++;

            bool eliminate_agent = false;
            if (col != RED) {
                eliminate_agent = true;
            } else {
                if(consecutive_wins > 500){
                    int dont_care_how_many_branches = 0;
                    if(steady_states[idx].validate_steady_state(rep, dont_care_how_many_branches)) {
                        cout << "Steady state found after " << games_played << " games." << endl;
                        shared_ptr<SteadyState> ss = make_shared<SteadyState>(steady_states[idx]);
                        ss->print();
                        string filename = "steady_states/" + ss_hash + ".ss";
                        ss->write_to_file(filename);
                        return ss;
                    } else {
                        eliminate_agent = true;
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
                            steady_states[random_idx].steadystate[y][x] = steady_states[idx].steadystate[y][x];
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


// Method to populate char** array from array of strings
void populate_char_array(const array<string, C4_HEIGHT>& source, char dest[C4_HEIGHT][C4_WIDTH]) {
    for (int i = 0; i < C4_HEIGHT; ++i) {
        for(int j = 0; j < C4_WIDTH; ++j){
            dest[i][j] = source[i][j];
        }
    }
}

void steady_state_unit_tests_problem_1() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        "       ",
        "       ",
        " #1  ++",
        " 12  ==",
        "#21  --",
        "212  @@"
    };
    SteadyState ss(ss_list);
    int actual = -1;







    C4Board b1("233332212");
    actual = ss.query_steady_state(b1);
    cout << actual << endl;
    assert(actual == 1);
    cout << "Passed test 1!" << endl;


    C4Board b2("233332211");
    actual = ss.query_steady_state(b2);
    cout << actual << endl;
    assert(actual == 2);
    cout << "Passed test 2!" << endl;


    C4Board b3("233332216");
    actual = ss.query_steady_state(b3);
    cout << actual << endl;
    assert(actual == 7);
    cout << "Passed test 3!" << endl;


    C4Board b4("233332217");
    actual = ss.query_steady_state(b4);
    cout << actual << endl;
    assert(actual == 6);
    cout << "Passed test 4!" << endl;


    C4Board b5("233332213");
    actual = ss.query_steady_state(b5);
    cout << actual << endl;
    assert(actual == 3);
    cout << "Passed test 5!" << endl;


    C4Board b6("233332214");
    actual = ss.query_steady_state(b6);
    cout << actual << endl;
    assert(actual == 4);
    cout << "Passed test 6!" << endl;


    C4Board b7("23333221676");
    actual = ss.query_steady_state(b7);
    cout << actual << endl;
    assert(actual == 6);
    cout << "Passed test 7!" << endl;


    C4Board b8("233332217677");
    actual = ss.query_steady_state(b8);
    cout << actual << endl;
    assert(actual == 7);
    cout << "Passed test 8!" << endl;
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







    C4Board b1("43667555535337");
    actual = ss.query_steady_state(b1);
    cout << actual << endl;
    assert(actual == 5);
    cout << "Passed test 1!" << endl;


    C4Board b2("43667555535335");
    actual = ss.query_steady_state(b2);
    cout << actual << endl;
    assert(actual == 7);
    cout << "Passed test 2!" << endl;


    C4Board b3("43667555535334");
    actual = ss.query_steady_state(b3);
    cout << actual << endl;
    assert(actual == 4);
    cout << "Passed test 3!" << endl;


    C4Board b4("43667555535332");
    actual = ss.query_steady_state(b4);
    cout << actual << endl;
    assert(actual == 2);
    cout << "Passed test 4!" << endl;
}

void steady_state_unit_tests_problem_3() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        "       ",
        "       ",
        "       ",
        "       ",
        "      @",
        "       "
    };
    SteadyState ss(ss_list);
    int actual = -1;



    
    C4Board b1("");
    actual = ss.query_steady_state(b1);
    cout << actual << endl;
    assert(actual == -2);
    cout << "Passed test 1!" << endl;
}

void steady_state_unit_tests_problem_4() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        " =+2++|",
        " @211||",
        " 11221|",
        " 22112|",
        "-21212|",
        "112112|"
    };
    SteadyState ss(ss_list);
    int actual = -1;



    C4Board b1("");
    actual = ss.query_steady_state(b1);
    cout << actual << endl;
    assert(actual == -2);
    cout << "Passed test 1!" << endl;
}

void steady_state_unit_tests_problem_5() {
    // Define the initial board configuration
    array<string, C4_HEIGHT> ss_list = {
        "------+",
        "------+",
        "------+",
        "------+",
        "------+",
        "------+"
    };
    SteadyState ss(ss_list);
    int actual = -1;







    C4Board b1("121212");
    actual = ss.query_steady_state(b1);
    cout << actual << endl;
    assert(actual == 1);
    cout << "Passed test 1!" << endl;


    C4Board b2("123212");
    actual = ss.query_steady_state(b2);
    cout << actual << endl;
    assert(actual == 2);
    cout << "Passed test 2!" << endl;

    C4Board b3("4332322441");
    actual = ss.query_steady_state(b3);
    cout << actual << endl;
    assert(actual == 1);
    cout << "Passed test 3!" << endl;


    C4Board b4("55567667364242");
    actual = ss.query_steady_state(b4);
    cout << actual << endl;
    assert(actual == 4);
    cout << "Passed test 4!" << endl;
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







    C4Board b1("44444435525221");
    actual = ss.query_steady_state(b1);
    cout << actual << endl;
    assert(actual == 1);
    cout << "Passed test 1!" << endl;


    C4Board b2("44444435525222");
    actual = ss.query_steady_state(b2);
    cout << actual << endl;
    assert(actual == 5);
    cout << "Passed test 2!" << endl;


    C4Board b3("44444435525223");
    actual = ss.query_steady_state(b3);
    cout << actual << endl;
    assert(actual == 3);
    cout << "Passed test 3!" << endl;


    C4Board b4("44444435525225");
    actual = ss.query_steady_state(b4);
    cout << actual << endl;
    assert(actual == 2);
    cout << "Passed test 4!" << endl;


    C4Board b5("44444435525226");
    actual = ss.query_steady_state(b5);
    cout << actual << endl;
    assert(actual == 6);
    cout << "Passed test 5!" << endl;


    C4Board b6("44444435525227");
    actual = ss.query_steady_state(b6);
    cout << actual << endl;
    assert(actual == 7);
    cout << "Passed test 6!" << endl;


    C4Board b7("444444355252252116");
    actual = ss.query_steady_state(b7);
    cout << actual << endl;
    assert(actual == 6);
    cout << "Passed test 7!" << endl;

    string s = "";
    for(int i = 0; i < 1000; i++){
        assert(ss.play_one_game("4444443552522") == RED);
    }
}

void steady_state_unit_tests(){
    steady_state_unit_tests_problem_1();
    steady_state_unit_tests_problem_2();
    //steady_state_unit_tests_problem_3();
    //steady_state_unit_tests_problem_4();
    //steady_state_unit_tests_problem_5();
    steady_state_unit_tests_problem_6();
}
