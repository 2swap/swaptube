#include "../Scenes/Common/MicrotoneScene.cpp"

typedef pair<int, int> Point;
typedef vector<Point> Tetromino;

vector<Tetromino> tetrominoes = {
    {{0,0}, {1,0}, {2,0}, {3,0}}, // I piece
    {{0,0}, {1,0}, {1,1}, {2,1}}, // S piece
    {{0,0}, {1,0}, {0,1}, {1,1}}, // O piece
    {{0,0}, {1,0}, {2,0}, {1,1}}, // T piece
    {{0,0}, {1,0}, {2,0}, {2,1}}, // L piece
    {{0,0}, {1,0}, {2,0}, {0,1}}, // J piece
    {{0,1}, {1,1}, {1,0}, {2,0}} // Z piece
};

Tetromino rotateTetromino(const Tetromino& tetromino) {
    Tetromino rotated;
    for (const auto& point : tetromino) {
        rotated.push_back({-point.second, point.first}); // 90-degree rotation
    }
    return rotated;
}

Tetromino shiftTetromino(const Tetromino& tetromino, int dx, int dy) {
    Tetromino shifted;
    for (const auto& point : tetromino) {
        shifted.push_back({point.first + dx, point.second + dy});
    }
    return shifted;
}

Tetromino generateTetromino() {
    int index = rand() % tetrominoes.size();
    Tetromino tetromino = tetrominoes[index];

    // Apply random rotations (0 to 3 times)
    int rotations = rand() % 4;
    for (int i = 0; i < rotations; ++i) {
        tetromino = rotateTetromino(tetromino);
    }

    // Apply random shift
    int dx = rand() % 3 - 1; // Shift between -2 and 2 on x-axis
    int dy = rand() % 3 - 1; // Shift between -2 and 2 on y-axis
    tetromino = shiftTetromino(tetromino, dx, dy);

    return tetromino;
}

Tetromino generateScatteredTetromino(int num_sounds) {
    Tetromino scattered;
    while (scattered.size() < num_sounds) {
        int x = rand() % 2-1; // Random x in range [-1, 1]
        int y = rand() % 3-1; // Random y in range [-1, 1]
        Point p = {x, y};

        // Ensure uniqueness
        bool unique = true;
        for (const auto& point : scattered) {
            if (point == p) {
                unique = false;
                break;
            }
        }
        if (unique) {
            scattered.push_back(p);
        }
    }
    /*int dx = rand() % 3 - 1; // Shift between -2 and 2 on x-axis
    int dy = rand() % 3 - 1; // Shift between -2 and 2 on y-axis
    scattered = shiftTetromino(scattered, dx, dy);*/
    return scattered;
}

void render_video() {
    srand(time(0));
    MicrotoneScene ms;

    FourierSound fs("recip.fft");
    StateSet init{ {"zoom", ".2"}, };
    int num_sounds = 1;
    for(int j = 0; j < num_sounds; j++){
        ms.add_sound(fs);
        init["circle"+to_string(j)+"_x"] = "0";
        init["circle"+to_string(j)+"_y"] = "0";
        init["circle"+to_string(j)+"_r"] = ".1";
    }
    ms.state_manager.set(init);
    for(int i = 0; i < 10; i++) {
        Tetromino rand_tet = generateScatteredTetromino(num_sounds);
        StateSet ss;
        for(int j = 0; j < num_sounds; j++){
            ss["circle"+to_string(j)+"_x"] = to_string(rand_tet[j].first);
            ss["circle"+to_string(j)+"_y"] = to_string(rand_tet[j].second);
        }
        ms.state_manager.macroblock_transition(ss);
        ms.inject_audio_and_render(SilenceSegment(.2));
        ms.inject_audio_and_render(SilenceSegment(.8));
    }
}
