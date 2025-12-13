#pragma once

#include "../Common/CoordinateScene.cpp"
#include "../../Host_Device_Shared/find_roots.c"

typedef uint64_t Bitboard;

extern "C" void iterate_conway(Bitboard* h_board, int w_bitboards, int h_bitboards);
extern "C" void draw_conway(Bitboard* h_board, int w_bitboards, int h_bitboards, unsigned int* h_pixels, int pixels_w, int pixels_h, glm::vec2 lx_ty, glm::vec2 rx_by);

class ConwayScene : public CoordinateScene {
private:
    // 10 by 10 grid of conway cells
    int grid_width = 5000;
    int grid_height = 5000;
    vector<Bitboard> grid = vector<Bitboard>(grid_width * grid_height, 0);
public:
    ConwayScene(const float width = 1, const float height = 1) : CoordinateScene(width, height) {
        Bitboard glider = 0;
        glider |= (1ULL << (1 + 0 * 8));
        glider |= (1ULL << (2 + 1 * 8));
        glider |= (1ULL << (0 + 2 * 8));
        glider |= (1ULL << (1 + 2 * 8));
        glider |= (1ULL << (2 + 2 * 8));
        // Place glider in the middle of the grid
        grid[grid_width / 2 + (grid_height / 2) * grid_width] = glider;

        for(int i = 0; i < grid_width * grid_height; i++) {
            // Inititialze as a random integer
            grid[i] = rand() & rand() & rand() & rand();
            grid[i] <<= 32;
            grid[i] |= rand() & rand() & rand() & rand();
        }
    }

    void draw() override {
        draw_conway(grid.data(), grid_width, grid_height, pix.pixels.data(), pix.w, pix.h,
                          glm::vec2(state["left_x"], state["top_y"]),
                          glm::vec2(state["right_x"], state["bottom_y"]));
        CoordinateScene::draw();
        iterate_conway(grid.data(), grid_width, grid_height);
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        return sq;
    }

    bool check_if_data_changed() const override { return CoordinateScene::check_if_data_changed() || true; }
};
