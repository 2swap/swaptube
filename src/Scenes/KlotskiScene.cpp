#pragma once

#include "Scene.cpp"
#include "../DataObjects/KlotskiBoard.h"
#include <string>
#include <vector>

class KlotskiScene : public Scene {
private:
    KlotskiBoard kb;
    char staged_char;
    int staged_dx;
    int staged_dy;

public:
    char highlight_char = '.';
    KlotskiScene(const int bw, const int bh, const string& rep, const bool rushhour, const double width = 1, const double height = 1)
        : Scene(width, height), kb(bw, bh, rep, rushhour) {
            state_manager.set({
                {"margin", "0.2"},
                {"dots", "1"},
            });
        }

    // No interactive state is needed for this static klotski scene.
    void on_end_transition(bool is_macroblock) override {
        if (staged_char != '.') {
            kb = kb.move_piece(staged_char, staged_dx, staged_dy);
        }
        staged_char = '.';
        highlight_char = '.';
    }
    KlotskiBoard copy_board() {
        return kb;
    }
    const StateQuery populate_state_query() const override {
        return StateQuery{"dots", "margin", "microblock_fraction", "t"};
    }
    void mark_data_unchanged() override { }
    void change_data() override { }
    bool check_if_data_changed() const override { return false; }

    void stage_move(char c, int dx, int dy) {
        staged_char = c;
        staged_dx = dx;
        staged_dy = dy;
    }

    void stage_random_move() {
        char sc;
        int sdx;
        int sdy;
        kb.get_random_move(sc, sdx, sdy);
        stage_move(sc, sdx, sdy);
    }

    // The draw method renders the board.
    void draw() override {
        // Get the viewframe dimensions.
        double view_width = get_width();
        double view_height = get_height();

        // Fixed size of a cell in pixels.
        double square_size = min(view_width / kb.w, view_height / kb.h) * .8;

        // Calculate total board size in pixels.
        double board_width = kb.w * square_size;
        double board_height = kb.h * square_size;

        // Center the board in the view.
        double offset_x = (view_width - board_width) / 2;
        double offset_y = (view_height - board_height) / 2;

        double margin = state["margin"] * square_size;
        double qm = margin * .25;
        double hm = margin * .5;
        double dots   = (kb.rushhour ? 1 : 0) * state["dots"];
        double microblock_fraction = state["microblock_fraction"];
        double micro  = smoothlerp(0,1,microblock_fraction);

        pix.fill_rect(offset_x-margin, offset_y-margin, hm, 2*margin + board_height, OPAQUE_WHITE);
        pix.fill_rect(offset_x-margin, offset_y-margin, 2*margin + board_width, hm, OPAQUE_WHITE);
        pix.fill_rect(offset_x+board_width+hm, offset_y-margin, hm, 2*margin + board_height, OPAQUE_WHITE);
        pix.fill_rect(offset_x-margin, offset_y+board_height+hm, 2*margin + board_width, hm, OPAQUE_WHITE);
        if (kb.rushhour && kb.w == 6 && kb.h == 6) {
            pix.fill_rect(offset_x+board_width+hm, offset_y+qm+square_size*2, hm, square_size-hm, TRANSPARENT_BLACK);
        }
        if (!kb.rushhour && kb.w == 4 && kb.h == 5) {
            pix.fill_rect(offset_x+qm+square_size, offset_y+board_height+hm, 2*square_size-hm, hm, TRANSPARENT_BLACK);
        }
        if(dots > 0.01){
            // Loop over every cell in the board.
            for (int y = 1; y < kb.h; y++) {
                for (int x = 1; x < kb.w; x++) {

                    // Calculate pixel coordinates for the block.
                    double rect_x = (offset_x + x * square_size) - qm;
                    double rect_y = (offset_y + y * square_size) - qm;

                    // Draw the block.
                    pix.fill_rect(rect_x, rect_y, dots*hm, dots*hm, OPAQUE_WHITE);
                }
            }
        }

        // Loop over every cell in the board.
        for (int y = 0; y < kb.h; y++) {
            for (int x = 0; x < kb.w; x++) {
                int index = y * kb.w + x;

                char cell = kb.representation[index];
                // Skip empty spaces.
                if (cell == '.')
                    continue;

                double mx = staged_char == cell ? staged_dx * micro * square_size : 0;
                double my = staged_char == cell ? staged_dy * micro * square_size : 0;

                // Determine the horizontal extent of the block.
                bool hor_ext = x < kb.w - 1 && cell == kb.representation[index+1];
                bool ver_ext = y < kb.h - 1 && cell == kb.representation[index+kb.w];
                bool mid_squ = x < kb.w - 1
                            && y < kb.h - 1
                            && cell == kb.representation[index+1+kb.w]
                            && cell == kb.representation[index+  kb.w]
                            && cell == kb.representation[index+1     ];

                // Calculate pixel coordinates for the block.
                double rect_x = (offset_x + x * square_size) + hm;
                double rect_y = (offset_y + y * square_size) + hm;
                double rect_width = square_size - margin;
                double rect_height = square_size - margin;

                // Simple pseudo-random color based on the character value.
                uint32_t color = rainbow(cell*.618034);

                double add = (cell==highlight_char ? .25-square(microblock_fraction - .5) : 0)*square_size;

                // Draw the block.
                            pix.fill_rect(mx+rect_x             -add, my+rect_y              -add, rect_width+add*2, rect_height+add*2, color);
                if(hor_ext) pix.fill_rect(mx+rect_x+rect_width-1-add, my+rect_y              -add, margin+2  +add*2, rect_height+add*2, color);
                if(ver_ext) pix.fill_rect(mx+rect_x             -add, my+rect_y+rect_height-1-add, rect_width+add*2, margin+2   +add*2, color);
                if(mid_squ) pix.fill_rect(mx+rect_x+rect_width-1-add, my+rect_y+rect_height-1-add, margin+2  +add*2, margin+2   +add*2, color);
            }
        }
    }
};

