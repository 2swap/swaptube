#include "KlotskiScene.h"
#include "../../Core/Color.h"

int piece_color(int cell){
    return rainbow(cell*.618034);
}

KlotskiScene::KlotskiScene(const KlotskiBoard& _kb, const vec2& dimensions)
    : Scene(dimensions), kb(_kb) {
        manager.set({
            {"margin", "0.2"},
            {"dots", kb.rushhour?"1":"0"},
            {"rainbow", "1"},
        });
    }

// No interactive state is needed for this static klotski scene.
void KlotskiScene::on_end_transition_extra_behavior(const TransitionType tt) {
    if (staged_move.piece != '.') kb = copy_staged_board();
    staged_move = {'.', 0, 0};
    if (tt == MACRO) highlight_char = '.';
}
KlotskiBoard KlotskiScene::copy_staged_board() {
    return kb.move_piece(staged_move);
}
KlotskiBoard KlotskiScene::copy_board() {
    return kb;
}
const StateQuery KlotskiScene::populate_state_query() const {
    return StateQuery{"dots", "margin", "microblock_fraction", "t", "rainbow"};
}
void KlotskiScene::mark_data_unchanged() { }
void KlotskiScene::change_data() { }
bool KlotskiScene::check_if_data_changed() const { return false; }

void KlotskiScene::stage_move(const KlotskiMove& km) {
    staged_move = km;
}

void KlotskiScene::stage_random_move() {
    KlotskiMove km{'.',0,0};
    kb.get_random_move(km);
    stage_move(km);
}

// The draw method renders the board.
void KlotskiScene::draw() {
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
    double dots   = state["dots"];
    double microblock_fraction = state["microblock_fraction"];
    double micro  = smoothlerp(0,1,microblock_fraction);

    vec2 horizontal_line_size(2*margin + board_width, hm);
    vec2 vertical_line_size(hm, 2*margin + board_height);
    vec2 top_left(offset_x-margin, offset_y-margin);
    vec2 top_right(offset_x+board_width+hm, offset_y-margin);
    vec2 bottom_left(offset_x-margin, offset_y+board_height+hm);
    pix.fill_rect(top_left, vertical_line_size, OPAQUE_WHITE);
    pix.fill_rect(top_left, horizontal_line_size, OPAQUE_WHITE);
    pix.fill_rect(top_right, vertical_line_size, OPAQUE_WHITE);
    pix.fill_rect(bottom_left, horizontal_line_size, OPAQUE_WHITE);
    if (kb.rushhour && kb.w == 6 && kb.h == 6) {
        pix.fill_rect(vec2(offset_x+board_width+hm, offset_y+qm+square_size*2), vec2(hm, square_size-hm), TRANSPARENT_BLACK);
    }
    if (!kb.rushhour && kb.w == 4 && kb.h == 5) {
        pix.fill_rect(vec2(offset_x+qm+square_size, offset_y+board_height+hm), vec2(2*square_size-hm, hm), TRANSPARENT_BLACK);
        pix.fill_rect(vec2(offset_x+hm+square_size, offset_y+board_height+hm), vec2(2*square_size-margin, hm), piece_color('b'));
    }
    if(dots > 0.01){
        vec2 dot_size(dots*hm, dots*hm);
        // Loop over every cell in the board.
        for (int y = 1; y < kb.h; y++) {
            for (int x = 1; x < kb.w; x++) {

                // Calculate pixel coordinates for the block.
                double rect_x = (offset_x + x * square_size) - dots*qm;
                double rect_y = (offset_y + y * square_size) - dots*qm;

                // Draw the block.
                pix.fill_rect(vec2(rect_x, rect_y), dot_size, OPAQUE_WHITE);
            }
        }
    }

    double rainbow_pct = state["rainbow"];

    // Loop over every cell in the board.
    for (int y = 0; y < kb.h; y++) {
        for (int x = 0; x < kb.w; x++) {
            int index = y * kb.w + x;

            char cell = kb.representation[index];
            // Skip empty spaces.
            if (cell == '.')
                continue;

            double mx = staged_move.piece == cell ? staged_move.dx * micro * square_size : 0;
            double my = staged_move.piece == cell ? staged_move.dy * micro * square_size : 0;

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
            uint32_t color = piece_color(cell);
            if(rainbow_pct<0.999) color = colorlerp(OPAQUE_WHITE, color, rainbow_pct);

            double triple_micro = microblock_fraction*3;
            double frac_triple_micro = triple_micro - static_cast<int>(triple_micro);
            if(cell!=highlight_char && highlight_char != '.') color = colorlerp(color, 0xff000000, .75);

            // Draw the block.
                        pix.fill_rect(vec2(mx+rect_x             , my+rect_y              ), vec2(rect_width, rect_height), color);
            if(hor_ext) pix.fill_rect(vec2(mx+rect_x+rect_width-1, my+rect_y              ), vec2(margin+2  , rect_height), color);
            if(ver_ext) pix.fill_rect(vec2(mx+rect_x             , my+rect_y+rect_height-1), vec2(rect_width, margin+2   ), color);
            if(mid_squ) pix.fill_rect(vec2(mx+rect_x+rect_width-1, my+rect_y+rect_height-1), vec2(margin+2  , margin+2   ), color);
        }
    }
}
