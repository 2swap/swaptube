
/**
 * C4Frame class representing a frame of video with C4 color format (32-bit RGBA).
 */
class C4Frame : public Frame {
public:
    double threat_diagram = 0;
    double spread = 0;
    vector<vector<char> > reduction_chars;
    vector<vector<char> > reduction_colors;
    // Constructor and other member functions specific to C4Frame go here...

    C4Frame(json curr){
        // caller does this: json curr = contents["sequence"][json_index];
        double threat_diagram = curr.contains("reduction") ? 1 : 0;
        double curr_spread = curr.value("spread", false) ? 1 : 0;
        reduction = curr["reduction"];
        reduction = curr["reduction_colors"];
    }

    /**
     * Implementation of the render() function for C4Frame.
     */
    Pixels render() const override {
        int json_index = content_index_to_json_index(board_index);
        pixels.fill(BLACK);
        Board b = boards[board_index];
        render_c4_board(b, static_cast<double>(time_in_this_block)/framerate);
        if(threat_diagram > 0) render_threat_diagram(-1, 0);
    }

    void draw_c4_disk(int px, int py, int col_id, bool blink, char annotation, bool highlighted, double t, double stonewidth, bool hide_blink){
        int cols[] = {C4_EMPTY, C4_RED, C4_YELLOW};
        int col = cols[col_id];

        double ringsize = 1;

        if(col_id != 0){
            if((blink && !hide_blink) || (col_id == 1 && (annotation == 'B' || annotation == 'R')) || (col_id == 2 && (annotation == 'B' || annotation == 'Y'))){
                double blink = bound(0, t, 1) * 2 * 3.14159; // one second transition from 0 to 2pi
                ringsize = 1-.3*(.5*(-cos(blink)+1));
            }
            int piece_fill_radius = ceil(stonewidth*(.4*ringsize));
            int piece_stroke_radius = ceil(stonewidth*(.4*ringsize+.07));
            p.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, col);
            p.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, BLACK, .4));
            return;
        }

        if(highlighted) col = BLACK;

        switch (annotation) {
            case '+':
                p.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, col);  // Draw two rectangles to form a plus sign
                p.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, col);
                break;
            case '-':
                p.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, col);  // Draw a rectangle to form a minus sign
                break;
            case '|':
                p.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, col);  // Draw a rectangle to form a vertical bar
                break;
            case '=':
                p.fill_rect(px - stonewidth/4 , py - 3*stonewidth/16, stonewidth/2, stonewidth/8, col);  // Draw two rectangles to form an equal sign
                p.fill_rect(px - stonewidth/4 , py + stonewidth/16, stonewidth/2, stonewidth/8, col);
                break;
            case 'r':
                p.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, C4_RED);  // Draw a rectangle to form a 't'
                p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_RED);
                break;
            case 'R':
                py += stonewidth/16;
                p.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, C4_RED);  // Draw a rectangle to form a 'T'
                p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_RED);
                break;
            case 'y':
                p.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, C4_YELLOW);  // Draw a rectangle to form a 't'
                p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_YELLOW);
                break;
            case 'Y':
                py += stonewidth/16;
                p.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, C4_YELLOW);  // Draw a rectangle to form a 'T'
                p.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, C4_YELLOW);
                break;
            case 'b':
                px -= stonewidth/8;
                py -= stonewidth/12;
                p.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 't'
                p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
                px += stonewidth/4;
                py += stonewidth/6;
                p.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 't'
                p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
                break;
            case 'B':
                px -= stonewidth/12;
                py -= stonewidth/32;
                p.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 'T'
                p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
                px += stonewidth/4;
                py += stonewidth/6;
                p.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 'T'
                p.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
                break;
            case ':':
                p.fill_ellipse(px, py - stonewidth / 8, stonewidth / 12, stonewidth / 12, col);
                p.fill_ellipse(px, py + stonewidth / 8, stonewidth / 12, stonewidth / 12, col);
                break;
            case '0':
                p.fill_ellipse(px, py, stonewidth / 4, stonewidth / 3, col);
                p.fill_ellipse(px, py, stonewidth / 9, stonewidth / 5, col);
                break;
            case 'o':
            case 'O':
                p.fill_ellipse(px, py, stonewidth / 3, stonewidth / 3, col);
                p.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, BLACK);
                break;
            case 'c': // c is an o but colorful
                p.fill_ellipse(px, py, stonewidth / 3, stonewidth / 3, 0xff0099cc);
                p.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, BLACK);
                break;
            case 'x':
                {
                    double rw = stonewidth*.3;
                    double rh = stonewidth*.3;
                    for(double dx = -rw+1; dx < rw; dx++)
                        for(double dy = -rh+1; dy < rh; dy++)
                            if(square(dx/rw)+square(dy/rh) < 1 && (abs(dx - dy) < stonewidth*.1 || abs(dx + dy) < stonewidth*.1))
                                p.set_pixel_with_transparency(px+dx, py+dy, col);
                    break;
                }
            case '.':
                if(col_id == 0)
                    p.fill_ellipse(px, py, stonewidth*.2, stonewidth*.2, col);
                break;
            default:
                break;
        }
    }

    void render_c4_board(Board b, double t){
        // background
        p.fill(BLACK);
        for(int stonex = 0; stonex < WIDTH; stonex++)
            for(int stoney = 0; stoney < HEIGHT; stoney++){
                char this_highlight = b.get_highlight(stonex, stoney);
                bool small = stoney == 0;
                char prev_highlight = small ? ' ' : b.get_highlight(stonex, stoney-1);
                if(this_highlight != prev_highlight && !small) this_highlight = ' ';
                double px = 0, py = 0, stonewidth = 0;
                get_disk_screen_coordinates(p, stonex, stoney, px, py, stonewidth);
                draw_highlight(px, py, this_highlight, stonewidth, small);
            }
        for(int stonex = 0; stonex < WIDTH; stonex++)
            for(int stoney = 0; stoney < HEIGHT; stoney++){
                char this_highlight = b.get_highlight(stonex, stoney);
                double px = 0, py = 0, stonewidth = 0;
                get_disk_screen_coordinates(p, stonex, stoney, px, py, stonewidth);
                draw_c4_disk(px, py, b.grid[stoney][stonex], b.blink[stoney][stonex], b.get_annotation(stonex, stoney), this_highlight != ' ', t, stonewidth, spread == 1);
            }
    }

    void render_threat_diagram(int diff_index, double weight){
        if(threat_diagram == 0) return;
        for(int x = 0; x < reduction.size(); x++){
            string column = reduction[x];
            for(int y = 0; y < column.size(); y++){
                char r = column.at(y);
                char rc = reduction_colors[x][y];
                int color = 0xff226688;
                if(rc == 'R' || rc == 'r') color = C4_RED;
                if(rc == 'Y' || rc == 'y') color = C4_YELLOW;
                string s = string(1,r);
                Pixels latex = eqn_to_pix("\\text{"+s+"}", p.w/640);
                latex.recolor(color);
                latex.mult_alpha(threat_diagram);
                double stonewidth = p.w/16.;
                double shifty = (x==diff_index && y != 0)?weight:0;
                double spreadx = lerp(1, 2.1, spread);
                double px = round((x-reduction.size()/2.+.5)*stonewidth*spreadx+p.w/2);
                double py = round((-(y-shifty)-.5)*stonewidth+p.h);
                if(y == 0 && x == diff_index)
                    latex.mult_alpha(1-weight);
                p.copy(latex, px-latex.w/2, py-latex.h/2, 1);
            }
        }
    }

    void get_disk_screen_coordinates(int stonex, int stoney, double& px, double& py, double& stonewidth){
        stonewidth = p.w/16.;
        double spreadx = lerp(1, 2.1, spread);
        double spready = lerp(0, -.75 + 1.5 * (stonex%2), spread*(1-threat_diagram));
        px = round((stonex-WIDTH/2.+.5)*stonewidth*spreadx+p.w/2);
        py = round((-stoney+spready  +HEIGHT/2.-.5)*stonewidth+p.h/2) - (threat_diagram*p.h/8);
    }

    void draw_highlight(Pixels& p, int px, int py, char highlight, double stonewidth, bool small){
        if (highlight == ' ') return;
        int highlight_color = WHITE;
        if(highlight == 'd' || highlight == 'D') highlight_color = IBM_PURPLE; // dead space
        if(highlight == 't' || highlight == 'T') highlight_color = IBM_BLUE  ; // terminal threat
        if(highlight == 'z' || highlight == 'Z') highlight_color = IBM_GREEN ; // zugzwang controlling threat
        if(highlight == 'n' || highlight == 'N') highlight_color = IBM_ORANGE; // non-controlling threat
        p.rounded_rect(px - stonewidth * .4, // left x coord
                       py - stonewidth * .4, // top y coord
                       stonewidth * .8, // width
                       stonewidth * ((small?0:1)+.8), // height
                       stonewidth * .4, // circle radius
                       highlight_color);
        highlight_color = lerp(highlight_color, BLACK, 0.75);
        p.rounded_rect(px - stonewidth * .33, // left x coord
                       py - stonewidth * .33, // top y coord
                       stonewidth * .66, // width
                       stonewidth * ((small?0:1)+.66), // height
                       stonewidth * .33, // circle radius
                       highlight_color);
    }
};

/**
 * Implementation of the interpolate_frame function for C4Frame.
 * Interpolates between two C4Frames and returns the resulting interpolated frame.
 *
 * @param frame1 The first frame for interpolation.
 * @param frame2 The second frame for interpolation.
 * @param t The interpolation parameter (ranging from 0 to 1).
 * @return The interpolated C4Frame at the given parameter t.
 */
static C4Frame interpolate_frame(const C4Frame& frame1, const C4Frame& frame2, double t) {
    // Implementation of interpolation specific to C4Frame...
    // You can access the members of frame1 and frame2, and perform the interpolation here.
    // Return the resulting interpolated C4Frame.
}
