inline int C4_RED           = 0xffff0000;
inline int C4_YELLOW        = 0xffffff00;
inline int C4_EMPTY         = 0xff222222;

inline int IBM_ORANGE = 0xFFFE6100;
inline int IBM_PURPLE = 0xFF785EF0;
inline int IBM_BLUE   = 0xFF648FFF;
inline int IBM_GREEN  = 0xFF5EB134;

#include <algorithm>

class C4Scene : public Scene {
private:
    bool rendered = false;

    string representation = "";
    string annotations = "......."
                         "......."
                         "......."
                         "......."
                         "......."
                         ".......";
    string highlight  =  "       "
                         "       "
                         "       "
                         "       "
                         "       "
                         "       ";

public:
    Board board;
    int diff_index = -1;

    bool is_transition = false;
    Board b1, b2;
    string annotations1, annotations2;
    string highlight1, highlight2;

    C4Scene(const int width, const int height, const string& rep):Scene(width, height), representation(rep), board(representation) {}
    C4Scene(const string& rep):Scene(VIDEO_WIDTH, VIDEO_HEIGHT), representation(rep), board(representation) {}

    //interpolative constructor
    C4Scene(const C4Scene& subscene1, const C4Scene& subscene2):Scene(subscene1.w, subscene1.h) {
        assert(subscene1.w == subscene2.w);
        assert(subscene1.h == subscene2.h);
        cout << subscene1.w << subscene2.h << endl;
        is_transition = true;
        b1 = subscene1.board;
        b2 = subscene2.board;
        annotations1 = subscene1.annotations;
        annotations2 = subscene2.annotations;
        highlight1 = subscene1.highlight;
        highlight2 = subscene2.highlight;
    }

    void play(string s){
        representation += s;
        board = Board(representation);
        rendered = false;
    }

    void set_highlight(string s){highlight = s; rendered = false;}
    string get_highlight(){return highlight;}

    void highlight_column(int index, char c, int bottom_row, int top_row) {
        // Ensure the index is within bounds
        if (index <= 0 || index > WIDTH) {
            cerr << "Highlight index out of bounds!" << endl;
            exit(0);
        }

        // For every row, set the character at the given column to c
        for (int i = 6-top_row; i < 6-bottom_row; i++) {
            highlight[i * 7 + index-1] = c;
        }
        rendered = false;
    }

    void highlight_unfilled(char c) {
        for (int x = 0; x < WIDTH; x++) {
            // Count the number of occurrences of the character 'x+1' in representation
            char columnChar = '1' + x; // since column numbers are 1-indexed
            int count = std::count(representation.begin(), representation.end(), columnChar);

            // Fill the unfilled parts of the highlight string
            for (int y = 6 - count - 1; y >= 0; y--) {
                highlight[y * 7 + x] = c;
            }
        }
        rendered = false;
    }

    void unhighlight() {
        highlight.assign(highlight.size(), ' ');
        rendered = false;
    }

    void interpolate(){
        double w = static_cast<double>(time)/scene_duration_frames;
        board = c4lerp(b1, b2, w);
        annotations = w>.5?annotations1:annotations2;
        highlight   = w>.5?highlight1  :highlight2  ;
        rendered = false;
    }

    char get_annotation(int x, int y){
        return annotations[x+(HEIGHT-1-y)*WIDTH];
    }
    char get_highlight(int x, int y){
        return highlight[x+(HEIGHT-1-y)*WIDTH];
    }

    void draw_c4_disk(int px, int py, int col_id, bool blink, char annotation, bool highlighted, double stonewidth){
        int cols[] = {C4_EMPTY, C4_RED, C4_YELLOW};
        int col = cols[col_id];

        if(col_id != 0){
            bool any_blink = (col_id == 1 && (annotation == 'B' || annotation == 'R')) || (col_id == 2 && (annotation == 'B' || annotation == 'Y')) || blink;
            double piece_fill_radius = ceil(stonewidth*.4);
            double piece_stroke_radius = ceil(stonewidth*(.47));
            double blink_radius = ceil(stonewidth*(.2));
            pix.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, col);
            pix.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, BLACK, any_blink?.8:.4));
        }

        if(highlighted) col = BLACK;

        int annotation_color = 0;
        switch (annotation){
            case 'r':
            case 'R':
                annotation_color = C4_RED;
                break;
            case 'y':
            case 'Y':
                annotation_color = C4_YELLOW;
                break;
            case 'c': // c is an o but colorful
                annotation_color = 0xff0099cc;
                break;
            default:
                annotation_color = col;
                break;
        }

        switch (annotation) {
            case '+':
                pix.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);  // Draw two rectangles to form a plus sign
                pix.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, annotation_color);
                break;
            case '-':
                pix.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);  // Draw a rectangle to form a minus sign
                break;
            case '|':
                pix.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, annotation_color);  // Draw a rectangle to form a vertical bar
                break;
            case '=':
                pix.fill_rect(px - stonewidth/4 , py - 3*stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);  // Draw two rectangles to form an equal sign
                pix.fill_rect(px - stonewidth/4 , py + stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);
                break;
            case 'r':
            case 'y':
                pix.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, annotation_color);  // Draw a rectangle to form a 't'
                pix.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, annotation_color);
                break;
            case 'R':
            case 'Y':
                py += stonewidth/16;
                pix.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, annotation_color);  // Draw a rectangle to form a 'T'
                pix.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, annotation_color);
                break;
            case 'b':
                px -= stonewidth/8;
                py -= stonewidth/12;
                pix.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 't'
                pix.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
                px += stonewidth/4;
                py += stonewidth/6;
                pix.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 't'
                pix.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
                break;
            case 'B':
                px -= stonewidth/12;
                py -= stonewidth/32;
                pix.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 'T'
                pix.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
                px += stonewidth/4;
                py += stonewidth/6;
                pix.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 'T'
                pix.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
                break;
            case ':':
                pix.fill_ellipse(px, py - stonewidth / 8, stonewidth / 12, stonewidth / 12, annotation_color);
                pix.fill_ellipse(px, py + stonewidth / 8, stonewidth / 12, stonewidth / 12, annotation_color);
                break;
            case '0':
                pix.fill_ellipse(px, py, stonewidth / 4, stonewidth / 3, annotation_color);
                pix.fill_ellipse(px, py, stonewidth / 9, stonewidth / 5, annotation_color);
                break;
            case 'o':
            case 'O':
            case 'c': // c is an o but colorful
                pix.fill_ellipse(px, py, stonewidth / 3, stonewidth / 3, annotation_color);
                pix.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, BLACK);
                break;
            case 'x':
                {
                    double rw = stonewidth*.3;
                    double rh = stonewidth*.3;
                    for(double dx = -rw+1; dx < rw; dx++)
                        for(double dy = -rh+1; dy < rh; dy++)
                            if(square(dx/rw)+square(dy/rh) < 1 && (abs(dx - dy) < stonewidth*.1 || abs(dx + dy) < stonewidth*.1))
                                pix.set_pixel_with_transparency(px+dx, py+dy, annotation_color);
                    break;
                }
            case '.':
                if(col_id == 0)
                    pix.fill_ellipse(px, py, stonewidth*.2, stonewidth*.2, annotation_color);
                break;
            default:
                break;
        }
    }

    void get_disk_screen_coordinates(int stonex, int stoney, double& px, double& py, double& stonewidth){
        stonewidth = pix.w/16.;
        px = round((stonex-WIDTH/2.+.5)*stonewidth+pix.w/2);
        py = round((-stoney + HEIGHT/2.-.5)*stonewidth+pix.h/2);
    }

    void draw_highlight(int px, int py, char highlight, double stonewidth, int height){
        int highlight_color = WHITE;
        if(highlight == 'd' || highlight == 'D') highlight_color = IBM_PURPLE; // dead space
        if(highlight == 't' || highlight == 'T') highlight_color = IBM_BLUE  ; // terminal threat
        if(highlight == 'z' || highlight == 'Z') highlight_color = IBM_GREEN ; // zugzwang controlling threat
        if(highlight == 'n' || highlight == 'N') highlight_color = IBM_ORANGE; // non-controlling threat
        double u = stonewidth * .4;
        for(int i = 0; i < 2; i++){
            highlight_color = colorlerp(highlight_color, BLACK, 0.6);
            pix.rounded_rect(
                           px - u, // left x coord
                           py - u, // top y coord
                           2*u, // width
                           2*u + stonewidth*height, // height
                           u, // circle radius
                           highlight_color);
            u = stonewidth * .32;
        }
    }

    void draw_highlights(){
        for(int stonex = 0; stonex < WIDTH; stonex++)
            for(int stoney = 0; stoney < HEIGHT; stoney++){
                int height = -1;
                char this_highlight = get_highlight(stonex, stoney);
                if (this_highlight == ' ') continue;
                char next_highlight = 'x';
                while(stoney < HEIGHT){
                    next_highlight = get_highlight(stonex, stoney);
                    if(next_highlight != this_highlight) break;
                    stoney++;
                    height++;
                }
                stoney--;
                double px = 0, py = 0, stonewidth = 0;
                get_disk_screen_coordinates(stonex, stoney, px, py, stonewidth);
                draw_highlight(px, py, this_highlight, stonewidth, height);
            }
    }

    void draw_board(){
        for(int stonex = 0; stonex < WIDTH; stonex++)
            for(int stoney = 0; stoney < HEIGHT; stoney++){
                double px = 0, py = 0, stonewidth = 0;
                get_disk_screen_coordinates(stonex, stoney, px, py, stonewidth);
                draw_c4_disk(px, py, board.grid[stoney][stonex], board.blink[stoney][stonex], get_annotation(stonex, stoney), get_highlight(stonex, stoney) != ' ', stonewidth);
            }
    }

    void render_c4() {
        pix.fill(BLACK);
        draw_highlights();
        draw_board();
    }

    Pixels* query(bool& done_scene) override {
        if(is_transition) interpolate();
        if (!rendered) {
            render_c4();
            rendered = true;
        }
        done_scene = time++>=scene_duration_frames;
        return &pix;
    }
};
