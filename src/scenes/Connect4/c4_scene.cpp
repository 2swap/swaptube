#pragma once

inline int C4_RED           = 0xffff0000;
inline int C4_YELLOW        = 0xffffff00;
inline int C4_EMPTY         = 0xff222222;

#include <algorithm>

inline string empty_annotations = "......."
                                  "......."
                                  "......."
                                  "......."
                                  "......."
                                  ".......";
inline string empty_highlights =  "       "
                                  "       "
                                  "       "
                                  "       "
                                  "       "
                                  "       ";

class C4Scene : public Scene {
private:
    // transitions
    Board b1, b2;
    string annotations1, annotations2;
    string highlights1, highlights2;

    // non-transitions
    Board board;
    string representation = "";
    string annotations = empty_annotations;
    string highlights  = empty_highlights ;

public:
    C4Scene(const string& rep, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        :Scene(width, height), representation(rep), board(representation) {}

    void stage_transition(string final_rep){
        is_transition = true;
        b1 = board;
        b2 = Board(final_rep);
        annotations1 = annotations;
        annotations2 = annotations;
        highlights1 = highlights;
        highlights2 = highlights;

        rendered = false;
    }

    void post_transition(){
        board = b2;
        annotations = annotations2;
        highlights = highlights2;
        is_transition = false;
        representation = b2.representation;
    }

    void set_highlights(string s){highlights = s; rendered = false;}
    void set_annotations(string s){annotations = s; rendered = false;}
    string get_highlights(){return highlights;}
    string get_annotations(){return annotations;}

    void highlight_column(int index, char c, int bottom_row, int top_row) {
        // Ensure the index is within bounds
        if (index <= 0 || index > WIDTH) {
            cerr << "Highlight index out of bounds!" << endl;
            exit(0);
        }

        // For every row, set the character at the given column to c
        for (int i = 6-top_row; i < 6-bottom_row; i++) {
            highlights[i * 7 + index-1] = c;
        }
        rendered = false;
    }

    void highlight_unfilled(char c) {
        for (int x = 0; x < WIDTH; x++) {
            // Count the number of occurrences of the character 'x+1' in representation
            char columnChar = '1' + x; // since column numbers are 1-indexed
            int count = std::count(representation.begin(), representation.end(), columnChar);

            // Fill the unfilled parts of the highlights string
            for (int y = 6 - count - 1; y >= 0; y--) {
                highlights[y * 7 + x] = c;
            }
        }
        rendered = false;
    }

    void unhighlight() {
        highlights.assign(highlights.size(), ' ');
        rendered = false;
    }

    void unannotate() {
        annotations.assign(annotations.size(), '.');
        rendered = false;
    }

    void interpolate(){
        double w = static_cast<double>(time)/scene_duration_frames;
        board = c4lerp(b1, b2, w);
        annotations = w>.5?annotations1:annotations2;
        highlights  = w>.5?highlights1 :highlights2 ;
        rendered = false;
    }

    char get_annotation(int x, int y){
        return annotations[x+(HEIGHT-1-y)*WIDTH];
    }
    char get_highlight(int x, int y){
        return highlights[x+(HEIGHT-1-y)*WIDTH];
    }

    void draw_c4_disk(int px, int py, int col_id, bool blink, char annotation, bool highlighted, double stonewidth){
        int cols[] = {C4_EMPTY, C4_RED, C4_YELLOW};
        int col = cols[col_id];

        if(col_id != 0){
            bool any_blink = (col_id == 1 && (annotation == 'B' || annotation == 'R')) || (col_id == 2 && (annotation == 'B' || annotation == 'Y')) || blink;
            double piece_fill_radius = ceil(stonewidth*.4);
            double piece_stroke_radius = ceil(stonewidth*(.47));
            double blink_radius = ceil(stonewidth*(.2));
            if(annotation == '%') col = colorlerp(col, OPAQUE_BLACK, 0.8);
            pix.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, col);
            pix.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, OPAQUE_BLACK, any_blink?.8:.4));
        }

        Pixels latex = eqn_to_pix(latex_text(to_string(annotation)));
        pix.copy(latex, px, py);
    }

    void get_disk_screen_coordinates(int stonex, int stoney, double& px, double& py, double& stonewidth){
        stonewidth = min(pix.w/10., pix.h/10.);
        px = round((stonex-WIDTH/2.+.5)*stonewidth+pix.w/2);
        py = round((-stoney + HEIGHT/2.-.5)*stonewidth+pix.h/2);
    }

    void draw_highlight(int px, int py, char highlight, double stonewidth, int height){
        int highlight_color = OPAQUE_WHITE;
        if(highlight == 'd' || highlight == 'D') highlight_color = IBM_PURPLE; // dead space
        if(highlight == 't' || highlight == 'T') highlight_color = IBM_BLUE  ; // terminal threat
        if(highlight == 'z' || highlight == 'Z') highlight_color = IBM_GREEN ; // zugzwang controlling threat
        if(highlight == 'n' || highlight == 'N') highlight_color = IBM_ORANGE; // non-controlling threat
        double u = stonewidth * .4;
        for(int i = 0; i < 2; i++){
            highlight_color = colorlerp(highlight_color, OPAQUE_BLACK, 0.6);
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
        pix.fill(TRANSPARENT_BLACK);

        draw_highlights();
        draw_board();
    }

    void query(bool& done_scene, Pixels*& p) override {
        if (is_transition) interpolate();
        if (!rendered) {
            render_c4();
            rendered = true;
        }
        done_scene = time++>=scene_duration_frames;
        if(done_scene && is_transition) post_transition();
        p = &pix;
    }
};
