#include "Subscene.cpp"

inline int C4_RED           = 0xffdc267f;
inline int C4_YELLOW        = 0xffffb000;
inline int C4_EMPTY         = 0xff222222;

inline int IBM_ORANGE = 0xFFFE6100;
inline int IBM_PURPLE = 0xFF785EF0;
inline int IBM_BLUE   = 0xFF648FFF;
inline int IBM_GREEN  = 0xFF5EB134;

/**
 * C4Subscene class representing a subscene of video with C4 color format (32-bit RGBA).
 */
class C4Subscene : public Subscene {
public:
    double threat_diagram = 0;
    double spread = 0;
    vector<string> reduction_chars;
    vector<string> reduction_colors;
    string annotations = "                                          ";
    string highlight = "                                          ";
    Board board;
    double weight = 0;
    int threat_diagram_transition = 0; // 0=no transition 1=fadein 2=fadeout 3=collapse
    int diff_index = -1;
    // Constructor and other member functions specific to C4Subscene go here...

    C4Subscene(int width, int height, double td, double s, vector<string> r, vector<string> rc, Board b, string a, string h):Subscene(width, height),threat_diagram(td),spread(s),reduction_chars(r),reduction_colors(rc),board(b),annotations(a),highlight(h){}

    C4Subscene(const C4Subscene& subscene1, const C4Subscene& subscene2, double w):Subscene(subscene1.pixels.w, subscene1.pixels.h){
        threat_diagram =   lerp(subscene1.threat_diagram, subscene2.threat_diagram, smoother2(w));
        spread         =   lerp(subscene1.spread        , subscene2.spread        , smoother2(w));
        board          = c4lerp(subscene1.board         , subscene2.board         , w           );
        annotations = w>.5?subscene2.annotations:subscene1.annotations;
        highlight = w>.5?subscene2.highlight:subscene1.highlight;
        weight = w;

        if(subscene1.threat_diagram == 1 && subscene2.threat_diagram == 1){
            for(int i = 0; i < subscene1.reduction_chars.size(); i++)
                if(subscene1.reduction_chars[i] != subscene2.reduction_chars[i]){
                    diff_index = i;
                    break;
                }
            reduction_chars = subscene1.reduction_chars;
            reduction_colors = subscene1.reduction_colors;
            threat_diagram_transition = 3;
        }
        if(subscene1.threat_diagram == 1 && subscene2.threat_diagram == 0){
            reduction_chars = subscene1.reduction_chars;
            reduction_colors = subscene1.reduction_colors;
            threat_diagram_transition = 2;
        }
        if(subscene1.threat_diagram == 0 && subscene2.threat_diagram == 1){
            reduction_chars = subscene2.reduction_chars;
            reduction_colors = subscene2.reduction_colors;
            threat_diagram_transition = 1;
        }
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
            double ringsize = 1;
            if((col_id == 1 && (annotation == 'B' || annotation == 'R')) || (col_id == 2 && (annotation == 'B' || annotation == 'Y')) || (blink && spread == 0)){
                ringsize = .75;
            }
            int piece_fill_radius = ceil(stonewidth*(.4*ringsize));
            int piece_stroke_radius = ceil(stonewidth*(.4*ringsize+.07));
            pixels.fill_ellipse(px, py, piece_stroke_radius, piece_stroke_radius, col);
            pixels.fill_ellipse(px, py, piece_fill_radius  , piece_fill_radius  , colorlerp(col, BLACK, .4));
            return;
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
                pixels.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);  // Draw two rectangles to form a plus sign
                pixels.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, annotation_color);
                break;
            case '-':
                pixels.fill_rect(px - stonewidth/4 , py - stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);  // Draw a rectangle to form a minus sign
                break;
            case '|':
                pixels.fill_rect(px - stonewidth/16, py - stonewidth/4 , stonewidth/8, stonewidth/2, annotation_color);  // Draw a rectangle to form a vertical bar
                break;
            case '=':
                pixels.fill_rect(px - stonewidth/4 , py - 3*stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);  // Draw two rectangles to form an equal sign
                pixels.fill_rect(px - stonewidth/4 , py + stonewidth/16, stonewidth/2, stonewidth/8, annotation_color);
                break;
            case 'r':
            case 'y':
                pixels.fill_rect(px - stonewidth / 6, py - stonewidth / 8, stonewidth / 3, stonewidth / 12, annotation_color);  // Draw a rectangle to form a 't'
                pixels.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, annotation_color);
                break;
            case 'R':
            case 'Y':
                py += stonewidth/16;
                pixels.fill_rect(px - stonewidth / 4, py - stonewidth / 4, stonewidth / 2, stonewidth / 12, annotation_color);  // Draw a rectangle to form a 'T'
                pixels.fill_rect(px - stonewidth / 24, py - stonewidth / 4, stonewidth / 12, stonewidth / 2, annotation_color);
                break;
            case 'b':
                px -= stonewidth/8;
                py -= stonewidth/12;
                pixels.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 't'
                pixels.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
                px += stonewidth/4;
                py += stonewidth/6;
                pixels.fill_rect(px - stonewidth / 8, py - stonewidth / 12, stonewidth / 4, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 't'
                pixels.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
                break;
            case 'B':
                px -= stonewidth/12;
                py -= stonewidth/32;
                pixels.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_RED);  // Draw a rectangle to form a 'T'
                pixels.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_RED);
                px += stonewidth/4;
                py += stonewidth/6;
                pixels.fill_rect(px - stonewidth / 6, py - stonewidth / 6, stonewidth / 3, stonewidth / 16, C4_YELLOW);  // Draw a rectangle to form a 'T'
                pixels.fill_rect(px - stonewidth / 32, py - stonewidth / 6, stonewidth / 16, stonewidth / 3, C4_YELLOW);
                break;
            case ':':
                pixels.fill_ellipse(px, py - stonewidth / 8, stonewidth / 12, stonewidth / 12, annotation_color);
                pixels.fill_ellipse(px, py + stonewidth / 8, stonewidth / 12, stonewidth / 12, annotation_color);
                break;
            case '0':
                pixels.fill_ellipse(px, py, stonewidth / 4, stonewidth / 3, annotation_color);
                pixels.fill_ellipse(px, py, stonewidth / 9, stonewidth / 5, annotation_color);
                break;
            case 'o':
            case 'O':
            case 'c': // c is an o but colorful
                pixels.fill_ellipse(px, py, stonewidth / 3, stonewidth / 3, annotation_color);
                pixels.fill_ellipse(px, py, stonewidth / 6, stonewidth / 6, BLACK);
                break;
            case 'x':
                {
                    double rw = stonewidth*.3;
                    double rh = stonewidth*.3;
                    for(double dx = -rw+1; dx < rw; dx++)
                        for(double dy = -rh+1; dy < rh; dy++)
                            if(square(dx/rw)+square(dy/rh) < 1 && (abs(dx - dy) < stonewidth*.1 || abs(dx + dy) < stonewidth*.1))
                                pixels.set_pixel_with_transparency(px+dx, py+dy, annotation_color);
                    break;
                }
            case '.':
                if(col_id == 0)
                    pixels.fill_ellipse(px, py, stonewidth*.2, stonewidth*.2, annotation_color);
                break;
            default:
                break;
        }
    }

    void render_threat_diagram(){
        if(threat_diagram == 0) return;
        for(int x = 0; x < reduction_chars.size(); x++){
            string column = reduction_chars[x];
            for(int y = 0; y < column.size(); y++){
                char r = column.at(y);
                char rc = reduction_colors[x].at(y);
                int color = 0xff226688;
                if(rc == 'R' || rc == 'r') color = C4_RED;
                if(rc == 'Y' || rc == 'y') color = C4_YELLOW;
                string s = string(1,r);
                Pixels latex = eqn_to_pix("\\text{"+s+"}", pixels.w/640);
                latex.recolor(color);
                latex.mult_alpha(threat_diagram);
                double stonewidth = pixels.w/16.;
                double shifty = (x==diff_index && y != 0)?weight:0;
                double spreadx = lerp(1, 2.1, spread);
                double px = round((x-reduction_chars.size()/2.+.5)*stonewidth*spreadx+pixels.w/2);
                double py = round((-(y-shifty)-.5)*stonewidth+pixels.h);
                if(y == 0 && x == diff_index)
                    latex.mult_alpha(1-weight);
                pixels.copy(latex, px-latex.w/2, py-latex.h/2, 1);
            }
        }
    }

    void get_disk_screen_coordinates(int stonex, int stoney, double& px, double& py, double& stonewidth){
        stonewidth = pixels.w/16.;
        double spreadx = lerp(1, 2.1, spread);
        double spready = lerp(0, -.75 + 1.5 * (stonex%2), spread*(1-threat_diagram));
        px = round((stonex-WIDTH/2.+.5)*stonewidth*spreadx+pixels.w/2);
        py = round((-stoney+spready  +HEIGHT/2.-.5)*stonewidth+pixels.h/2) - (threat_diagram*pixels.h/8);
    }

    void draw_highlight(int px, int py, char highlight, double stonewidth, int height){
        int highlight_color = WHITE;
        if(highlight == 'd' || highlight == 'D') highlight_color = IBM_PURPLE; // dead space
        if(highlight == 't' || highlight == 'T') highlight_color = IBM_BLUE  ; // terminal threat
        if(highlight == 'z' || highlight == 'Z') highlight_color = IBM_GREEN ; // zugzwang controlling threat
        if(highlight == 'n' || highlight == 'N') highlight_color = IBM_ORANGE; // non-controlling threat
        double u = stonewidth * .4;
        for(int i = 0; i < 2; i++){
            pixels.rounded_rect(
                           px - u, // left x coord
                           py - u, // top y coord
                           2*u, // width
                           2*u + stonewidth*height, // height
                           u, // circle radius
                           highlight_color);
            highlight_color = colorlerp(highlight_color, BLACK, 0.75);
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

    /**
     * Implementation of the render() function for C4Subscene.
     */
    void render() override {
        // background
        pixels.fill(BLACK);
        draw_highlights();
        draw_board();
        render_threat_diagram();
    }
};
