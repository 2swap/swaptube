class ConwayGrid {
    ConwayGrid(int w, int h){
        if(w % 4 != 0 || w < 8) return;
        if(h % 4 != 0 || h < 8) return;
        w_cells = w;
        h_cells = h;
        w_bitboards = w/4-1;
        h_bitboards = h/4-1;
        board = new long long int[w_bitboards * h_bitboards];
        if (sizeof(long long int) != 8) throw runtime_error("long long int is not 8 bytes");
    }
    ~ConwayGrid(){
        delete[] board;
    }
private:
    long long int* board;
    int w_cells;
    int h_cells;
    int w_bitboards;
    int h_bitboards;
};
