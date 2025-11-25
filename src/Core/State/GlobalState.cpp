#pragma once

static unordered_map<string, double> global_state{
    {"frame_number", 0},
    {"t", 0},
    {"macroblock_number", 0},
    {"microblock_number", 0},
    {"macroblock_fraction", 0},
    {"microblock_fraction", 0},
};
void print_global_state(){
    cout << endl << endl << "=====GLOBAL STATE=====" << endl;
    for(const auto& pair : global_state){
        cout << pair.first << ": " << pair.second << endl;
    }
    cout << "======================" << endl << endl;
}
double get_global_state(string key){
    const auto& pair = global_state.find(key);
    if(pair == global_state.end()){
        // I used to error on this always. However, there is a general pattern in which
        // we give some scene "A" the ability to write to global state and give
        // scene "B"'s state manager a dependency on that variable. Depending on
        // the order in which the scenes are rendered, we get indeterminate behavior.
        // We could fix this by enforcing a render order, but I don't think
        // the complexity is, or ever will be worth it.
        // This zero-default behavior fixes the problem for the first frame,
        // after which scene "A" will have published to global state.
        if (global_state["microblock_fraction"] != 0) {
            print_global_state();
            throw runtime_error("global state access failed on element " + key);
        }
        return 0;
    }
    return pair->second;
}
