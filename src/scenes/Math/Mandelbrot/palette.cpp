#pragma once

class Palette{
    std::vector<int> cols;
    public:
        Palette(){
            //Generate a random palette
            for(int i = 0; i < 30; i++){
                cols.push_back(rainbow(i/30.));
            }
        }
        int prompt(int i){
            return cols[i%cols.size()];
        }
};
