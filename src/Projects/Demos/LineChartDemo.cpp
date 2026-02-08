#include "../Scenes/Math/LineChartScene.cpp"

void render_video() {
    LineChartScene lcs;
    vector<int> values = {1, 7, 49, 238, 1120, 4263, 16422, 54859, 184275, 558186, 1662623};
    stage_macroblock(SilenceBlock(10), values.size());
    for(int val : values){
        lcs.add_data_point(val);
        lcs.render_microblock();
    }
}
