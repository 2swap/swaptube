#include "../Scenes/KlotskiScene.cpp"
#include "../DataObjects/KlotskiBoard.cpp"

void render_video() {
    KlotskiBoard kb(6, 6, "a.cddda.ce....gebb..gffi...hhi..jjji", true);
    KlotskiScene ks(kb);

    ks.stage_macroblock(SilenceBlock(1), 1);
    ks.render_microblock();
    ks.export_png("thumb");
}
