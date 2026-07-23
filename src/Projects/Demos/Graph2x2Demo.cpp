#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/PngScene.h"
#include "../Scenes/Math/RubiksGraphScene.h"
#include "../Core/Smoketest.h"

void render_video() {
    RubiksGraphScene rgs;

    rgs.manager.set({
        {"physics_multiplier", "1"},
    });

    stage_macroblock(SilenceBlock(10), 2);
    rgs.add_cube("");
    rgs.render_microblock();
    rgs.render_microblock();
}
