#include "../DataObjects/Rubiks.h"
#include "../Scenes/Math/RubiksScene.h"
#include "../Scenes/Media/LatexScene.h"
#include "../Scenes/Media/Mp4Scene.h"
#include "../Scenes/Common/CompositeScene.h"
#include <memory>
#include "../Scenes/Math/RubiksGraphScene.h"
#include "../Scenes/Physics/RopeScene.h"
#include "../Core/State/StateTester.h"




void render_video(){
    RopeScene rs("io_in/loop_example_0", vec2(1, 1));
    rs.add_pin(vec2(0.3,0.2));
    rs.add_pin(vec2(0.7,0.2));
    rs.add_pin(vec2(0.9,0.7));

    stage_macroblock(SilenceBlock(5), 1);
    rs.render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    rs.remove_pin(1);
    rs.render_microblock();
}