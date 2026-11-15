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
    RubiksScene rs;
    stage_macroblock(SilenceBlock(1));

    quat yaw_quat = quat(cos(0.125 * M_PI), 0, sin(0.125 * M_PI), 0);
    quat pitch_quat = quat(cos(-0.098 * M_PI), sin(-0.098 * M_PI), 0, 0);
    quat combined_quat = pitch_quat * yaw_quat;

    rs.manager.transition(MACRO, {
        {"q1", to_string(combined_quat.u)},
        {"qi", to_string(combined_quat.i)},
        {"qj", to_string(combined_quat.j)},
        {"qk", to_string(combined_quat.k)},
        {"d", "1.4"},
        {"fov", "0.25"}
    });

    // d = 1.4, fov = 0.25

    
    rs.render_microblock();
    // open_ui(rs);


    stage_macroblock(SilenceBlock(10));
    
    rs.exec_move_from_slice("R");
    rs.render_microblock();

    rs.exec_move_from_slice("B'");
    rs.render_microblock();

    rs.exec_move_from_slice("B2");
    rs.render_microblock();

    rs.exec_move_from_slice("U");
    rs.render_microblock();

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.manager.set("internal_plastic_opacity", "0");

    rs.exec_move_from_slice("D");
    rs.render_microblock();

    rs.exec_move_from_slice("R");
    rs.render_microblock();

    rs.exec_move_from_slice("U'");
    rs.render_microblock();
    
    rs.manager.set("internal_plastic_opacity", "1");

    rs.exec_move_from_slice("R'");
    rs.render_microblock();

    rs.exec_move_from_slice("D'");
    rs.render_microblock();

    // get the hash of the cube after the T perm and print it
    double hash = rs.the_cube.get_hash(3);
    std::cout << "Hash of the cube after T perm: " << setprecision(10)<< hash << std::endl;






    stage_macroblock(SilenceBlock(10));
    rs.render_microblock();
}
