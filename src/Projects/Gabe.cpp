#include "../Scenes/Math/LambdaScene.cpp"

void render_video() {
    //string loop_part = "(\\x. (\\y. (((\\i. (i i)) x) y)))";
    //string loop_part = "(\\x. (\\y. ((x x) y)))";
    //string gabe = "(" + loop_part + " " + loop_part + ")";
    string pqpqp = "(\\p. (\\q. ((p q) p)))";
    string gabe = "((\\y. ((" + pqpqp + " y) " + pqpqp + ")) " + pqpqp + ")";
    shared_ptr<LambdaExpression> term = parse_lambda_from_string(gabe);
    term = term->reduce();

    term->flush_uid_recursive();

    LambdaScene ls(term);
    ls.stage_macroblock(SilenceBlock(1.33), 2);
    while(ls.microblocks_remaining()) {
        ls.reduce();
        ls.render_microblock();
    }
}
