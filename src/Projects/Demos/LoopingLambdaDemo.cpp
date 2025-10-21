#include "../Scenes/Math/LambdaScene.cpp"

void render_video() {

    // Option 1: period length 3
    //string loop_part = "(\\x. (\\y. (((\\i. (i i)) x) y)))";
    //string term_string = "(" + loop_part + " " + loop_part + ")";

    // Option 2: period length 2
    //string loop_part = "(\\x. (\\y. ((x x) y)))";
    //string term_string = "(" + loop_part + " " + loop_part + ")";

    // Option 3: period length 2
    string pqpqp = "(\\p. (\\q. ((p q) p)))";
    string term_string = "((\\y. ((" + pqpqp + " y) " + pqpqp + ")) " + pqpqp + ")";

    shared_ptr<LambdaExpression> term = parse_lambda_from_string(term_string);
    term = term->reduce();

    term->flush_uid_recursive();

    LambdaScene ls(term);
    ls.stage_macroblock(SilenceBlock(1.33), 2);
    while(remaining_microblocks_in_macroblock) {
        ls.reduce();
        ls.render_microblock();
    }
}
