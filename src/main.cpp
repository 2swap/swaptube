using namespace std;

#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <chrono> // chrono library for timing.

int WIDTH_BASE = 640;
int HEIGHT_BASE = 360;
int MULT = 2;

int VIDEO_WIDTH = WIDTH_BASE*MULT;
int VIDEO_HEIGHT = HEIGHT_BASE*MULT;
int VIDEO_FRAMERATE = 30;
double video_time_s = 0;
int video_num_frames = 0;

#include "audio_video/AudioSegment.cpp"
#include "audio_video/writer.cpp"

MovieWriter* WRITER;

#include "misc/inlines.h"
#include "scenes/Connect4/c4.h"
#include "misc/pixels.h"
#include "misc/convolver.cpp"
#include "misc/convolution_tests.cpp"
#include "scenes/scene.cpp"

void run_unit_tests(){
    run_inlines_unit_tests();
    run_convolution_unit_tests();
    run_c4_unit_tests();
}

void setup_writer(const string& project_name){
    // Create a new MovieWriter object and assign it to the pointer
    WRITER = new MovieWriter("../out/" + project_name + ".mp4",
                             "../out/" + project_name + ".srt",
                           "../media/" + project_name + "/");
    WRITER->init("../media/testaudio.mp3");
}

void render_claimeven() {
    CompositeScene composite;

    C4Scene c40("43334444343773");
    C4Scene c41("4344444636");
    C4Scene c42("43333734474443");
    C4Scene c43("233332123225632561");

    C4Scene transition1(c40, c41);
    C4Scene transition2(c41, c42);
    C4Scene transition3(c42, c43);

    c40.inject_audio_and_render(AudioSegment("intro_1.mp3", "To the untrained eye, this connect 4 position might look unassuming."));
    transition1.inject_audio_and_render(AudioSegment(1));
    c41.inject_audio_and_render(AudioSegment("intro_2.mp3", "This one, you might not even be able to tell who is winning."));
    transition2.inject_audio_and_render(AudioSegment(1));
    c42.inject_audio_and_render(AudioSegment("intro_3.mp3", "But an expert could tell you, all of these positions share a special property,"));
    transition3.inject_audio_and_render(AudioSegment(1));
    c43.inject_audio_and_render(AudioSegment("intro_4.mp3", "one which is actually quite extraordinary."));

    C4Scene tl = c40;
    C4Scene tr = c41;
    C4Scene bl = c42;
    C4Scene br = c43;

    composite.add_scene(&tl,             0,              0, .5);
    composite.add_scene(&tr, VIDEO_WIDTH/2,              0, .5);
    composite.add_scene(&bl,             0, VIDEO_HEIGHT/2, .5);
    composite.add_scene(&br, VIDEO_WIDTH/2, VIDEO_HEIGHT/2, .5);

    composite.inject_audio_and_render(AudioSegment("intro_5.mp3", "One feature of all of these positions is that the amount of space remaining in each column is even."));

    tl.highlight_column(1, 'd', 0, 6);
    composite.inject_audio_and_render(AudioSegment("intro_6.mp3", "This column has 6 empty spots."));
    tl.unhighlight();

    tl.highlight_column(7, 'd', 2, 6);
    composite.inject_audio_and_render(AudioSegment("intro_7.mp3", "This column has 4."));
    tl.unhighlight();

    composite.inject_audio_and_render(AudioSegment("intro_8.mp3", "This is true for all of the games I've shown."));
    composite.inject_audio_and_render(AudioSegment("intro_9.mp3", "Not a single column with an odd number of pieces."));
    composite.inject_audio_and_render(AudioSegment("intro_10.mp3", "What's more, in each of these cases, it so happens that Yellow, player 2, is winning."));
    composite.inject_audio_and_render(AudioSegment("intro_11.mp3", "It might not look like it, but Yellow is winning so spectacularly in these games, that even a fool could beat an expert. As long as that fool knows what makes these positions special."));


    c40.highlight_unfilled('d');
    c40.inject_audio_and_render(AudioSegment("intro_12.mp3", "Remember how all of the empty space is even?"));
    c40.set_highlight("dd  ddd"
                      "dd  ddd"
                      "DD  DDD"
                      "DD  DDD"
                      "dd  dd "
                      "dd  dd ");
    c40.inject_audio_and_render(AudioSegment("intro_13.mp3", "That means that we can cut those remaining columns up into groups of 2."));
    c40.inject_audio_and_render(AudioSegment("intro_14.mp3", "Now, since it's red's turn in all of these cases, let's let Red start by making a move."));
    c40.play("1");
    c40.inject_audio_and_render(AudioSegment("intro_15.mp3", "When drawn like this, yellow is almost beckoned to fill in the uncoupled pair."));
    c40.play("1");
    c40.inject_audio_and_render(AudioSegment("intro_16.mp3", "And, since the entire board is divied up into pairs, Yellow can just keep doing this forever."));
    c40.play("5");
    c40.inject_audio_and_render(AudioSegment("intro_17.mp3", "Wherever Red goes, yellow fills in the unpaired spot."));
    c40.play("5");
    c40.inject_audio_and_render(AudioSegment("We can even take away the pairings."));
    c40.unhighlight();
    c40.inject_audio_and_render(AudioSegment("All Yellow is doing is responding, thoughtlessly, to Red's last move."));
    c40.play("2");
    c40.inject_audio_and_render(AudioSegment("A randomly placed red stone, is followed by a Yellow stone immediately above."));

    C4Scene c40done("4333444434377311552222117777115555666622");
    C4Scene c40transition_to_done(c40, c40done);
    c40transition_to_done.inject_audio_and_render(AudioSegment("So on, and so forth."));
    c40done.inject_audio_and_render(AudioSegment("Wait a minute... did yellow just win?"));

    c41.inject_audio_and_render(AudioSegment("Let's try again, with a different board that we saw at the beginning."));
    c41.play("1");
    c41.inject_audio_and_render(AudioSegment("Remember, nothing fancy. Red initiates, then Yellow naively responds."));
    c41.play("1");

    C4Scene c41done("4344444636112222667777775511112266555533");
    C4Scene c41transition_to_done(c41, c41done);
    c41transition_to_done.inject_audio_and_render(AudioSegment("On and on, until the end of the game."));
    c41done.inject_audio_and_render(AudioSegment("Like magic, Yellow has won again."));

    composite.inject_audio_and_render(AudioSegment("Now, I promised you those boards at the beginning had something magical about them."));
    composite.inject_audio_and_render(AudioSegment("Does the columns having an even amount of empty space necessitate this winning condition for Yellow?"));

    C4Scene counterexample("4377345644465545");
    counterexample.inject_audio_and_render(AudioSegment("Yellow is winning in this position."));
    counterexample.inject_audio_and_render(AudioSegment("and it also has all even columns."));
    counterexample.inject_audio_and_render(AudioSegment("However, it is not one of the special ones I showed before. Let's try our strategy here!"));

    C4Scene redwinning("437734564446554533331111");
    C4Scene transitionredwinning(counterexample, redwinning);
    transitionredwinning.inject_audio_and_render(AudioSegment("Just like before, red then yellow."));
    redwinning.play("2");
    redwinning.inject_audio_and_render(AudioSegment("But with this last Red move, Yellow is starting to panic."));
    redwinning.inject_audio_and_render(AudioSegment("If Yellow continues our simple strategy, Red will immediately win."));

    C4Scene redwon("437734564446554533331111222");
    C4Scene transitionredwon(redwinning, redwon);
    transitionredwon.inject_audio_and_render(AudioSegment(1));
    C4Scene transitionrednotwon(redwon, redwinning);
    transitionrednotwon.inject_audio_and_render(AudioSegment(1));

    redwinning.inject_audio_and_render(AudioSegment("So, Yellow, in a last ditch effort, breaks our strategy."));
    redwinning.inject_audio_and_render(AudioSegment("Perhaps, by making some threats on the right side, Yellow can regain the tempo?"));

    C4Scene redwon2("437734564446554533331111255667");
    C4Scene transitionredwon2(redwinning, redwon2);
    transitionredwon2.inject_audio_and_render(AudioSegment(1));

    redwon2.inject_audio_and_render(AudioSegment("Unfortunately for Yellow, the damage is already done."));
    redwon2.play("2");
    redwon2.inject_audio_and_render(AudioSegment("Red immediately plays in the second column..."));
    redwon2.play("2");
    redwon2.inject_audio_and_render(AudioSegment("Yellow is forced to block the horizontal threat,"));
    redwon2.play("2");
    redwon2.inject_audio_and_render(AudioSegment("but Red wins on a diagonal anyways."));

    C4Scene goback(redwon2, counterexample);
    goback.inject_audio_and_render(AudioSegment("So, our naive call-and-response strategy didn't work in this case."));
    goback.inject_audio_and_render(AudioSegment("Despite the fact that yellow is winning, and all of the columns are even."));
    goback.inject_audio_and_render(AudioSegment("Indeed, it is the case that the 4 boards at the beginning weren't just any boards with all even columns."));
    goback.inject_audio_and_render(AudioSegment("They were specially selected, as boards which can be won with our simple call-and-response strategy."));
    goback.inject_audio_and_render(AudioSegment("If Yellow could always win by just playing call-and-response, that wouldn't be much fun, would it?"));

    composite.inject_audio_and_render(AudioSegment("So, what is it about these positions which make them special?"));
    c41.inject_audio_and_render(AudioSegment("When we play our naive follow-up strategy, let's take note of what happens."));
    c41.set_annotations("...x..."
                        "...x..."
                        "...x..."
                        "...x..."
                        "..xx..x"
                        "..xx..x");
    c41.inject_audio_and_render(AudioSegment("Let's mark all of the disks already played before we start."));
    c41done = C4Scene("4333444434377311552222117777115555666622");
    c41transition_to_done = C4Scene(c41, c41done);
    c41transition_to_done.inject_audio_and_render(AudioSegment(5));
    c41done.inject_audio_and_render(AudioSegment("Regardless of what Red chooses, after the point where we start to use our strategy,"));
    c41done.set_annotations("..xx..."
                            "ooxxooo"
                            "..xx..."
                            "ooxxooo"
                            "..xx..x"
                            "ooxxoox");
    c41done.inject_audio_and_render(AudioSegment("all of the red disks end up on an odd row."));
    c41done.inject_audio_and_render(AudioSegment("That is, the first, third, and fifth rows from the bottom."));
    c41done.set_annotations("ooxxooo"
                            "..xx..."
                            "ooxxooo"
                            "..xx..."
                            "ooxxoox"
                            "..xx..x");
    c41done.inject_audio_and_render(AudioSegment("Similarly, yellow inevitably gets all of the even rows."));
    c41done.inject_audio_and_render(AudioSegment("Since Yellow is the one employing this strategy, let's call it 'claimeven'."));
    c41done.set_annotations("..xx..."
                            "..xx..."
                            "..xx..."
                            "..xx..."
                            "..xx..x"
                            "..xx..x");
    c41done.inject_audio_and_render(AudioSegment("Yellow, by merely following Red, can get hold of all of the remaining even-row spaces."));
    c41done.inject_audio_and_render(AudioSegment("What makes this board special, is that even when we fill in all of the Red pieces on the odd rows, no line of 4 disks is made."));
    c41done.set_annotations("ooooooo"
                            "..xx..."
                            "..xx..."
                            "..xx..."
                            "..xx..x"
                            "..xx..x");
    c41done.inject_audio_and_render(AudioSegment("However, Yellow is able to lay claim to the entire top row."));
    c41done.inject_audio_and_render(AudioSegment("So, it doesn't matter the order which Red plays. The result is already guaranteed for Yellow, as long as Yellow plays Claimeven."));

    c42.inject_audio_and_render(AudioSegment("Well, what about this slightly different board?"));
    c42.inject_audio_and_render(AudioSegment("It was another one of the 4 at the beginning, so it also wins by claimeven."));
    c42.inject_audio_and_render(AudioSegment("If we imagine all the red stones filling in the odd rows..."));
    c42.set_annotations(  "..xx..."
                          "..xx..."
                          "..xx..."
                          "..xoooo"
                          "..xx..x"
                          "..xx..x");
    c42.inject_audio_and_render(AudioSegment("There is a line of 4!"));
    c42.set_annotations(  "..xx..."
                          "..xx..."
                          "..xx..."
                          "..xx..."
                          "..xoooo"
                          "..xx..x");
    c42.inject_audio_and_render(AudioSegment("But... in this case, it is hovering immediately over a Yellow line of 4."));
    c42.inject_audio_and_render(AudioSegment("We say that the red line of 4 is undercut."));
    c42.inject_audio_and_render(AudioSegment("So, for Red to win... Yellow would already have to have won."));
    c42.inject_audio_and_render(AudioSegment("This is what makes these boards special. If there are red winning lines on odd rows, they are all undercut by yellow winning chains on even rows."));

    C4Scene example1("4344377");
    example1.inject_audio_and_render(AudioSegment("Let's try some examples. It's Yellow's turn. How can we force a win using Claimeven?"));
    example1.inject_audio_and_render(AudioSegment("By placing this disk in the center column, Yellow can kill two birds with one stone. This move makes all of the columns even, and guarantees that Red can't make a line of 4 until after Yellow gets a line of 4 on the 4th row."));
    C4Scene example2("44137371444");
    C4Scene example3("43444446");

    HeaderScene h("uhhh");
    h.inject_audio_and_render(AudioSegment("Reducing a complex position to a simple one is difficult."));
    h.inject_audio_and_render(AudioSegment("However, an expert player up against a newbie might make it look easy."));
    h.inject_audio_and_render(AudioSegment("As expected, capitalizing on these situations on the fly is the trademark of expertise."));
    h.inject_audio_and_render(AudioSegment("Active study and deliberate practice can help you see it coming."));

    h = HeaderScene("Steady State Solutions");
    h.inject_audio_and_render(AudioSegment("Understanding these positions is not only crucial for the learner, but also for the connect 4 researcher."));
    h.inject_audio_and_render(AudioSegment("Solving a game requires an immense search of billions of options on behalf of either player."));
    h.inject_audio_and_render(AudioSegment("However, these positions from which a player can win via Claimeven prove to be a small island of simplicity in an otherwise chaotic sea of complex situations."));
    h.inject_audio_and_render(AudioSegment("These positions will serve as a foothold, an indispensable landmark for further reduction."));
    h.inject_audio_and_render(AudioSegment("Accompany me, dear viewer, and subscribe to follow along on how we can use this to deeper understand the structure of Connect 4."));

    C4Scene claimodd("");
    claimodd.inject_audio_and_render(AudioSegment("Before we part, I'll give you a taste of how this can be generalized, with one last puzzle."));
    C4Scene claimodd("43667555355335117");
    claimodd.inject_audio_and_render(AudioSegment("In this case, and unlike the others, Red is winning."));
    claimodd.inject_audio_and_render(AudioSegment("[solve the puzzle]"));

    TwoswapScene swap;
    swap.inject_audio(AudioSegment("outtro.mp3", "This has been 2swap."));
    swap.render();
}

int main(int argc, char* argv[]) {
    run_unit_tests();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config_file_without_.json>" << endl;
        exit(1);
    }

    const string project_name = string(argv[1]);

    setup_writer(project_name);

    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();
    render_claimeven();
    // Stop the timer.
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // Print out time stats
    double render_time_minutes = duration.count() / 60000000.0;
    double video_length_minutes = video_time_s/60.;
    cout << "Render time:  " << render_time_minutes << " minutes." << endl;
    cout << "Video length: " << video_length_minutes << " minutes." << endl;

    delete WRITER;

    return 0;
}