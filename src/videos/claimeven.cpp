
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

    tl.highlight_unfilled('d');
    tr.highlight_unfilled('d');
    bl.highlight_unfilled('d');
    br.highlight_unfilled('d');
    composite.inject_audio_and_render(AudioSegment("intro_8.mp3", "This is true for all of the games I've shown."));
    composite.inject_audio_and_render(AudioSegment("intro_9.mp3", "Not a single column with an odd number of pieces."));
    tl.unhighlight();
    tr.unhighlight();
    bl.unhighlight();
    br.unhighlight();
    composite.inject_audio_and_render(AudioSegment("intro_10.mp3", "What's more, in each of these cases, it so happens that Yellow, player 2, is winning."));
    composite.inject_audio_and_render(AudioSegment("intro_11.mp3", "It might not look like it, but Yellow is winning so spectacularly in these games, that even a fool could beat an expert. As long as that fool knows what makes these positions special."));


    c40.highlight_unfilled('d');
    c40.inject_audio_and_render(AudioSegment("intro_12.mp3", "Remember how all of the empty space is even?"));
    c40.set_highlights("dd  ddd"
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
    c40.inject_audio_and_render(AudioSegment("intro_18.mp3", "We can even take away the pairings."));
    c40.unhighlight();
    c40.inject_audio_and_render(AudioSegment("intro_19.mp3", "All Yellow is doing is responding, thoughtlessly, to Red's last move."));
    c40.play("2");
    c40.inject_audio_and_render(AudioSegment("intro_20.mp3", "A randomly placed red stone, is followed by a Yellow stone immediately above."));

    c40.stage_transition("4333444434377311552222117777115555666622");
    c40.inject_audio_and_render(AudioSegment("intro_21.mp3", "So on, and so forth. [silence]"));
    c40.inject_audio_and_render(AudioSegment("intro_22.mp3", "Wait a minute... did yellow just win?"));

    c41 = C4Scene(c40, c41);
    c41.inject_audio_and_render(AudioSegment("intro_23.mp3", "Let's try again, with a different board which we saw at the beginning."));
    c41.play("1");
    c41.inject_audio_and_render(AudioSegment("intro_24.mp3", "Remember, nothing fancy. Red initiates, then Yellow naively responds."));
    c41.play("1");

    c41.stage_transition("4344444636112222667777775511112266555533");
    c41.inject_audio_and_render(AudioSegment("intro_25.mp3", "On and on, until the end of the game. [silence]"));
    c41.inject_audio_and_render(AudioSegment("intro_26.mp3", "Like magic, Yellow has won again."));

    composite.inject_audio_and_render(AudioSegment("counterexample_1.mp3", "Now, I promised you those boards at the beginning had something magical about them."));
    composite.inject_audio_and_render(AudioSegment("counterexample_2.mp3", "Does the columns having an even amount of empty space necessitate this winning condition for Yellow?"));

    C4Scene counterexample("4377345644465545");
    counterexample.inject_audio_and_render(AudioSegment("counterexample_3.mp3", "Yellow is winning in this position."));
    counterexample.highlight_unfilled('d');
    counterexample.inject_audio_and_render(AudioSegment("counterexample_4.mp3", "and it also has all even columns."));
    counterexample.unhighlight();
    counterexample.inject_audio_and_render(AudioSegment("counterexample_5.mp3", "However, it is not one of the special ones I showed before. Let's try our strategy here!"));

    counterexample.stage_transition("437734564446554533331111");
    counterexample.inject_audio_and_render(AudioSegment("counterexample_6.mp3", "Just like before, red then yellow."));
    counterexample.play("2");
    counterexample.inject_audio_and_render(AudioSegment("counterexample_7.mp3", "But with this last Red move, Yellow is starting to panic."));
    counterexample.inject_audio_and_render(AudioSegment("counterexample_8.mp3", "If Yellow continues our simple strategy, Red will immediately win."));

    counterexample.stage_transition("437734564446554533331111222");
    counterexample.inject_audio_and_render(AudioSegment(3));
    counterexample.stage_transition("4377345644465545333311112");
    counterexample.inject_audio_and_render(AudioSegment(3));

    counterexample.inject_audio_and_render(AudioSegment("counterexample_9.mp3", "So, Yellow, in a last ditch effort, breaks our strategy."));
    counterexample.inject_audio_and_render(AudioSegment("counterexample_10.mp3", "Perhaps, by making some threats on the right side, Yellow can regain the tempo?"));

    counterexample.stage_transition("437734564446554533331111255667");
    counterexample.inject_audio_and_render(AudioSegment(4));

    counterexample.inject_audio_and_render(AudioSegment("counterexample_11.mp3", "Unfortunately for Yellow, the damage is already done."));
    counterexample.play("2");
    counterexample.inject_audio_and_render(AudioSegment("counterexample_12.mp3", "Red immediately plays in the second column..."));
    counterexample.play("2");
    counterexample.inject_audio_and_render(AudioSegment("counterexample_13.mp3", "Yellow is forced to block the horizontal threat,"));
    counterexample.play("2");
    counterexample.inject_audio_and_render(AudioSegment("counterexample_14.mp3", "but Red wins on a diagonal anyways."));

    counterexample.stage_transition("4377345644465545");
    counterexample.inject_audio_and_render(AudioSegment(2));
    counterexample.inject_audio_and_render(AudioSegment("counterexample_15.mp3", "So, our naive just-follow-red strategy didn't work in this case,"));
    counterexample.inject_audio_and_render(AudioSegment("counterexample_16.mp3", "Despite the fact that yellow is winning, and all of the columns are even."));
    counterexample.inject_audio_and_render(AudioSegment("counterexample_17.mp3", "Indeed, it is the case that the 4 boards at the beginning weren't just any boards with all even columns."));
    counterexample.inject_audio_and_render(AudioSegment("counterexample_18.mp3", "They were especially selected, as boards which can be won with our simple follow-red strategy."));
    counterexample.inject_audio_and_render(AudioSegment("counterexample_19.mp3", "If Yellow could always win by just playing call-and-response, that wouldn't be much fun, would it?"));

    composite.inject_audio_and_render(AudioSegment("analysis_1.mp3", "So, what is it about these positions which makes them special?"));
    c41 = C4Scene("43334444343773");
    c41.inject_audio_and_render(AudioSegment("analysis_2.mp3", "When we play our naive strategy, let's take note of what happens."));
    c41.set_annotations("..%%..."
                        "..%%..."
                        "..%%..."
                        "..%%..."
                        "..%%..%"
                        "..%%..%");
    c41.inject_audio_and_render(AudioSegment("analysis_3.mp3", "Let's mark all of the disks already played before we start."));
    c41.stage_transition("4333444434377311552222117777115555666622");
    c41.inject_audio_and_render(AudioSegment(7));
    c41.inject_audio_and_render(AudioSegment("analysis_4.mp3", "Regardless of what Red chooses, after the point where we start to use our strategy,"));
    c41.set_annotations("%%%%%%%"
                        "..%%..."
                        "%%%%%%%"
                        "..%%..."
                        "%%%%%%%"
                        "..%%..%");
    c41.inject_audio_and_render(AudioSegment("analysis_5.mp3", "all of the red disks end up on an odd row."));
    c41.inject_audio_and_render(AudioSegment("analysis_6.mp3", "That is, the first, third, and fifth rows from the bottom."));
    c41.set_annotations("..%%..."
                        "%%%%%.%"
                        "..%%..."
                        "%%%%%%%"
                        "..%%..%"
                        "%%%%%%%");
    c41.inject_audio_and_render(AudioSegment("analysis_7.mp3", "Similarly, yellow inevitably gets all of the even rows."));
    c41.inject_audio_and_render(AudioSegment("analysis_8.mp3", "Since Yellow is the one employing this strategy, let's call it 'claimeven'."));
    c41.inject_audio_and_render(AudioSegment("analysis_9.mp3", "Yellow, by merely following Red, can get hold of all of the remaining even-row spaces."));
    c41.set_annotations("%%%%%.%"
                        "...%..."
                        "%%..%%%"
                        "..%%..."
                        "%%..%%."
                        "..%...%");
    c41.inject_audio_and_render(AudioSegment("analysis_10.mp3", "What makes this board special, is that even when we fill in all of the Red pieces on the odd rows, no line of 4 disks is made."));
    c41.set_annotations("......."
                        "%%%%%.%"
                        "%%%%%%%"
                        "%%%%%%%"
                        "%%%%%%%"
                        "%%%%%%%");
    c41.inject_audio_and_render(AudioSegment("analysis_11.mp3", "However, Yellow is able to lay claim to the entire top row."));
    c41.unannotate();
    c41.inject_audio_and_render(AudioSegment("analysis_12.mp3", "So, it doesn't matter the order which Red plays. The result is already guaranteed for Yellow, as long as Yellow plays Claimeven."));

    c42 = C4Scene(c41, c42);
    c42.inject_audio_and_render(AudioSegment("analysis_13.mp3", "Well, what about this slightly different board?"));
    c42.inject_audio_and_render(AudioSegment("analysis_14.mp3", "It was another one of the 4 at the beginning, so it also wins by claimeven."));
    c42.set_annotations("......."
                        "......."
                        "......."
                        "....rrr"
                        "......."
                        ".......");
    c42.inject_audio_and_render(AudioSegment("analysis_15.mp3", "If we imagine all the red stones filling in the odd rows..."));
    c42.inject_audio_and_render(AudioSegment("analysis_16.mp3", "There's a red line of 4!"));
    c42.set_annotations("......."
                        "......."
                        "......."
                        "......."
                        "....yy."
                        ".......");
    c42.inject_audio_and_render(AudioSegment("analysis_17.mp3", "But... in this case, it is hovering immediately over a Yellow line of 4."));
    c42.inject_audio_and_render(AudioSegment("analysis_18.mp3", "So, for Red to win... Yellow would already have to have won."));
    c42.inject_audio_and_render(AudioSegment("analysis_19.mp3", "We say that the red line of 4 is undercut."));
    c42.unhighlight();
    c42.inject_audio_and_render(AudioSegment("analysis_20.mp3", "Let's play it out!"));
    c42.stage_transition("43333734474443227711227722555511551166");
    c42.inject_audio_and_render(AudioSegment(7));
    c42.inject_audio_and_render(AudioSegment("analysis_21.mp3", "Sure enough, we win on the precise line that we predicted!"));
    c42.inject_audio_and_render(AudioSegment("analysis_22.mp3", "This is what makes these boards special. If there are red winning lines on odd rows, they are all undercut by yellow winning chains on even rows."));
    c42.stage_transition("");
    c42.inject_audio_and_render(AudioSegment(1));

    C4Scene example("");
    example.stage_transition("4344377");
    example.inject_audio_and_render(AudioSegment("examples_1.mp3", "Let's try some examples."));
    example.inject_audio_and_render(AudioSegment("examples_2.mp3", "It's Yellow's turn. How can we force a win using Claimeven?"));
    example.inject_audio_and_render(AudioSegment("examples_3.mp3", "Pause now if you want to think about it."));
    example.inject_audio_and_render(AudioSegment(2));
    example.play("4");
    example.inject_audio_and_render(AudioSegment("examples_4.mp3", "By placing this disk in the center column, Yellow can kill two birds with one stone."));
    example.set_highlights("ddddddd"
                           "ddddddd"
                           "DDD DDD"
                           "DDD DDD"
                           "dd  dd "
                           "dd  dd ");
    example.inject_audio_and_render(AudioSegment("examples_5.mp3", "This move makes all of the columns even,"));
    example.unhighlight();
    example.stage_transition("434437742233116644667777113366551122");
    example.inject_audio_and_render(AudioSegment("examples_6.mp3", "and guarantees that Red can't make a line of 4 until after Yellow gets a line of 4 on the 4th row."));
    example.inject_audio_and_render(AudioSegment(4));
    example.stage_transition("434444466");
    example.inject_audio_and_render(AudioSegment(2));
    example.inject_audio_and_render(AudioSegment("examples_7.mp3", "Let's try this trickier example."));
    example.stage_transition("4344444663");
    example.inject_audio_and_render(AudioSegment("examples_8.mp3", "We may naively play the spot which yields even amounts of empty space in each column."));
    example.inject_audio_and_render(AudioSegment("examples_9.mp3", "But red has other plans in mind. Let's see what happens if we try to play Claimeven."));
    example.play("7");
    example.inject_audio_and_render(AudioSegment("examples_10.mp3", "Red claims this spot on the seventh column."));
    example.set_annotations("......."
                           "......."
                           "......."
                           "....R.."
                           "......."
                           ".......");
    example.inject_audio_and_render(AudioSegment("examples_11.mp3", "See how that makes this threat on an odd row? Pay attention to that threat."));
    example.play("7");
    example.inject_audio_and_render(AudioSegment("examples_12.mp3", "Oblivious, Yellow continues playing Claimeven. Let's see what happens."));
    example.stage_transition("434444466377555");
    example.inject_audio_and_render(AudioSegment(2));
    example.inject_audio_and_render(AudioSegment("examples_13.mp3", "Well, that didn't work."));
    example.unannotate();
    example.stage_transition("434444466");
    example.inject_audio_and_render(AudioSegment("examples_14.mp3", "So then, what was yellow supposed to do?"));
    example.inject_audio_and_render(AudioSegment("examples_15.mp3", "Pause now if you want to think about it."));
    example.inject_audio_and_render(AudioSegment(2));
    example.stage_transition("4344444667");
    example.inject_audio_and_render(AudioSegment("examples_16.mp3", "The trick is blocking that third row threat before it comes to fruition."));
    example.set_highlights("dd  dd "
                           "dd  dd "
                           "DD  DD "
                           "DD  DD "
                           "dd  d  "
                           "dd  d  ");
    example.inject_audio_and_render(AudioSegment("examples_17.mp3", "Now, most columns have an even amount of empty space."));
    example.set_highlights("dd  dd "
                           "dd  dd "
                           "DD  DD "
                           "DD  DD "
                           "ddz d z"
                           "dd  d  ");
    example.inject_audio_and_render(AudioSegment("examples_18.mp3", "But two columns don't."));
    example.inject_audio_and_render(AudioSegment("examples_19.mp3", "Is there any hope of using Claimeven?"));
    example.inject_audio_and_render(AudioSegment("examples_20.mp3", "It turns out, here, the trick is as follows:"));
    example.stage_transition("434444466737");
    example.inject_audio_and_render(AudioSegment("examples_21.mp3", "If Red plays in one of these spots, then Yellow plays the other."));
    example.set_highlights("ddd ddd"
                           "ddd ddd"
                           "DDD DDD"
                           "DDD DDD"
                           "ddz d z"
                           "dd  d  ");
    example.inject_audio_and_render(AudioSegment("examples_22.mp3", "From there, we can continue with Claimeven."));
    example.unhighlight();
    example.stage_transition("43444446673722112233667766772233111155");
    example.inject_audio_and_render(AudioSegment(5));
    example.inject_audio_and_render(AudioSegment("examples_23.mp3", "Reducing a complex position to a simple one is difficult."));

    example.stage_transition("4344444667");
    example.inject_audio_and_render(AudioSegment("examples_24.mp3", "However, an expert player up against a newbie might make it look easy."));
    example.set_highlights("       "
                           "       "
                           "       "
                           "       "
                           "  z   z"
                           "       ");
    example.inject_audio_and_render(AudioSegment("examples_25.mp3", "As expected, capitalizing on these situations on the fly is the hallmark of expertise."));
    example.stage_transition("43444446673722112233667766772233111155");
    example.inject_audio_and_render(AudioSegment("examples_26.mp3", "Attentive practice can help you see it coming."));
    example.inject_audio_and_render(AudioSegment(2));
    example.unhighlight();

    C4Scene claimodd(example, C4Scene(""));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_2.mp3", "Before we part, I'll give you a taste of how this can be generalized, with one last puzzle."));
    claimodd.stage_transition("43667555355335117");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_3.mp3", "In this case, and unlike the others, Red, AKA player 1, is winning. But for now, it's Yellow's turn."));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_4.mp3", "How can we apply our understanding of claimeven to this position?"));
    claimodd.set_highlights("   d   "
                            "   d   "
                            "   d   "
                            "   d   "
                            "   d   "
                            "       ");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_5.mp3", "For one, we notice that there is precisely one column that doesn't have an even amount of spaces left."));
    claimodd.unhighlight();
    claimodd.inject_audio_and_render(AudioSegment("claimodd_6.mp3", "This means we can't play claimeven."));
    claimodd.stage_transition("436675553553351171");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_7.mp3", "If Yellow goes anywhere but the middle,"));
    claimodd.set_highlights("       "
                            "       "
                            "d      "
                            "d      "
                            "   d   "
                            "   d   ");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_8.mp3", "we find that there are two even spaces we need to take."));
    claimodd.unhighlight();
    claimodd.undo(1);
    claimodd.inject_audio_and_render(AudioSegment("claimodd_10.mp3", "So, Claimeven doesn't exactly work when you are player 1."));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_11.mp3", "Well, what happens if we just naively respond to Yellow anyways?"));
    claimodd.set_highlights("DDDd DD"
                            "DDDD DD"
                            "dd D dd"
                            "dd d dd"
                            " D d   "
                            " D     ");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_12.mp3", "We'll follow this pattern here, Red will get the top of each coupled pair of spaces."));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_13.mp3", "In other words, Red will play Claimeven, except, on the center column, it will instead be Claimodd."));
    claimodd.set_highlights("DDDz DD"
                            "DDDD DD"
                            "dd D dd"
                            "dd d dd"
                            " D d   "
                            " D     ");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_14.mp3", "But, something weird is going on on the middle column, there's an unpaired spot..."));
    claimodd.set_annotations("......."
                             "......."
                             "......."
                             "...R..."
                             "......."
                             ".......");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_15.mp3", "One other thing we notice is that there is an odd-row red threat on the center column."));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_16.mp3", "Since that is on the top half of one of the coupled spaces, Red will get that space."));
    claimodd.set_highlights("DDD  DD"
                            "DDD  DD"
                            "dd   dd"
                            "dd d dd"
                            " D d   "
                            " D     ");
    claimodd.inject_audio_and_render(AudioSegment("claimodd_17.mp3", "That means the game will end there, and consequently, we don't need to worry about what happens on the top of the middle column."));
    claimodd.unhighlight();
    claimodd.inject_audio_and_render(AudioSegment("claimodd_18.mp3", "Alright, let's try it! Claimeven, except claimodd in the middle."));
    claimodd.stage_transition("436675553553351173311667766221122772244");
    claimodd.inject_audio_and_render(AudioSegment(4));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_19.mp3", "Once again, this only works because this position was, in some sense, simple to begin with."));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_20.mp3", "Red, through training and expertise, was able to force a position which permits such a simple strategy."));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_21.mp3", "Most positions aren't like this- most positions can't be naively solved just using Claimeven and/or Claimodd."));
    claimodd.inject_audio_and_render(AudioSegment("claimodd_22.mp3", "But let this serve as a taste, an idea from which we could devise a more expressive positional language which permits the solution of a much larger class of boards."));

    HeaderScene h("Steady State Solutions", "");
    h.inject_audio_and_render(AudioSegment("claimodd_23.mp3", "Understanding these positions is not only crucial for the learner, but also for the connect 4 researcher."));
    h.inject_audio_and_render(AudioSegment("claimodd_24.mp3", "Solving a game requires an immense search of billions of options on behalf of either player."));
    h.inject_audio_and_render(AudioSegment("claimodd_25.mp3", "However, these positions from which a player can win via Claimeven prove to be a small island of simplicity in an otherwise chaotic sea of complex situations."));
    h.inject_audio_and_render(AudioSegment("claimodd_26.mp3", "These positions will serve as a foothold, an indispensable landmark for further reduction."));

    TwoswapScene swap;
    swap.inject_audio_and_render(AudioSegment("staytunedthishasbeen2swap.mp3", "Stay tuned! This has been 2swap."));
}
