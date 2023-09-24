{
  "config": {
    "width": 640,
    "height": 360,
    "framerate": 30
  },
  "scenes": [
    {
      "type": "c4",
      "omit":false,
      "sequence": [
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "audio":"intro_1.mp3",
          "script":"To the untrained eye, this connect 4 position might look unassuming."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4344444636",
          "audio":"intro_2.mp3",
          "script":"This one, you might not be able to tell who is winning."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43333734474443",
          "audio":"intro_3.mp3",
          "script":"But an expert player could tell you, that all of these positions share a few special properties,"
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "233332123225632561",
          "audio":"intro_4.mp3",
          "script":"properties which are actually quite extraordinary."
        },
        {
          "transition": true,
          "duration_seconds":2
        }
      ]
    },
    {
      "type":"composite",
      "omit":true,
      "subscenes":
      [
        {
          "x": 0,
          "y": 0,
          "width": 0.5,
          "height": 0.5,
          "subscene": {
            "type": "c4",
            "sequence": [
              {
                "transition": true,
                "duration_seconds":1
              },
              {
                "representation": "43334444343773",
                "audio":"intro_5.mp3",
                "script":"One obvious feature of all of these positions is that the amount of space remaining in each column is even."
              },
              {
                "transition": true,
                "duration_seconds":1
              }
            ]
          }
        },
        {
          "x": 0.5,
          "y": 0,
          "width": 0.5,
          "height": 0.5,
          "subscene": {
            "type": "c4",
            "sequence": [
              {
                "transition": true,
                "duration_seconds":1
              },
              {
                "representation": "4344444636",
                "audio":"intro_6.mp3",
                "script":"This column has 6 empty spots."
              },
              {
                "transition": true,
                "duration_seconds":1
              },
              {
                "representation": "4344444636",
                "audio":"intro_7.mp3",
                "script":"This column has 2."
              },
              {
                "transition": true,
                "duration_seconds":1
              }
            ]
          }
        },
        {
          "x": 0,
          "y": 0.5,
          "width": 0.5,
          "height": 0.5,
          "subscene": {
            "type": "c4",
            "sequence": [
              {
                "transition": true,
                "duration_seconds":1
              },
              {
                "representation": "43333734474443",
                "audio":"intro_8.mp3",
                "script":"This is true for all of the games I've shown."
              },
              {
                "transition": true,
                "duration_seconds":1
              }
            ]
          }
        },
        {
          "x": 0.5,
          "y": 0.5,
          "width": 0.5,
          "height": 0.5,
          "subscene": {
            "type": "c4",
            "sequence": [
              {
                "transition": true,
                "duration_seconds":1
              },
              {
                "representation": "233332123225632561",
                "audio":"intro_9.mp3",
                "script":"Not a single column with an odd number of pieces."
              },
              {
                "transition": true,
                "duration_seconds":1
              }
            ]
          }
        }
      ]
    },
    {
      "type": "c4",
      "omit":false,
      "sequence": [
        {
          "transition": true,
          "duration_seconds":2
        },
        {
          "representation": "43334444343773",
          "audio":"intro_10.mp3",
          "script":"What's more, in each of these cases, it so happens that Yellow, player 2, is winning."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "audio":"intro_11.mp3",
          "script":"It might not look like it, but Yellow is winning so spectacularly in these games, that even a fool could beat an expert. As long as that fool knows what makes these positions special."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "highlight":   ["dd  ddd",
                          "dd  ddd",
                          "dd  ddd",
                          "dd  ddd",
                          "dd  dd ",
                          "dd  dd "],
          "audio":"intro_12.mp3",
          "script":"Remember how all of the empty space is even?"
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "highlight":   ["dd  ddd",
                          "dd  ddd",
                          "DD  DDD",
                          "DD  DDD",
                          "dd  dd ",
                          "dd  dd "],
          "audio":"intro_13.mp3",
          "script":"That means that we can cut those remaining columns up into groups of 2."
        },
        {
          "transition": true,
          "audio":"intro_14.mp3",
          "script":"Now, since it's red's turn in all of these cases, let's let Red start by making a move."
        },
        {
          "representation": "433344443437731",
          "highlight":   ["dd  ddd",
                          "dd  ddd",
                          "DD  DDD",
                          "DD  DDD",
                          "dd  dd ",
                          "dd  dd "],
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_15.mp3",
          "script":"When drawn like this, yellow is almost beckoned to fill in the uncoupled pair."
        },
        {
          "representation": "4333444434377311",
          "highlight":   ["dd  ddd",
                          "dd  ddd",
                          "DD  DDD",
                          "DD  DDD",
                          "dd  dd ",
                          "dd  dd "],
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_16.mp3",
          "script":"And, since the entire board is divied up into pairs, Yellow can just keep doing this forever."
        },
        {
          "representation": "433344443437731155",
          "highlight":   ["dd  ddd",
                          "dd  ddd",
                          "DD  DDD",
                          "DD  DDD",
                          "dd  dd ",
                          "dd  dd "],
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_17.mp3",
          "script":"Wherever Red goes, yellow fills in the unpaired spot."
        },
        {
          "representation": "43334444343773115522",
          "highlight":   ["dd  ddd",
                          "dd  ddd",
                          "DD  DDD",
                          "DD  DDD",
                          "dd  dd ",
                          "dd  dd "],
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_17.mp3",
          "script":"We can even take away the pairings."
        },
        {
          "representation": "43334444343773115522",
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_17.mp3",
          "script":"All Yellow is doing is responding, thoughtlessly, to Red's last move."
        },
        {
          "representation": "4333444434377311552222",
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_17.mp3",
          "script":"A randomly placed red stone, is followed by a Yellow stone immediately above."
        },
        {
          "representation": "433344443437731155222211",
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_17.mp3",
          "script":"So on, and so forth."
        },
        {
          "representation": "4333444434377311552222117777115555666622",
          "audio":"intro_18.mp3",
          "script":"Wait a minute... did yellow just win?"
        },
        {
          "transition": true,
          "duration_seconds":5
        },
        {
          "representation": "4344444636",
          "audio":"intro_18.mp3",
          "script":"Let's try again, with a different board that we saw at the beginning."
        },
        {
          "transition": true,
          "audio":"intro_18.mp3",
          "script":"Remember, nothing fancy. Red initiates, then Yellow naively responds."
        },
        {
          "representation": "434444463611",
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_18.mp3",
          "script":"On and on, until the end of the game."
        },
        {
          "representation": "4344444636112222667777775511112266555533",
          "audio":"intro_18.mp3",
          "script":"Like magic, Yellow has won again."
        },
        {
          "transition": true,
          "duration_seconds":5
        },
        {
          "representation": "4377345644465545",
          "audio":"intro_18.mp3",
          "script":"Now, I promised you those boards at the beginning had something magical about them. Does the columns having an even amount of empty space necessitate this winning condition for Yellow?"
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4377345644465545",
          "audio":"intro_18.mp3",
          "script":"Yellow is winning in this position, and it also has all even columns. However, it is not one of the special ones I showed before. Let's try our strategy here!"
        },
        {
          "transition": true,
          "audio":"intro_18.mp3",
          "script":"Just like before, red then yellow."
        },
        {
          "representation": "437734564446554533331111",
          "duration_seconds":1
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4377345644465545333311112",
          "audio":"intro_18.mp3",
          "script":"But with this last Red move, Yellow is starting to panic."
        },
        {
          "transition": true,
          "audio":"intro_18.mp3",
          "script":"If Yellow continues our simple strategy, Red will immediately win."
        },
        {
          "representation": "437734564446554533331111222",
          "duration_seconds":1
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4377345644465545333311112",
          "audio":"intro_18.mp3",
          "script":"So, Yellow, in a last ditch effort, breaks our strategy."
        },
        {
          "transition": true,
          "audio":"intro_18.mp3",
          "script":"Perhaps, by making some threats on the right side, Yellow can regain the tempo?"
        },
        {
          "representation": "437734564446554533331111255667",
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_18.mp3",
          "script":"Unfortunately for Yellow, the damage is already done. Red immediately plays in the second column..."
        },
        {
          "representation": "4377345644465545333311112556672",
          "duration_seconds":1
        },
        {
          "transition": true,
          "audio":"intro_18.mp3",
          "script":"Yellow is forced to block, but Red wins on a diagonal anyways."
        },
        {
          "representation": "437734564446554533331111255667222",
          "audio":"intro_18.mp3",
          "script":"Indeed, it is the case that the 4 boards at the beginning weren't just any boards with all even columns."
        },
        {
          "transition": true,
                "audio":"intro_5.mp3",
                "script":"If Yellow could always win by just playing call-and-response, that wouldn't be much fun, would it?"
        }
      ]
    },
    {
      "type": "c4",
      "omit":false,
      "sequence": [
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "audio":"intro_1.mp3",
          "script":"So, what is it about these positions which make them special?"
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "audio":"intro_2.mp3",
          "script":"When we play our naive follow-up strategy, let's take note of what happens. Let's mark all of the disks already played before we start."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "annotations": ["..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx..x",
                          "..xx..x"],
          "duration_seconds":1
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4333444434377311552222117777115555666622",
          "annotations": ["..xx...",
                          "ooxxooo",
                          "..xx...",
                          "ooxxooo",
                          "..xx..x",
                          "ooxxoox"],
          "audio":"intro_3.mp3",
          "script":"Regardless of what Red chooses, after the point where we start to use our strategy, all of the red disks end up on an odd row. That is, the first, third, and fifth rows from the bottom."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4333444434377311552222117777115555666622",
          "annotations": ["ooxxooo",
                          "..xx...",
                          "ooxxooo",
                          "..xx...",
                          "ooxxoox",
                          "..xx..x"],
          "audio":"intro_4.mp3",
          "script":"Yellow inevitably gets all of the even rows."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "annotations": ["..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx..x",
                          "..xx..x"],
          "audio":"intro_4.mp3",
          "script":"It is for this reason that this strategy is known as Claimeven."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "annotations": ["..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx..x",
                          "..xx..x"],
          "audio":"intro_4.mp3",
          "script":"Yellow, by merely following Red, can get hold of all of the remaining even-row spaces."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "annotations": ["..xx...",
                          "ooxxooo",
                          "..xx...",
                          "ooxxooo",
                          "..xx..x",
                          "ooxxoox"],
          "audio":"intro_4.mp3",
          "script":"What makes this board special, is that even when we fill in all of the Red pieces on the odd rows, no line of 4 disks is made."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "annotations": ["ooooooo",
                          "..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx..x",
                          "..xx..x"],
          "audio":"intro_4.mp3",
          "script":"However, Yellow is able to lay claim to the entire top row."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43334444343773",
          "audio":"intro_4.mp3",
          "script":"So, it doesn't matter the order which Red plays. It _can't_ matter. The result is already guaranteed for Yellow."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43333734474443",
          "annotations": ["..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx..x",
                          "..xx..x"],
          "audio":"intro_4.mp3",
          "script":"Well, what about this slightly different board, which also wins by claimeven? If we imagine all the red stones filling in the odd rows..."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43333734474443",
          "annotations": ["..xx...",
                          "..xx...",
                          "..xx...",
                          "..xoooo",
                          "..xx..x",
                          "..xx..x"],
          "audio":"intro_4.mp3",
          "script":"There is a line of 4!"
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43333734474443",
          "annotations": ["..xx...",
                          "..xx...",
                          "..xx...",
                          "..xx...",
                          "..xoooo",
                          "..xx..x"],
          "audio":"intro_4.mp3",
          "script":"But... in this case, it is hovering immediately over a Yellow line of 4."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43333734474443",
          "audio":"intro_4.mp3",
          "script":"So, for Red to win... Yellow would already have to have won."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43333734474443111122112255225555777766",
          "audio":"intro_4.mp3",
          "script":"This is what makes these boards special. The red winning chains on odd rows are all undercut by yellow winning chains on even rows."
        },
        {
          "transition": true,
          "duration_seconds":1
        }
      ]
    },
    {
      "type": "c4",
      "omit":false,
      "sequence": [
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4344377",
          "audio":"intro_4.mp3",
          "script":"Let's try some examples. It's Yellow's turn. How can we force a win using Claimeven?"
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "4344377",
          "audio":"intro_4.mp3",
          "script":"By placing this disk in the center column, Yellow can kill two birds with one stone. This move makes all of the columns even, and guarantees that Red can't make a line of 4 until after Yellow gets a line of 4 on the 4th row."
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "44137371444",
          "audio":"intro_4.mp3",
          "script":"Tricky"
        },
        {
          "transition": true,
          "duration_seconds":1
        },
        {
          "representation": "43444446",
          "audio":"intro_4.mp3",
          "script":"Trickier"
        },
        {
          "transition": true,
          "duration_seconds":1
        }
      ]
    },
    {
      "type": "header",
      "omit":true,
      "header": "TODO",
      "subheader": "",
      "audio":"84.mp3",
      "script":"Creating a position which yields these easy wins for yellow is difficult. However, an expert player up against a newbie might make it look easy. As expected, capitalizing on these situations on the fly is the trademark of expertise. Active study and deliberate practice can help you see it coming."
    },
    {
      "type": "header",
      "omit":true,
      "header": "Steady State Solutions",
      "subheader": "",
      "audio":"84.mp3",
      "script":"Understanding these positions is not only crucial for the learner, but also for the connect 4 researcher. Solving a game requires an immense search of billions of options on behalf of either player. However, these positions from which a player can win via Claimeven prove to be a small island of simplicity in an otherwise chaotic sea of complex situations. These positions will serve as a foothold, an indispensable landmark for further reduction."
    },
    {
      "type": "header",
      "omit":true,
      "header": "TODO",
      "subheader": "",
      "audio":"84.mp3",
      "script":"Before we part, I'll give you a taste of how this can be generalized, with one last puzzle. 43667555355335117 In this case, and unlike the others, Red is winning. How can we apply our understanding of claimeven to this position? [solve the puzzle]"
    },
    {
      "type": "2swap",
      "omit":true,
      "audio":"85.mp3",
      "script":"This has been 2swap."
    }
  ]
}
