# CONNECT 4

## Directory structure
The directory structure is quite simple, inside the folder *scripts* you can see a bunch of scripts.
Some of them provide you with all the functionalities offered to play Connect4 and the AIs for solving the game, some others provide you usage examples to play with.

## Playground
Inside *game_utility.py*, you can find a plug-to-play function **play_connect4** to start a full game providing two *Player* objects (either human or AI).

## AIs provided
There are a bunch of different fully parameterizable AIs implemented:
- *Minimax depth-limited (without pruning)*: you have to provide an estimation function (among the ones already offered inside the script)
- *Minimax depth-limited (with Alpha-Beta pruning)*: you have to provide an estimation function plus a second estimation function used to sort the possible moves in order for the alpha-beta pruning to be effective
- *Monte Carlo Tree Search*: fully configurable, you can provide the number of iterations or the time given to the algorithm to compute the move, along with a rollout function
- *Minimax (alpha-beta pruned) plus Monte Carlo Tree Search*: you can use a special version of minimax integrated with MCTS.  
The algorithm starts with the standard Minimax and it will use a MCTS to estimate non-terminal node when *max_depth* is reached.

*NB: Using the drawing of Monte Carlo trees from within a notebook can cause some problems, please don't use the drawing functionality from within Google Colab. If you run the script it should be fine.*

## Example usage
Every script which starts with *game_test_* provides a usage example of the different algorithms.
It is easy as writing 3 lines of code to make different AIs fighting against each other.

## Notebook
If you want, you can import these scripts and run the functions from a notebook (suggested Google Colab).
Please note that notebooks inside VSCode can be used but they are ugly and inconvenient because VSCode compress the output of each cell, so playing a full game can be very inconvenient to see and follow.

## Statistics
It is provided a script to compute some statistics in order to compare different configurations for the hyperparameters of the different algorithms and to compare different AI algorithms too.
It is a bit rusty as it is the first version, but running the example will compare 5 different algorithms with some configurations of the hyperparameters.
The comparison is round-based, every player fights against each other for *rounds* times. At the end, the results are reported in a table.  
*NB: the process can take a long time if you provide a lot of rounds and a lot of players.*  
*NB2: for a faster execution of statistics, please, use **generate_statistics** inside a google colab notebook.*

