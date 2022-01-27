# Computational Intelligence 2021-2022

Exam of computational intelligence 2021 - 2022. It requires teaching the client to play the game of Hanabi (rules can be found [here](https://www.spillehulen.dk/media/102616/hanabi-card-game-rules.pdf)).

# Learning Classifier System
The system is a learning classifier system which tries to exploit the reinforcement learning technique to obtain long-term good results in handling large-space problems like the Hanabi game.
The complete description is available in `DOCUMENTATION.md`.
## Usage
The only script that you need to launch is: `client.py`. You can find all the details to launch the script typing `python client.py -h`.
However, this is a small recap:
- The first parameter is the only required, it is a positional argument indicating the player name;
- The second positional argument indicates the type of the player, it can be one of
  - human
  - dummy
  - zcs
- The other important parameter is `--train` : it allows to enable the training, the rewards will be dispatched and the GA algorithm will work
- `--load_from_file` allows to load the model from file using the *model_path* parameter, the default path is: `ZCS_Data/<player_name>_model.zcs`, so, if you have a model named `zcs1_model.zcs`, you can load the model launching: `python client.py zcs1 zcs --load_from_file` and the work will be done for you
- `model_path` allows to indicate the model to load and / or save after the training. Read the previous note for the default.
- `show_reason` allows you to visualize the rule that caused the action in a textual format.
- ..other parameters allow to tune the model capacities and hyperparameters

**Please, note that the system is producing some files:**
- logs/.. which will contain all the output of the various clients
- game_states/.. if enabled, it contains all the game states in a picke format
- ZCS_data/.. it will contain the trained models and the correspective file *<..>_rules.txt* which contains all the rules in a textual readable format.

**Please, to interrupt the client, just type exit and press enter, it will save the model in the current state (if in training mode) and produce the rules.txt file. It could require some seconds to do this, please, do not interrupt forcely**

## Folder structure
Here is a brief recap of the folder structure:
- client.py - provides the basic entrance
- client_wrapper.py - provides a wrapper for the server connection and other useful utility functions to disentangle the logic from the communication
- BasePlayer.py (BasePlayer class) - provides all the basic functionalities for a player, offering some callbacks that a Derived class can overwrite to perform custom functionalities
- BaseAI.py ( BaseAI (BasePlayer) class) - offers a basic entry point for building an AI system, such as automatic ready and others
- ZCS_AI.py ( ZCS_AI (BaseAI) class) - it contains all the functionalities of the learning classifier system
- ZCS_Data.py - it offers all the building blocks (like rules, actions, etc.) and the genetic operators used from the ZCS_AI class

## Basic concepts
Here is a brief recap of how the system is made and how it works.
We start with the building blocks, we have these concepts:
- Rule: it represents all the constraints that a certain feature of the game (like cards) needs to met in order to be triggered, the most important parameters are:
  - constraints.. of course
  - activations
  - reward
  - last_fitness
  - **action**
- CompoundRule (Rule): it is simply a special rule that incorporates other rules
- Action: it contains the type of the action (play, discard, hint_color, hint_value) and an additional **Targeting rule** which is capable of identifying the target of the action to perform (like a specific card in our hand or a specific fellow player card, etc.)
- We have different rule types like CardRule, StochasticCardRule, etc.
- It is worth noting that a rule can be triggered multiple times by different features of the game 1 Rule -> multiple triggers. So, we introduced an additional concept of **strength** which represents the amount of force with which the rule has been triggered. This allows to simply picking the best trigger for a certain triggered rule.
- Genetic algorithm: we have a genetic algorithm which runs after *ga_rate* games and is capable of:
  - Deleting and regenerating the weak dead rules (survival selection)
  - Rule specialization and generalization, which allow to specialize weak rules with high coverage and generalize strong rules with low coverage
  - Rule merging, which picks a subpart of the population and try to match certain criterions to merge the best similar rules together
  - All the parameters which vary inside the rules are combined with a Mutation object (either continuous or discrete) which allows to leverage the self-adaptation strategy.  
To fully understand, here is a complete example:
- I have a certain probability of having 1-red and 2-blue in my hand.
- A strong rule, with the highest utility function (a momentum adapted fitness), is claiming that I need to have more than 75% of having a 1-red and more than 50 % of having a 2-blue in my hand. It happens so.
- The rule is triggered, and  the correspective action is selected. The action is playing a card, but to understand which card it has another rule in it:
  - Target rule: the card which has at least 50 % of being a 1-red
  - action: play
  - -> it plays the triggered card (1-red)

It is of course oversimplified for conciseness. You can find more comments in the code.
