# Learning Classifier System Documentation
This is a brief description of the principles behind the design of the system.
First, we can skip all the parts that involve *client_wrapper.py*, *BasePlayer.py*, *BaseAI.py* and *DummyAI.py*, because
they are simply providing a playground and a basic extended knowledge representation of the game state, in order for the agent
design, to disentangle completely the management and communication stuff from the logic part.
## What we have in our pocket
Here we provide a brief recap of the agent's vision of the game state to provide the pieces that we can work on after.
- Basic game stuff like current storm and note tokens, the table cards and the discard pile
- Fellow players' hands
- Our hand represented as a bunch of incomplete cards (instances of HintCard) filled with the partial received hints
- The hints that we have given to other fellow players

Among the basic functionalities, we offer a Stochastic Card representation. This can be useful to represent partial known cards or
deck cards (i.e. what is the probability that the next drawn card is of this color or that value?). We therefore provide
 a basic probability reasoning. Still, it is missing the conditioning on partial known cards which can increase the precision
of our probabilities.

What it is missing: the hints that other fellow players give to each other, this can be a huge improvement over the
vision that our agent can have.

## How it works
To give a high-level vision over the whole system, we provide all the components here, and next we describe how they work.
- Rules
- Actions
- Triggering System
- Action selection
- Credit assignment
- Reinforcement learning
- Genetic Algorithm

### Rules
The very base part is the concept of rule. The most critical and difficult part is the understanding of how a **general rule** can be applied 
effectively on the real state of the game.
In a standard reinforcement learning algorithm, we theoretically need to store in the agent's memory the whole set of
game states (i.e. all the states that a game can be in at any time), and for each of them, the list of actions that the agent
can perform when it is in that state. This is not applicable in most of the cases, therefore,
we need a way to compress the game states without explicitly listing all of them.
Having a general rule, in place of a very specific type of rule, allows to compress the internal agent's decision
making table, that otherwise would become gigantic since the game is very large. The difficulty stays in the fact that we
want a good compromise of generality and good capacity on representing key-parts of the game.
In this system, we have strongly-typed rules, each of them acting on a different feature of the game.
#### Card Rule
The first type of rule if the card rule. To avoid confusion, we state it clearly, this card rule applies only on completely known
cards; therefore, this type of rule is triggered by the table cards and the discard pile cards. For the fellow players' cards, we
reserve a specific type of rule because we want to add more specific conditioning. Here are the fields of this type of rule.
- Owner type: it can be discard pile or table, and it actually represents where the rule wants to apply on.
- Card Pattern: basically it defines the pattern of the card, a specific color, a specific value, or both.
- Number of requested triggers: the number of minimum requested triggers in order for this rule to be considered active (e.g.
  if you require that in the table there are two red cards, you have 2 minimum requested triggers)
- NB: the constraints can be reversed (type must be of this specific color or must NOT be of this color etc.)

#### Fellow Card Rule
It is basically an extension of the Card Rule, but it provides the fellow player number as an addition.
Notice that we can define both rules with a specific fellow player number and rules without specifying it.
The fellow player number express the specific fellow player we want to target (like: +1: the fellow player
which plays 1 turn after me, and so on)

#### Stochastic Card Rule
This basically applies on the agent's cards and the deck cards.
- Owner type: myself or deck
- Stochastic Card Pattern: indicates the pattern that we want to respect
  (e.g. the card must have a probability of being red greater or equal than 50%, and so on)
- Triggers requested: read above

#### Token Rule
It expresses a rule over the current storm or note token used.

#### Players number Rule
It expresses a rule over the current players number

#### Compound rule
This is a very important concept, it allows putting together different rules to have a more complex rule.

### Actions
In the last version of the design, we've decided to put an explicit action to disentangle the rule triggering from the action itself.
Again, the main problem is: how to express a general action that must be applied on specific cards or features in the game?
To tackle this problem, we have decided to put an action type and another **targeting rule**.
- Action type: play, discard, hint_color, hint_value
- Targeting rule: it decides which card to apply the action to, it can be a Stochastic Card Rule with owner type: myself,
or a Fellow Card Rule (the idea is to give hints based on the target fellow player card) 
  
This simple system allows to link the general rule to the specific feature of the game.

### Triggering System
As a consequence, having expressed general rules and actions, we need a way to effectively trigger these rules starting from
the current game state. The idea is that a rule can have, in most of the cases, more than one trigger.
To cope with this, we have added a new field to the triggers, the **strength**.
Basically the strength is representing how much the rule is triggered by a certain feature of the game.
For example, let's say that we have a Stochastic Card Rule which says: the card must have a probability of being red greater
than 50 %, and we have, in our hand, some cards with hints and some card without hints. We could have more than one card that 
match the constraints; therefore, we will have one trigger for each card in our hand that match the constraints and they could have
a different strength (say, one with 80 % and one with 60 %).

### Action selection
Up to now, we have rules, actions, and a way to trigger the rules and to connect the rules with the specific feature of the game (through the triggers).
It is now time to make the move, and to accomplish this, we have decided to have a fully deterministic way of doing that.
We basically pick the rule with the highest utility (we will explain later what it is), and since the rule has an action,
we pick the trigger of the targeting rule inside the action which has the highest strength.
For example, we can have selected one compound rule, which has an action of type *play* and a targeting rule, of course,
of type Stochastic Card Rule. Therefore, we have more than one trigger for that rule, we simply pick the highest trigger strength,
and we play that card.
Since it is not taken for granted that a rule covers the whole state, we may have invalid actions for the current game state.
Therefore, we sort the triggered rules by their utility, and instead of picking the one with the highest utility, we try every rule
in decreasing order until we reach a valid action that we can effectively perform.

### Credit assignment
A key part is how we assign the credits to the different rules. We have decided to keep it simple.
We have basically two types of rewards: immediate rewards and final reward.
The immediate rewards are given during the game if the chosen action are good or bad.
The final reward, which is higher enough to counterbalance the immediate rewards, is given after the game is end based
on the score obtained (with a multiplier), and it is backpropagated to every (rule->action) pair selected to perform the actions.

### Reinforcement Learning
In this way, we may have very bad immediate rewards but very good final rewards; therefore, when we choose the action based on the utility,
which is also based on the reward, we are effectively choosing the rule which leads to a state that optimizes the expected return
of rewards, obtaining a reinforcement learning method in the end.

### Genetic Algorithm
So far, we have rules, actions, a triggering system, a credit assignment system, and a reinforcement learning method that
allows to pick the action which maximizes the expected return of rewards. It seems that we could stop here our system, but
we have to remember that the rules composing our classifier may be completely inappropriate to represent the game state,
leading to uncovered game states; moreover, wrong connected actions make impossible the optimization of the rewards and we
will never reach convergence in this way.

Therefore, we need a way to evolve the rules and the actions using a systematic method, hoping to reach a good convergence point.
The good principle for a genetic algorithm is, in general, to operate on the genotype without looking at the obtained results.
This may seem a good idea, but it leads to some problems too. The main problem here is that we do not have a population of
distinct individuals trying to fight each other to survive, instead, we have a set of rules which, **together**, compose the
single individual. The thing is, how to optimize the total utility (i.e. the sum of the utilities of all the rules) of the rules?
First, for each rule type, we have designed three genetic operators, each of them acting differently to be as less destructive
as possible:
- Creep mutation: the basic concept here is that we can have either a small mutation (exploitation) or a big mutation (exploration).
This is obtained by self-adapting the mutation step σ of a normal Gaussian distribution according to the following:
  `σ = σ' * exp( (lr / sqrt(n)) * N(0, 1) )`  
  where *n* is the number of mutations already performed and *lr* is the learning rate.
  We recall that in LCSs we should have little mutations and we should exploit crossover operators for exploration purposes.
  However, since it is not easy to put all the necessary genetic material inside the initial rules, as we'll see in a moment,
  we preferred to use a self-adapted mutation step for each modifiable parameter.
  The division by the square root of the number of mutations already performed tries to combine the just explained concept.
  The idea behind this concept of temperature is that we want initially to create the necessary genetic material putting them
  inside the population, and once achieved this goal, reducing to have only little creep mutations.  
  *NB: it is worth trying CMA-ES for floating point parameters optimization*  
- Crossover: we have employed two types of crossover: uniform and average crossover. Since we have strongly-typed rules,
we crossover only the rules of the same type and sharing the same action. The idea is to merge similar strong rules sharing the
  same action trying to achieve a better result. Notice that crossovering very similar rules may not favor exploration at all,
  therefore, we need a way to keep diversity inside the population.
  
The just described genetic operators act at rule level, but as just said, since we have rules trying to reach a common goal
(optimizing the total utility), we have designed four fixed steps that compose our genetic algorithm pipeline, taking some
inspiration from Arthur Samuel; notice that we store, for each rule, the number of times it has been triggered.
#### 1. Rule deletion
The original goal was to delete only old rules, however, we redesigned this step to delete and regenerate
weak rules, achieving a similar result as a survival selection method. This may seem wrong, and actually it could be wrong,
   but we have done that because the whole system was so slow to train that we'd needed a way to destroy wrong unuseful genetic
   material and recreate it until some good candidate is generated. This has a problem as a consequence, newly generated rules can
   be killed by the good rules already present inside the population. However, this effect is actually mitagated, but still not
   completely solved, by the other genetic steps which actually modify also the good rules, resetting the utility as a consequence
#### 2. Rule specialization
The idea is to pick *X %* of the rules, using a tournament selection, favoring the rules with low utility and
high coverage. Here we want to specialize rules which achieve scars results and are triggered very often.
   We use a tournament selection to add some exploration factor, giving the chance to every rule to be specialized.
For the compound rules, we specialize the same factor of the nested rules and we add an extra rule.
For the basic rules, we do a creep mutation, leaving the balance between exploration and exploitation to the self-adapted step.
#### 3. Rule generalization
As the opposite of specialization, here we try to generalize the rules in the exact same but opposite way. If we can, we also
remove a rule from the compound rules.
#### 4. Rule merging
As the last step, we want to merge good rules which are **similar** and share the same action. The idea, as just said, is to
improve the fitness crossovering good rules (based on utility) hoping to pick good traits from both the parents. This is pushed by the fact
that we are crossovering similar rules, similar in the sense that they share the same type, and the same structure
(e.g. two rules that express the same condition over the same color but in a different quantity).

The fitness in this system is a simple momentum-adapted total reward. I.e. to update the fitness, we pick an *alpha*
factor of the previous fitness and a *(1-alpha)* factor of the reward.

#### Bootstrap
We have leaved it as the last thing but it is actually a critical part. Up to now, we generate randomly (tweaking probabilities)
every type of rules and we combine them in Compound rules (most of the times), expressing a maximum number of nested rules.
The problem, as it happens in Genetic Programming, is that in the first generation we want to have all the genetic material
that we need. This gives a reason to the applied modifications of our previous steps; unfortunately, even tweaking the
generation probabilities, we don't have enough control over the created genetic material and, more probably, we don't have
enough coverage in the boostrap phase. We have tried to tackle this problem by re-creating newly fresh genetic material in a
kind of survival selection step; but actually this is not the best method we can have.

## Known problems
The first immediate problem is that, up to now, there is no rule that tries to express a constraint on the already given hints.
This leads to a huge and immediate problem, if we find a good rule in charge of giving some good hint in some kind of situation,
we are positively rewarded when we give the hint, and in the immediate next turn, we risk to do the exact same action again,
receiving a negative reward for duplicated hint.  
This gives light to a more general problem of having incomplete LCS systems. The rules must have the possibility to cover
the whole game state. If we, for any reason, neglect something like the above example, the convergence will not be reached,
since the rules have no chance to be adapted specifically enough to a certain situation being consequently triggered and good only in that situation.
This explains the concept that we have introduced before, we want a right compromise between generality and specificity;
furthermore, we want coverage, which is another important problem.
Moreover, the presence of a lot of hyperparameters (like the generation probabilities, alpha, and so on) gives us another problem.
We may add them as endogenous parameters or find a right strategy (e.g. cross-validation) to choose a good set.  
Adding the missing rule type (GivenHint Rule) may complete the vision of the agent over the whole game state.
All of that sounds reasonable and, with the proper modification over the genetic pipeline, it could work. However,
we recall that the GA paradigm is extremely slow, especially because in an LCS trying to face big problems, we need a
huge amount of rules to have the chance to cover in a meaningful way all the game states without ending in loops of positive and negative effects,
as we have just seen.
