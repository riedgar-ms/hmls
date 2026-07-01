# Planning for Squads

The goal of this directory is to provide support for a computer player which can manage a squad of multiple tanks. This should ultimately be able to connect as a player to a game hosted by `hmls-server`.

## High Level Design

The computer player should be split into two parts:

- Planner (single instance)
- Executor (instance per active tank)

The executors can use the same neural net design and weights, and just have their own internal state vectors. On each turn, the planner runs first, and produces an order. This is then sent to the executor model for the currently active tank, to generate the actual move. The planner should be discouraged from changing the order for a given tank too frequently.

### Available Orders

The planner can issue the following orders:

- Move to location (x, y)
- Explore
- Hunt

The last two are similar, but 'explore' would be trained to prioritise movement, while 'hunt' would be more willing to stay in one place.

## Model Design

### Executors

The executors can follow the same basic structure as the Mk-I tanks, namely CNN -> GRU -> Order classifier. At some point the model will need to be fed the current location, orientation and the order (which may itself contain a target location).

### Planner

The planner should be stateful. It can use the incoming patch data to build up an overall map (subsequently processable by a CNN), as well as an internal state vector. It would also have to have some way of knowing which is the active tank for the current turn (and order history...). And it needs to cope with a variable number of active tanks. The details are very unclear.

## Training

Initial training of the executors can be done per-order. We can have one trainer for the 'move' order, one for explore, and one for hunting. The last should use one of the existing singletanks (or the randomtank) as an opponent. Similar to the single tank trainer, this should be done on multiple maps of multiple sizes.

Once the basic executor training is done, then we can start training the planner (although there will be continued executor training happening at the same time). Initially, it can play against groups of single tanks (random or Mk-n), but over time this should be switchable to another squad player. Again, each training run will need to have multiple maps of differing sizes.