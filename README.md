# hmls

A simple tank game, written in Python.
Two players take turns moving their tanks around a landscape (with cells which are either passable or impassable), subject to fog-of-war.
On each turn, the active tank can move forward, turn left, turn right, fire (with a range of one cell forward) or pass.

Of course, there's a couple of ulterior motives.
The first is to serve as a framework for learning how to create and train models to play a game.
Fog-of-war generally makes gameplay very hard for a computer, so even this simple game is a non-trivial problem.
The complexity will multiply as each player controls more tanks.

The second motive is to experiment with coding agents.
Almost none of the code in this repository is manually written.
Instead, Copilot has been given instructions, and asked to plan an implementation.
After a round or two of clarifying questions, Copilot then writes the code.
The code is subject to manual review, and several refactorings have been performed in consequence (although Copilot still does the actual work of the refactor).

## Getting started

This repository is configured for use with the [`uv` package manager](https://docs.astral.sh/uv/).
To set up, run:
```bash
# Clone and install
git clone https://github.com/riedgar-ms/hmls.git
cd hmls
uv sync --all-packages
```

To play a game, you'll first need to create a map.
Run the map generator with:

```bash
uv run hmls-mapgenerator
```

This will start up a [Textual](https://textual.textualize.io/) UI in your console.
Use this to generate a map and save it to disk (it will be a JSON file).

In order to get a sense for how the game works, use this map with the test harness.
This isn't for proper games, since it shows in a single (Textual) view then entire map, plus the individual fog-of-war patches for each tank.
Start up the harness with

```bash
uv run hmls-testharness path/to/map.json 3
```

This will open the map, and place down three tanks for each player.
Play proceeds by alternating between the two players, each moving one of their tanks (the one highlighted).
The controls are:

| Key | Action |
|---|---|
| `W` | Move forward |
| `A` | Turn left |
| `D` | Turn right |
| `Space` | Fire |
| `Tab` | Pass (skip turn) |
| `Q` | Quit |

When the game ends, a summary is shown and you are prompted to save the
full game history as JSON.
To replay the game history, run:

```bash
uv run hmls-replayviewer path/to/history.json
```