# SO-101 poker-playing robot

I'm building an SO-101 arm that plays heads-up No-Limit Texas Hold'em against a human.

## Status

The work is split in two: what needs the arm, and what does not.

### Without the arm

These parts only need a computer, a camera, cards and chips.

| Part | State |
|---|---|
| Poker engine | Works: full heads-up hands, playable in the terminal |
| Equilibrium solver (CFR) | Works on Kuhn and Leduc poker, checked against their known values |
| Short-stack strategy | Works: exact push-or-fold equilibrium, playable in the terminal |
| Deep-stack strategy | Not started |
| Adapting to the opponent | Not started |
| Reading cards | Works on three decks, through a camera under a glass plate |
| Learning a new deck | Works: one picture of each card, then a minute of training |
| Collecting chip pictures | Works: guided capture with exact labels |
| Counting chips | Not working yet: about 1.5 chips of error per colour |
| Link between vision and brain | Not started |

### With the arm

| Part | State |
|---|---|
| Locating the arm relative to the camera | Not started |
| Moving chips: bet, call, push to the pot | Not started |
| Full game at the table: see, decide, act | Not started |

## Layout

Two independent Python projects, meant for two containers that talk to each other
over the network, so that they can run on different machines:

- `brain/` (package `pokerbot`, Python 3.11+): game rules, strategy, decisions.
- `vision/` (package `pokervision`, Python 3.10): reading the cards and the chips.
  Python 3.10 is imposed by PyTorch with GPU support on the Jetson Orin Nano.

Pictures (`data/`) and trained weights (`models/`) are not tracked in git.

## Brain

```
cd brain
uv run pokerbot                                     # play the push-or-fold bot, 10 bb each
uv run pokerbot --bot random --stack 100            # play a random bot, 100 bb each
uv run python -m pokerbot.strategy.kuhn             # CFR on Kuhn poker
uv run python -m pokerbot.strategy.leduc            # CFR on Leduc poker
uv run python -m pokerbot.strategy.leduc_rl         # self-play RL on Leduc, for comparison
uv run python -m pokerbot.strategy.pushfold 10 15   # push-or-fold charts for 10 and 15 bb
uv run pytest
```

- `engine.py`: heads-up No-Limit Hold'em on top of PokerKit. A hand advances step by
  step (hole cards, action, board, showdown), with cards coming either from a random
  deal or from the camera. The opponent's hand can stay unknown until showdown.
- `bots.py`: the opponents of the terminal game. The push-or-fold bot plays the
  equilibrium for the current stack depth. When I step outside of that small game
  (a limp, a small raise, a flop), it falls back on simple rules that are not an
  equilibrium.
- `strategy/kuhn.py`, `strategy/leduc.py`: CFR with exact exploitability. Both reach
  the known value of their game.
- `strategy/leduc_rl.py`: a self-play policy-gradient agent, kept as a comparison.
  It stays far more exploitable than CFR (about 1 chip per hand against 0.01).
- `strategy/evaluator.py`: seven-card hand evaluator, vectorised with numpy
  (about 3 million hands per second), checked against PokerKit.
- `strategy/preflop.py`, `strategy/pushfold.py`: the 169 preflop classes, their
  all-in equities, and the exact push-or-fold equilibrium for short stacks.
- `strategy/equity.py`: chance of winning against a random hand, by sampling.

Deeper stacks and adapting to the opponent are still to do. The match format is a
freeze-out: play until one player has all the chips.

## Vision

The robot's hole cards lie face down on a glass plate, with a camera underneath, so
that the opponent cannot see them. The camera is a USB webcam on the Jetson, driven
over SSH from the development machine.

```
cd vision
uv run python -m pokervision.collect --host <jetson> --out ../data/cards/pass1 --preview-port 8765
uv run python -m pokervision.train_cards
uv run python -m pokervision.read_cards picture.jpg            # names of the cards in a picture
uv run python -m pokervision.read_cards --check ../data/cards  # compare with every label
uv run python -m pokervision.collect_chips --host <jetson> --out ../data/chips \
    --chips white:1:25,red:5:25,black:25:25,blue:10:25
uv run pytest
```

### Cards

- `collect.py` takes a picture each time two cards are in place and still, and serves
  a live annotated view of the camera. `collect_remote.py` is its camera-side half.
- `cards.py` finds the cards, straightens them and cuts out their two index corners.
- `card_model.py` is a small network (81,000 parameters) that reads rank and suit from
  a corner. The two corners of a card vote together.
- `train_cards.py` trains on every labelled folder of `data/cards/` and tests on the
  folders whose name ends in `test`.
- `read_cards.py` goes from a picture to card names in a few milliseconds, and says
  which readings to doubt: low confidence, two corners that disagree, or the same
  card twice.

Results on 20 held-out cards per deck, with three different decks:

| Deck | Before seeing the deck | After one picture of each card |
|---|---|---|
| First | - | 20 / 20 |
| Second (large index, very different design) | 7 / 20 | 20 / 20 |
| Third (design close to the first) | 19 / 20 | 20 / 20 |

Showing a new deck once takes about five minutes of pictures and one minute of
training. Without that step the result depends on the deck: training on two decks
and reading every card of the third gives 117 / 132, 41 / 72 and 72 / 72. The reader
flags 41 of those 46 mistakes as doubtful, and none of the 60 held-out cards of decks
it knows. I took every picture in one evening with the same camera and lamp, so I
don't know yet how it copes with other lighting.

### Chips

- `collect_chips.py` dictates scenes (which stacks to build), takes the pictures and
  writes the exact content of each one at capture time, since a tall stack cannot be
  recounted from a picture. `collect_chips_remote.py` is its camera-side half.

I collected 122 pictures over 27 scenes with four chip colours, plus 8 test pictures,
and tried three ways of counting. The best one, based on the visible area of each
colour at a standard scale, is still off by about 1.5 chips per colour on unseen
scenes, which is not good enough to read a bet. White chips are the worst: their
edges blend into my light wooden table. Stacks that touch or hide each other also
hurt, and 27 scenes is not much to learn from. A dark plain mat and separated stacks
are the next things to try. No chip counter is in the repository yet.

## Data

```
data/cards/<folder>/        pictures and labels.csv (file, left card, right card)
data/chips/                 pictures, plan.json and labels.jsonl
models/cards.pt             weights of the card reader
```
