# SO-101 poker-playing robot

An SO-101 arm that plays heads-up No-Limit Texas Hold'em against a human.

The repository contains two independent Python projects, meant for two containers
that talk to each other over the network:

- `brain/` (package `pokerbot`, Python 3.11+): game rules, strategy, decision making.
- `vision/` (package `pokervision`, Python 3.10): reading the cards and the chips.
  Python 3.10 is required by PyTorch with GPU support on Jetson.

Images (`data/`) and trained weights (`models/`) are not tracked in git.

## Brain

```
cd brain
uv run pokerbot                            # game against a random bot
uv run python -m pokerbot.strategy.kuhn    # CFR on Kuhn poker
uv run pytest
```

## Vision

```
cd vision
uv run python -m pokervision.capture --camera 0 --out ../data/cards
```
