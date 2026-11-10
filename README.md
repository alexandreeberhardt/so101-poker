# pokerbot

Robot SO-101 qui joue au Texas Hold'em No-Limit en tête-à-tête contre un humain.

## Etat

- `pokerbot.engine` : règles du jeu (au-dessus de PokerKit)
- `pokerbot.cli` : partie en ligne de commande contre un adversaire aléatoire
- `pokerbot.strategy.kuhn` : CFR validé sur le Kuhn poker
- `pokerbot.vision.capture` : capture d'images depuis une webcam

## Utilisation

```
uv run pokerbot
uv run python -m pokerbot.strategy.kuhn
uv run python -m pokerbot.vision.capture --camera 0 --out data/cartes
uv run pytest
```
