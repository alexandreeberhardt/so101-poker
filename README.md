# Robot SO-101 joueur de poker

Un bras SO-101 qui joue au Texas Hold'em No-Limit en tête-à-tête contre un humain.

Le dépôt contient deux projets Python indépendants, destinés à deux conteneurs
qui se parlent par le réseau :

- `brain/` (paquet `pokerbot`, Python 3.11+) : règles du jeu, stratégie, décision.
- `vision/` (paquet `pokervision`, Python 3.10) : lecture des cartes et des jetons.
  Python 3.10 est imposé par PyTorch avec GPU sur Jetson.

Les images (`data/`) et les poids entraînés (`models/`) ne sont pas versionnés.

## Cerveau

```
cd brain
uv run pokerbot                            # partie contre un bot aléatoire
uv run python -m pokerbot.strategy.kuhn    # CFR sur le Kuhn poker
uv run pytest
```

## Vision

```
cd vision
uv run python -m pokervision.capture --camera 0 --out ../data/cartes
```
