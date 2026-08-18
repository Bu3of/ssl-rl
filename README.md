Self-Supervised Representations for Sample-Efficient RL

Question: Keeping encoder, algorithm and environment the same, which self-supervised algorithm is the most effective in improving results and preventing representation collapse?

Algorithm: Rainbow-style DQN
env: MinAtar (possibly Atari-100k if no hardware issues)
Loss functions:

Contrastive - CURL
Predictive - SPR
Quantized - iQRL


Related work

https://arxiv.org/abs/2206.05266
https://arxiv.org/pdf/1312.5602

