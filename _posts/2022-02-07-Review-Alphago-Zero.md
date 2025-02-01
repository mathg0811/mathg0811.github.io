---
title: Review - Article Alphago Zero
author: DS Jung
date: 2022-02-07 16:00:00 +0900
categories: [Review, Artificial Intelligence]
tags: [reinforcementlearning, review, alphago]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Link :

[![thumb1](/assets/pic/Review_Alphago_zero.JPG){: width="400px" height="200px"}](https://www.nature.com/articles/nature24270?sf123103138=1)
{: .text-center}

일단 읽으면서 생각 흐름대로 끄적이기부터

## Alphago

이세돌을 이긴 첫번째 alphgo는 두 개의 deep neural network를 사용

- Policy network : Outputs move probabilities
- Value network : predict the winner of games played by the policy network

Train 후에는 Monte Carlo Tree Saerch (MCTS) 로 통합, 가능성이 높은 수들을 위주로 예측하고 이 예측된 수들의 승리가능성을 value network로 판단

## Alphago Zero

Alphago Zero는 각각 판후이와 이세돌을 이긴 Alphago Fan 과 Alphago Lee 와는 중요한 점에서 다름

- Self-play reinforcement learning 으로만 학습하여 지도를 완전히 배제함
- Domain 지식 등이 들어가는 feature 변수를 완전히 제외하였고 오로지 바둑판과 돌 위치로만 학습시킴
- 두개의 network를 사용하지 않고 하나의 nearal network만 사용함 MC rollout 없이 single neural network로만 수를 계산하는 더 간단한 tree search를 사용함 - Algorithm that incorporates lookahead search inside the training loop

### Reinforcement learning

Deep neural network $f_\theta$ with parameters $\theta$

- Input : raw board representation $s$ of the position and its history
- Output : probabilities and a value $(p ,v) = f_\theta(s)$
- vector of move probabilities $p$ represents the probability of selecting each move $a$,$p_a = Pr(a\vert s)$
- neural network consists of residual blocks of convolutional layers with batch normalization and rectifier nonlinearities

- Train from self-play game
- In each position $s$, an MCTS search is executed, guided by neural network $f_\theta$
- MCTS search output probabilities $\pi$ of playing each move.
- neural network parameters $\theta$ are updated to maximize the similarity of the policy vector $p_t$ to the search probabilities $\pi_t$ and to minimize the error between the predicted winner $v_t$ and the game winner $z$
