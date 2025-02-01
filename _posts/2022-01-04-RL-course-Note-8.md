---
title: RLcourse note - Lecture 8 Integrating Learning and Planning
author: DS Jung
date: 2022-01-04 18:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec8_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/ItMutbeOHtc)
{: .text-center}

### 8강 소감

Tree Search 의 장점은 알 수 없는 Model을 가정하여 거기에서 현재까지 학습된 policy에 따라 여러 시나리오들을 진행해보고 그를 바탕으로 Value를 빠르게 계산해볼 수 있다는 것이다. 강의에서는 Model로부터 Sample Experience를 확보하여 빠른 학습이 가능하다는 것을 중점으로 두고 설명하고 있으나, 사실 이 부분은 논리적으로 빠른 학습은 어려워 보인다. 특히 모델의 오차를 더 강화할 수도 있다는 부분에서 그렇다. 그러나 Tree Search 과정은 상당히 유의미한데, 이 방법을 통해 실제 Policy를 기반으로 유의미한 미래 예측과 그 Return 예측을 통해 Action을 선택할 수 있게 함으로써 무가치한 미래 예측에 할당될 자원을 절약할 수 있고 보다 빠르게 실시간으로 Model을 완벽히 모르는 상태에서도 최선의 선택을 할 수 있다는 점이다. 이 구조는 실제 인간 등이 선택을 하는 사고 과정과도 유사하지 않을까?

## Introduction

Learn model directly from experience, and use planning to construct a value function or policy. Integrate learning and planning into a single architecture

model을 경험으로부터 학습한다고 하는데, 이게 제일 중요한 부분인거 같은데 잘 다뤄져있지 않다. 이 실제 environment에 근접하는 model을 결정하는 부분이야말로 RL의 학습을 결정하는 가장 중요한 부분이 아닐까?

### Model-Based and Model-Free RL

- Model-Free RL
  - No model
  - Learn value function (and/or policy) from experience
- Model-Based RL
  - Learn a model from experience
  - Plan value function (and/or policy) from model

## Model-Based Reinforcement Learning

Advantages:

- Can efficiently learn model by supervised learning methods
- Can reason about model uncertainty

Disadvantages:

- First learn a model, then construct a value function
  <br>$\Rightarrow$ two sources of approximation error

### Model

- A model $\mathcal{M}$ is a representation of an MDP $\langle\mathcal{S,A,P,R}\rangle$ parametrized by $\eta$
- We will assume state space $\mathcal{S}$ and action space $\mathcal{A}$ are known
- So a model $\mathcal{M=\langle P_\eta ,R_\eta \rangle}$ represents state transitions $\mathcal{P_\eta \approx P}$ and rewards $\mathcal{R_\eta \approx R}$

$$\begin{aligned}
\mathsf{ S_{t+1} } & \sim \mathsf{ \mathcal{P}_\eta(S_{t+1}\;\vert\;S_t, A_t) } \\
\mathsf{ R_{t+1} } &= \mathsf{ \mathcal{R}_\eta(R_{t+1}\;\vert\;S_t, A_t) }
\end{aligned}$$


- Typically assume conditional independence between state transitions and rewards

$$ \mathsf{ \mathbb{P}[S_{t+1}, R_{t+1}\;\vert\;S_t, A_t] = \mathbb{P}[S_{t+1}\;\vert\;S_t, A_t]\mathbb{P}[R_{t+1}\;\vert\;S_t, A_t] } $$

경험으로부터 Model의 P, R을 기록하고 이를 통해 비슷한 결과를 내는 모델을 만든다. 즉 모델을 확률적인 state 변화와 Reward로 준다는 것이다. 이보다 중요한 구성 요소가 충분히 있을 것 같은데 전부 필드 개발자의 직관에 맡기겠다고 말하는 것 같다. State와 Action, Probability와 Reward를 잘 구성하기 어려울 것 같은데. 단순한 문제 또는 Model을 완벽하게 알고 있는 문제가 아니고서야.

또한 Action의 결과로 어느 State로 전이되는지를 은근슬쩍 무시하기 시작했다. 어쩐지 지난 강의에서 Reward중 next state 부분을 없애버리더라. 이렇게 은근슬쩍 하나씩 맘대로 하는게 쌓이고 쌓인다. 물론 그 모든 조건을 만족하도록 Model을 만들어낼 수 있을지도 모른다. AI가 점점 어려워지겠지

#### Model Learning

- Goal: estimate model $\mathcal{M}_\eta$ from experience $\lbrace \mathsf{ S_1, A_1, R_2, \dots, S_T } \rbrace $
- This is a supervised learning problem

$$\begin{aligned}
\mathsf{ S_1, A_1 } &\rightarrow \mathsf{ R_2,S_2 } \\
\mathsf{ S_2, A_2 } &\rightarrow \mathsf{ R_3,S_3 } \\
&\vdots\\
\mathsf{ S_{T-1}, A_{T-1} } &\rightarrow \mathsf{ R_T,S_T }
\end{aligned}$$

- Learning $\mathsf{s,a,\rightarrow r}$ is a regression problem
- Learning $\mathsf{s,a,\rightarrow s'}$ is a density estimation problem
- Pick loss function, e.g. mean-squared error, KL divergence, $\dots$
- Find parameters $\eta$ that minimise empirical loss

Reward가 항상 Regression으로 결정될 수 있는 건 아닐 것 같은데, 전이 되는 state 에 따라 Reward가 바뀐다면 s,a 만 가지고는 결정할 수 없다.

#### Examples of Models

- Table Lookup Model
- Linear Expectation Model
- Linear Gaussian Model
- Gaussian Process Model
- Deep Belief Network Model
- ...

#### Table Lookup Model

- Model is an explicit MDP, $\mathcal{\hat{P}, \hat{R}}$
- Count visits $\mathsf{N(s,a)}$ to each state action pair

$$\begin{aligned}
\mathsf{ \mathcal{\hat{P}}^a_{s,s'} } &= \mathsf{ \frac{1}{N(s,a)} \displaystyle\sum^T_{t=1} 1(S_t,A_t,S_{t+1} = s,a,s') } \\
\mathsf{ \mathcal{\hat{R}}^a_s } &= \mathsf{ \frac{1}{N(s,a)} \displaystyle\sum^T_{t=1} 1(S_t,A_t = s,a)R_t }
\end{aligned}$$

- Alternatively
  - At each time-step t, record experience tuple
  <br>$$ \mathsf{ \langle S_t, A_t, R_{t+1}, S_{t+1} \rangle } $$
  - To sample, model, randomly pick tuple matching $ \mathsf{ \langle s,a, \cdot, \cdot \rangle } $

### Planning with a Model

- Given a model $\mathcal{M_\eta = \langle P_\eta, R_\eta \rangle}$
- Solve the MDP $\langle \mathcal{S,A,P_\eta, R_\eta} \rangle$
- Using favourite planning algorithm
  - Value iteration
  - Policy iteration
  - Tree search
  - ...

#### Sample-Based Planning

- A simple but powerful approach to planning
- Use the model only to generate samples
- Sample experience from model

$$\begin{aligned}
\mathsf{ S_{t+1} } & \sim \mathsf{ \mathcal{P}_\eta(S_{t+1}\;\vert\;S_t, A_t) } \\
\mathsf{ R_{t+1} } &= \mathsf{ \mathcal{R}_\eta(R_{t+1}\;\vert\;S_t, A_t) }
\end{aligned}$$

- Apply model-free RL to samples, e.g.:
  - Monte-Carlo control
  - Sarsa
  - Q-learning
- Sample-based planning methods are often more efficient

#### Planning with an Inaccurate Model

- Given an imperfect model $\langle \mathcal{P_\eta, R_\eta} \rangle \neq \langle \mathcal{P, R} \rangle$
- Performance of model-based RL is limited to optimal policy for approximate MDP $\langle \mathcal{S,A,P_\eta, R_\eta} \rangle$
- i.e. Model-based RL is only as good as the estimated model
- When the model is inaccurate, planning process will compute a suboptimal policy
- Solution 1: when model is wrong, use model-free RL
- Solution 2: reason explicitly about model uncertainty

이런 사례가 많이 생길것 같은데 그 이유는 여태까지 쌓아온 가정이 너무 많으면서 굳이 필요없어보이는 부분까지도 가정하기 때문이라는 느낌

## Integrated Architectures

### Dyna

#### Real and Simulated Experience

We consider two sources of experience

Real experience - Sample from envrionment (True MDP)

$$\begin{aligned}
\mathsf{ S' } & \sim \mathsf{ \mathcal{P}^a_{s,s'} } \\
\mathsf{ R } &= \mathsf{ \mathcal{R}^a_s }
\end{aligned}$$

Simulated experience - Sampled from model (approximate MDP)

$$\begin{aligned}
\mathsf{ S' } & \sim \mathsf{ \mathcal{P}_\eta(S'\;\vert\;S, A) } \\
\mathsf{ R } &= \mathsf{ \mathcal{R}_\eta(R\;\vert\;S, A) }
\end{aligned}$$

#### Integrating Learning and Planning

- Model-Free RL
  - No model
  - Learn value funciton (and/or policy) from real experience
- Model-Based RL (using Sample-Based Planning)
  - Learn a model from real experience
  - Plan value function (and/or policy) from simulated experience
- Dyna
  - Learn a model from real experience
  - Learn and plan value function (and/or policy) from real and simlated experience

#### Dyna-Q Algorithm

|Initialize Q(s,a) and Model(s,a) for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$<br>Do forever:<br>$\quad$ (a) $S\leftarrow$ current (nonterminal) state<br>$\quad$ (b) $A\leftarrow\epsilon$-greedy (S,A) <br>$\quad$ (c) Execute action A; observe resultant reward, R, and state, S'<br>$\quad$ (d) $Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma\; max_a Q(S',a)-Q(S,A)]$<br>$\quad$ (e) $Model(S,A)\leftarrow R,S'$ (assuming deterministic environment)<br>$\quad$ (f) Repeat n times: <br> $\quad\quad$ $S\leftarrow$ random previously observed state<br> $\quad\quad$ $A\leftarrow$ random action previously taken in S <br> $\quad\quad$ $R,S'\leftarrow Model(S,A)$ <br> $\quad\quad$ $Q(S,A)\leftarrow Q(S,A) + \alpha [R + \gamma max_a Q(S',a)-Q(S,A)]$|

Model에서 추가적인 exeprience를 얻어 Value를 학습시키는 것은 적은 episode로도 빠르게 Value를 학습시킬 수 있을 지 모르나, Model의 정확성은 Real Experience가 충분히 많은 수로 확보되고 Model이 실제 Environment에 가까울 때에만 믿을 수 있다. 때로 초반에 잘못된 Model을 구성하도록 유도할 수도 있으며 장기적으로도 크게 유의미할것 같지는 않다.

## Simulation-Based Search

### Forward Search

- Forward search algorithms select the best action by lookahead
- They build a search tree with the current state $s_t$ at the root
- Using a model of the MDP to look ahead
- No need to solve whole MDP, just sub-MDP starting from now

Simulation-Based Search

- Forward search paradigm using sample-based planning
- Simulate episodes of experience from now with the model
- Apply model-free RL to simulated episodes
- Simulate episodes of experience from now with the model

$$ \mathsf{ \lbrace s^k_t, A^k_t, R^k_{t+1}, \dots, S^k_T \rbrace ^K_{k=1} \sim \mathcal{M}_v } $$

- Apply model-free RL to simulated episodes
  - Monte-Carlo control $\rightarrow$ Monte-Carlo search
  - Sarsa $\rightarrow$ TD search

### Monte-Carlo Search

#### Simple Monter-Carlo Search

- Given a model $\mathcal{M}_v$ and a simulation policy $\pi$
- For each action $\mathsf{a}\in\mathcal{A}$
  - Simulate K episodes from current (real) state $\mathsf{s_t}$
  <br><center>$$ \mathsf{ \lbrace s_t, a, R^k_{t+1}, S^k_{t+1}, A^k_{t+1} \dots, S^k_T \rbrace ^K_{k=1} \sim \mathcal{M}_v,\pi } $$</center>
  - Evaluate actions by mean return (Monte-Carlo evaluation)
  <br><center>$$ \mathsf{ Q(s_t,a) = \frac{1}{K} \displaystyle\sum^K_{k=1} G_t \overset{P}{\rightarrow} q_\pi(s_t,a) } $$</center>
- Select current (real) action with maximum value
<br><center>$$ \mathsf{ a_t = \underset{a\in\mathcal{A}}{argmax}\; Q(s_t,a) } $$</center>

#### Monte-Carlo Tree Search (Evaluation)

- Given a model $\mathcal{M}_v$
- Simulate K episodes from current state $\mathsf{s_t}$ using current simulation policy $\pi$
<br><center>$$ \mathsf{ \lbrace s_t, A^k_{t+1}, R^k_{t+1}, S^k_{t+1}, \dots, S^k_T \rbrace ^K_{k=1} \sim \mathcal{M}_v,\pi } $$</center>
- Build a search tree containing visited states and actions
- Evaluate states Q(s,a) by mean return of episodes from s, a
<br><center>$$ \mathsf{ Q(s,a) = \frac{1}{N(s,a)} \displaystyle\sum^K_{k=1}\sum^T_{u=t} 1(S_u, A_u = s,a)G_u \overset{P}{\rightarrow} q_\pi(s_t,a) } $$</center>
- After search is finished, select current (real) action with maximum value in search tree
<br><center>$$ \mathsf{ a_t = \underset{a\in\mathcal{A}}{argmax}\; Q(s_t,a) } $$</center>

#### Monte-Carlo Tree Search (Simulation)

- MCTS, the simulation policy $\pi$ improves
- Each simulation consists of two phases (in-tree, out-of-tree)
  - Tree policy (improves): pick actions to maximise Q(S,A)
  - Default policy (fixed): pick actions randomly
- Repeat (each simulation)
  - Evaluate states Q(S,A) by Monte-Carlo evaluation
  - Improve tree policy, e.g. by $\epsilon$-greedy(Q)
- Monte-Carlo control applied to simulated experience
- Converges on the optimal search tree, $\mathsf{ Q(S,A) \rightarrow q_*(S,A)}$

#### Position Evaluation in Go

- How good is a position s?
- Reward function (undiscounted):

$$\begin{aligned}
\mathsf{ R_t } &= \mathsf{ 0 \text{ for all non-terminal steps } t<T } \\
\mathsf{ R_T } &= \begin{cases}
1 & \text{ if Balck wins } \\
0 & \text{ if White wins }
\end{cases}
\end{aligned}$$

- Policy $\mathsf{\pi = \langle\pi_B , \pi_W\rangle}$ selects moves for both players
- Value function (how good is position s):

$$\begin{aligned}
\mathsf{v_\pi(s)} &= \mathsf{ \mathbb{E}_\pi [R_T \vert S=s] = \mathbb{P} [\text{Black wins } \vert S=s] }  \\
\mathsf{v_*(s)} &= \mathsf{ \underset{\pi_B}{max}\; \underset{\pi_W}{min}\; v_\pi(s) }
\end{aligned}$$

#### Advantages of MC Tree Search

- Highly selective best-first search
- Evaluates states dynamically (unlike e.g. DP)
- Uses sampling to break curse of dimensionality
- Works for "black-box" models (only requires samples)
- Computationally efficient, anytime, parallelisable

### Temporal-Difference Search

- Simulation-based search
- Using TD instead of MC (bootstrapping)
- MC tree search applies MC control to sub-MDP from now
- TD search applies Sarsa to sub-MDP from now

#### MC vs. TD search

- For model-free reinforcement learning, bootstrapping is helpful
  - TD learning reduces variance but increases bias
  - TD learning is usually more efficient than MC
  - TD($\lambda$) can be much more efficient than MC
- For simulation-based search, bootstrapping is also helpful
  - TD search reduces variance but increase bias
  - TD search is usually more efficient than MC search
  - TD($\lambda$) search can be much more efficient than MC search

#### TD Search

- Simulate episodes from the current (real) state $\mathsf{s_t}$
- Estimate action-value function $\mathsf{Q(s,a)}$
- For each step of simulation, update action-values by Sarsa

$$ \mathsf{ \Delta Q(S,A) = \alpha (R+ \gamma Q(S',A') - Q(S,A)) } $$

- Select actions based on action-values $\mathsf{Q(s,a)}
  - e.g. $\epsilon$-greedy
- May also use function approximation for Q

#### Dyna-2

- In Dyna-2, the agent stores two sets of feature weights
  - Long-term memory
  - Short-term (working) memory
- Long-term memory is updated from real experience using TD learning
  - General domain knowledgd that applies to any episod
- Short-term memory is updated from simulated experience using TD search
  - Specific local knowledge about the currrent situation
- Over value function is sum of long and short-term memories
