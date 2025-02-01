---
title: RLcourse note - Lecture 1 Introduction to Reinforcement Learning
author: DS Jung
date: 2021-11-26 16:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]    # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec1_thumb.JPG){: width="200px" height="100px"}](https://www.youtube.com/watch?v=2pWv7GOvuf0)
{: .text-center}

## Characteristics

- Label이 필요하지 않다. Reward 설계를 통해 Model이 학습한다.
- Feedback is delayed. 지도학습과 비교해 RL은 Reward 설정으로 부터 가시적인 변화를 얻기까지 시간이 걸린다.
- Time really matters 학습된 모델이 즉각적으로 환경에 반응하여 동작하는 만큼 시간과의 관계 또한 중요하다.
- Agent's actions affect the subsequent data it recieves 위와 비슷한 맥락

### Rewards

- reward $R_t$ 는 scalar feedback 신호로 정의한다. 왜 벡터로는 안되는 걸까. 그게 정의하기나 학습이 용이한 건 알겠으나 Vector는 안된다고 선을 그어야 할 이유는 아직 모르겠다.
- Agent는 Reward를 극대화 시켜야 한다.

Reinforcement Learning은 Reward Hypothesis로 시작한다.

|Definition (Reward Hypothesis)|
|---|
|All goals can be described by the maximisation of expected cumulative reward|

### Sequential Decision Making

- Goal : select actions to maximise total future reward
- Actions may have long term consequences
- Reward may be delayed
- It may be better to sacrifice immediate reward to gain more long-term reward
- 당연한 내용들이랄까

## Agent and Environment

- At each step $t$ the agent :
  - Executes action $A_t$
  - Receives observation $O_t$ from environmnet
  - Receives scalar reward $R_t$
- The environmen :
  - Receives actions $A_t$ from agent
  - Emits observation $O_{t+1}$
  - Emits scalar reward $R_{t+1}$ calculated

### History and State

- The history is sequence of observations, actions, rewards

$$H_t = A_1, O_1, R_1, ... , A_t, O_t, R_t $$

- State is the information used to determine what happens next
- Formally, state is a function of the history:

$$S_t = f(H_t)$$

#### Environment State

- The environment state $S^e_t$ is the environment's private representation
- not usually visible to the agent
- including irrelevant information

#### Agent State

- The agent state $S^\mathsf{a}_t$ is the agent's internal representation
- information used by reinforcement learning algorithms
- It can be any function of history:

$$S^\mathsf{a}_t = f(H_t)$$

#### Information State

An information state (a.k.a Markov state) contains all useful information from the history.

|Definition||
|:---|---:|
|A state $$S_t$$ is Markov if and only if | $$ \mathbb{P}[S_{t+1} \| S_t] = \mathbb{P}[S_{t+1} \| S_1, ...,S_t] $$|

### Fully Observable Environments

- Full observability: agent directly observes envrionment state

$$ O_t = S^\mathsf{a}_t - S^e_t $$

- Agent state = Environment state = information State
- Formally, this is a Markov decision process (MDP) 이부분은 약간의 논리적 오류가..?

### Partially Observable Environments

- Partial observability : agent indirectly observes environment 실질적인 경우 - 환경의 일부 정보만 습득 가능
- agent state $\neq$ environment state
- Formally this is called partially observable Markov decision process (POMDP)
- Agent must construct its own state representation $S^\mathsf{a}_t$
  - Complete history $S^\mathsf{a}_t = H_t$
  - Beliefs of envrionment state : $S^\mathsf{a}_t = (\mathbb{P}[S^e_t = s^1], ... , \mathbb{P}[S^e_t = s^n])$ 분포 형태의 state
  - Recurrent neural network : $S^\mathsf{a}_t = \sigma (S^\mathsf{a}_{t-1}W_s + O_tW_o)$ 회귀 신경망

## Major Components of an RL Agent

- An RL agent may include one or more of these components
  - Policy: agent's behaviour function
  - Value function: how good is each state and/or action
  - Model: agents's representation of the environment

### Policy

- A policy isthe agent's behaviour
- map from state to action
- Deterministic policy: $\mathsf{a}=\pi(s)$ - 확정적으로 action을 결정
- Stochastic policy: $ \pi(\mathsf{a}\|s) = \mathbb{P}[A=\mathsf{a}\|S=s] $ - 확률적으로 action을 선택

### Value Function

- Value function is a prediction of future reward
- Used to evaluate the goodness/badness of states
- And therefore to select between actions

$$ v_\pi(s) = \mathbb{E}_\pi[R_t + \gamma R_{t+1}+ \gamma ^2R_{t+2}| S_t = s] $$

### Model

- A model predicts what the environment will do next
- $\mathcal{P}$ predicts the next state
- $\mathcal{R}$ predicts the next (immediate) reward

$$ \mathcal{P}^\mathsf{a}_{ss'} = \mathbb{P}[S_{t+1} =s' | S_t =s, A_t=\mathsf{a}] $$

$$ \mathcal{R}^\mathsf{a}_{s} = \mathbb{E}[R_{t+1} | S_t =s, A_t=\mathsf{a}] $$

### Categorizing RL agents 1

- Value Based
  - [ ] ~~Policy~~
  - [x] Value Function
- Policy Based
  - [x] Policy
  - [ ] ~~Value Function~~
- Actor Critic
  - [x] Policy
  - [x] Value Function

### Categorizing RL agents 2

- Model Free
  - [x] Policy and/or Value Function
  - [ ] ~~Model~~
- Model Based
  - [x] Policy and/or Value Function
  - [x] Model

### RL Agent Taxonomy

![RL Agent Taxonomy](/assets/pic/RL_agent_taxonomy.jpg){: width="400px" height="400 px"}

## Learning and Planning

Two fundamental probelms in sequential decision making

- Reinforcement Learning
  - The environmnet is initially unknown
  - The agent interacts with the environment
  - The agent imporves its policy
- Planning
  - A model of the environment is known
  - The agent performs computations with its model (without any external interaction)
  - The agent imporves its policy

### Exploration and Exploitation

- **Exploration** finds more information about the envrionment
- **Exploitation** expolits known information to maximise reward
- 둘다 적절히 시행하는 것이 중요

### Prediction and Control

- Prediction: evaluate the future with given policy
- Control: optimise the future finding the best policy
