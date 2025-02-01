---
title: RLcourse note - Lecture 2 Markov Decision Processes
author: DS Jung
date: 2021-11-27 10:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec2_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/lfHX2hHRMVQ)
{: .text-center}

## 2강 소감 및 생각

**결국 이렇게 정의된 MDP, Bellman equation을 푸는 것. 일반적인 모든 문제에서 Reward라는 것과 Action이라는 것으로 최대한 수학적으로 해를 구하기 쉽도록 학습 문제를 정의하기 위한 첫번째 단계인 듯하다. 최대한 Linear하게 만들고 싶은 것. 그러나 나는 AI 학습이 모든 것을 Linear하게 정의하려고 노력하는 것이 모든 상황을 커버할 수 있을지는 모르겠다. 물론 대부분의 수학적 접근은 Convolution과 Linear한 식의 combination으로 표현될 수 있다고 대충 느낌상 기억하고 있긴 하다만 그러면 Convolution과 Dense layer를 잘 구성해야 하며 수많은 Unrelated 변수가 생겨난다. 물론 그를 통해서 미처 파악하지 못한 변수를 이용해 추상적 접근을 모델링해내는 것이 가능할 수도 있다. 그러나 이 부분을 구현하기 위해서 Laplace transform과 거기서 표현되는 여러가지 수학적 접근을 구현할 수 있도록 Layer를 구현해야 할 것이다.**

## Introduction ot MDPs

- *Markov decision processes* formally describe and envrionment for reinforcement learning
- Where the envrionment is *fully observable*
- i.e. The current state completely characterise the process - state는 현재 상태 변수들로 markov 상태가 될 수 있도록 충분한 변수들을 갖추어야 함
- Almost all RL problems can be formalised as MDPs 뭐 잘 설계해서 MDP 맞추란 소린데 누가 이런말 못함

### Markov Property

"The future is independent of the past given the present" 문제를 자꾸 단정하고 싶어한다. 그게 속도나 효율면에서 맞긴 하지만 이게 다른 문제를 만들진 않을까? Markov로 만들수 없는 경우도 있을 것 같지만 복잡하다고 불가능하진 않을것 같은데, 혹은 간혹 더 간단한 모델을 만들 수도 있지 않을까

|Definition||
|:---|---:|
|A state $$S_t$$ is Markov if and only if | $$ \mathbb{P}[S_{t+1} \| S_t] = \mathbb{P}[S_{t+1} \| S_1, ...,S_t] $$|

- The state captures all relevant information from the history
- Once the state is known, the history may be thrown away
- 흐음... 그보다는 environment의 information에 적절히 history data를 축적한 변수를 만들어야 하지 않을까. 그 변수를 전부 현재의 변수라고 설정하는 것 뿐 현재의 envrionment에서 관찰할 수 있는 것은 아니지 않으려나.

### State Transition Matrix

For a Markov state *s* and successor state *s'*, the *state transition probability* is defined by,

$$ \mathcal{P}_{ss'}=\mathbb{P}[S_{t+1}=s'|S_t =s] $$

State transition matrix $\mathcal{P}$ defines transition probabilities from all states $s$ to all successor states $s'$

$$ \mathcal{P} = from \begin{bmatrix} \mathcal{P}_{11}&\dots&\mathcal{P}_{1n}\\ \vdots & & \\ \mathcal{P}_{n1}&\dots&\mathcal{P}_{nn} \end{bmatrix}$$

- state 의 갯수가 n개로 제한된 경우에만 사용하는 건가?
- 확률로 state 변화가 표현된 만큼 action이 개입할 여지가 없다. 결국 이 확률 $\mathcal{P}$ 는 이후 가변 성질의 policy로 대체되는 것 같다.

## Markov Process (1단계)

A Markov process is a memoryless random process
마르코브 형태의 프로세스에 대한 설명, 확률만 가지고 state의 진행만을 보여주므로 agent, reward 이런건 없음. memoryless 이전 state과 필요없음. random, sampling 할 수 있다.

|Definition|
|---|
|a *Markov* Process (or *Markov Chain*) is a tuple $\langle\mathcal{S,P}\rangle$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{S}}\;$ is a (finite) set of states<br>$\tiny{\blacksquare}\quad\normalsize{\mathcal{P}}\;$ is a state transition probability matrix,<br> $$\quad\mathcal{P}_{ss'}=\mathbb{P}[S_{t+1}=s'\|S_t=s]$$|

## Markov Reward Process (MRP)

A Markov reward process is a Markov chain with values
Reward와 Discount factor를 정의함.

|Definition|
|---|
|a *Markov Reward* Process is a tuple $\langle\mathcal{S,P,}$<span style="color:red">$\mathcal{R,\gamma}$</span>$\rangle$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{S}}\;$ is a (finite) set of states<br>$\tiny{\blacksquare}\quad\normalsize{\mathcal{P}}\;$ is a state transition probability matrix,<br> $$\quad\mathcal{P}_{ss'}=\mathbb{P}[S_{t+1}=s'\|S_t=s]$$ <br> $\tiny{\blacksquare}\quad$ <span style="color:red">$\normalsize{\mathcal{R}}\;$ is a reward function, $$\mathcal{R}_s=\mathbb{E}[R_{t+1} \| S_t=s] $$</span><br>$\tiny{\blacksquare}\quad$ <span style="color:red">$\normalsize{\mathcal{\gamma}}\;$ is a discount factor, $$\gamma\in[0,1]$$</span>|

### Return $G_t$

|Definition|
|---|
|The return $G_t$ is the total discounted reward from tiem-step $t$.<br><center>$$G_t = R_{t+1}+\gamma R_{t+2}+... = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$</center>|

- The discount $\gamma \in [0,1]$ is the present value of future rewards
- The value of receiving reward $R$ after $k+1$ time-steps is $\gamma^k R$
- This values immediate reward above delayed reward
  - $\gamma$ close to 0 leads to "myopic" evaluation
  - $\gamma$ close to 1 leads to "far-sighted" evaluation
- 단순 급수적인 형태로 감가를 시행하고 있지만 경우에 따라서 감가 함수를 별도로 함수로 만들어 사용할 수 있을 듯. 크게 필요한 경우가 있을 진 모르겠지만..
- 여기부터 이상해 $R_t$는 어디간거야 계산할 땐 다 쓰면서... 별로 중요한건 아닌데 거슬리네

### Reason of discount

- 수학적으로 편리함
- 무한히 증가하는 return을 제거
- state가 항상 terminate 될 것이 보장된다면 discount를 1로 해도 괜찮을 수 있음

### Value Function

The value function $v(s)$ gives the long-term value of state $s$
기댓값(평균)

|Definition|
|---|
|The *state* vlaue function $v(s)$ of an MRP is the expected return starting from state $s$<br> <center>$$\mathsf{v}(s)=\mathbb{E}[G_t\;\|\;S_t=s] $$</center>|

## Bellman Equation for MRPs

The value function can be decomposed into two parts:
Value function을 학습시키기 위한 기본 논리
근데 이상한게 $R_{t+1}$ 써놓고 계산한거 보면 다 $R_t$ 쓴 것 같음. 그리고 예시를 보면 그게 맞는거 같기도 함. 식 중에서 중간에 뜬금없이 $\mathbb{E}[R_{t+1}]$을 $R_s$로 바꿔놓음 아주 지멋대로야.

- immediate reward $R_{t+1}$
- discounted value of successor state $\gamma \mathsf{v}(S_{t+1})$

$$\quad\quad\quad\quad\mathsf{v}(s) =\mathbb{E}\,[\,G_t\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma R_{t+2} + \gamma ^2 R_{t+3} + \dots\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma (R_{t+2} + \gamma R_{t+3} + \dots)\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma G_{t+1}\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\quad\quad=\mathbb{E}\,[\,R_{t+1}+\gamma \mathsf{v} (S_{t+1})\;\|\;S_t=s\,]$$<br>
$$\quad\quad\quad\quad\mathsf{v}(s) =\mathcal{R}_s + \gamma \displaystyle\sum_{s' \in S} ^{}\mathcal{P}_{ss'} \mathsf{v} (s')$$

### Bellman Equation in Matrix Form

The Bellman equation can be expressed concisely using matrices.

$$ \mathsf{v} = \mathcal{R} + \gamma \mathcal{P} \mathsf{v} $$

where $\mathsf{v}$ is a column vector with one entry per state

$$  \begin{bmatrix} \mathsf{v} (1) \\ \vdots \\ \mathsf{v} (n) \end{bmatrix} = \begin{bmatrix} \mathcal{R}_1 \\ \vdots \\ \mathcal{R}_n \end{bmatrix} + \gamma \begin{bmatrix} \mathcal{P}_{11} & \dots & \mathcal{P}_{1n} \\ \vdots & & \\ \mathcal{P}_{n1} & \dots & \mathcal{P}_{nn} \end{bmatrix} \begin{bmatrix} \mathsf{v} (1) \\ \vdots \\ \mathsf{v} (n) \end{bmatrix} $$

각 state 1~n에서의 value function은 각 state의 reward와 dot (각 state로의 확률), (각 state의 value) * discount factor 의 합이다.
너무 흔한 형태의 식... 그러나 수치해석해야함...

### Solving the Bellman Equation

- Linear equation

$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \mathsf{v} = \mathcal{R+ \gamma P}\mathsf{v}$$<br>
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\; (1 - \gamma \mathcal{P}) \mathsf{v} = \mathcal{R}$$<br>
$$\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \mathsf{v} = (1 - \gamma \mathcal{P})^{-1} \mathcal{R}$$

- Computational complexiy is $O(n^3)$ for n state
- Direct solution only possible for small MRPs 근데 되겠냐
- There are many iterative methods for large MRPs, e.g.
  - Dynamic programming
  - Monte-Carlo evaluation
  - Temporal-Difference learning

## Markov Decision Process (main)

드디어 Action으로 개입이 추가되었다. 근데 policy는 왜 빼놓고 하냐 논리적으로 문제가 생기잖아 이미 probability를 정의해 놓고서는 Action을 어떻게 해, 결국 예제에서는 Action을 정의하지 않는 자동 전이 state에서만 Probability를 사용한다. 물론 실제로 둘이 공존할수 있기는 하다. 그리고 Action을 만드는 애는 policy라서 이미 있는 것이다.

|Definition|
|---|
|a *Markov Reward* Process is a tuple $\langle\mathcal{S, }$<span style="color:red">$\mathcal{A}$</span>$\mathcal{, P, R, \gamma \rangle}$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{S}}\;$ is a finite set of states<br>$\tiny{\blacksquare}\quad$ <span style="color:red">$\normalsize{\mathcal{A}}\;$ is a finite set of actions</span><br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{P}}\;$ is a state transition probability matrix,<br> $$\quad\mathcal{P}^a_{ss'}=\mathbb{P}[S_{t+1}=s'\|S_t=s,$$ <span style="color:red">$$A_t =a$$</span>$$]$$ <br> $\tiny{\blacksquare}\quad \normalsize{\mathcal{R}}\;$ is a reward function, $$\mathcal{R}^a_s=\mathbb{E}[R_{t+1} \| S_t=s,$$ <span style="color:red">$$A_t =a$$</span>$$] $$<br>$\tiny{\blacksquare}\quad \normalsize{\mathcal{\gamma}}\;$ is a discount factor, $$\gamma\in[0,1]$$|

## Policies

Policy는 stochastic으로 시작해 확률을 정의하는 것처럼 해놓았지만 결국엔 deterministic으로 은근슬쩍 바뀐다. 물론 내부 policy 정의는 양쪽을 다 가다가 우선순위 높은 놈으로 바뀌는 식이니 둘다 맞는 말이긴 하다.

|Definition|
|---|
|A policy $\pi$ is a distribution over actions given states.<br><centor>$$\pi (a\| s) = \mathbb{P} [A_t = a \| S_t = s]$$|

- A policy fully defines the behaviour of an agent
- MDP policies depend on the current state (not the history)
- Policies are stationary (time-independent)

$\quad A_t ~ \pi(. \| S_t), \forall t > 0$

- Given an MDP $\mathcal{M = \langle S, A, P ,R ,\gamma \rangle}$ and a policy $ \pi$
- The state sequence $\mathcal{S_1, S_2, \dots }$ is a Markov process $\mathcal{\langle S, P^{\pi} \rangle}$
- The state and reward sewuence $\mathcal{S_1, R_2, S_2, \dots}$ is a Markov reward process $\mathcal{\langle S, P^{\pi}, R^{\pi}, \gamma \rangle }$
- 슬쩍 A를 뺐지만 policy에 포함된 것이므로 없어진 게 아니다 바뀐것 없다
- where

<center>
$$ \mathcal{P^{\pi}_{s, s'}= \displaystyle\sum_{a \in A} \pi (a \| s) P^a_{ss'}}$$
$$ \mathcal{R^{\pi}_{s}= \displaystyle\sum_{a \in A} \pi (a \| s) R^a_{s}}$$</center>

## Value Function 2

각 state와 action의 value를 정의한다.

The state-value function $\mathsf{v_{\pi} (s)} $ of an MDP is the expected return starting from state s, and then following policy $\pi$

$$\mathsf{v_{\pi} (s) = \mathbb{E}_{\pi} [G_t | S_t = s]} $$

The action-value function $\mathsf{q_{\pi} (s,a)}$ is the expected return starting from state s, taking action a, and then following policy $\pi$

$$\mathsf{q_{\pi} (s,a) = \mathbb{e}_{\pi} [G_t | S_t = s, A_t = a] }$$

### Bellman Expectation Equation

필요에 따라 연쇄적으로 대충 정의할 수 있다.
Action에 따라서 따라오는 State도 stochastic할 수 있다.

The state-value function can again be decomposed into immediate reward plus discounted value of successor state

$$ \mathsf{v_{\pi} (s) = \mathbb{E}_{\pi} [R_{t+1} + \gamma v_{\pi} (S_{t+1}) | S_t = s]}$$

The action-value function can similarly be decomposed

$$ \mathsf{q_{\pi} (s,a) = \mathbb{E}_{\pi}[R_{t+1} + \gamma q_{\pi} (S_{t+1}, A_{t+1}) | S_t = s, A_t=a]} $$

#### Bellman Expectation Equation Matrix Form

The Bellman expectation equation can be expressed concisely using the induced MRP

$$ \mathsf{v_{\pi} = \mathcal{R}^{\pi} + \gamma \mathcal{P}^{\pi} v_{\pi}}$$

with direct solution

$$ \mathsf{v_{\pi} = (1- \gamma \mathcal{P}^{\pi})^{-1} \mathcal{R}^{\pi}} $$

### Optimal Value Function

The optimal state-value function $\mathsf{v_* (s)}$ is the maximum value function over all policies

$$ \mathsf{v_* (s) = \displaystyle\max_{\pi} v_{\pi} (s)} $$

The optimal action-value function $\mathsf{a_* (s,a)}$ is the maximum action-value function over all policies

$$ \mathsf{q_* (s,a) = \displaystyle\max_{\pi} q_{\pi} (s,a)} $$

- The optimal value function specifies the best possible performance in the MDP
- An MDP is "solved" when we know the optimal value fn

### Optimal Policy

증명생략 - 하여튼 존재하고 반드시 그렇다. 흠 잘못 수렴되는 경우가 발생할 수 있을 것 같은데..

Define a partial ordering over policies

$$ \mathsf{\pi \ge \pi ' \quad if \quad v_{\pi} (s) \ge v_{\pi '} (s), \forall s} $$

|Theorem|
|---|
|For any Markov Decision Process<br>$\bullet\;$ There exists an optimal policy $\pi$, that is better than or equal to all other policies, $\pi _* \geq  \pi, \forall \pi $<br>$\bullet\;$  All optimal policies achieve the optimal value function, $$\mathsf{v_{\pi_*} (s) = v_* (s)}$$<br>$\bullet\;$  All optimal policies achieve the optimal action-value function, $$\mathsf{q_{\pi_*} (s,a) = q_* (s,a)}$$|

#### Finding an Optimal Policy

An optimal policy can be found by maximising over $$\mathsf{q_* (s,a)}$$

$$ \mathsf{\pi_*(a\vert s)} = \begin{cases} \mathsf{1\quad if \; a = \underset{a\in\mathcal{A}}{argmax} \; q_* (s,a)} \\ \mathsf{0 \quad otherwise} \end{cases}$$

- There is always a deterministic optimal policy for any MDP
- If we know $\mathsf{q_* (s,a)}$, we immediately have the optimal policy
- 당연하지만 optimal이 되면 deterministic이 된다.

### Bellman Optimality Equation

The optimal value functions are recursively related by the Bellman optimality equations:

$$ \mathsf{v_* (s) = \underset{a}{max} \;q_* (s,a)}$$

$$ \mathsf{q_* (s,a) = \mathcal{R}^a_s + \gamma \displaystyle\sum_{s'\in S} \mathcal{P}^a_{ss'} v_* (s')}$$

$$ \mathsf{v_* (s) = \underset{a}{max} \; \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in S} \mathcal{P}^a_{ss'}v_* (s')}$$

$$ \mathsf{q_* (s,a) = \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in S} \mathcal{P}^a_{ss'} \underset{a'}{max} \; q_* (s', a')} $$

#### Solving the Bellman Optimality Equation

- 수치해석 하라는 소리. 오랜만에 복습해야겠는 걸 잘 기억이안나네 쉽긴한데
- Bellman Optimality Equation is non-linear
- No closed form solution (in general)
- Many iterative solution methods
  - Value Iteration
  - Policy Iteration
  - Q-learning
  - Sarsa

## Extensions to MDPs

여기부턴 강의는 없다. PPT만 있다.

- Infinite and continuous MDPs
- Partially observable MDPs
- Undiscounted, average reward MDPs

### Infinite MDPs

Continuous한 MDP, 물리적상태 같은건가 HJB equation 다시한번 봐야겠다

The following extensions are all possible:

- Countably infinite state and/or action spaces
  - Straightforward
- Continuous state and/or action spaces
  - Closed form for linear quadratic model(LQR)
- Continuous Time
  - Requires partial differential equations
  - Hamilton-Jaccobian-Bellman (HJB) equation
  - Limiting case of Bellman equation as time-step $\rightarrow 0$

### POMDPs

실제 모든것이 관측 가능하지 않은 문제에 대한 정의로 observation이 들어간다. 실제로 이런게 많을것 같은데 정의를 어떻게 할 것인가

A Partially Observab;e Markov Decision Process is an MDP with hidden states. It is a hidden Markov model with actions.

| Definition|
|---|
|A POMDP is a tuple $\mathcal{\langle S, A, \textcolor{red}{O}, P, R,\textcolor{red}{Z}, \gamma \rangle}$<br>$\bullet\;\; \mathcal{S}\;$ is a finite set of states<br>$\bullet\;\; \mathcal{A}\;$ is a finite set of actions<br>$\bullet\;\; \textcolor{red}{\mathcal{A}}\;$ is a finite set of observation<br>$\bullet\;\; \mathcal{P}\;$ is a state transition probability matrix<br>$$\quad\,\mathsf{\mathcal{P}^a_{ss'} = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a] }$$<br>$\bullet\;\; \mathcal{R}\;$ is a reward function<br>$$\quad\,\mathsf{\mathcal{R}^a_s = \mathbb{E} [R_{t+1} \vert S_t = s, A_t = a] }$$<br>$\bullet\;\; \mathcal{\textcolor{red}{Z}}\;$ is an observation function<br>$$\quad\,\mathsf{\mathcal{Z}^a_{s'o} = \mathbb{P} [O_{t+1} = o \vert S_{t+1} = s', A_t = a] }$$<br>$\bullet\;\; \mathcal{\gamma}\;$ is a discount factor $\gamma \in [0, 1]$|

### Belief States

현재 state가 어딘지 모르고 history에 따라 확률로 정의한다는 뜻인가? 흐음 어떤 경우인지 아직 모르겠다.

A history $H_t$ is a sequence of actions, observations and rewards

$$\mathsf{H_t = A_0, O_1, R_1, \dots , A_{t-1}, O_t, R_t}$$

A belief state $b(h) is a probablity distribution over states, conditioned on the history h

$$\mathsf{b(h) = (\mathbb{P} [ S_t = s^1 \vert H_t = h ], \dots , \mathbb{P}[S_t = s^n \vert H_t = h])}$$

### Reductinos of POMDPs

State tree로 표현하는게 뭘까 흠 bellman equation으로 표현하기 어려운 경우가 있는건가

- The history $\mathsf{H_t}$ satisfies the Markov property
- The belief state $\mathsf{b(H_t)}$ satisfies the Markov property
- A POMDP can be reduced to an (infinite) history tree
- A POMDP can be reduced to an (infinite) belief state tree

### Ergodic Markov Process

각 policy에 대해서 각 state에 나타나는 확률적 분포에 대한 이야기같다. 나중에 다시 생각해볼까..

An ergodic Markov process is

- Recurrent: each state is visted an infinite number of times
- Aperiodic: each state is visited without any systematic period

|Theorem|
|---|
|An ergodic Markov process has a limiting stationary distribution $\mathsf{d^{\pi}(s)}$ with the property<br><center>$$ \mathsf{d^{\pi} (s) = \displaystyle\sum_{s' \in S} d^{\pi} (s') \mathcal{P}_{s's}}$$</center>|

### Ergodic MDP

그렇게 state의 확률 분포가 수렴하는 형태의 MDP에 대해서 policy에 의한 각 타임 step의 평균적인 reward가 존재한다 흠... 이게 마이너스면 time이 흘러갈수록 reward는 계속 낮아지고 그런거일 뿐 아닌가...? 이 평균은 어따쓴담

|Definition|
|---|
| An MDP is ergodic if the Markov chain induced by any policy is ergodic|

For any policy $\pi$, and ergodic MDP has an average reward per time-step $\rho ^{\pi}$ that is independent of start state.

$$ \mathsf{\rho^{\pi} = \underset{T\rightarrow \infty}{lim} \; {1\over T}\mathbb{E} \left[\displaystyle\sum^T_{t=1} R_t\right]} $$

### Average Reward Value Function

Ergodic 내용을 적용한 Value Function

- The value function of an undiscounted, ergodic MDP can be expressed in terms of average reward.
- $\mathsf{\tilde{v}_{\pi} (s)$ is the extra reward due to starting from state s,

$$\mathsf{\tilde{v}_{\pi} (s) = \mathbb{E}_{\pi} \left[ \displaystyle\sum^{\infty}_{k=1} (R_{t+k} - \rho^{\pi})\;\vert\;S_t =s \right ]}$$

There is a corresponding average reward Bellman equation,

$$\mathsf{\tilde{v}_{\pi} (s) = \mathbb{E}_{\pi} \left[(R_{t+1} - \rho^{\pi}) + \displaystyle\sum^{\infty}_{k=1} (R_{t+k} - \rho^{\pi})\;\vert\;S_t =s \right ]}$$

$$\quad\quad = \mathsf{\mathbb{E}_{\pi} \left[(R_{t+1} - \rho^{\pi}) + \tilde{v}_{\pi} (s+1)\;\vert\;S_t =s \right ]}$$
