---
title: RLcourse note - Lecture 3 Planning by Dynamic Programming
author: DS Jung
date: 2021-12-02 10:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec3_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/Nd1-UUMVfz4)
{: .text-center}

## Introduction of Dynamic Programming

**Dynamic** sequential or temporal component to the problem

**Programming** optimising a "program", i.e. a policy

- A method for solving complex problems
- By breaking them down into subproblems
  - Solve the subproblems
  - Combine solutions to subproblems
- Dynamic programming이라 이름이 흠 어울리는것 같지 않은데

### 3강 소감

3강은 Bellman equation에 기초하여 policy와 value를 iteration으로 계산하는 기본적인 논리를 보여준다. 기초적인 방법인 만큼 조금 더 깔끔하게 정리가 되어 있으면 좋겠지만 적당히 개념위주로 보여주고 수학적 논리적 기반은 가볍게 넘어가는것 같거나 조금 허점?이 있는 것도 같다. 어쨌거나 학습 문제를 풀기 위해 정의한 state, policy, action, value의 첫번째 활용이니 이번엔 그냥저냥 넘어가볼까.

### Requirements for Dynamic Programming

Dynamic Programming is a very general solution method for problems which have two properties:

- Optimal substructure
  - *Principle of optimality* applies
  - Optimal solution can be decomposed into subproblems
- Overlapping subproblems
  - Subproblems recure many times
  - Solutions can be cached and reused
- Markov decision processes satisfy both properties
  - Bellman equation gives recursive decomposition
  - Value function stores and reuses solutions

### Planning by Dynamic Programming

- Dynamic programming assumes full knowledge of the MDP
- It is used for *planning* in an MDP
- For prediction:
  - Input: MDP $\mathcal{\langle S, A, P, R, \gamma \rangle} $ and policy $\pi$
  - &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or: MRP $\mathcal{\langle S, P^{\pi}, R^{\pi}, \gamma \rangle}$
  - Output: value function $\mathsf{v_{\pi}}$
- Or for control:
  - Input: MDP $\mathcal{\langle S, A, P, R, \gamma \rangle} $
  - Output: optimal value function $\mathsf{v_*}$
  - &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and: optimal policy $\pi_*$

## Iterative Policy Evaluation

To evaluate given policy $\pi$, iterative application of Bellman expectation backup is applied

Reward를 모르는 것(0)으로 간주하고 backup을 반복하여 Reward 계산 - Policy 변경 없음

### synchronous backup

- At each iteration $k+1$, For all states $\mathsf{s} \in \mathcal{S}$
- Update $\mathsf{v_{k+1}(s)}$ from $\mathsf{v_{k}(s')}$
- where $\mathsf{s'}$ is a successor state of s

$$\begin{aligned}
\mathsf{v_{k+1}(s)} &= \mathsf{\displaystyle\sum_{a \in \mathcal{A}}\pi(a \vert s)\left( \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'}v_k(s')\right)} \\
\mathsf{\boldsymbol{v}^{k+1}} &= \mathsf{\mathcal{\boldsymbol{R}}^{\pi} + \gamma \mathcal{\boldsymbol{P} }^{\pi} v^k}
\end{aligned}$$

- $\mathcal{R}$ 은 $\sum$ *( policy * action-reward )*
- $\mathcal{P}$ 는 $\sum$ *( policy * $\sum$ ( Probability of transition from s to s' by action * value of s' state ) )*
- 단순화 하겠다고 쫌 너무 막나가는거 아니냐

## Policy Iteration

- Given a policy $\pi$
  - Evaluate the policy $\pi$

$$ \mathsf{v_{\pi} (s) = \mathbb{E} \left[ R_{t+1} + \gamma R_{t+2} + \dots \vert S_t = s \right]} $$

  - Improve the policy by acting greedily with respect to $\mathsf{v_{\pi}}$ - 그냥 value 높은 쪽으로 가게 하는 policy라는 뜻

$$ \mathsf{\pi ' = greedy(v_{\pi})} $$

- This process of policy iteration always converges to $\pi_*$

### Policy Improvement

- Consider a determinisitc policy, $\mathsf{a = \pi (s)}$
- Improve the policy by acting greedily

$$ \mathsf{\pi ' (s) = \underset{a \in \mathcal{A}}{argmax}\; q_{\pi} (s,a)} $$

- This improves the value from any state $\mathsf{s}$ over one step

$$ \mathsf{q_{\pi} (s, \pi ' (s)) = \underset{a \in \mathcal {A}}{mas} \; q_{\pi}(s,a) \geq q_{\pi}(s,\pi (s)) = v_{\pi} (s)} $$

- This imporves the value from any state s over one step

$$ \mathsf{q_{\pi} (s, \pi '(s)) = \underset{a \in \mathcal{A}}{max} \; q_{\pi} (s,a) \geq q_\pi (s,\pi(s)) = v_\pi (s)}$$

- 이 action value와 state value를 동일시하는 이 부분은 deterministic일 때만 성립한다.

- It therefor imporves the value function, $\mathsf{v_{\pi'}(s) \geq v_\pi(s)}$

$$\begin{aligned}
\mathsf{v_\pi(s)} &\leq \mathsf{q_\pi(s,\pi'(s)) = \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma v_\pi(S_{t+1})\;\vert\;S_t = s \right]} \\
&\leq \mathsf{\mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma q_\pi (S_{t+1}, \pi '(S_{t+1}))\;\vert\;S_t = s \right]}\\
&\leq \mathsf{\mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} +\gamma ^2q_\pi (S_{t+2}, \pi '(S_{t+2}))\;\vert\;S_t = s \right]}\\
&\leq \mathsf{\mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \dots \;\vert\;S_t = s \right] = v_{\pi '} (s)}
\end{aligned}$$

- Deterministic인 경우에 한정하는 식 전개로써 $\mathsf{R}$은 $\mathbb{E}$내부에 있을 필요가 없다. 이 강의의 수식부분들이 때때로 찜찜한 이유는 이런 부분을 명확히 하지 않기 때문이다. 수학적인 논리 전개를 좀 제대로 해주면 좋겠다. small grid 같은 예제는 deterministic이 아니기 때문에 이 경우에 적절하지않고 $\mathsf{v}$가 점점 작아질 수 있는 이유가 deterministic이 아니기 때문에 위 식이 성립하지 않기 때문이다. 이 방법으로 반복하면 deterministic이 아니어도 optimal에 가까워질 수 있을까? 반례가 있을지 궁금하긴 한데 강의 다 듣고나서로 미뤄두겠다.

- If improvements stop,

$$ \mathsf{q_{\pi} (s, \pi '(s)) = \underset{a \in \mathcal{A}}{max} \; q_{\pi} (s,a) = q_\pi (s,\pi(s)) = v_\pi (s)}$$

- Then the Bellman optimality equation has been satisfied

$$ \mathsf{ v_\pi(s) = \underset{a \in \mathcal{A}}{max} \; q_\pi(s,a)} $$

- 이 정의는 적절하지 않다. $\mathsf{v}$가 최대 action value일 때라고 하는건 바보같은 정의이며 그보다는 Value Iteration을 반복해도 Value가 바뀌지 않으며 모든 state에서 Bellman expectation과 현재 value가 같을 때 optimality가 성립한다고 하는게 적절할 것이다.

- Therefore $\mathsf{v_\pi(s)=v_*(s)}$ for all $\mathsf{s} \in \mathcal{S}$
- so $\pi$ is an optimal policy

- 주어진 Policy에 대해서 Value를 계산한다. 그 방법 중 대표로써 Bellman Expectation Backup 을 이용해 Iterative한 Value를 계산하는데 이 방법이 무조건 optimal 해질 수 있는지에 대해서는 Deterministic한 경우에 대해서만 증명이 된 것이다.

### Modified Policy Iteration

- Does policy evaluation need to converge to $\mathsf{v_\pi}$?
- Or should we introduce  a stopping condition
  - e.g. $\epsilon$-convergence of value function
- Or simply stop after $\mathsf{k}$ iterations of iterative policy evaluation?
- Why not update policy every iteration?
  - This is equivalent to value iteration

### Principle of Optimality

An optimal policy can be subdivided into two components:

- An optimal first action $\mathsf{A_*}$
- Followed by an optimal policy from successor state $\mathsf{S'}$

|Theorem (Principle of Optimality)|
|---|
|A policy $\pi\mathsf{(a \vert s)} $ achieves the optimal value from state s, $\mathsf{v_\pi (s) = v_* (s)}$, if and only if<br>$\quad\bullet\;\;$ For any state $\mathsf{s'}$ teachable from $\mathsf{s}$<br>$\quad\bullet\;\;$ $\pi$ achieves the optimal value from state $\mathsf{s', v_\pi(s')=v_*(s')}$|

이 Theorem은 당연한 소리를 써놓은거 같은데 왜 Theorem인지는 나중에 생각해볼 필요가 있겠다.

## Value Iteration

### Deterministic Value Iteration

- If we know the solution to subproblems $\mathsf{v_*(s')}$
- solution $\mathsf{v_*(s)}$ can be found by one-step lookahead

$$\mathsf{v_*(s) \leftarrow \underset{a \in \mathcal{A}}{max} \; \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in \mathcal{S}}\mathcal{P}^a_{ss'}v_*(s')}$$

- The idea of value iteration is to apply these updates iteratively
- Intuition: start with final rewards and work backwards
- Still works with loopy, stochastic MDPs
- 증명이 복잡해서 보여주지 않는건가 싶지만.... 앞에 stochastic으로 계산하던 거에서 deterministic으로 backup만 바뀌었다. policy iteration Value iteration 큰 차이도 없는것 같다. 그저 policy를 명시하지 않았을 뿐. greedy만 했던 poicy는 있으나 없으나 차이를 모르겠다.

### Value Iteration - Define

- Probleam: find optimal policy $\pi$
- Solution: ierative application of Bellman optimality backup
- $\mathsf{v_1 \rightarrow v_2 \rightarrow \dots \rightarrow v_*}$
- Using synchronous backups
  - At each iteration $\mathsf{k+1}$
  - For all states $\mathsf{s \in \mathcal{S}}$
  - Update $\mathsf{v_{k+1}(s)}$ from $\mathsf{v_k(s')}$
- Convergence to $\mathsf{v_*}$ will be proven later
- Unlike policy iteration, there is no explicit policy
- Intermediate value functions may not correspond to any policy

$$\begin{aligned}
\mathsf{v_{k+1}(s)} &= \mathsf{\underset{a \in \mathcal{A}}{max}\left( \mathcal{R}^a_s + \gamma \displaystyle\sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'}v_k(s')\right)} \\
\mathsf{\boldsymbol{v}_{k+1}} &= \mathsf{\underset{a \in \mathcal{A}}{max}\;\mathcal{\boldsymbol{R}}^a+ \gamma \mathcal{\boldsymbol{P} }^a v^k}
\end{aligned}$$

## Extensions to Dynamic Programming

### Synchronous Dynamic Programming Algorithms

| Problem | Bellman Equation | Algorithm |
| :--- | :--- | :---:|
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation|
| Control | Bellman Expectation Equation + Greedy Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Equation | Value Iteration |

- Algorithms are based on state-value function $\mathsf{v_\pi(s)}$ or $\mathsf{v_*(s)}$
- Complexiy $\mathcal{O}\mathsf{mc^2}$ per iteration, for $\mathsf{m}$ actions and $\mathsf{n}$ states
- Could also apply to action-valuefunction $\mathsf{q_\pi(s,a)}$ or $\mathsf{q_*(s,a)}$
- Complexity $\mathcal{O}\mathsf{(m^2n^2)}$ per iteration

- Expectation에서는 Stochastic 할수 있는 경우를 포함하여 Value를 계산하고 Control 하고 싶은경우 policy를 greedy 로 deterministic하게 바꿔주는 것이고 Optimality에서는 policy를 계속 deterministic으로 간주하고 value를 계산하고 policy를 바꿔주는 거라서 식 자체는 거의다 그대로인데 policy를 바꾸지 않는게 prediction이고 Control은 deterministic하게 해주는 것 뿐이다. Policy는 기본적으로 deterministic을 가정하고 하는건가? Probability는 내부적으로 유지한 채로 value를 계산하면서 policy를 greedy하게 설정해주는것과 얼마나 다를지 생각해볼 필요도 있겠다.

### ASynchronous Dynamic Programming

- DP methods described so far used synchronous backups
- i.e. all states are backed up in parallel
- Asynchronous DP backs up states individually, in any order
- For each selected state, apply the appropriate backup
- Can significantly reduce computation
- Guaranteed to converge if all states continue to be selected

- Update의 방법과 순서를 모델링함으로써 성능 개선

Three simple ideas for asynchronous dynamic programming:

- In-placedynamic programming
- Prioritised sweeping
- Real-time dynamic programming

#### In-Place Dynamic Programming

- Synchronous value iteration stores two copies of value function

for all $\mathsf{s}$ in $\mathcal{S}$

$$\mathsf{v_{new}(s)\leftarrow\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_s+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v_{old}(s')\right)}$$

$$ \mathsf{ v_{old} \leftarrow v_{new} }$$

- In-place value iteration only stores one copy of value function

for all $\mathsf{s}$ in $\mathcal{S}$

$$\mathsf{v(s)\leftarrow\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_s+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v(s')\right)}$$

Value function을 즉각 교체

#### Prioritised Sweeping

- Use magnitude of Bellman error to guide state selection, e.g.

$$ \mathsf{\left\vert \underset{a\in\mathcal{A}}{max}\; \left( \mathcal{R}^a_s+\gamma \displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v(s') \right) - v(s) \right\vert }$$

- Backup the state with the largest remaining Bellman error
- Update Bellman error of affected states after each backup
- Requires knowledge of reverse dynamics (predecessor states)
- Can be implemented efficiently by maintaining a priority queue

Bellman error가 큰 state부터 backup

#### Real-Time dynamic Programming

- Idea: only states that are relevant to agent
- Use agent's experience to guide the selection of states
- After each time-step $\mathsf{S_t, A_t, R_{t+1}}$
- Backup the stae $\mathsf{S_t}$

$$\mathsf{v(S_t)\leftarrow\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_{S_t}+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{S_t s'}v(s')\right)}$$

Agent가 사용한 state들을 backup

### Full-Width Backups

- DP uses full-width backups
- For each backup (sync or async)
  - Every succesor state and action is considered
  - Using knowledge of the MDP transitions and reward function
- DP is effective for medium-sized probelms (millons of states)
- For large problems DP suffers Bellman's curse of dimensionality
  - Number of states $\mathsf{n= \vert \mathcal{S} \vert}$ grows exponentially with number of state variables
- Even one backup can be too expensive

모든 state의 전이를 고려해서 backup을 하는 것은 비용이 너무 크다.

### Sample Backups

- Using sample rewards and sample transitions $\mathsf{ \langle S, A, R, S' \rangle}$
- Instead of reward function $\mathcal{R}$ and transition dynamics $\mathcal{P}$
- Advantages:
  - Model-free: no advance knowledge of MDP required
  - Break the curse of dimensionality through sampling
  - Cost of backup is constant, independent of $\mathsf{n} = \vert \mathcal{S} \vert$

모델을 몰라도 sample만 이용해서 backup. 단점은?

여기부터는 강의에서는 다루지 않음

### Approximate Dynamic Programming

- Approximate the value function
- Using a function approximator $\mathsf{\hat{v}(s,\boldsymbol{w})}$
- Apply dynamic programming to $\mathsf{\hat{v}(\cdot ,\boldsymbol{w})}$
- e.g. Fitted VAlue Iteration repeats at each iteration $\mathsf{k}$,
  - Sample states $\mathcal{\tilde{S} \subseteq S}$
  - For each state $\mathcal{s \in \tilde{S}}$, estimate target value using Bellman optimality equation,
$$\mathsf{\tilde{v}_k(s)=\underset{a\in\mathcal{A}}{max}\;\left(\mathcal{R}^a_s+\gamma\displaystyle\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}\hat{v}(s', \boldsymbol{w_k})\right)}$$
  - Train next value function $\mathsf{\hat{v}(s', \boldsymbol{w_{k+1}})}$ using targets $\mathsf{ \lbrace \langle s,\tilde{v}_k(s)\rangle\\rbrace}$

function approximator가 뭔지는 써줘야 하는거 아니냐

## Contraction Mapping

contraction mapping theorem resolve convergence of value iteration $v_*$, policy evaluation $v_\pi$, policy iteration, uniqueness of solution, convergence speed.

수렴성과 uniqueness를 보여주겠다 함

### Value Function Space

- Consider the vector space $\mathcal{V} $ over value functions, $\mathcal{\vert S \vert}$ dimensions
- Each point in this space fully specifies a value function $\mathsf{v(s)}$
- Bellman backup brings value function closer
- therefore backup must converge on a unique solution

Unique optimal solution을 포함하는 value function space

### Value Function $\infty$-Norm

- distance between state-value functions $\mathsf{u}$ and $\mathsf{v}$ by $\infty$-norm is largest difference

$$ \mathsf{\Vert u-v\Vert_\infty = \underset{s\in\mathcal{S}}{max}\;\vert u(s) -v(s)\vert}$$

Error가 제일 큰게 $\infty$-norm

### Bellman Expectation Backup is a Contraction

- Define the Bellman expectation backup operator $\mathsf{T^\pi}$,

$$ \mathsf{T^\pi(v) = \mathcal{R}^\pi + \gamma \mathcal{P}^\pi v} $$

- This operator is a $\gamma$-contraction, i/e/ it makes value functions closer by at least $\gamma$,

$$\begin{aligned}
\mathsf{\Vert T^\pi(u) - T^\pi (v) \Vert_\infty} &= \mathsf{\Vert (\mathcal{R}^\pi + \gamma \mathcal{P}^\pi u) - (\mathcal{R}^\pi + \gamma\mathcal{P}^\pi v)\Vert_\infty} \\
&= \mathsf{\Vert\gamma\mathcal{P}^\pi (u-v) \Vert_\infty} \\
&\leq \mathsf{\Vert\gamma\mathcal{P}^\pi \Vert u-v \Vert_\infty \Vert_\infty} \\
&\leq \mathsf{\gamma \Vert u-v \Vert_\infty} \\
\end{aligned}$$

backup을 통해 항상 최소한 최대 error의 $\gamma$배 만큼 가까워진다. 근데 식은 u의 value와 v의 value를 backup했을 때 value 차이가 가장 큰 것이 backup 하기 전 u와 v의 value 차이가 가장 큰 곳의 $\gamma$배 보다 작거나 같다는 뜻. $\mathcal{R}$이 소거된 걸 보면 u와 v는 action reward가 같으니 동일 state 및 동일 action이고 actino이 바뀌지 않는다는 건 policy evaluation에 한하는 증명이다. u가 optimal value라고 가정해보면 v가 점점 optimal에 가까워진다는 뜻이기는 하다. 하지만 표기를 보면 그것만 의도한 것은 아닌 것 같다. 그러나 만약 서로 다른 state라면 R이 소거되지 않으니 성립하지 않는다. 거기다가 optimal할 때와 action이 같다고 가정해야하니 그것마저 버리면 evaluation 외에는 정말 무의미한 일이 되어버린다. 아이고.... 여기도 강의 다 보고나서 다시 한번 볼까... 아니면 이게 나왔다는 리처드 서튼 Reinforcement learning an introduction 책이라도 찾아봐야 하나

### Contraction Mapping Theorem

|Theorem (Contraction Mapping Theorem)|
|---|
|For any metric space $\mathcal{V}$ that is complete (i.e. closed) under an operator $\mathsf{T(v)}$, where T is a $\gamma$-contraction,<br>$\quad\bullet\;\;\mathsf{T}$ converges to a unique fixed point<br>$\quad\bullet\;\;$ At a linear convergence rate of $\gamma$|

### Convergence of Iter. Policy Evaluation and Policy Iteration

- The Bellman expectation operator $\mathsf{T^\pi}$ has a unique fixed point
- $\mathsf{v_\pi}$ is a fixed point of $\mathsf{T^\pi}$ (by Bellman expectation equation)
- By contraction mapping theorem
- Iterative policy evaluation converges on $\mathsf{v_\pi}$
- Policy iteration converges on $\mathsf{v_*}$

### Bellman Optimality Backup is a Contraction

- Define the Bellman optimality backup operator $\mathsf{T^*}$,

$$ \mathsf{T^* (v) = \underset{a\in\mathcal{A}}{max}\;\mathcal{R}^a + \gamma \mathcal{P}^a v}$$

- This operator is a $\gamma$-contraction, i.e. it makes value functions closer by at least $\gamma$ (similar to previous proof)

$$\mathsf{\Vert T^*(u)-T^* (v) \Vert_\infty \leq \gamma \Vert u-v \Vert_\infty }$$

### Convergence of Value Iteration

- The Bellman optimality operator $\mathsf{T^*}$ has a unique fixed point
- $\mathsf{v_*}$ is a fixed point of $\mathsf{T^*}$ (by Bellman expectation equation)
- By contraction mapping theorem
- Value iteration converges on $\mathsf{v_*}$
