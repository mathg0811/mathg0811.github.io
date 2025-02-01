---
title: RLcourse note - Lecture 5 Model-Control
author: DS Jung
date: 2021-12-16 06:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec5_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/0g4j2k_Ggc4)
{: .text-center}

## Introduction of Model-Free Control

Optimise the Value function of an unknown MDP<br>
Model-free control can solve Some MDP problems which modelled:

- MDP model is unknown, but experience can be sampled
- MDP model is known, but is too big to use, except by samples

### 5강 소감

Q-Learning으로 간단하지만 깔끔한 update 방법으로 정리되는 과정을 하나하나 짚어간다. 이해하고 보니 별거 아니긴 하네. 결국 여기까지 발전하기까지 누군가가 Idea를 냈다는 것 뿐. Q-learning이 유명하고 많이 보이는데 Return이 action에서 오는 경우가 많은가 보다. 그래서 action value를 사용하고 update를 조금 더 효율적으로 하면서 off-policy를 사용하도록 Algorithm이 정리된 것이다. 근데 Off policy는 이전에 학습된 episode에서도 배울수 있다는 게 내가 보기엔 중요한 장점인 것 같은데, 결국 여기까지에선 같은 state와 action을 공유하는 상황에서 확률이 다른 것들을 서로 다른 policy로 본다는 뜻이고 한두번만 개선되어도 다른 policy로 정의한다는 뜻이라서 내 기대와는 조금 다르다. 물론 설계만 잘 된다면 model 자체가 조금 달라지더라도 어느정도 공유하는 부분에 대해서는 off policy를 활용할 방법을 찾을 수도 있을 것이다. 또 다르게 학습된 policy를 사용한다면 여기서 말하는 Q-Learning과는 다른 부분이 되겠지만 이렇게 되면 target policy는 개선되지 않고 고정이 되는 만큼 다른 문제가 발생할 수 있다. 이 부분이 오히려 개선을 방해할 수 있는 만큼 policy 또한 exploration이 가능하도록 target policy를 $\epsilon$ 방법을 활용하는 방법도 체계화할 수도 있을 것이다.
 Q-Learning에서 Off Policy 중 다른 학습 결과를 활용하는 방법이 나오면 좋겠다.

## On-Policy Monte-Carlo Control

- On-policy Learning
  - Learn on the job
  - Learn about policy $\pi$ from experience sampled from $\pi$
- Off-policy Learning
  - Look over someone's shoulder
  - Learn about poilcy $\pi$ from experience sampled from $\mu$

### Generalised Policy Iteration

#### Generalised Policy Iteration With Monte-Carlo Evaluation

Policy evaluation - Monte-Carlo policy evaluation<br>
Policy improvement - Greedy policy improvement?

#### Model-Free Policy Iteration Using Action-Value Function

- Greedy policy improvement over $\mathsf{V(s)}$ requires model of MDP

$$ \mathsf{\pi'(s) = \underset{a \in \mathcal{A}}{argmax}\;\mathcal{R}^a_s + \mathcal{P}^a_{ss'}V(s') } $$

- Greedy policy improvement over $\mathsf{Q(s,a)}$ is model-free

$$ \mathsf{\pi'(s) = \underset{a \in \mathcal{A}}{argmax}\;Q(s,a) } $$

이 부분이 아직 이해가 안된거같기도 하다. 왜 state value로 업데이트 하는건 MDP 모델이 필요하고 Action value로 업데이트하는건 model-free 인가... 왜 Q는 알수 있는데 P V는 알 수 없는가...

### Exporation

#### $\epsilon$-Greedy Exploration

- Simplest idea for ensuring continual exploration
- All m actions are tried with non-zero probability
- With probability $1-\epsilon$ choose the greedy action
- With probability $\epsilon$ choose an action at random

$$ \mathsf{\pi(a\vert s)=} \begin{cases}
\epsilon/\mathsf{m} +1 - \epsilon & \mathsf{if\;a^*=\underset{a\in \mathcal{A}}{argmax}\;Q(s,a)} \\
\epsilon/\mathsf{m} & \mathsf{otherwise}
\end{cases} $$

또 또 식 이상하게 쓴다... policy 결과가 왜 숫자가 나와..... 이번엔 $\pi$ output이 action이 아니라 action을 선택할 확률이다 이거지.... 좀 문자 다른거 써라..

#### $\epsilon$-Greedy Policy Improvement

|Theorem|
|---|
|For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $\mathsf{q_\pi}$ is an improvement, $\mathsf{v_{\pi'}(s)\geq v_\pi(s)}$|

$$\begin{aligned}
\mathsf{q_\pi (s,\pi'(s))} &= \mathsf{\displaystyle\sum_{a\in\mathcal{A}}\pi'(a\vert s)q_\pi(s,a)}\\
&=\mathsf{\epsilon/m \displaystyle\sum_{a\in\mathcal{A}}q_\pi (s,a)+(1-\epsilon)\underset{a\in\mathcal{A}}{max}\;q_\pi(s,a)}\\
&\leq \mathsf{ \epsilon/m \displaystyle\sum_{a\in\mathcal{A}}q_\pi (s,a)+(1-\epsilon) \displaystyle\sum_{a\in\mathcal{A}} \frac{\pi(a\vert s)-\epsilon/m}{1-\epsilon}q_\pi(s,a) } \\
&=\mathsf{\displaystyle\sum_{a\in\mathcal{A}}\pi(a\vert s)q_\pi(s,a)=v_\pi(s)}
\end{aligned}$$

Therefor from policy improvement theorem, $\mathsf{v_{\pi'}(s)\geq v_\pi(s)}$

사실 greedy만 해도 improvement가 보장되었었는데 일부만 greedy 나머지는 random 할때도 보장되는 건 당연한거긴 하다. 굳이 이런건 또 증명해주네

#### Monte-Carlo Policy Iteration

Policy evaluation - Monte-Carlo policy evaluation $\mathsf{Q=q_\pi}$<br>
Policy improvement $\epsilon$-greedy policy improvement

#### Monte-Carlo Policy Control

Every episode:<br>
Policy evaluation - Monte-Carlo policy evaluation $\mathsf{Q\approx q_\pi}$<br>
Policy improvement $\epsilon$-greedy policy improvement

Monte-Carlo evaluation 할때 iteration 하지 않고 every episode 마다 improvement 진행

### GLIE

|Definition|
|---|
|Greedy in the Limit with Infinite Exploration(GLIE)<br>$\quad \bullet\;$ All state-action pairs are explored infinitely many times,<br><center>$$ \mathsf{ \underset{k\rightarrow\infty}{lim}\;\, N_k(s,a) = \infty } $$</center><br>$\quad \bullet\;$ The policy converges on a greedy policy,<br><center>$$ \mathsf{ \underset{k\rightarrow\infty}{lim}\;\, \pi_k(a\vert a) = 1(a=\underset{a'\in\mathcal{A}}{argmax}\;Q_k(s,a')) } $$</center>|

- For example, $\epsilon$-greedy is GLIE if $\epsilon$ reduces to zero at $\mathsf{\epsilon_k = \frac{1}{k}}$

#### GLIE Monte-Carlo Control

- Sample *k*th episode using $\pi:\mathsf{ \lbrace S_1,A_1,R_2,\dots,S_T \rbrace \sim \pi}$
- For each state $\mathsf{S_t}$ and action $\mathsf{A_t}$ in the episode,

$$\begin{aligned}
&\mathsf{ N(S_t,A_t)\leftarrow N(S_t,A_t)+1 }\\
&\mathsf{ Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\frac{1}{N(S_t,A_t)}(G_t - Q(S_t,A_t)) }
\end{aligned}$$

- Improve policy based on new action-value function

$$\begin{aligned}
\epsilon &\leftarrow \mathsf{ 1/k }\\
\pi &\leftarrow \mathsf{ \epsilon-greedy(Q) }
\end{aligned}$$

$\epsilon$ 값이 episode 마다 점점 작아짐 - exploration 감소

|Theorem|
|---|
|GLIE Monte-Carlo control converges to the optimal action-value functino, $\mathsf{ Q(s,a)\rightarrow q_*(s,a) }$

## On-Policy Temporal-Difference Learning

### MC vs. TD Control

- Temporal-difference (TD) learning has several advantages over Monte-Carlo (MC)
  - Lower variance
  - Online
  - Incomplete sequence
- Natural idea: use TD instead of MC in our control loop
  - Apply TD to $\mathsf{Q(S,A)}$
  - Use $\epsilon$-greedy policy improvement
  - Update every time-step

### Sarsa($\lambda$)

#### Updating Action-Value Functions with Sarsa

$$ \mathsf{ Q(S,A) \leftarrow Q(S,A) + \alpha (R+ \gamma Q(S',A') - Q(S,A)) } $$

#### On-Policy Control With Sarsa

Every time-step<br>
Policy evaluation Sarsa, $\mathsf{Q \approx q_\pi}$<br>
Policy improvement $\epsilon$-greedy policy improvement

#### Sarsa Algorithm for On-Policy Control

|Initialize $\mathsf{Q(s,a), \forall s \in S, a \in A(s)}$, arbitrarily,and $\mathsf{Q}(terminal state, \cdot)=0$<br>Repeat (for each episode):<br>$\quad$Initialize S<br>$\quad$Choose A from S using policy derived from Q (e.g., $\epsilon$-greedy)<br>$\quad$Repeat (for each step of episode)<br>$\quad\quad$Take action A, observe R, S'<br>$\quad\quad$Choose A' from S' using policy derived from Q (e.g., $\epsilon$-greedy)<br>$$\quad\quad\mathsf{Q(S,A)\leftarrow Q(S,A) + \alpha[R+\gamma Q(S',Q') - Q(S,A)] }$$<br>$$\quad\quad\mathsf{ S\leftarrow S'; A \leftarrow A';} $$<br>$\quad$until S is terminal|

다음 state의 action 까지 포함하는 TD의 action 확장 버전

#### Convergence of Sarsa

|Theorem|
|---|
| Sarsa converges to the optimal action-value function,<br>$\mathsf{Q(s,a)\rightarrow q_\pi(s,a)}$, under the following conditions:<br>$\quad \bullet\;$ GLOE sequence of policies $\mathsf{ \pi_t(a\vert s) }$<br>$\quad \bullet\;$ Robbins-Monro sequence of step-sizes $\alpha_t$<br><center>$$\begin{aligned} &\mathsf{ \displaystyle\sum^\infty_{t=1} \alpha_t = \infty } \\ &\mathsf{ \displaystyle\sum^\infty_{t=1} \alpha^2_t < \infty } \end{aligned} $$</center>|

#### n-Step Sarsa

- Consider the following n-step returns for $n = 1, 2, \infty$:

$$\begin{aligned}
\mathsf{n=1\;\; (Sarsa)\;\;} & \mathsf{q^{(1)}_t = R_{t+1} + \gamma Q(S_{t+1})} \\
\mathsf{n=2\quad \quad\quad\quad\,} & \mathsf{q^{(2)}_t = R_{t+1} + \gamma R_{t+2} + \gamma ^2 Q(S_{t+2})} \\
\vdots \quad\quad\quad\quad\quad\; & \;\, \vdots \\
\mathsf{n=\infty \;\, (MC)\quad} & \mathsf{q^{(\infty)}_t = R_{t+1} + \gamma R_{t+2} +\dots + \gamma ^{T-1} R_T}
\end{aligned}$$

- Define the n-step return

$$ \mathsf{q^{(n)}_t = R_{t+1} + \gamma R_{t+2} + \dots + \gamma ^{n-1} R_{t+n} + \gamma ^n Q(S_{t+n})} $$

- n-step Sarsa updates $\mathsf{Q(s,a)}$ towards the n-step Q-return

$$ \mathsf{ Q(S_t, A_t) \leftarrow Q(S_t, A_t) +\alpha \left(q^{(n)}_t - Q(S_t,A_t)\right) }$$

#### Forward View Sarsa($\lambda$)

- The $q^\lambda$ return combines all n-step Q-reutrns $\mathsf{q^{(n)}_t}$
- Using weight $\mathsf{(1-\lambda)\lambda ^{n-1}}$

$$ \mathsf{q^\lambda_t = (1-\lambda) \displaystyle\sum^\infty_{n=1} \lambda^{n-1} q^{(n)}_t} $$

- Forward-view Sarsa($\lambda$)

$$ \mathsf{ Q(S_t,A_t) \leftarrow Q(S_t,A_t) +\alpha \left(q^\lambda_t - Q(S_t,A_t)\right)}$$

#### Backward View Sarsa($\lambda$)

- Just like TD($\lambda$), we use eligibility traces in an online algorithm
- But Sarsa($\lambda$) has one eligibility trace for each state-action pair

$$\begin{aligned}
&\mathsf{E_0(s,a) =0} \\
&\mathsf{E_t(s,a) =\gamma \lambda E_{t-1}(s,a) + 1(S_t=s, A_t=a)}
\end{aligned}$$

이거 $\gamma$는 알겠는데 $\lambda$는 왜 곱해진거지. 그리고 식 제발.... 나야 이해하겠지만 진짜 쉽게 갈걸 어렵게가게 만드네, 수식을 개발자마인드로 쓴건가

- $\mathsf{Q(s,a)}$ is updated for every state s and action a
- In proportion to TD-error $\delta_t$ and eligibility trace $\mathsf{E_t(s,a)}$

$$\begin{aligned}
&\delta_t = \mathsf{R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))}\\
&\mathsf{Q(s,a) \leftarrow Q(s,a) + \alpha\delta _t E_t(s,a)}
\end{aligned}$$

#### Sarsa($\lambda$) Algorithm

|Initialize $\mathsf{Q(s,a)}$ aribitrarily, for all $s \in S, a \in A(s)$<br>Repeat (for each episode):<br>$\quad$E(s,a) = 0, for all $s \in S, a \in A(s)$<br>$\quad$Initialize S, A<br>$\quad$Repeat (for each step of episode)<br>$\quad\quad$Take action A, observe R, S'<br>$\quad\quad$Choose A' from S' using policy derived from Q (e.g., $\epsilon$-greedy)<br>$$\quad\quad \delta \leftarrow R+ \gamma Q(S', A') - Q(S,A)$$<br> $$\quad\quad E(S,A) \leftarrow E(S,A) + 1 $$<br> $ \quad\quad$ For all $s \in S, a \in A(s)$:<br>$$\quad\quad\quad Q(s,a)\leftarrow Q(s,a) + \alpha\delta E(s,a)$$<br>$$\quad\quad\quad E(s,a) \leftarrow \gamma\lambda E(s,a)$$<br>$$\quad\quad S\leftarrow S'; A \leftarrow A'; $$<br>$\quad$until S is terminal|

$\delta$는 S,A 에 대한 error 값인데 이 값을 episode 내에서 지나온 모든 value를 업데이트하는데 사용한다. Eligibility factor에 의해 감소되어 멀수록 점점 영향은 적어지고 최근에 영향을 미친 곳일수록 크게 작용하기는 한다. 이 업데이트는 때로는 방해가 되기도 하고 잘못된 방향을 강화하는 것도 가능하다. 그러나 충분한 양을 한다고 했을 때 결국 최적 값으로 수렴하긴 할것이다. 그러나 Exploration 비율과 개성되는 값에 따라서는 어딘가에 물려서 개선되지 못하는 것도 가능할 것 같다.

## Off-Policy Learning

- Evaluate target policy $\mathsf{ \pi(a \vert s)}$ to compute $\mathsf{v_\pi(s)}$ or $\mathsf{q_\pi(s,a)}$
- While following behaviour policy $\mathsf{\mu(a\vert s)}$

$$ \mathsf{ \lbrace S_1, A_1, R_2, \dots, S_T \rbrace \sim \mu } $$

Why is this important

- Learn from observing humans or other agents
- Re-use experience generated from old policies $\pi_1, \pi_2, \dots, \pi_{t-1}$
- Learn about optimal policy while following exploratory policy
- Learn about multiple policies while following one policy
- Policy가 다르면 Value가 어느정도 다르게 계산될 텐데 그래도 그 결과를 사용할수 있다라.. 매우 중요하고 도움이 될 내용이긴 하다

### Importance Sampling

- Estimate the expectation of a different distribution

$$\begin{aligned}
\mathsf{ \mathbb{E}_{X \sim P}[f(X)] } &= \sum \mathsf{P(X)f(X)} \\
&= \sum \mathsf{Q(X) \frac{P(X)}{Q(X)} f(X) } \\
&= \mathsf{ \mathbb{E}_{X \sim Q} \left[ \frac{P(X)}{Q(X)}f(X)\right]}
\end{aligned}$$

plicy가 정하는 확률에 대해서 Environment에서 Action reward 는 고정이고 policy에 의한 확률만 고정이므로 이 확률 비율만 보정해주면 return을 다른 policy에 대해서도 구할 수 있다는 뜻인데 이는 기본적으로 같은 Action을 따라 갈 때만 가능하다. $\epsilon$-greedy 등으로 exploration을 포함하는 policy를 통해 충분히 많은 action 가짓 수를 확보하고 모든 episode 데이터가 있을 때에만 return을 계산할 수 있으므로 deterministic policy의 data는 학습이 어렵다고 볼 수 있다.

#### Importance Sampling for Off-Policy Monte-Carlo

- Use returns generated from $\mu$ to evaluate $\pi$
- Weight return $\mathsf{G_t}$ according to similarity between policies
- Multiply importance sampling corrections along whole episode

$$ \mathsf{ G^{\pi/\mu}_t = \frac{\pi(A_t \vert S_t)}{\mu(A_t \vert S_t)} \frac{\pi(A_{t+1} \vert S_{t+1})}{\mu(A_{t+1} \vert S_{t+1})} \dots \frac{\pi(A_T \vert S_T)}{\mu(A_T \vert S_T)} G_t } $$

- Update value towards corrected return

$$ \mathsf{ V(S_t) \leftarrow V(S_T) + \alpha\left(G^{\pi/\mu}_t - V(S_t)\right) } $$

- Cannot use if $\mu$ is zero when $\pi$ is non-zero
- Importance sampling can dramatically increase variance

#### Importance Sampling for Off-Pollicy TD

- Use TD targets generated from $\mu$ to evaluate $\pi$
- Weight TD target $\mathsf{R+\gamma V(S')}$ by importance sampling
- Only need a single importance sampling correction

$$ \mathsf{ V(S_t) \leftarrow V(S_t) + \alpha \left( \frac{\pi(A_t\vert S_t)}{\mu(A_t\vert S_t)} (R_{t+1} + \gamma V(S_{t+1})) - V(S_t)\right) } $$

- Much lower variance than Monte-Carlo importance sampling
- Policies only need to be similar over a single step

즉 $\mu$ 가 0이거나 너무 작으면 사용하기 힘들다고 하는데 이건 내용을 잘못 파악한것 같다. 물론 episode의 다양성이 커질수록 연산이 힘들어지는 문제가 있다고 볼 수도 있지만 조금만 더 접근하면 충분히 가능할것 같은데. TD를 통해서 solution을 구하는 방법과도 엄청나게 다르지 않을것같은데. 제약조건만 만족하면 상상속에서나 가능한 건 아닐듯

### Q-Learning

- We now consider off-policy learning of action-values $\mathsf{Q(s,a)}$
- No importance sampling is required
- Next action ischosen using begaviour policy $\mathsf{ A_{t+1} \sim \mu(\cdot \vert S_t) }$
- But we consider alternative successor action $\mathsf{ A' \sim \pi(\cdot \vert S_t) }$
- And update $\mathsf{Q(S_t,A_t)}$ towards value of alternative action

$$ \mathsf{ Q(S_t,A_t)\leftarrow Q(S_t,A_t) + \alpha \left(R_{t+1} + \gamma Q(S_{t+1},A') - Q(S_t,A_t)\right) } $$

Error를 다른 action 결과값에서 가져와서 update 해도 된다는건데 그럼 결국 그 다른 policy에만 가까워지는거 아닌가 학습하는 policy가 아니라는 말은 그 결과를 가져오는 target policy는 개선이 안된다는거 같은데 그게 optimal하다고 가정하는건가 그럼 의미가 없는데

#### Off-Policy Control with Q-Learning

- We now allow both behaviour and target policies to improve
- The target policy $\pi$ is greedy w.r.t. $\mathsf{Q(s,a)}$

$$ \mathsf{ \pi (S_{t+1}) = \underset{a'}{argmax}\;Q(S_{t+1}, a') } $$

- The behaviour policy $\mu$ is e.g. $\epsilon$-greedy w.r.t. $\mathsf{Q(s,a)}$
- The Q-learning target then simplifies:

$$\begin{aligned}
& \mathsf{ R_{t+1} + \gamma Q(S_{t+1}, A') } \\
=& \mathsf{ R_{t+1} + \gamma Q(S_{t+1}, \underset{a'}{argmax}\; Q(S_{t+1}, a')) } \\
=& \mathsf{ R_{t+1} + \underset{a'}{max}\;\gamma Q(S_{t+1}, a') } \\
\end{aligned}$$

target은 greedy, behaviour은 $\epsilon$-greedy라서 target policy도 결국 개선되긴 한다는 말이네 그니까 결국 exploration을 유지하되 update에 사용되는 return은 greedy로 유지해서 error term이 잘못되지 않도록 한다는 말이군. 그럼 이미 축적된 학습 Data는 Behaviour로 사용되는건가

#### Q-Learning Control Algorithm

$$ \mathsf{ Q(S,A) \leftarrow Q(S,A) + \alpha \left( R + \gamma\; \underset{a'}{max}\; Q(S',a') - Q(S,A)\right) } $$

|Theorem|
|---|
|Q-learning control converges to the optimal action-value function, $\mathsf{Q(s,a)\rightarrow q_*(s,a)}$

Q-learning을 Sarsa max라고도 부른다고 함 결국 action value를 improve하는 방법에서 behaviour는 exploration을 포함하고 update는 greedy로 한다는 뜻

#### Q-Learning Algorithm for Off-Policy Control

|Initialize $\mathsf{Q(s,a) \forall s \in S, a \in A(s)}$, arbitrarily, and $\mathsf{Q(terminal\, state,\cdot)=0}$<br>Repeat (for each episode):<br>$\quad$Initialize S<br>$\quad$Repeat (for each step of episode)<br>$\quad\quad$Choose A from S using policy derived from Q (e.g., $\epsilon$-greedy)<br>$\quad\quad$Take action A, observe R, S'<br>$$\quad\quad Q(S,A)\leftarrow Q(S,A) + \alpha [ R + \gamma\; max_a Q(S',a)-Q(S,A)]$$<br>$$\quad\quad S\leftarrow S'; $$<br>$\quad$until S is terminal|

## Summary

||Full Backup (DP)|Sample Backup (TD)|
|---|---|---|
|Bellman Expectation<br> Equation for $\mathsf{v_\pi(s)}$|Iterative Policy Evaluation<br>$$ \mathsf{ V(s) \leftarrow \mathbb{E} [R+\gamma V(S') \vert s] } $$ | TD Learning <br> $$ \mathsf{ V(S)\;\overset{\alpha}{\leftarrow}\; R+\gamma V(S')} $$|
|Bellman Expectation<br> Equation for $\mathsf{q_\pi(s,a)}$|Q-Policy Iteration<br>$$ \mathsf{ Q(s,a) \leftarrow \mathbb{E} [R+\gamma Q(S',A') \vert s,a] } $$ | Sarsa <br> $$ \mathsf{ Q(S,A)\;\overset{\alpha}{\leftarrow}\; R+\gamma Q(S',A')} $$|
| Bellman Optimality<br> Equation for $\mathsf{q_*(s,a)}$|Q-Value Iteration<br>$$ \mathsf{ Q(s,a) \leftarrow \mathbb{E} \left[ R+\gamma \;\underset{a'\in\mathcal{A}}{max}\;Q(S',a') \vert s,a \right] } $$ |Q-Learning <br> $$ \mathsf{ Q(S,A)\;\overset{\alpha}{\leftarrow}\; R+\gamma \;\underset{a'\in\mathcal{A}}{max}\; Q(S',a')} $$|

where $$ \mathsf{ x \overset{\alpha}{\leftarrow} y \equiv x \leftarrow x + \alpha(y-x) } $$
