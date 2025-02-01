---
title: RLcourse note - Lecture 9 Exploration and Exploitation
author: DS Jung
date: 2022-01-14 19:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec9_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/sGuiWX07sKw)
{: .text-center}

### 9강 소감
Exploration과 Exploitation을 밸런스를 맞추면서 학습하도록하는 몇가지 기법에 대한 내용인데 별로 중요하진 않은것같다..? 솔직히 몇가지 배웠어도 큰 성능차이가 나는 부분도 아니고 몇 가지 환경적 상황 변수를 파악하고 거기에 맞게 설계한다면 이런 문제는 훨씬 효율적이고 적합하게 해결될 수 있다. 몇개 내용설명이 미흡하여 따로 공부해볼까했지만 중요하지 않아보여서 패스

## Introduction

Exploration vs. Exploitation Dilemma

- Inline decision-making invloves a fundamental choice:
  - Exploitation : Make the best decision given current information
  - Exploration : Gather more information
- The best long-term strategy may involve short-term sacrifices
- Gather enough information to make the best overall decisions

### Principles

- Naive Exploration
  - Add noise to greedy policy (e.g. $\epsilon$-greedy)
- Optimistic Initialisation
  - Assume the best until proven otherwise
- Optimism in the Face of Uncertainty
  - Prefer actions with uncertain values
- Probability Matching
  - Select actions according to probability they are best
- Infromation State Search
  - Lookahead search incorporating value of information

## Multi-Armed Bandits

- A multi-armed bandit is tuple $\langle\mathcal{A,R}\rangle$
- $\mathcal{A}$ is a known set of m actions (or "arms")
- $\mathsf{\mathcal{R}^a(r) = \mathbb{P}[r\vert a]}$ is an unknown probability distribution over rewards
- At each step t the agent selects an action $a_t \in \mathcal{A}$
- The environment generates a reward $r_t \sim \mathcal{R}^{a_t}$
- The goal is to maximise cumulative reward $\sum^t_{\tau = 1} r_\tau$

tuple에서 P를 빼놓고 내용에서는 정작 사용하고 있는 아이러니, state transition probability와 reward probability가 다를 수는 있지만 state에서 정의될 수 있는 부분이기도 하다. reward varation을 state에서 정의하지 못할 만한 경우가 있을까?

### Regret

- The action-value is the mean reward for action a,

$$ \mathsf{ Q(a) = \mathbb{E} [r \vert a] } $$

- The optimal value $\mathsf{V}^*$ is

$$ \mathsf{ V^* = Q(a^*) = \underset{a\in\mathcal{A}}{max}\; Q(a) } $$

- The regret is the opportunity loss for one step

$$ \mathsf{ l_t = \mathbb{E} [V^* - Q(a_t)] } $$

- The total regret is the total opportunity loss

$$ \mathsf{ L_t = \mathbb{E} \left[ \displaystyle\sum^t_{\tau = 1 } V^* - Q(a_\tau) \right] } $$

- Maximise cumulative reward $\equiv$ minimise total regret

Optimal도 그냥 Q 쓰면되지 난데없이 왜 V야. mean action value는 따로 표시하면 되지 왜 똑같이 q를 쓰려고 하는가. 여기서 regret은 경우에 따라 minus인 경우도 생길 것이다. 근데 regret은 왜 Expected value인가. one step 이라면서 왜 expected야 짜증나게. Summation 해놓고 E 붙이는것도 마찬가지고

#### Counting Regret

- The count $\mathsf{N_t(a)}$ is expected number of elections for action a
- The gap $\Delta _a$ is the difference in value between action a and optimal action $\mathsf{ a^*, \Delta_a = V^* - Q(a) }$
- Regret is a function of gaps and the counts

$$\begin{aligned}
\mathsf{L_t}&= \mathsf{ \mathbb{E} \left[ \displaystyle\sum^t_{\tau=1} V^* - Q(a_\tau) \right] } \\
&= \mathsf{ \displaystyle\sum_{a\in\mathcal{A}} \mathbb{E} [N_t(a)] (V^* - Q(a)) } \\
&= \mathsf{ \displaystyle\sum_{a\in\mathcal{A}} \mathbb{E} [N_t(a)] \Delta_a }
\end{aligned}$$

- A good algorithm ensures small counts for large gaps
- Problem: gaps are not known!

이미 Summation을 했는데 $\mathbb{E}$는 왜 붙어있는건가...

#### Linear or Sublinear Regret

- If an algorithm forever explores it will have linear total regret
- If an algorithm never explores it will have linear total regret
- Is it possible to achieve sublinear total regret?

### Greedy and $\epsilon$-greedy algorithms

#### Greedy Algorithm

- We consider algorithms that estimate $\mathsf{\hat{Q}_t(a)\approx Q(a)}$
- Estimate the value of each action by Monte-Carlo evaluation

$$ \mathsf{ \hat{Q}_t(a) = \frac{1}{N_t(a)} \displaystyle\sum^T_{t=1} r_t 1(a_t=a) } $$

- The greedy algorithm selects action with highest value

$$ \mathsf{ a^*_t = \underset{a\in\mathcal{A}}{argmax}\; \hat{Q}_t(a) } $$

- Greedy can lock onto a suboptimal action forever
- $\Rightarrow$ Greedy has linear total regret

#### $\epsilon$-greedy Algorithm

- The $\epsilon$-greedy algorithm continues to explore forever
  - With probability $1-\epsilon$ select $\mathsf{a=\underset{a\in\mathcal{A}}{argmax}\;\hat{Q}(a)}$
  - With probability $\epsilon$ select a random action
- Constant $\epsilon$ ensures minimum regret

$$ \mathsf{ l_t \geq \frac{\epsilon}{\mathcal{A}} \displaystyle\sum_{a\in\mathcal{A}}\Delta_a } $$

- $\Rightarrow$ $\epsilon$-greedy has linear total regret

#### Optimistic Initialisation

- Simple and practical idea: initialise $\mathsf{Q(a)}$ to high value
- Update action value by incremental Monte-Carlo evaluation
- Starting with $\mathsf{N(a) > 0}$

$$\mathsf{ \hat{Q}_t(a_t) = \hat{Q}_{t-1} + \frac{1}{N_t(a_t)}(r_t - \hat{Q}_{t-1}) }$$

- Encourages systematic exploration early on
- But can still lock onto suboptimal action
- $\Rightarrow$ greedy + optimistic initialisation has linear total regret
- $\Rightarrow$ $\epsilon$-greedy + optimistic initialisation has linear total regret

초기 Value를 높게 설정해두고 시작함으로써 Exploration이 저절로 일어나지만 여전히 suboptimal 수렴은 가능하다. 간단하지만 꽤 괜찮은 접근법

#### Decating $\epsilon_t$-Greedy Algorithm

- Pick a decay schedule for $\epsilon_1, \epsilon_2, \dots$
- Consider the following schedule

$$\begin{aligned}
\mathsf{ c } &> \mathsf{ 0 } \\
\mathsf{ d } &= \mathsf{ \underset{a\vert\Delta_a>0}{min}\; \Delta_i } \\
\mathsf{ \epsilon_t } &= \mathsf{ min \lbrace 1, \frac{c\vert \mathcal{A}\vert}{d^2t} \rbrace }
\end{aligned}$$

- Decaying $\epsilon_t$-greedy has logarithmic asymptotic total regret!
- Unfortunately, schedule requires advance knowledge of gaps
- Goal: find an algorithm with sublinear regret for any multi-armed bandit (without knowledge of $\mathcal{R}$)

gap notation 또 엉망이다 action 끼리의 regret 차이라고 한다. 그리고 regret은 실제로 알수 없는 값이므로 이론적인 내용일뿐 아직 의미는 없다. 그냥 exploration이 decay해서 regret이 logarithmic 해지고 기회비용이 줄었을 뿐

### Lower Bound

- The performance of any algorithm is determined by similarity between optimal arm and other arms
- Hard problems have similar-looking arms with different means
- This is described formally by the gap $\Delta_a$ and the similarity in distributions $\mathsf{KL(\mathcal{R}^a\Vert\mathcal{R}^a *)}$

|Theorem (Lai and Robbins|
|---|
|Asymptotic total regret is at least logarithmic in number of steps<br><center>$$ \mathsf{ \underset{t\rightarrow\infty}{\lim}\; L_t \leq \log\, t \displaystyle\sum_{a\vert\Delta_a>0} \frac{\Delta_a}{KL(\mathcal{R}^a\Vert \mathcal{R}^{a^*})} } $$</center>|

흠 상황과 모델을 만들기에 따라 극복 가능할것 같은 느낌도 있지만 굳이. 근데 이걸 왜 하는지 나오려나?

### Upper Confidence Bound

#### Optimism in the Face of Uncertainty

![graph](/assets/pic/Note9_figure1.JPG){: width="700px" height="400px"}{: .text-center}

- Which action should we pick?
- The more uncertain we are about an action-value
- The more important it is to eplore that action
- It could turn out to be the best action

- After picking blue action
- We are less uncertain about the value
- And more likely to pick another action
- Until we home in on best action

흠 충분히 data를 수집한 뒤에도 1번 그래프가 유지된다면 그래도 1번을 고를 것인가

#### Upper Confidence Bounds

- Estimate an upper confidence $\mathsf{\hat{U}_t(a)}$ for each action value
- Such that $\mathsf{Q(a)\leq\hat{Q}_t(a)+\hat{U}_t(a)}$ with high probability
- This depends on the number of times $\mathsf{N(a)}$ has been selected
  - Small $\mathsf{N_t(a) \Rightarrow}$ large $\mathsf{\hat{U}_t(a)}$ (estimated value is uncertain)
  - Large $\mathsf{N_t(a) \Rightarrow}$ small $\mathsf{\hat{U}_t(a)}$ (estimated value is accurate)
- Select action maximising Upper Confidence Bound (UCB)

$$\mathsf{ a_t = \underset{a\in\mathcal{A}}{argmax}\; \hat{Q}_t(a) + \hat{U}_t(a) }$$

평균적인 reward 기대값이 큰 action이 아니라불확실성이 있는 action을 선택할 가능성을 높이기 위한 방법으로써 U를 사용하는 것.

#### Hoeffding's Inequality

|Theorem (Hoeffding's Inequality)|
|---|
|Let $$\mathsf{X_1,\dots ,X_t}$$ be i.i.d. random variables in [0,1], and let <br>$$\mathsf{ \bar{X}_t = \frac{1}{t}} \sum^t_{\tau=1} X_\tau }$$  be the sample mean. Then<br><center>$$\mathsf{ \mathbb{P}[\mathbb{E}[X]>\bar{X}_t+u]\leq e^{-2tu^2} }$$</center>|

- We will apply Hoeffding's Inequality to rewards of the bandit
- conditioned on selecting action a

$$\mathsf{ \mathbb{P} \left[ Q(a) > \hat{Q}_t(a) + U_t(a) \right] \leq e^{-2N_t(a)U_t(a)^2} }$$

#### Calculating Upper Confidence Bounds

- Pick a probability p that true value exceeds UCB
- Now solve for $\mathsf{U_t(a)}$

$$\begin{aligned}
\mathsf{ e^{-2N_t(a)U_t(a)^2} } &= \mathsf{ p } \\
\mathsf{ U_t(a) } &= \mathsf{ \sqrt{\frac{-\log p}{2N_t(a)}} }
\end{aligned}$$

- Reduce p as we observe more rewards, e.g. $\mathsf{p=t^{-4}}$
- Ensures we select optimal action as $\mathsf{t\rightarrow \infty}$

$$\mathsf{ U_t(a) = \sqrt{\frac{-\log p}{2N_t(a)}} }$$

#### UCB1

- This leads to the UCB1 algorithm

$$\mathsf{ a_t = \underset{a\in\mathcal{A}}{argmax}\; Q(a) + \sqrt{\frac{2\log t}{N_t(a)}} }$$

|Theorem|
|---|
|The UCB algorithm achieves logarithmic asymtotic total regret<br><center>$$\mathsf{ \displaystyle\lim_{t\rightarrow\infty} L_t \leq 8\log t \displaystyle\sum_{a\vert\Delta_a>0} \Delta_a }$$</center>|

UCB중 확률이 decay하는 $$t^{-4}$$를 사용하는 알고리즘

### Bayesisan Bandits

- So far we have made no assumptions about the reward distribution $\mathcal{R}$
  - Except bounds on rewards
- Bayesian bandits exploit prior knowledge of rewards, $\mathsf{p}[\mathcal{R}]$
- They compute posterior distribution of reward $\mathsf{p[\mathcal{R}\vert h_t]}$
  - where $\mathsf{h_t=a_1,r_1,\dots a_{t-1},r_{t-1}}$ is the history
- Use posterior to guide exploration
  - Upper confidence bounds (Bayesian UCB)
  - Probability matching (Thompson sampling)
- Better performance if prior knowledge is accurate

#### Bayesian UCB Example: Independent Gaussians

- Assume reward distribution is Gaussian, $\mathsf{\mathcal{R}_a(r) = \mathcal{N}(r;\mu_a,\sigma^2_a)}$
- Compute Gaussian posterior over $\mu_a$ and $\sigma_a^2$ (by Bayes law)

$$ \mathsf{ p[\mu_a, \sigma^2_a \vert h_t] \propto p[\mu_a,\sigma^2_a] \displaystyle\prod_{t\vert a_t=a} \mathcal{N}(r_t; \mu_a,\sigma^2_a) } $$

- Pick action that maximises standard deviation of $\mathsf{Q(a)}$

$$\mathsf{ a_t = argmax \;\mu_a + c\sigma_a/\sqrt{N(a)} }$$

#### Probability Matching

- Probability matching selects action a according to probability that a is the optimal action

$$ \mathsf{ \pi(a\vert h_t) = \mathbb{P} [Q(a) > Q(a'), \forall a' \neq a \vert h_t] } $$

- Probability matching is optimistic in the face of uncertainty
  - Uncertain actions have higher probability of being max
- Can be difficult to compute anlytically from posterior

#### Thompson Sampling

- Thompson sampling implements probability matching

$$\begin{aligned}
\mathsf{ \pi(a\vert h_t) } &= \mathsf{ \mathbb{P} [ Q(a) > Q(a'), \forall a' \neq a \vert h_t ] } \\
&= \mathsf{ \mathbb{E}_{\mathcal{R}\vert h_t} \left[ 1(a=\underset{a\in\mathcal{A}}{argmax}\; Q(a))\right] }
\end{aligned}$$

- Use Bayes law to compute posterior distribution $\mathsf{p[\mathcal{R}\vert h_t]}$
- Sample a reward distribution $\mathcal{R}$ from posterior
- Compute action-value function $\mathsf{Q(a) = \mathbb{E}[\mathcal{R}_a]}$
- Select action maximising value on ssample, $\mathsf{a_t = \underset{a\in\mathcal{A}}{argmax}\; Q(a)}$
- Thompson sampling achieves Lai and Robbins lower bound!

#### Value of Information

- Exploration is useful because it gains information
- Can we quantify the value of information?
  - How much reward a decision-maker would be prepared to pay in order to have that information, prior to making a decision
  - Long-term reward after getting information - immediate reward
- Information gain is higher in uncertain situations
- Therefor it makes sense to explore uncertain situations more
- If we know value of information, we can trade-off exploration and exploitation optimally

exploration은 거의 random하게 시행되지만 좀더 smart, logically efficient한 방법이 있지않을까, 여기서는 uncertatinty만 쫓아서 모든 tree를 검증하는 방향으로만 하고 있다.

### Information State Search

- We have viewed bandits as one-step decision-making problems
- Can also view as sequential decision-making problems
- At each step there is an information state $\mathsf{\tilde{s}}$
  - $\mathsf{\tilde{s}}$ is a statistic of the history, $\mathsf{\tilde{s_t} = f(h_t)}$
  - summarising all information accumulated so far
- Each action a causes a transition to a new information state $\mathsf{\tilde{s}'}$ (by adding information), with probability $\mathsf{\mathcal{\tilde{P}}^a_{\tilde{s},\tilde{s}'}}$
- This defines MDP $\mathcal{\tilde{M}}$ in augmented information state space

$$\mathcal{ \tilde{M} = \langle \tilde{S}, A, \tilde{P}, R, \gamma\rangle }$$

#### Example: Bernoulli Bandits

- Consider a Bernoulli bandit, such that $\mathsf{\mathcal{R}^a = \mathcal{B}(\mu_a)}$
- e.g. Win or lose a game with probability $\mu_a$
- Want to find which arm has the highest $\mu_a$
- The information state is $\mathsf{\tilde{s}} = \langle\alpha,\beta\rangle$
  - $\alpha_a$ counts the pulls of arm a where reward was 0
  - $\beta_a$ counts the pulls of arm a where reward was 1

#### Solving Information State Space Bandits

- We now have an infinite MDP over information states
- This MDP can be solved by reinforcement learning
- Model-free reinforcement learning
  - e.g. Q-learning (Duff, 1994)
- Bayesian model-based reinforcement learning
  - e.g. Gittins indices (Gittins, 1979)
  - This approach is known as Bayes-adaptive RL
  - Finds Bayes-optimal exploration/exploitation trade-off with respect to prior distribution

#### Bayes-Adaptive Bernoulli Bandits

- Start with Beta$(\alpha_a, \beta_a)$ prior over reward function $\mathcal{R}^a$
- Each time a is selected, update posterior for $$\mathcal{R}^a$$
  - Beta($\alpha_a + 1 ,\beta_a$) if $r= 0$
  - Beta($$\alpha_a,\beta_a+1$$) if $$r=1$$
- This defines transition function $$\mathcal{\tilde{P}}$$ for the Bayes-adaptive MDP
- Information state $$\langle\alpha,\beta\rangle$$ corresponds to reward model Beta($$\alpha,\beta$$)
- Each state transition corresponds to a Bayesian model update

#### Gittins Indices for Bercoulli Bandits

- Bayes-adaptive MDP can be solved by dynamic programming
- The solution is known as the Gittins index
- Exact solution to Bayes-adaptive MDP is typically intractable
  - Information state space is too large
- Recent idea: apply simulation-based search (Guez et al. 2012)
  - Forward search in information state space
  - Using simulations from current information state

## Contextual Bandits

- A contextual bandit is a tuple $$\langle\mathcal{A,S,R}\rangle$$
- $$\mathcal{A}$$ is a known set of actions (or "arms")
- $$\mathcal{S= \mathbb{P}}\mathsf{[s]}$$ is an unknown distribution over states (or "contexts")
- $$\mathsf{\mathcal{R}^a_s(r) = \mathbb{P}[r\vert s,a]}$$ is an unknown probability distribution over rewards
- At each step t
  - Environmnet generates state $$s_t\sim \mathcal{S}$$
  - Agent selects action $$\mathsf{a_t}\in\mathcal{A}$$
  - Environment generates reward $$\mathsf{r_t}\sim\mathcal{R}^{a_t}_{s_t}$$
- Goal is to maximise cumulative reward $$\mathsf{\sum^t_{\tau=1} r_\tau} $$

### Linear UCB

#### Linear Regression

- Action-value function is expected reward for state s and action a

$$\mathsf{Q(s,a) = \mathbb{E}[r\vert s,a]}$$

- Estimate value function with a linear function approximator

$$\mathsf{ Q_\theta(s,a) = \phi (s,a)^\top \theta \approx Q(s,a) }$$

- Estimate parameters by least squares regression

$$\begin{aligned}
\mathsf{ A_t } &= \mathsf{ \displaystyle\sum^t_{\tau=1} \phi(s_\tau , a_\tau) \phi(s_\tau, a_\tau)^\top } \\
\mathsf{ b_t } &= \mathsf{ \displaystyle\sum^t_{\tau=1} \phi(s_\tau,a_\tau)r_\tau } \\
\mathsf{ \theta_t } &= \mathsf{ A_t^{-1} b_t }
\end{aligned}$$

#### Linear Upper Confidence Bounds

- Least squares regression estimates the mean action-value $$\mathsf{Q_\theta(s,a)}$$
- But it can also estimate the variance of the action-value $$\mathsf{\sigma^2_\theta(s,a)}$$
- i.e. the uncertainty due to parameter estimation error
- Add on a bonus for uncertainty, $$\mathsf{U_\theta(s,a) = c\sigma}$$
- i.e. define UCB to be c standard deviations above the mean

#### Geometric Interpretation

- Define confidence ellipsoid $$\mathcal{E}_t$$ around parameters $$\theta_t$$
- Such that $$\mathcal{E}_t$$ includes true parameters $$\theta^*$$ with high probability
- Use this ellipsoid to estimate the uncertainty of action values
- Pick parameters within ellipsoid that maximise action value

$$\mathsf{\underset{\theta\in\mathcal{E}}{argmax}\; Q_\theta(s,a)}$$

#### Calculating Linear Upper Confdence Bounds

- For least quares regression, parameter covariance is $$\mathsf{A^{-1}}$$
- Action-value is linear in features, $$\mathsf{Q_\theta(s,a) = \phi (s,a)^\top \theta}$$
- So action-value variance is quadratic, $$\mathsf{\sigma^2_\theta(s,a) = \phi(s,a)^\top A^{-1} \phi(s,a)}$$
- Upper confidence bound is $$\mathsf{Q_\theta(s_t,a) + c\sqrt{\phi(s,a)^\top A^{-1} \phi(s,a)}}$$
- Select action maximising upper confidence bound

$$ \mathsf{ a_t = \underset{a\in\mathcal{A}}{argmax}\; Q_\theta(s_t,a)+ c\sqrt{\phi(s_t,a)^\top A^{-1}_t \phi(s_t,a)}} $$

## MDPs

Exploration/Exploitation Principles to MDPS

The same principles for exploration/exploitation apply to MDPs

- Naive Exploration
- Optimistic Initialisation
- Optimism in the Face of Uncertainty
- Probability Matching
- Information State Search

### Optimistic Initialisations

#### Optimistic Initialisation: Model-Free RL

- Initialise action-value function Q(s,a) to $$\frac{r_{max}}{1-\gamma}$$
- Run favourite model-free RL algorithm
  - Monte-Carlo control
  - Sarsa
  - Q-learning
  - ...
- Encourages systematic exploration of states and actions

#### Optimistic Initialisation: Model-Based RL

- Construct an optimistic model of the MDP
- Initialise transitions to go to heaven
  - (i.e. transition to terminal state with $$ r_{max} $$ reward)
- Solve optimistic MDP by favourite planning algorithm
  - policy iteration
  - value iteration
  - tree search
  - ...
- Encourages systematic exploration of states and actions
- e.g. RMax algorithm (Brafman and Tennenholtz)

### Optimism in the Face of Uncertainty

#### Upper Confidence Bounds: Model-Free RL

- Maximise UCB on action-value function $$ \mathsf{Q^\pi (s,a)}$$
  <br><center>$$\mathsf{ a_t = \underset{a\in\mathcal{A}}{argmax}\; Q(s_t,a) + U(s_t,a) }$$</center>
  - Estimate uncertainty in policy evaluation (easy)
  - Ignores uncertainty from policy improvement
- Maximise UCB on optimal action-value function $$\mathsf{Q^*(s,a)}$$
  <br><center>$$\mathsf{ a_t = \underset{a\in\mathcal{A}}{argmax}\; Q(s_t,a) + U_1(s_t,a) + U_2(s_t,a) }$$</center>
  - Estimate uncertainty in policy evaluation (easy)
  - plus uncertainty from policy improvement (hard)

#### Bayesian Model-Based RL

- Maintain posterior distribution over MDP models
- Estimate both transitions and rewards, $$\mathsf{p[\mathcal{P,R}\vert h_t]}$$
  - where $$\mathsf{h_t = s_1,a_1,r_2,\dots ,s_t}$$ is the history
- Use posterior to guide exploration
  - Upper confidence bounds (Bayesian UCB)
  - Probability matching (Thompson sampling)

### Probability Matching

#### Thompson Sampling: Model-Based RL

- Thompson sampling implements probability matching

$$\begin{aligned}
\mathsf{ \pi(s,a,\vert h_t)} &= \mathsf{ \mathbb{P} [Q^*(s,a) > Q^*(s,a'), \forall a' \neq a \vert h_t] } \\
 &= \mathsf{ \mathbb{E}_{\mathcal{P,R}\vert h_t} \left[ 1(a=\underset{a\in\mathcal{A}}{argmax}\; Q^*(s,a))\right] }
\end{aligned}$$

- Use Bayes law to compute posterior distribution $$\mathsf{p[\mathcal{P,R}\vert h_t]}$$
- Sample an MDP $$\mathcal{P,R}$$ from posterior
- Solve MDP using favourite planning algorithm to get $$\mathsf{Q^*(s,a)}$$
- Select optimal action for sample MDP, $$\mathsf{a_t = \underset{a\in\mathcal{A}}{argmax}\; Q^*(s_t,a)}$$

### Information State Search

#### Information State Search in MDPs

- MDPs can be augmented to include infromation state
- Now the augmented state is $$\langle s,\tilde{s}\rangle$$
  - where s is original state within MDP
  - and $$\tilde{s}$$ is a statistic of the history (accumulated information)
- Each action a causes a transition
  - to a new state s' with probability $$\mathcal{P}^a_{s,s'}$$
  - to a new information state $$\tilde{s}'$$
- Defines MDP $$\mathcal{\tilde{M}}$$ in augmented information state space

$$\mathcal{\tilde{M} = \langle \tilde{S},A,\tilde{P},R,\gamma\rangle}$$

#### Bayes Adaptive MDPs

- Posterior distribution over MDP model is an information state

$$\mathsf{\tilde{s}_t = \mathbb{P}[\mathcal{P,R}\vert h_t]}$$

- Augmented MDP over $$\mathsf{\langle s, \tilde{s}\rangle}$$ is called Bayes-adaptive MDP
- Solve this MDP to find optimal exploration/exploitation trade-off (with respect to prior)
- However, Bayes-adaptive MDP is typically enormous
- Simulation-based search has proven effective (Guez et al.)

#### Conclusion

- Have covered several principles for exploration/exploitation
  - Naive methods such as $\epsilon$-greedy
  - Optimistic initialisation
  - Upper confidence bounds
  - Probability matching
  - Information state search
- Each principle was developed in bandit setting
- But same principles also apply to MDP setting
