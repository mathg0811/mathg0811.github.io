---
title: RLcourse note - Lecture 7 Policy Gradient
author: DS Jung
date: 2021-12-24 01:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec7_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/KHZVXao4qXs)
{: .text-center}

## Introduction of Policy Gradient

### 7강 소감

마지막으로 제일 어려운 부분이라고 하는데 사실 진짜 어렵다기보다는 논리적으로 중요한 연결점들을 다 생략해버리고 이게 결국 이렇게 된다라는 결과적인 내용만 정리한 데다가 notation 은 이상하게 작성되고 시도때도없이 변하니 이해하기 어려운 게 당연하다. 그래도 컨셉은 거의 다 이해했지만 베이스가 되는 논리가 많이 빠져있으니 제대로 기억해낼지는 모르겠다. 나머지 강의에서도 크게 보충이 안될것 같고 결국엔 무언가 해보던지 좀더 심화 강의를 들어보던지 책을 독학하던지 해야 한다. 물론 이정도 지식이면 실제 활용에도 문제 없을 정도는 될것같지만 그래도 찝찝한게 싫은게 나니까 어쩔수없다.

### Policy-Based Reinforcement Learning

- In the last lecture we approximated the value or action-value function using parameters $\theta$
- 이번엔 $\theta$ notation이 변수로 들어가지 않고 underbar로 들어갔다. 의미는 똑같다

$$ \begin{aligned}
\mathsf{V_\theta (s)} &\approx \mathsf{V^\pi (s)} \\
\mathsf{Q_\theta (s,a)} &\approx \mathsf{Q^\pi (s,a)}
\end{aligned} $$

- A policy was generated directly from the value function
  - e.g. using $\epsilon$-greedy
- In this lecture we will directly parametrise the policy

$$ \mathsf{ \pi_\theta (s,a) = \mathbb{P} [a\vert s, \theta] } $$

- We will focus again on model-free reinforcement learning

### Value-Based and Policy-Based RL

- Value Based
  - Learnt Value Function
  - Implicit policy (e.g. $\epsilon$-greedy)
- Policy Based
  - No Value Function
  - Learnt Policy
- Actor-Critic
  - Learnt Value Function
  - Learnt Policy

### Advantages of Policy-Based RL

Advantages:

- Better convergence properties
- Effective in high-dimensional or continuous action spaces
- Can learn stochastic policies
- 왜 Value based는 Stochastic이 안된다고 하는걸까 걍 하면되지

Disadvantages:

- Typically converge to a local rather than global optimum
- Evaluating a policy is typically inefficient and high variance
- 이거에 대한 부분은 차후에 고민해보는걸로

### Policy Search

#### Policy Objective Functions

- Goal : given policy $\mathsf{\pi_\theta (s,a)}$ with parameters $\theta$, find best $\theta$
- But how do we measure the quality of a policy $\pi_\theta$?
- In episodic environments we can use the start value

$$ \mathsf{ J_1 (\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta} [v_1] } $$

- In continuing envrionments we can use the average value

$$ \mathsf{ J_{avV} (\theta) = \displaystyle\sum_s d^{\pi_\theta} (s) V^{\pi_\theta} (s) } $$

- Or the average reward per time-step

$$ \mathsf{ J_{avR} (\theta) = \displaystyle\sum_s d^{\pi_\theta} (s) \displaystyle\sum_a \pi_\theta (s,a) \mathcal{R}^a_s } $$

- where $\mathsf{d^{\pi_\theta} (s) }$ is stationary distribution of Markov chain for $\pi_\theta$
- Value 안쓴다더니 Value Function만 안쓰고 Value는 쓰는건가 approximation 만 안하는거네, 근데 state Value는 빼고 Action reward 만 가져다 쓰는

#### Policy Optimisation

- Policy based reinforcement learning is an optimisation problem
- Find $\theta$ that maximises $\mathsf{J(\theta)}$
- Some approaches do not use gradient
  - Hill climbing
  - Simplex / amoeba / Nelder Mead
  - Genetic algorithms
- Greater efficiency often possible using gradient
  - Gradient descent
  - Conjugate gradient
  - Quasi-newton
- We focus on gradient descent, many extensions possible
- And on methods that exploit sequential structure

## Finite Difference Policy Gradient

### Policy Gradient

- Let $\mathsf{J(\theta)}$ be a policy objective function
- Policy gradient algorithms search for a local maximum in $\mathsf{J(\theta)}$ by ascending the gradient of the policy, w.r.t parameters $\theta$

$$ \Delta \theta = \alpha \nabla _\theta \mathsf{J}(\theta) $$

- Where $\nabla _\theta \mathsf{J}(\theta)$ is the policy gradient

사실 이 방법의 local maximise 방법은 적절한 알파값을 찾는게 쉽지 않고 그래프 형태에 따라 달라지기 때문에 사실 그렇게 좋은 방법이라고 하긴 어렵다. 물론 최대한 작은 숫자를 넣어주면 안정적이기는 하지만 그런 만큼 느려지기도 한다. 최대한 간단한 방법의 접근이긴한데 나중에 보충하려나

$$ \mathsf{ \nabla _\theta J(\theta) } = \left( \begin{array} \; \mathsf{ \frac{\partial J(\theta)}{\partial \theta _1} } \\ \vdots \\ \mathsf{ \frac{\partial J(\theta)}{\partial \theta _n} } \end{array} \right) $$

- and $\alpha$ is a step-size parameter

조금 더 좋은 수치해석 접근방법이 있었던거 같은데 수치해석 다시 찾아볼까..

### Computing Gradients By Finite Differences

- To evaluate policy gradient of $\mathsf{\pi_\theta (s,a)}$
- For each dimension $\mathsf{k\in [1,n]}$
  - Estimate kth partial derivative of objective function w.r.t. $\theta$
  - By perturbing $\theta$ by small amount $\epsilon$ in kth dimension
  <br><center>$$ \mathsf{ \frac{\partial J(\theta)}{\partial \theta_k} \approx \frac{J(\theta + \epsilon u_k) - J(\theta)}{ \epsilon} } $$</center>
  <br> where $\mathsf{u_k}$ is unit vector with 1 in kth component, 0 elsewhere
- Uses n evaluations to compute policy gradient in n dimensions
- Simple, noisy, inefficient - but sometimes effective
- Works for arbitrary policies, even if policy is not differentiable

하나씩 조금씩 바꿔본다는 그런 방법

## Monte-Carlo Policy Gradient

### Likelihood Ratios

#### Score Function

- We now compute the policy gradient analytically
- Assume policy $\pi_\theta$ is differentiable whenever it is non-zero
- and we know the gradient $\nabla_\theta \pi_\theta \mathsf{(s,a)}$
- Likelihood ratios exploit the following identity

$$\begin{aligned}
\nabla_\theta \pi_\theta \mathsf{(s,a)} &= \mathsf{ \pi_\theta (s,a) \frac{ \nabla_\theta \pi_\theta (s,a) }{\pi_\theta (s,a)} } \\
&= \mathsf{ \pi_\theta (s,a) \nabla_\theta \log \pi_\theta (s,a) }
\end{aligned}$$

- The score function is $\nabla_\theta \log \pi_\theta \mathsf{(s,a)}$

여기서 Policy를 approximation, Score function 이라는 이름의 의미는 이 값이 policy의 각 확률에 대해서 gradient 보정 방향마다의 보정 크기 비율을 곱해주는 값이기 때문인 듯 Gradient = policy * score

#### Softmax Policy

- We will use a softmax policy as a running example
- Weight actions using linear combination of features $ \phi \mathsf{(s,a)^T} \theta $
- Probability of action is proportional to exponentiated weight

$$ \mathsf{ \pi_\theta (s,a) \propto e^{\phi (s,a)^T \theta} } $$

- The score function is

$$ \mathsf{ \nabla_\theta \log \pi_\theta (s,a) = \phi (s,a) - \mathbb{E}_{\pi_\theta}[\phi(s,\cdot)] } $$

과정이 안적혀있어서 나중에 찾아봐야

#### Gaussian Policy

- In continuous action spaces, a Gaussian policy is natural
- Mean is a linear combination of state features $\mathsf{\mu(s) = \phi(s)^T\theta}$
- Variance may be fixed $\rho^2$, or can also parametrised
- Policy is Gaussian, $\mathsf{a \sim \mathcal{N}(\mu(s),\rho^2)}$
- The scroe function is

$$ \mathsf{ \nabla_\theta \log \pi_\theta (s,a) = \frac{ (a-\mu(s))\phi(s) }{ \rho^2 } } $$

Continuous한 action을 선택해야하여 한 방법으로 Gaussain 형태를 사용한다

### Policy Gradient Theorem

#### One-Step MDPs

- Consider a simple class of one-step MDPs
  - Starting in state $\mathsf{s\sim d(s)}$
  - Terminating after on time-step with reward $\mathsf{r=\mathcal{R}_{s,a}}$
- Use likelihood ratios to compute the policy gradient

$$\begin{aligned}
\mathsf{J(\theta)} &= \mathbb{E}_{\pi_\theta} [\mathsf{r}] \\
&= \mathsf{\displaystyle\sum_{s\in \mathcal{S}} d(s) \displaystyle\sum_{a\in\mathcal{A}} \pi_\theta (s,a) \mathcal{R}_{s,a} } \\
\mathsf{\nabla_\theta J(\theta)} &= \mathsf{\displaystyle\sum_{s\in \mathcal{S}} d(s) \displaystyle\sum_{a\in\mathcal{A}} \pi_\theta (s,a) \nabla_\theta \log \pi_\theta (s,a) \mathcal{R}_{s,a} } \\
&=  \mathsf{\mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta (s,a) r]}
\end{aligned}$$

이거 그냥 Value Function 아니냐고

근데 왜 one step return에 next step state value는 은근슬쩍 빠지고 action reward만 남아있는가 자꾸 이랬다 저랬다 할거야? Terminateing state value가 0이라고 써주던지.. 만약에 state value 포함하면 sigma도 한개 더 들어가야함

#### Policy Gradient Theorem -

- The policy gradient theorem generalises the likelihood ratio approach to multi-step MDPs
- Replaces instantaneous reward r with long-term value $\mathsf{Q^\pi(s,a)}$ - r은 action reward 였는데 멋대로 Q action value로 바꾸기 있냐
- Policy gradient theorem applies to start state objective, average reward and average value objective

|Theorem|
|---|
|For any differentiable policy $\mathsf{\pi_\theta(s,a)}$,<br> for any of the policy objective functions $\mathsf{J = J_1, J_{avR} \text{, or} \frac{1}{1-\gamma}J_{avR}}$, <br> the policy gradient is<br><center>$$ \mathsf{ \nabla_\theta J(\theta) = {\color{Red} \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta (s,a) Q^{\pi_\theta}(s,a)]} } $$</center>|

하여튼 score와 action value 곱의 기대값이다. 증명은 생략되어 있으므로 나중에 책을 찾아본다, 사실상 sarsa나 Q-learning을 parameterise한 것인듯

#### Monte-Carlo Policy Gradient (REINFORCE)

- Update parameters by stochastic gradient ascent
- Using policy gradient theorem
- Using return $\mathsf{v_t}$ as an unbiased sample of $\mathsf{Q^{\pi_\theta} (s_t,a_t)}$

$$ \mathsf{\Delta \theta_t = \alpha \nabla_\theta \log \pi_\theta (s_t,a_t) v_t } $$

| function REINFORCE <br> $\quad$ Initialise $\theta$ arbitrarily <br> $\quad$ for each episode $\mathsf{ \lbrace s_q,a_q,r_2, \dots, s_{T-1}, a_{T-1}, r_T \rbrace \sim \pi_\theta }$ do <br>$\quad\quad$ for $\mathsf{t=1}$ to $\mathsf{T-1}$ do <br>$\quad\quad\quad$ $\mathsf{ \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta (s_t,a_t) v_t }$ <br>$\quad\quad$ end for <br>$\quad$ end for <br>$\quad$ return $\theta$ <br> end function |

에피소드 중 각 step에 대해서 score 곱하기 return으로 parameter를 업데이트한다. 오래전에 나온 알고리즘이라 딱 이 알고리즘을 Reinforce 알고리즘이라고 한다. 알파고1 에 쓰인 알고리즘

improvement가 부드럽지만 variance가 커서 느리다

## Actor-Critic Policy Gradient

### Reducing Variance Using a Critic

- Monte-Carlo policy gradient still has high variance
- We use a critic to estimate the action-value function,

$$ \mathsf{ Q_w (s,a) \approx Q^{\pi_\theta} (s,a)} $$

- Actor-critic algorithms maintain two sets of parameters
  - Critic - Updates action-value function parameters w
  - Actor - Updates policy parameters $\theta$, in direction suggested by critic
- Actor-critic algorithms follow an approximate policy gradient

$$\begin{aligned}
\mathsf{\nabla_\theta J(\theta)} &\approx \mathsf{\mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta (s,a) Q_w(s,a)]} \\
\Delta \theta &=  \mathsf{ \alpha \nabla_\theta \log \pi_\theta (s,a) Q_w(s,a) }
\end{aligned}$$

J 는 average return for time step인데 왜 return 값의 sample로 parameter를 업데이트 하는 부분 이거 자꾸 신경쓰이긴 한데.. alpha값 꽤 중요하겠다.

### Estimating the Action-Value Function

- The critic is solving a familiar probelm: policy evaluation
- How good is policy $\pi_\theta$ for current parameters $\theta$?
- This problem was explored in previous two lectures, e.g.
  - Monte-Carlo policy evaluation
  - Temporal-Difference learning
  - TD($\lambda$)
- Could also use e.g. least-squares policy evaluation

### Action-Value Actor-Critic

- Simple actor-critic algorithm based on action-value critic
- Using linear value fn approx. $\mathsf{Q_w(s,a) =\phi(s,a)^T w}$
  - Critic - Update w by linear TD(0)
  - Actor - Updates $\theta$ by policy gradient

| function QAC <br> $\quad$ Initialise $\mathsf{s}, \theta$ <br> $\quad$ Sample $\mathsf{ a \sim \pi_\theta }$ <br> $\quad$ For each step do <br>$\quad\quad$ Sample reward $\mathsf{r=\mathcal{R}^a_s}$; sample transition $\mathsf{s' \sim \mathcal{P}^a_s}$ <br>$\quad\quad$ Sample action $\mathsf{a' \sim \pi_\theta (s', a')}$ <br>$\quad\quad$ $\mathsf{ \delta = r + \gamma Q_w (s',a') - Q_w (s,a)}$ <br>$\quad\quad$ $\mathsf{ \theta = \theta + \alpha \nabla_\theta \log \pi_\theta (s,a) Q_w(s,a)}$ <br>$\quad\quad$ $\mathsf{ w \leftarrow w + \beta \delta \phi (s,a) }$ <br>$\quad\quad$ $\mathsf{ a \leftarrow a', s \leftarrow s' }$ <br>$\quad$ end for <br> end function |

policy와 Value를 둘다 동시에 학습한다는 건데 결국 하나씩 하나씩 조립한 것. TD error를 구하고 policy 업데이트하고 linear model에 error 곱해서 Q parameter weight 업데이트한다.

### Compatible Function Approximation

#### Bias in Actor-Critic Algorithms

- Approximatin the policy gradient introduces bias
- A biased policy gradient may not find the right solution
  - e.g. if $\mathsf{Q_w(s,a)}$ uses aliased features, can we solve gridworld example?
- Luckily, if we choose value function approximation carefully
- Then we can avoid introducing any bias
- i.e. We can still follow the exact policy gradient

Compatible Function Approximation

|Theorem (Compatible Function Approximation Theorem)|
|---|
|If the following two conditions are satisfied:<br> 1. Value function approximator is compatible to the policy<br><center>$$ \mathsf{ \nabla _w Q_w(s,a) = \nabla_\theta \log \pi_\theta (s,a) } $$</center><br>2. Value function parameters w  minimise the mean-squared error <br><center>$$ \mathsf{ \epsilon = \mathbb{E}_{\pi_\theta} [(Q^{\pi_\theta}(s,a)-Q_w(s,a))^2] } $$</center><br> Then the policy gradient is exact,<br><center>$$\mathsf{ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log_\theta \pi_\theta(s,a) Q_w (s,a)]} $$</center>|

왜 에러가 parameterised policy의 action Value와 parameterised Action Value의 차이가 되어있을까. 설명이 없어도 너무 없다. improved Value 와 이전 단계 Value의 에러가 되어야 하는 거 아닌가
근데 이부분 강의에서 안다루고 대충넘어감

#### Proof of Compatible Function Approximation Therorem

If w is chosen to minimise mean-squared error, gradient of $\epsilon$ w.r.t. w must be zero,

$$\begin{aligned}
\mathsf{ \nabla_w \epsilon } &= 0 \\
\mathsf{ \mathbb{E}_{\pi_\theta} [(Q^{\theta}(s,a)-Q_w(s,a))\nabla_w Q_w (s,a)] } &= 0 \\
\mathsf{ \mathbb{E}_{\pi_\theta} [(Q^{\theta}(s,a)-Q_w(s,a)) \nabla_\theta \log \pi_\theta (s,a)] } &= 0 \\
\mathsf{ \mathbb{E}_{\pi_\theta} [Q^{\theta}(s,a) \nabla_\theta \log \pi_\theta (s,a)] } &= \mathsf{ \mathbb{E}_{\pi_\theta} [Q_w(s,a) \nabla_\theta \log \pi_\theta (s,a)] }
\end{aligned}$$

So $\mathsf{Q_w(s,a)}$ can be substituted directly into policy gradient,

$$ \mathsf{ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log_\theta \pi_\theta(s,a) Q_w (s,a)] } $$

### Advantage Function Critic

#### Reducing Variance Using a Baseline

- We subtract a baseline function B(s) from the policy gradient
- This can reduce variance, without changing expectation

$$\begin{aligned}
\mathsf{  \mathbb{E}_{\pi_\theta} [\nabla_\theta \log_\theta \pi_\theta(s,a) B(s)] } &= \mathsf{ \displaystyle\sum_{s\in\mathcal{S}} d^{\pi_\theta}(s) \displaystyle\sum_{a} \nabla_\theta \pi_\theta(s,a)B(s) } \\
&= \mathsf{ \displaystyle\sum_{s\in\mathcal{S}} d^{\pi_\theta}B(s) \nabla_\theta \displaystyle\sum_{a\in\mathcal{A}} \pi_\theta(s,a) } \\
&= 0
\end{aligned}$$

- A good baseline is the state value function $\mathsf{B(s) = V^{\pi_\theta}(s)}$
- So we can rewrite the policy gradient using the advantage function  $\mathsf{A^{\pi_\theta}(s,a)}$

$$\begin{aligned}
\mathsf{A^{\pi_\theta}(s,a)} &= \mathsf{ Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s) } \\
\mathsf{ \nabla_\theta J(\theta)} &= {\color{Red} \mathsf{ \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s,a) A^{\pi_\theta}(s,a)] }}
\end{aligned}$$

그냥 action과 관계없는 state value 를 빼면 평균적으로 영향이 0에 수렴한다는 뜻이라서 충분한 수를 sample할 때는 괜찮다는 것

#### Estimating the Advantage Function

- The advantage function can significantly reduce variance of policy gradient
- So the critic should really estimate the advantage function
- For example, by estimating both $\mathsf{V^{\pi_\theta}(s)}$ and $\mathsf{Q^{\pi_\theta}(s,a)}$
- Using two function approximators and two parameter vectors,

$$\begin{aligned}
\mathsf{ V_v(s) } &\approx \mathsf{ V^{\pi_\theta}(s) } \\
\mathsf{ Q_w(s,a) } &\approx \mathsf{ Q^{\pi_\theta}(s,a) } \\
\mathsf{ A(s,a) } &= \mathsf{ Q_w(s,a)-V_v(s) }
\end{aligned}$$

- And updating both value functions by e.g. TD learning
-  For the true value function $\mathsf{V^{\pi_\theta}(s)}$, the TD error $\delta^{\pi_\theta}$

$$ \mathsf{ \delta^{\pi_\theta} = r+\gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s) } $$

- is an unbiased estimate of the advantage function

$$\begin{aligned}
\mathsf{ \mathbb{E}_{\pi_\theta}[\delta^{\pi_\theta} \vert s,a ] } &= \mathsf{ \mathbb{E}_{\pi_\theta}[ r+\gamma V^{\pi_\theta}(s') \vert s,a ] - V^{\pi_\theta}(s) } \\
&= \mathsf{ Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s) } \\
&= \mathsf{ A^{\pi_\theta}(s,a) }
\end{aligned}$$

- So we can use the TD error to compute the policy gradient

$$ \mathsf{ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[ \nabla_\theta \log \pi_\theta(s,a) \delta^{\pi_\theta} ] } $$

- In practice we can use an approximate TD error

$$ \mathsf{ \delta_v = r+\gamma V_v(s') - V_v(s) } $$

- This approach only requires one set of critic parameters v

여기까지 하면 State value와 Action Value, Policy까지 전부 parameterise하는 것이었지만. Advantage는 TD error와 같고 이걸로 policy gradient를 계산할 수 있다. 근데 Q가 아니라 V를 parameterise 한다는 건 dimension이 하나 줄어든다는 건데 이거에 관해서 전에 model free가 불가능하다고 했던거 같은데 이거 한번 다시 맞춰봐야겠다.

### Eligibility Traces

#### Critics at Different Time-Scales

- Critic can estimate value function $\mathsf{V_\theta(s)}$ from many targets at different time-scales
  - For MC, the target is the return $\mathsf{v_t}$
  <br><center>$$ \mathsf{ \Delta\theta = \alpha({\color{Red} v_t} - V_\theta(s))\phi(s) } $$ </center>
  - For TD(0), the target is the TD target $\mathsf{ r+\gamma V(s') }$
  <br><center>$$ \mathsf{ \Delta\theta = \alpha({\color{Red} r+\gamma V(s') } - V_\theta(s))\phi(s) } $$ </center>
  - For forward-view TD(\lambda), the target is the $\lambda$-return $\mathsf{v^\lambda_t}$
  <br><center>$$ \mathsf{ \Delta\theta = \alpha({\color{Red} v^\lambda_t} - V_\theta(s))\phi(s) } $$ </center>
  - For backward-view TD, we use eligibility traces
  <br><center>$$\begin{aligned}
  \mathsf{ \delta_t } &= \mathsf{ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) } \\
  \mathsf{ e_t } &= \mathsf{ \gamma\lambda e_{t-1} + \phi(s_t) } \\
  \mathsf{ \Delta \theta } &= \mathsf{ \alpha\delta_t e_t }
  \end{aligned}$$ </center>

#### Actors at Different Time-Scales

- The policy gradient can also be estimated at many time-scales

$$ \mathsf{ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[ \nabla_\theta \log \pi_\theta(s,a) {\color{Red} A^{\pi_\theta}(s,a) } ] } $$

- Monte-Carlo policy gradient uses error from complete return

$$ \mathsf{ \Delta\theta = \alpha({\color{Red} v_t} - V_v(s_t)) \nabla_\theta \log \pi_\theta(s_t,a_t) } $$

- Actor-critic policy gradient uses the one-step TD error

$$ \mathsf{ \Delta\theta = \alpha({\color{Red} r+\gamma V_v(s_{t+1})} - V_v(s_t)) \nabla_\theta \log \pi_\theta(s_t,a_t) } $$

#### Policy Gradient with Eligibility Traces

- Just like forward-view TD($\lambda$), we can mix over time-scales

$$ \mathsf{ \Delta\theta = \alpha({\color{Red} v^\lambda_t} - V_v(s_t)) \nabla_\theta \log \pi_\theta(s_t,a_t) } $$

- where $\mathsf{ v^\lambda_t - V_v(s_t)}$ is a biased estimate of advantage fn
- Like backward-view TD($\lambda$), we can also use eligibility traces
  - By equivalence with TD($\lambda$), substituting $\mathsf{\phi(s) = \nabla_\theta \log \pi_\theta(s,a)}$

$$\begin{aligned}
\mathsf{ \delta } &= \mathsf{ r_{t+1} + \gamma V_v(s_{t+1}) - V_v(s_t) } \\
\mathsf{ e_{t+1} } &= \mathsf{ \lambda e_t + \nabla_\theta \log \pi_\theta(s,a) } \\
\mathsf{ \Delta \theta } &= \mathsf{ \alpha\delta_t e_t }
\end{aligned}$$

- This update can be applied online, to incomplete sequences

### Natural Policy Gradient

#### Alternative Policy Gradient Directions

- Gradient ascent algorithms can follow any ascent direction
- A good ascent direction can significantly speed convergence
- Also, a policy can often be reparametrised without changing action probabilities
- For example, increasing score of all actions in a softmax policy
- The vanilla gradient is sensitive to these reparametrisations

Natural Policy Gradient

- The natural policy gradient is parametrisation independent
- It finds ascent direction that is closest to vanilla gradient, when changing policy by a small, fixed amount

$$\mathsf{ \nabla^{nat}_\theta \pi_\theta (s,a) = G^{-1}_\theta \nabla_\theta \pi_\theta(s,a)} $$

- where $\mathsf{G_\theta}$ is the Fisher information matrix

$$\mathsf{ G_\theta = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a)\nabla_\theta \log \pi_\theta(s,a)^T \right] }$$

#### Natural Actor-Critic

- Using compatible function approximation,

$$\mathsf{ \nabla_w A_w (s,a) = \nabla_\theta\log\pi_\theta(s,a) }$$

- So the natural policy gradient simplifies,

$$\begin{aligned}
\mathsf{ \nabla_\theta J(\theta) } &= \mathsf{ \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) A^{\pi_\theta}(s,a) \right] } \\
&= \mathsf{ \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a)\nabla_\theta \log \pi_\theta(s,a)^T w \right] } \\
&= \mathsf{ G_\theta w } \\
\mathsf{ \nabla^{nat}_\theta J(\theta) } &= \mathsf{ w } \\
\end{aligned}$$

- i.e. update actor parameters in direction of critic parameters

## Summary of Policy Gradient Algorithms

- The policy gradient has many equivalent forms

$$\begin{alignat*}{3}
\mathsf{ \nabla_\theta J(\theta) } &= \mathsf{ \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a)\; {\color{Red} v_t } \right] } \quad & \text{REINFORCE} \\
&= \mathsf{ \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \;{\color{Red} Q^w(s,a) } \right] } \quad & \text{Q Actor-Critic} \\
&= \mathsf{ \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \; {\color{Red} A^w(s,a) } \right] } \quad & \text{Advantage Actor-Critic} \\
&= \mathsf{ \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \;{\color{Red} \delta } \right] } \quad & \text{TD Actor-Critic} \\
&= \mathsf{ \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(s,a) \;{\color{Red} \delta e } \right] } \quad & \text{TD($\lambda$) Actor-Critic} \\
\mathsf{ G^{-1}_\theta\nabla_\theta J(\theta) } &= \mathsf{ w } & \text{Natural Actor-Critic}
\end{alignat*}$$

- Each leads a stochastic gradient ascent algorithm
- Critic uses policy evaluation (e.g. MC or TD learning) to estimate $\mathsf{Q^\pi (s,a), A^\pi(s,a) \text{ or } V^\pi(s)}$
