---
title: RLcourse note - Lecture 6 Value Function Approximation
author: DS Jung
date: 2021-12-20 17:00:00 +0900
categories: [RLcourse, Note]
tags: [reinforcementlearning, lecturenote]     # TAG names should always be lowercase
comment: true
math: true
mermail: false
pin: true
---

Video Link :

[![thumb1](/assets/pic/RL_lec6_thumb.JPG){: width="400px" height="200px"}](https://youtu.be/UoPei5o4fps)
{: .text-center}

## Introduction of Value Function Approximation

Reinforcement learning can be used to solve *large* problems, e.g.

### 6강 소감

6강은 어느정도 이해했지만 좀 찜찜한 부분들도 있다 슬슬 실전에 사용되는 내용에 많아지는 만큼 세심한 수식들이 많은데 나중에 다시한번 봐야할듯

### Value Function Approximation

- So far we gave represented value function by a lookup table
  - Every state s has an entry $\mathsf{V(s)}$
  - Or every state-action pair s, a has an entry $\mathsf{Q(s,a)}$
- Probelm with large MDPs:
  - There are too many states and/or actions to store in memory
  - It is too slow to learn the value of each state individually
- Solution for large MDPs:
  - Estimate value function with function approximation
<br><center>$$ \begin{aligned}
\mathsf{\hat{v}(s,w)} &\approx \mathsf{v_\pi(s)} \\
\mathsf{or \;\hat{q}(s,a,w)} &\approx \mathsf{q_\pi(s,a)}
\end{aligned}$$</center>
  - Generalise from seen states to unseen states
  - Update parameter w using MC or TD learning

Function Approximators

- Linear combinations of features
- Neural network
- Decision tree
- Nearest neighbour
- Fourier / wavelet bases
- ...

We require a training method that is suitable for non-stationary, non-iid data

## Incremental Methods

### Gradient Descent

- Let $\mathsf{J(w)}$ be a differentiable function of parameter vector w
- Define the gradient of $\mathsf{J(w)}$ to be
<br><center>$$ \mathsf{\nabla _w J(w)} = \left( \begin{array} \mathsf{ \frac{\partial J(w)}{\partial w_1} } \\ \vdots \\ \mathsf{ \frac{\partial J(w)}{\partial w_n} } \end{array} \right) $$ </center>
- To find a local minimum of $\mathsf{J(w)}$
- Adjust w in direction of -ve gradient
<br><center> $$ \mathsf{ \Delta w = -\frac{1}{2} \alpha \nabla _w J(w) } $$</center>
where $\alpha$ is a step-size parameter

Gradient의 minus값으로 향하는 vector이므로 local minimum을 찾게 됨

#### Value Funcion Approx. By Stochastic Gradient Descent

- Goal: find parameter vector w minimising mean-squared error between approximate value fn $\mathsf{\hat{v}(s,w)}$ and true value fn $\mathsf{v_\pi(s)}$

$$ \mathsf{ J(w) = \mathbb{E}_\pi [(v_\pi(S) - \hat{v} (S,w))^2] } $$

- Gradient descent finds a local minimum

$$ \begin{aligned}
\Delta \mathsf{w} &= \mathsf{-\frac{1}{2}\alpha \nabla _w J(w) } \\
&= \mathsf{ \alpha \mathbb{E}_\pi [(v_\pi(S) - \hat{v}(S,w))\nabla _w \hat{v} (S,w)] }
\end{aligned} $$

- Stochastic gradient descent samples the gradient

Error 의 Gradient descent를 취할 때 Target 은 fixed여야 한다

$$ \mathsf{ \Delta w = \alpha (v_\pi(S) - \hat{v}(S,w))\nabla _w \hat{v} (S,w) } $$

- Expected update is equal to full gradient update

### Linear Function Approximation

#### Feature Vectors

- Represent state by a feature vector

$$ \mathsf{ x(S)} = \left( \begin{array}\, \mathsf{x_1(S)} \\ \vdots \\ \mathsf{x_n(S)} \end{array} \right) $$

- For example:
  - Distance of robot from landmarks
  - Trends in the stock market
  - Piece and pawn configurations in chess

information의 집합으로 결국 가장 중요한 input임

#### Linear value Function Approximation

- Represent value function by a linear combination of feature

$$ \mathsf{ \hat{v}(S,w) = x(S)^T w=\displaystyle\sum^n_{j=1}x_j(S)w_j } $$

- Objective function is quadratic in parameters $\mathsf{w}$

$$ \mathsf{ J(w) = \mathbb{E}_\pi \left[ (v_\pi(S)-x(S)^T w)^2\right] } $$

- Stochastic gradient descent converges on global optimum
- Update rule is particularly simple

$$ \begin{aligned}
\mathsf{\nabla _w \hat{v}(S,w)} &= \mathsf{x(S)} \\
\mathsf{\Delta w} &= \mathsf{ \alpha (v_\pi(S) - \hat{v}(S,w))x(S) }
\end{aligned} $$

Update - step-size $\times$ prediction error $\times$ feature value

Linear한 경우를 굳이 보여주는 이유는 단순한 문제들에 대하여 문제를 Linear하게 설정해내서 해결해내는 경우가 많기 때문일듯

#### Table Lookup Features

- Table lookup is a special case of linear value function approximation
- Using table lookup features

$$ \mathsf{ x^{table}(S)} = \left( \begin{array}\, \mathsf{1(S=s_1)} \\ \vdots \\ \mathsf{1(S=s_n)} \end{array} \right) $$

- Parameter vector $\mathsf{w}$ gives value of each individual state

$$ \mathsf{\hat{v}(S,w)} = \left( \begin{array}\, \mathsf{1(S=s_1)} \\ \vdots \\ \mathsf{1(S=s_n)} \end{array} \right) \cdot \left( \begin{array}\, \mathsf{w_1} \\ \vdots \\ \mathsf{w_n} \end{array} \right) $$

이전 강의까지 하던 table lookup 방식의 연장선이라는 것

### Incremental Prediction Algorithm

- Have assumed true value function $\mathsf{v_\pi(s)}$ given by supervisor
- But in RL there is no supervisor, only rewards
- In practice, we substitute a target for $\mathsf{v_\pi(s)}$
  - For MC, the target is the return $\mathsf{G_t}$
  <br><center>$$ \mathsf{ \Delta w = \alpha({\color{red}G_t} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } $$ </center>
  - For TD(0), the target is the TD target $\mathsf{R_{t+1} + \gamma\hat{v}(S_{t+1},w)}$
  <br><center>$$ \mathsf{ \Delta w = \alpha({\color{red}R_{t+1} + \gamma\hat{v}(S_{t+1},w)} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } $$ </center>
  - For TD($\lambda$), the target is the $\lambda$-return $\mathsf{G^\lambda_t}$
  <br><center>$$ \mathsf{ \Delta w = \alpha({\color{red}G^\lambda_t} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } $$ </center>

target을 고르는 prediction 방법들<br>
E가 왜 feature를 더하는지도 생각해봐야

#### Monte-Carlo with Value Function Approximation

- Return $\mathsf{G_t}$ is an unbiased, noisy sample of true value $\mathsf{v_\pi(S_t)}$
- Can therefore apply supervised learning to "training data":

$$ \mathsf{ \langle S_1, G_1\rangle,\langle S_2,G_2\rangle, \dots , \langle S_T,G_T\rangle } $$

- For example, using linear Monte-Carlo policy evaluation

$$ \begin{aligned}
\mathsf{\Delta w} &= \mathsf{ \alpha({\color{red}G_t} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } \\
&= \mathsf{ \alpha(G_t - \hat{v}(S_t,w))x(S_t) }
\end{aligned} $$

- Monte-Carlo evaluation converges to a local optimum
- Even when using non-linear value function approximation

#### TD Learning with Value Function Approximation

- The TD-target $\mathsf{R_{t+1} + \gamma \hat{v}(S_{t+1},w)}$ is biased sample of true value $\mathsf{v_\pi(S_t)}$
- Can still apply supervised learning to "training data":

$$ \mathsf{ \langle S_1, R_2 + \gamma\hat{v}(S_2,w) \rangle,\langle S_2,R_3 + \gamma\hat{v}(S_3,w) \rangle, \dots , \langle S_T,R_T\rangle } $$

- For example, using linear TD(0)

$$ \begin{aligned}
\mathsf{\Delta w} &= \mathsf{ \alpha({\color{red}R + \gamma\hat{v}(S',w)} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } \\
&= \mathsf{ \alpha\delta x(S) }
\end{aligned} $$

- Linear TD(0) convergs (close) to global optimum

#### TD($\lambda$) with Value Function Approximation

- The $\lambda$-return $\mathsf{G^\lambda_t}$ is also a biased sample of true value $\mathsf{v_\pi(s)}$
- Can again apply supervised learning to "training data":

$$ \mathsf{ \langle S_1, G^\lambda_1\rangle,\langle S_2,G^\lambda_2\rangle, \dots , \langle S_{T-1},G^\lambda_{T-1}\rangle } $$

- Forward view linear TD($\lambda$)

$$ \begin{aligned}
\mathsf{\Delta w} &= \mathsf{ \alpha({\color{red}G^\gamma_t} - \hat{v}(S_t,w))\nabla _w\hat{v}(S_t,w) } \\
&= \mathsf{ \alpha({\color{red}G^\gamma_t} - \hat{v}(S_t,w))x(S_t) }
\end{aligned} $$

- Backward view linear TD($\lambda$)

$$ \begin{aligned}
\mathsf{\delta _t} &= \mathsf{ R_{t+1} +\gamma hat{v}(S_{t+1},w) - hat{v}(S_t,w) } \\
\mathsf{E_t } &= \mathsf{\gamma\lambda E_{t-1} +x(S_t) } \\
\mathsf{\Delta w} &= \alpha\delta _t E_t
\end{aligned} $$

이거 E는 모든 x에 대해서 정의되는거같은데 <S> 또는 <S,A> 로<br>
target은 왜 Gradient에 포함하지 않는가에 대해서 Time-reversal 이라고 설명하는데 이건 좀 고민을 해봐야겠다. 실제로 해봐도 그렇게 하면 결과가 나오지 않는다고 하는데.... <br>
Target 도 gradient를 취할 경우 %\lambda$가 크면 값이 아주 작아지거나 뒤바뀌기도 하는데 그러면 아예 의미가 달라져서 잘못된 접근이 되긴 한다.

Forward view and backward view linear TD($\lambda$) are equivalent

### Incremental Control Algorithm

#### Control with Value Function Approximation

Policy evaluation - Approximate policy evaluation $\mathsf{\hat{q}(\cdot, \cdot, w) \approx q_\pi}$
Policy improvement $\epsilon$-greedy policy improvement

#### Action-Value Function Approximation

- Approximate the action-value function

$$ \mathsf{ \hat{q}(S,A,w) \approx q_\pi(S,A) } $$

- Minimise mean-squred error between approximate action-value fn $\mathsf{\hat{q}(S,A,w)}$ and true action-value fn $\mathsf{q_\pi (S,A)}$

$$\mathsf{ J(w) = \mathbb{E}_\pi [(q_\pi(S,A) - \hat{q}(S,A,w))^2] }$$

- Use stochastic gradient descent to find a local minimum

$$ \begin{aligned}
\mathsf{-\frac{1}{2}\nabla _w J(w)} &= \mathsf{(q_\pi(S,A)-\hat{q}(S,A,w)) \Delta _w \hat{q}(S,A,w)} \\
\mathsf{\Delta w} &= \mathsf{\alpha(q_\pi(S,A)-\hat{q}(S,A,w)) \Delta _w \hat{q}(S,A,w)}
\end{aligned} $$

#### Linear Action-Value Function Approximation

- Represent state and action by a feature vector

$$ \mathsf{x(S,A)} = \left( \begin{array} \; \mathsf{x_1(S,A)} \\ \vdots \\ \mathsf{x_n(S,A)}  \end{array} \right) $$

- Represent action-value fn by linear combination of features

$$ \mathsf{ \hat{q}(S,A,w) = x(S,A)^T w = \displaystyle\sum^n_{j=1} x_j (S,A)w_j } $$

- Stochastic gradient descent update

$$ \begin{aligned}
\mathsf{ \nabla _w \hat{q}(S,A,w) } &= \mathsf{ x(S,A)} \\
\mathsf{\Delta w} &= \mathsf{\alpha(q_\pi(S,A)-\hat{q}(S,A,w))x(S,A)}
\end{aligned} $$

#### Incremental Control Algorithms

- Like prediction, we must substitute a target for $\mathsf{q_\pi(S,A)}$
  - For MC, the target is the return $\mathsf{G_t}$
  <br><center>$$ \mathsf{ \Delta w = \alpha ( {\color{red}G_t} - \hat{q}(S_t,A_t,w))\nabla _w \hat{q}(S_t,A_t,w) } $$ </center>
  - For TD(0), the target is the TD target $\mathsf{R_{t+1} + \gamma Q(S_{t+1}A_{t+1})}$
  <br><center>$$ \mathsf{ \Delta w = \alpha ( {\color{red}R_{t+1} + \gamma Q(S_{t+1}A_{t+1})} - \hat{q}(S_t,A_t,w))\nabla _w \hat{q}(S_t,A_t,w) } $$ </center>
  - For forward-view TD($\lambda$), target is the action-value $\lambda$-return
  <br><center>$$ \mathsf{ \Delta w = \alpha ( {\color{red}q^\lambda_t} - \hat{q}(S_t,A_t,w))\nabla _w \hat{q}(S_t,A_t,w) } $$ </center>
  - For backward-view TD($\lambda$), equialent update is
  <br><center>$$ \begin{aligned}
  \mathsf{\delta _t} &= \mathsf{ R_{t+1} +\gamma hat{v}(S_{t+1},A_{t+1},w) - hat{v}(S_t,A_t,w) } \\
  \mathsf{E_t } &= \mathsf{\gamma\lambda E_{t-1} +\nabla _w \hat{v}(S_t,A_t,w) } \\
  \mathsf{\Delta w} &= \alpha\delta _t E_t
  \end{aligned} $$

### Convergence

#### Convergence of Prodiction Algorithms

| On/Off-Policy | Algorithm | Table Lookup | Linear | Non-Linear |
| :---: | :---: | :---: | :---: | :---: |
| On-Policy | MC <br> TD(0) <br> TD($\lambda$) | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> X <br> X |
| Off-Policy | MC <br> TD(0) <br> TD($\lambda$) | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> X <br> X | $\checkmark$ <br> X <br> X |

#### Gradient Temporal-Difference Learning

- TD does not follow the gradient of any objective function
- This is why TD can diverge when off-policy or using non-linear function approximation
- Gradient TD follows true gradient of projected Bellman error

| On/Off-Policy | Algorithm | Table Lookup | Linear | Non-Linear |
| :---: | :---: | :---: | :---: | :---: |
| On-Policy | MC <br> TD(0) <br> Gradient TD | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> X <br> $\checkmark$ |
| Off-Policy | MC <br> TD(0) <br> Gradient TD | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> X <br> $\checkmark$ | $\checkmark$ <br> X <br> $\checkmark$ |

Gradient TD 가 뭔진 알아서 찾아보라는 건가

#### Convergence of Control Algorithms

| Algorithm | Table Lookup | Linear | Non-Linear |
| :---: | :---: | :---: | :---: |
| Monte-Carlo Control <br> Sarsa <br> Q-learning <br> Gradient Q-learning | $\checkmark$ <br>$\checkmark$ <br>$\checkmark$ <br>$\checkmark$ | ($\checkmark$) <br>($\checkmark$) <br>X <br>$\checkmark$ | X <br> X <br> X <br> X |

$(\checkmark) =$ chatters around near-optimal value function

## Batch Methods

Batch Reinforcement Learning

- Gradient descent is simple and appealing
- But it is not sample efficient
- Batch methods seek to find the best fitting value function
- Given the agent's experience ("training data")

### Least Squares Prediction

- Given value function approximation $\mathsf{\hat{v}(s,w) \approx v_\pi(s)}$
- And experience $\mathcal{D}$ consisting of <state, value> pairs

$$ \mathsf{ {\cal D} = \lbrace \langle s_1,v_1^\pi \rangle, \langle s_2,v_2^\pi \rangle , \dots, \langle s_T,v_T^\pi \rangle \rbrace }$$

- Which parameters $\mathsf{w}$ give the best fitting value fn $\mathsf{\hat{v}(s,w)}$?
- Least squares algorithms find parameter vector w minimising sum-squared error between $\mathsf{\hat{v}(s_t,w)}$ and target values $\mathsf{v^\pi_t}$

$$ \begin{aligned}
  \mathsf{ LS(w) } &= \mathsf{ \displaystyle\sum^T_{t=1} (v^\pi_t - \hat{v}(s_t, w))^2 } \\
  &= \mathsf{ \mathbb{E}_\mathcal{D} [(v^\pi - \hat{v}(s,w))^2] }
  \end{aligned} $$

#### Stochastic Gradient Descent with Experience Replay

Given experience consisting of <state, value> pairs

$$ \mathsf{ \mathcal{D} = \lbrace \langle s_1, v^\pi_1 \rangle , \langle s_2, v^\pi_2 \rangle , \dots , \langle s_T, v^\pi_T \rangle \rbrace } $$

Repeat:

1. Sample state, value from experience
<br><center>$$ \mathsf{ \langle s, v^\pi \rangle \sim \mathcal{D} } $$</center>
2. Apply stochastic gradient descent update
<br><center>$$ \mathsf{ \Delta w = \alpha (v^\pi - \hat{v} (s,w)) \nabla _w \hat{v} (s,w) } $$</center>

Converges to least squares solution

$$ \mathsf{ w^\pi = \underset{w}{argmin} \; LS(w) } $$

#### Experience Replay in Deep Q-Networks (DQN)

DQN uses experience replay and <font color='red'>fixed Q-targets</font>

- Take action $\mathsf{a_t}$ according to $\epsilon$-greedy policy
- Store transition $\mathsf{(s_t, a_t, r_{t+1}, s_{t+1})}$ in replay memory $\mathcal{D}$
- Sample random mini-batch of transitions $\mathsf{(s, a, r, s')}$ from $\mathcal{D}$
- Compute !-learning targets w.r.t. old, fixed parameters $w^-$
- Optimise MSE between Q-network and Q-learning targets

$$ \mathsf{ \mathcal{L}_i(w_i) = \mathbb{E}_{s,a,r,s' \sim \mathcal{D}_i} \left[ \left( r+\gamma \,\underset{a'}{max} \; Q(s', a'; w^-_i ) - Q(s,a,;w_i) \right)^2 \right] } $$

- Using variant of stochastic gradient descent

#### DQN in Atari

- End-toend learning of values Q(s,a) from pixels s
- Input state s is stack of raw pixels from last 4 frames
- Output is $\mathsf{Q(s,a)}$ for 18 joystick/button positions
- Reward is change in score for that step

#### Linear Least Squares Prediction

- Experience replay finds least squares solution
- But it may take many iterations
- Using linear value funciton approximation $\mathsf{\hat{v}(s,w) = x(s)^T w}$
- We can solve the least squares solution directly
- At minimum of $\mathsf{LS(w)}$, the expected update must be zero

$$\begin{aligned}
\mathbb{E}_\mathcal{D} [\Delta w] &= 0 \\
\mathsf{ \alpha \displaystyle\sum^T_{t=1} x(s_t) (v^\pi_t - x(s_t)^T w ) } &= 0 \\
\mathsf{\displaystyle\sum^T_{t=1} x(s_t) v^\pi_t} &= \mathsf{ \displaystyle\sum^T_{t=1} x(s_t)x(s_t)^T w } \\
\mathsf{w} &= \mathsf{ \left( \displaystyle\sum^T_{t=1} x(s_t)x(s_t)^T \right)^{-1} \displaystyle\sum^T_{t=1} x(s_t)v^\pi_t }
\end{aligned}$$

- For N features, direct solution time is $\mathsf{O(N^3)}$
- Incremental solution time is $\mathsf{O(N^2)}$ using Shermann-Morrison

#### Linear Least Squares Prediction Algorithms

- We do not know true value $\mathsf{v^\pi_t}$
- In practice, our "training data" must use noisy or biased samples of $\mathsf{v^\pi_t}$

$$\begin{alignat*}{2}
&\color{ProcessBlue}{\text{LSMC}}\; \; && \text{Least Squares Monte-Carlo uses return} \\
& \; && \mathsf{v^pi_t \approx \color{Red}{G_t}} \\
&\color{ProcessBlue}{\text{LSTD}}\; \; && \text{Least Squares Temporal-Difference uses TD target} \\
& \; && \mathsf{v^pi_t \approx \color{Red}{R_{t+1} + \gamma \hat{v} (S_{t+1}, w)}} \\
&\color{ProcessBlue}{\text{LSTD} (\lambda)}\; \; && \text{Least Squares TD($\lambda$) uses $\lambda$ -return} \\
& \; && \mathsf{v^pi_t \approx \color{Red}{G^\lambda_t}} \\
\end{alignat*}$$

- In each case solve directly for fixed point of MC / TD / TD($\lambda$)

$$\begin{align*}
\color{ProcessBlue}{\text{LSMC}} && \; 0\; & \mathsf{= \displaystyle\sum^T_{t=1} \alpha (G_t - \hat{v}(S_t,w))x(S_t)} \\
&& \; w\; & \mathsf{= \left( \displaystyle\sum^T_{t=1} x(S_t) x(S_t)^T \right)^{-1} \displaystyle\sum^T_{t=1}x(S_t)G_t } \\
\color{ProcessBlue}{\text{LSTD}} && \; 0\; & \mathsf{= \displaystyle\sum^T_{t=1} \alpha (R_{t+1} + \gamma \hat{v} (S_{t+1}, w) - \hat{v}(S_t,w))x(S_t)} \\
&& \; w\; & \mathsf{= \left( \displaystyle\sum^T_{t=1} x(S_t)( x(S_t) - \gamma x(S_{t+1}))^T \right)^{-1} \displaystyle\sum^T_{t=1}x(S_t)R_{t+1} } \\
\color{ProcessBlue}{\text{LSTD} (\lambda)} && \; 0\; & \mathsf{= \displaystyle\sum^T_{t=1} \alpha \delta _t E_t} \\
&& \; w\; & \mathsf{= \left( \displaystyle\sum^T_{t=1} E_t( x(S_t) - \gamma x(S_{t+1}) )^T \right)^{-1} \displaystyle\sum^T_{t=1} E_t R_{t+1} } \\
\end{align*}$$

#### Convergence of Linear Least Squares Prediction Algorithms

| On/Off-Policy | Algorithm | Table Lookup | Linear | Non-Linear |
| :---: | :---: | :---: | :---: | :---: |
| On-Policy | MC <br> LSMC <br> TD <br> LSTD | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> - <br> X <br> - |
| Off-Policy | MC <br> LSMC <br> TD <br> LSTD | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | $\checkmark$ <br> $\checkmark$ <br> X <br> $\checkmark$ | $\checkmark$ <br> - <br> X <br> - |

### Least Squares Control

#### Least Squares Policy Iteration

Policy evaluation - Policy evaluation by least squares Q-learning  \\
Policy improvement Greedy policy improvement

#### Least Squares Action-Value Function Approximation

- Approximate action-value function $\mathsf{q_\pi(s,a)}$
- using linear combination of features $\mathsf{x(s,a)}$

$$ \mathsf{ \hat{q}(s,a,w) = x(s,a)^T w \approx q_\pi(s,a) } $$

- Minimise least squares error between $\mathsf{\hat{q}(s,a,w)}$ and $\mathsf{q(s,a)}$
- from experience generated using policy $\pi$
- consisting of <(state, action), value> pairs

$$ \mathsf{ \mathcal{D} = \lbrace \langle (s_1,a_1), v^\pi_1\rangle, \langle (s_2,a_2), v^\pi_2\rangle, \dots, \langle (s_T,a_T), v^\pi_T\rangle \rbrace } $$

#### Least Square Control

- For policy evaluation, we want to efficiently use all experience
- For control, we also want to improve the policy
- This experience is generated from many policies
- So to evaluate $\mathsf{q_\pi(S,A)}$ we must learn off-policy
- We use the same idea as Q-learning:
  - Use experience generated by old policy
  <br>$$ \mathsf{ S_t, A_t, R_{t+1}, S_{t+1} \sim \pi_{old} } $$
  - Consider alternative successor action $\mathsf{A' = \pi_{new}(S_{t+1})}$
  - Update $\mathsf{\hat{q}(S_t,A_t,w)}$ towards value of alternative action
  <br>$$ \mathsf{ R_{t+1} + \gamma \hat{q}(S_{t+1}, A', w) } $$

#### Least Squares Q-Learning

- Consider the following linear Q-learning update

$$\begin{aligned}
\delta &= \mathsf{ R_{t+1} + \gamma \hat{q} (S_{t+1}, \pi(S_{t+1}),w) - \hat{q}(S_t,A_t,w) } \\
\Delta \mathsf{w} &= \mathsf{ \alpha \delta x(S_t, A_t) }
\end{aligned}$$

- LSTDQ algorithm: solve for total update = zero

$$\begin{aligned}
0 &= \mathsf{ \displaystyle\sum^T_{t=1} \alpha(R_{t+1} + \gamma \hat{q} (S_{t+1}, \pi(S_{t+1}),w) - \hat{q}(S_t,A_t,w)) x(S_t,A_t) } \\
\mathsf{w} &= \mathsf{ \left( \displaystyle\sum^T_{t=1} x(S_t,A_t)(x(S_t,A_t) - \gamma x(S_{t+1}, \pi(S_{t+1})))^T \right)^{-1} \displaystyle\sum^T_{t=1} x(S_t, A_t)R_{t+1} }
\end{aligned}$$

#### Least Squares Policy Iteration Algorithm

- The following pseudocode uses LSTDQ for policy evaluation
- It repeatedly re-evaluates experience $\mathcal{D}$ with different policies

| function LSPI-TD($\mathcal{D}, \pi_0$) <br> $\quad$ $\pi' \leftarrow \pi_0$ <br> $\quad$ Repeat <br>$\quad\quad$ $\pi \leftarrow \pi'$ <br>$\quad\quad$ $$\mathsf{ Q \leftarrow LSTDQ(\pi, \mathcal{D})} $$ <br>$\quad\quad$ for all $\mathsf{s} \in \mathcal{S}$ do <br>$\quad\quad\quad$ $ \mathsf{ \pi'(s) \leftarrow \underset{a\in\mathcal{A}}{argmax}\; Q(s,a) } $ <br>$\quad\quad$ end for <br>$\quad$ until $(\pi \approx \pi')$ <br>$\quad$ return $\pi$ <br> end function |

#### Convergence of Control Algorithms

| Algorithm | Table Lookup | Linear | Non-Linear |
| :---: | :---: | :---: | :---: |
| Monte-Carlo Control <br> Sarsa <br> Q-learning <br> LSPI | $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ <br> $\checkmark$ | ($\checkmark$) <br> ($\checkmark$) <br> X <br> ($\checkmark$) | X <br> X <br> X <br> - |

($\checkmark$) = chatters around near-optimal value function
