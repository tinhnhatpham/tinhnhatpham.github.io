---
layout: post
comments: false
title: The Central Limit Theorem
categories: [Likelihood Estimation]
---

Any linear combination of normal random variables is again normal.

* $X_1,X_2,...,X_n$ normal

  $\implies a + \sum_{i=1}^{n}a_iX_i$ normal

* This includes linear transformations of single normal random variables.

  $$
    X \sim N(\mu,\sigma^2) \quad \implies \quad Z:=\frac{X-\mu}{\sigma} \\
  $$

  $\implies Z \sim N(0,1) \quad \text{Standard Normal Dist.}$

  We can tranform the other way, from standard normal dist. to normal dist.

  $Z \sim N(0,1) \quad \implies \quad X:=\sigma Z+\mu\sim N(\mu,\sigma^2)$

**Normal Distribution**

$$
  X \sim N(\mu,\sigma^2)\\
  \implies f_X(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2\sigma^2}(x-\mu)^2} 
$$

**Standard Normal Distribution**

$$
  X \sim N(0,1)\\
  \implies f_X(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}z^2} 
$$

The cdf cannot be written down in closed form.

Notation:
  
$$
  Z \sim N(0,1) \quad \text{Standard Normal} \\
  \phi(z) = P(Z \le z)
$$

Can be intergrated numerically.

* Standard normal table.
* R code: pnorm()

Example: pnorm($1.23$) will give us $0.891$.

---

### Example 1

Suppose that $X\sim N(1,4)$.

Find $P(X\le 2)$.

$$
  \begin{align}
    P(X\le 2) &= P\Bigg(\frac{X-\mu}{\sigma}\le\frac{2-1}{\sqrt{4}}\Bigg)\\
    &=P(Z\le 0.5)\\
    &\approx 0.6915
  \end{align}
$$

### Example 2

Suppose that $X_1,X_2,...,X_n\sim N(1,4)$. Find $P(\bar{X}\le 2)$.

* \bar{X} has a normal distribution.
* $E[\bar{X}] = E[X_1] = 1$.
* $Var[\bar{X}] = \frac{Var(X_1)}{n} = \frac{4}{3}$.

$$
  \begin{align}
    P(\bar{X}\le 2) &= P\Bigg(\frac{\bar{X}-\mu_\bar{X}}{\sigma_\bar{X}}\le\frac{2-1}{2/\sqrt{3}}\Bigg)\\
    &= P(Z\le\sqrt{3}/2)\\
    &\approx 0.8068
  \end{align}
$$

---

### Convergence in Distribution

Let $X_1,X_2,...,X_n$ be a sequence of random variables when $X_n$ has some cdf 

$F_n(X)=P(X_n\le x)$.

Let $X$ be a random variable with cdf 

$F(X)=P(X\le x)$

The sequence **converges in distribution** if

$$
  \lim_{n\to\infty}F_n(X) = F(x)
$$

at all points of continuity of $F$.

We write $X_n \stackrel{d}{\rightarrow}X$.

This is a **weak** form of convergence.

The random variables in the sequence are **not** getting closer to each other.

Their **distribution** are getting close!

---

### The Central Limit Theorem

Let $X_1,X_2,X_3...$ be a sequence of random variables from any distribution with mean $\mu$ and variance $\sigma^2\lt\infty$.

Let

$$
  \bar{X}_n=\frac{1}{n}\sum_{i=1}^{n}X_i
$$

Then

$$
  \frac{\bar{X}_n-\mu}{\sigma/\sqrt{n}} \stackrel{d}{\rightarrow}N(0,1)
$$

**Definition/Notation**

A random variable $X_n$ is **asymptotically normal** if there exists sequences $\{a_n\}$ and $\{b_n\}$ of real numbers such that

$$
  \frac{X_n-a_n}{\sqrt{b_n}}\stackrel{d}{\rightarrow}N(0,1)
$$

We write $X_n\stackrel{asymp}{\sim}N(a_n,b_n)$.

Note: This does not mean that $X_n \rightarrow N(a_n,b_n)$.

The **Central Limit Theorem** is saying that $\bar{X}_n$ is **asymptotically normal**.

We write

$$
  \bar{X}_n\stackrel{asymp}{\sim}N(\mu,\sigma^2/n)
$$

**Example**

Let $\bar{X}$ be the sample mean for a random sample of size 100 from the $\gamma(3,2)$ distribution.

What is the approximate probability that $\bar{X}$ is greater than 1.4?

We already know that $\bar{X}$ has

* mean $E[\bar{X}] = E[X_1] = 3/2$
* variance $Var[\bar{X}]=\frac{Var[X_1]}{n}=\frac{3/4}{100}=\frac{3}{400}$

By the **CLT**, for this "large sample" $(n\ge30)$, the distribution of $\bar{X}$ is approximately normal.

So, we have that $\bar{X}$ has an approximately normal distribution with mean $3/2$ and variance $3/400$.

$$
  \begin{align}
    P(\bar{X}\gt1.4)&=P\Bigg(\frac{\bar{X}-\mu_\bar{X}}{\sigma_\bar{X}}\gt\frac{1.4-1.5}{\sqrt{3/400}}\Bigg)\\
    &\approx P(Z \gt-1.15)\\
    &\approx 0.87
  \end{align}
$$

$$
  \begin{align}
    P(Z\gt-1.15)&=1-P(Z\le-1.15)\\
    &=1-\phi(-1.15)\\
    &\approx0.8749
  \end{align}
$$

Computed in R using: ```1-pnorm(-1.15)```

Note that:

$$
  \frac{\bar{X}-\mu_\bar{X}}{\sigma_\bar{X}} = \frac{\bar{X}-\mu}{\sigma/\sqrt{n}}
$$