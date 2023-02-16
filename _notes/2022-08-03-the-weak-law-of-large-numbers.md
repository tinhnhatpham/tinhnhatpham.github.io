---
layout: post
comments: false
title: The Weak Law of Large Numbers
categories: [Likelihood Estimation]
---

### Convergence in Probability

The sequence of random variables:

$$
  X_1,X_2,X_3,...
$$

converges in probability to a radom variable $X$ if, for any of $\epsilon \gt 0$

$$
  \lim_{n\to\infty}P(\mid X_n-X\mid  \gt \epsilon) = 0
$$

We write $X_n \stackrel{P}{\rightarrow}X$.

---

An Integral Notation:

* Rewrite

  $$
    \int_{0}^{2}f(x)dx = \int_{A}f(x)dx
  $$

  Where $A = \{x:0 \le x \le 2\}$.

Suppose that $X$ has the exponential distribution with rate $\lambda$.

How can we find $P(\mid \sin(X)\mid  \gt 1/2)$?

We can have many ways to do it.

1. We can define a new random variable $Y=\mid \sin(X)\mid $, try to find its pdf and then

  $$
    \begin{align}
      P(\mid \sin(X)\mid  > 1/2) &= P(Y > 1/2)\\
      &= \int_{1/2}^{\infty}f_Y(y)dy
    \end{align}
  $$

2. We can intergrate the pdf for $X$ over the relevant region.
  ![](\assets\images\convergence-example-sin-x.jpg)

  $$
    \begin{align}
      &= \int_{\sin^{-1}(1/2)}^{\pi-sin^{-1}(1/2}f_X(x)dx + \int_{\pi+\sin^{-1}(1/2)}^{2\pi-sin^{-1}(1/2}f_X(x)+...\\
      &= \int_{\{x:\mid sin(x)\mid \gt 1/2\}}f_X(x)dx
    \end{align}
  $$

  **An Inequality:**

  Let $X$ be a random variable. Let $g$ be a no-negative function and let $c\gt 0$. Then

  $$
      P(g(X) \ge c) \le \frac{E[g(X)]}{c}
  $$

  When $g(x) = \mid x\mid $, this is known as **Markov's inequality**.

  Proof:

  $$
    \begin{align}
      E[g(X)] &= \int_{-\infty}^{\infty}g(x)f_X(x)dx\\
      &= \int_{\{x:g(x)\ge c\}}g(x)f_X(x)dx+\int_{\{x:g(x)\ge c\}}g(x)f_X(x)dx
    \end{align}
  $$

  We know that $g(X)$ is non-negative, so the pdf is also non-negative. So both of these integrands above are none-negative, and there for the integrals are non-negative.

  $$
    \begin{align}
      E[g(X)] &\ge \int_{\{x:g(x)\ge c\}}g(x)f_X(x)dx\\
      &\ge \int_{\{x:g(x)\ge c\}}cf_X(x)dx\\
      &= c\int_{\{x:g(x)\ge c\}}f_X(x)dx = cP(g(X) \ge c)
    \end{align}
  $$

  **Chebyshev's Inequality:**

  Let $X$ be a random variable with mean $\mu$ and variance $\sigma^2 \lt \infty$. Let $k \gt 0$. Then

  $$
    P(\mid X - \mu\mid  \ge k\sigma) \le \frac{1}{k^2}
  $$

  or, equivalently,

  $$
    P(\mid X - \mu\mid  \lt k\sigma) \ge 1-\frac{1}{k^2}
  $$

  This is the probability that $X$ is within $k$ standard deivations of its mean.

  Proof:

  $$
    \begin{align}
      P(\mid X - \mu\mid  \ge k\sigma) &= P((\underbrace{X-\mu}_{g(x)})^2 \ge \underbrace{k^2\sigma^2}_{c})\\
      \le \frac{E[g(X)]}{c}&=\frac{E[(X-\mu^2)^2]}{k^2\sigma^2}=\frac{\sigma^2}{k^2\sigma^2}\\
      &= \frac{1}{k^2}
    \end{align}
  $$

---

### The Weak Law of Large Numbers

Suppose that $X_1,X_2,X_3,...$ is a sequence of iid random variables from any distribution with mean $\mu$ and variance $\sigma^2 \lt \infty$.

Then

$$
  \bar{X} \stackrel{P}{\rightarrow} \mu
$$

**Proof**: Let $\epsilon \gt 0$.

Chebyshev: $P(\mid X - \mu\mid  \ge k\sigma) \le \frac{1}{k^2}$

Here:

$$
  P(\mid \bar{X} - \mu_{\bar{X}}\mid  \ge k\sigma_{\bar{X}}) \le \frac{1}{k^2}
$$

$$
  P
$$

Which is:

$$
  P(\mid \bar{X} - \mu\mid  \ge k\sigma/\sqrt{n}) \le \frac{1}{k^2}
$$

Choose $k$ so that $k\sigma/\sqrt{n} = \epsilon$

$$
  P(\mid \bar{X} - \mu\mid  \ge \epsilon) \le \frac{1}{(\epsilon \sqrt{n}/\sigma)^2} = \frac{\sigma^2}{\epsilon^2n}
$$

Now we take the limits of both sides w.r.t $n$ as $n$ go to infinity.

$$
  \lim_{n\to\infty}P(\mid \bar{X}-\mu\mid \ge\epsilon) \le \lim_{n\to\infty}\frac{\sigma^2}{\epsilon^2n}=0
$$

We can conclude that the limit probability of the left-hand side is at least less than or equal to 0. But it's probability, so it's always between 0 and 1, so it must be 0.

$$
  \lim_{n\to\infty}P(\mid \bar{X}-\mu\mid \ge\epsilon) = 0 \implies \bar{X} \stackrel{P}{\rightarrow} \mu
$$

### Example

$$
  X_1,X_2,X_3,... \stackrel{iid}{\sim} exp(\text{rate}=\lambda)
$$

We can say that the sample mean $\bar{X}$ converges in probability to $1/\lambda$.

$$
  \bar{X} \stackrel{P}{\rightarrow} 1/\lambda
$$

---

$$
  X_1,X_2,X_3,... \stackrel{iid}{\sim} \gamma(\alpha,\beta)
$$

$$
  \bar{X} \stackrel{P}{\rightarrow} \alpha/\beta
$$